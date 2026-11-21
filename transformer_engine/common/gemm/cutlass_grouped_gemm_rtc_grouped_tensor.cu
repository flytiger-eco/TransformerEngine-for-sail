/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <climits>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "config.h"
#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#include "cutlass/gemm/kernel/nvte_grouped_tensor.hpp"
#include "cutlass_grouped_gemm_rtc.h"
#include "cutlass_grouped_gemm_rtc_kernel.h"
#include "ppu_adaptive_tile.hpp"

namespace transformer_engine {
namespace grouped_gemm_rtc {
namespace {
using cutlass::gemm::kernel::NativeControl;
using cutlass::gemm::kernel::NativeExpert;
using cutlass::gemm::kernel::NativeParams;

struct NativeStorage {
  void* base;
  uint64_t capacity;  // elements
  const int64_t *first, *last, *offsets;
  int64_t common_first, common_last;
};

struct DiscreteOperands {
  NativeStorage* device;
  std::vector<NativeStorage> host;
};

struct NativeSetup {
  NativeStorage a, b, c, d;
  const NativeStorage *b_list, *c_list, *d_list;
  NativeExpert* experts;
  NativeControl* control;
  const float *alpha, *beta;
  int groups, scalar_count, ta, tb, output_bytes, sms, tile_m_shift, tile_n_shift;
  bool has_c, packed_in_place, ne4;
};

__device__ __forceinline__ void require_metadata(bool valid) {
  if (!valid) __assert_fail("invalid grouped GEMM metadata", __FILE__, __LINE__, __func__);
}

struct MatrixSlice {
  uint64_t ptr;
  int rows, cols;
};

// CSR, uniform and discrete operands have independent addresses. Only ragged
// storage without offsets needs a running prefix in expert order.
__device__ __forceinline__ MatrixSlice matrix_slice(const NativeStorage& packed,
                                                    const NativeStorage* list, int group, int bytes,
                                                    uint64_t& running) {
  const auto& s = list ? list[group] : packed;
  int64_t rows = s.first ? s.first[group] : s.common_first;
  int64_t cols = s.last ? s.last[group] : s.common_last;
  require_metadata(rows >= 0 && cols >= 0 && rows <= INT_MAX - 256 && cols <= INT_MAX - 256);
  uint64_t count = uint64_t(rows) * cols;
  uint64_t offset = 0;
  if (s.offsets) {
    int64_t start = s.offsets[group], end = s.offsets[group + 1];
    require_metadata(start >= 0 && end >= start && uint64_t(end) <= s.capacity &&
                     count <= uint64_t(end - start));
    offset = uint64_t(start);
  } else if (!list) {
    if (s.first || s.last) {
      offset = running;
      running += count;
    } else {
      require_metadata(__umul64hi(uint64_t(group), count) == 0);
      offset = uint64_t(group) * count;
    }
  }
  require_metadata(offset <= s.capacity && count <= s.capacity - offset);
  // The host checked the byte range of the complete allocation.
  uint64_t ptr = reinterpret_cast<uint64_t>(s.base) + offset * bytes;
  require_metadata(count == 0 || (ptr != 0 && ptr % 16 == 0));
  return {ptr, int(rows), int(cols)};
}

__host__ __device__ bool disjoint_ranges(uint64_t a, uint64_t bytes_a, uint64_t b,
                                         uint64_t bytes_b) {
  return bytes_a == 0 || bytes_b == 0 || (a <= b ? bytes_a <= b - a : bytes_b <= a - b);
}

using ByteRange = std::pair<uint64_t, uint64_t>;  // Address and size in bytes.

ByteRange byte_range(const void* p, uint64_t count, size_t element_bytes) {
  const uint64_t start = reinterpret_cast<uintptr_t>(p);
  NVTE_CHECK(count <= (UINT64_MAX - start) / element_bytes, "grouped storage byte range overflow");
  return {start, count * element_bytes};
}

bool disjoint(ByteRange x, ByteRange y) {
  return disjoint_ranges(x.first, x.second, y.first, y.second);
}

void check_device_pointer(const void* p, int device, size_t alignment = 1) {
  if (!p) return;
  cudaPointerAttributes attrs{};
  NVTE_CHECK_CUDA(cudaPointerGetAttributes(&attrs, p));
  NVTE_CHECK(attrs.type == cudaMemoryTypeDevice && attrs.device == device &&
                 reinterpret_cast<uintptr_t>(p) % alignment == 0,
             "grouped storage requires aligned pointers on the current device");
}

const int64_t* checked_metadata(const SimpleTensor& tensor, size_t count, int device) {
  if (!tensor.dptr) {
    NVTE_CHECK(tensor.numel() == 0, "null grouped metadata with nonzero length");
    return nullptr;
  }
  NVTE_CHECK(tensor.dtype == DType::kInt64 && tensor.shape.size() == 1 && tensor.shape[0] == count,
             "grouped metadata must be rank-1 int64 with the expected length");
  check_device_pointer(tensor.dptr, device, alignof(int64_t));
  return static_cast<const int64_t*>(tensor.dptr);
}

NativeStorage grouped_storage(const GroupedTensor* tensor, DType dtype, size_t groups, int device) {
  if (!tensor) return {};
  NVTE_CHECK(tensor->num_tensors == groups && tensor->logical_shape.ndim == 2 &&
                 tensor->data.dtype == dtype && tensor->data.shape.size() == 1 &&
                 !tensor->scale_inv.dptr && !tensor->columnwise_scale_inv.dptr,
             "native grouped GEMM requires rank-1 unquantized grouped storage");
  NativeStorage result{};
  result.base = tensor->data.dptr;
  result.capacity = tensor->data.numel();
  check_device_pointer(result.base, device);
  result.first = checked_metadata(tensor->first_dims, groups, device);
  result.last = checked_metadata(tensor->last_dims, groups, device);
  result.offsets = checked_metadata(tensor->tensor_offsets, groups + 1, device);
  if (!result.first) {
    NVTE_CHECK(result.last || tensor->logical_shape.data[0] % groups == 0,
               "native grouped GEMM uniform rows are not divisible by group count");
    auto value = tensor->get_common_first_dim();
    NVTE_CHECK(value <= size_t(INT_MAX - 256), "native grouped GEMM row dimension overflow");
    result.common_first = value;
  }
  if (!result.last) {
    NVTE_CHECK(tensor->logical_shape.data[1] <= size_t(INT_MAX - 256),
               "native grouped GEMM column dimension overflow");
    result.common_last = tensor->logical_shape.data[1];
  }
  NVTE_CHECK(result.capacity == 0 || result.base,
             "native grouped GEMM nonempty storage has a null pointer");
  return result;
}

std::pair<int64_t, int64_t> average_shape(const GroupedTensor& t) {
  const auto groups = int64_t(t.num_tensors);
  const int64_t cols = static_cast<int64_t>(t.logical_shape.data[1]);
  if (t.all_same_first_dim()) {
    return {t.get_common_first_dim(), t.all_same_last_dim() ? cols : cols / groups};
  }
  if (t.all_same_last_dim()) return {int64_t(t.logical_shape.data[0]) / groups, cols};
  return {1, cols / groups};
}

constexpr int kMetadataThreads = 32;
template <bool PackedInPlace>
__global__ void prepare_native_grouped_metadata(NativeSetup p) {
  auto needs_prefix = [](const NativeStorage& s) { return !s.offsets && (s.first || s.last); };
  int step = needs_prefix(p.a) || needs_prefix(p.b) || needs_prefix(p.c) || needs_prefix(p.d)
                 ? 1
                 : kMetadataThreads;
  int lane = threadIdx.x;
  uint64_t offsets[4]{};
  uint64_t total = 0, per_group = 0;
  for (int64_t start = 0; start < p.groups; start += step) {
    int count = int(p.groups - start < step ? p.groups - start : step);
    NativeExpert e{};
    if (lane < count) {
      int i = int(start) + lane;
      auto a = matrix_slice(p.a, nullptr, i, 2, offsets[0]);
      auto b = matrix_slice(p.b, PackedInPlace ? nullptr : p.b_list, i, 2, offsets[1]);
      auto d = matrix_slice(p.d, PackedInPlace ? nullptr : p.d_list, i, p.output_bytes, offsets[2]);
      int m = p.ta ? a.cols : a.rows, k = p.ta ? a.rows : a.cols;
      int n = p.tb ? b.rows : b.cols;
      float alpha = p.alpha[p.scalar_count == 1 ? 0 : i];
      float beta = p.beta[p.scalar_count == 1 ? 0 : i];
      require_metadata(k == (p.tb ? b.cols : b.rows) && d.rows == m && d.cols == n);
      require_metadata(!m || !n || !k || (a.cols % 8 == 0 && b.cols % 8 == 0));
      uint64_t cptr = 0;
      if constexpr (PackedInPlace) {
        cptr = d.ptr;
      } else if (p.has_c) {
        auto c = matrix_slice(p.c, p.c_list, i, p.output_bytes, offsets[3]);
        require_metadata(c.rows == m && c.cols == n);
        uint64_t bytes = uint64_t(m) * n * p.output_bytes;
        require_metadata(c.ptr == d.ptr || disjoint_ranges(c.ptr, bytes, d.ptr, bytes));
        require_metadata(!bytes || !p.d.base || p.c.base != p.d.base || c.ptr == d.ptr);
        cptr = c.ptr;
      } else {
        require_metadata(beta == 0 || !m || !n);
      }
      int columns = p.ne4 && ne4_shape(m, n, k) && alpha == 1 && beta == 0 ? 4 : 1;
      int mt = (m + (1 << p.tile_m_shift) - 1) >> p.tile_m_shift;
      int nt = (n + (1 << p.tile_n_shift) - 1) >> p.tile_n_shift;
      uint64_t tiles = uint64_t(mt) * (columns == 4 ? nt / 4 : nt);
      e = {a.ptr, b.ptr, cptr, d.ptr, tiles, m, n, k, columns, alpha, beta};
    }
    constexpr unsigned mask = 0xffffffff;
    if (start == 0) per_group = __shfl_sync(mask, e.end, 0);
    if (!__all_sync(mask, lane >= count || e.end == per_group)) per_group = 0;
      // A chunk contains at most 32 experts with 32-bit extents and tiles >= 128.
      // Its inclusive sum fits in 64 bits; check the carry between chunks.
#pragma unroll
    for (int delta = 1; delta < kMetadataThreads; delta *= 2) {
      uint64_t prior = __shfl_up_sync(mask, e.end, delta);
      if (lane >= delta) e.end += prior;
    }
    uint64_t chunk_tiles = __shfl_sync(mask, e.end, kMetadataThreads - 1);
    require_metadata(chunk_tiles <= UINT64_MAX - total);
    e.end += total;
    if (lane < count) p.experts[start + lane] = e;
    total += chunk_tiles;
  }
  // A list of independent outputs has no single allocation range to validate
  // on the host. Check it against all inputs and the other output slices.
  if (!PackedInPlace && p.d_list) {
    __syncthreads();
    for (int64_t i = lane; i < p.groups; i += kMetadataThreads) {
      auto e = p.experts[i];
      uint64_t bytes = uint64_t(e.m) * e.n * p.output_bytes;
      for (int j = 0; j < p.groups; ++j) {
        auto other = p.experts[j];
        uint64_t other_bytes = uint64_t(other.m) * other.n * p.output_bytes;
        require_metadata(disjoint_ranges(e.d, bytes, other.a, uint64_t(other.m) * other.k * 2) &&
                         disjoint_ranges(e.d, bytes, other.b, uint64_t(other.n) * other.k * 2));
        if (i != j) {
          require_metadata(disjoint_ranges(e.d, bytes, other.d, other_bytes));
          require_metadata(other.beta == 0 || disjoint_ranges(e.d, bytes, other.c, other_bytes));
        }
      }
    }
  }
  if (lane == 0) *p.control = {total, per_group};
}

// These are the geometries used by the native BF16 backend. Reuse the
// ordinary tile heuristic and snap an unsupported tuning override once.
template <bool TA, bool TB, bool FP32, int M, int N, int K, int S>
constexpr std::pair<int, int> native_launch_geometry() {
  static_assert((M & (M - 1)) == 0 && (N & (N - 1)) == 0,
                "Metadata setup requires power-of-two tile dimensions");
  using Traits = GemmTraits<TA, TB, 89, cutlass::bfloat16_t, M, N, K, S, FP32>;
  using Kernel =
      cutlass::gemm::kernel::NvteNativeGrouped<typename Traits::CollectiveMainloop,
                                               typename Traits::CollectiveEpilogue, false>;
  return {Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize};
}

template <bool TA, bool TB, bool FP32>
void launch_native(NativeSetup setup, TileConfig cfg, cudaStream_t stream,
                   const DiscreteOperands& lists, NVTETensor workspace) {
  if (cfg.stages == 0) cfg.stages = cfg.tile_m == 128 && cfg.tile_n == 128 ? 2 : 3;
  auto geometry = [&]() -> std::pair<int, int> {
#define NVTE_NATIVE_MATCH(M, N, K, S) \
  if (cfg.matches(M, N, K, S)) return native_launch_geometry<TA, TB, FP32, M, N, K, S>();
    NVTE_GG_GROUPED_TENSOR_TILES(NVTE_NATIVE_MATCH)
#undef NVTE_NATIVE_MATCH
    cfg = {256, 128, 64, 3};
    return native_launch_geometry<TA, TB, FP32, 256, 128, 64, 3>();
  };
  const auto [threads, smem] = geometry();
  const auto& kernel = get_or_compile(
      cuda::current_device(),
      {89, TA, TB, DType::kBFloat16, FP32 ? DType::kFloat32 : DType::kBFloat16, true}, cfg, smem);
  setup.tile_m_shift = cutlass::find_log2(cfg.tile_m);
  setup.tile_n_shift = cutlass::find_log2(cfg.tile_n);
  setup.ne4 = cfg.n_expand;
  NativeParams params{};
  auto packed = [&](const NativeStorage& s, int bytes) {
    if (s.first || s.last || s.offsets || !s.base || s.common_first <= 0 || s.common_last <= 0)
      return false;
    uint64_t count = uint64_t(s.common_first) * s.common_last;
    return count <= s.capacity / setup.groups && reinterpret_cast<uintptr_t>(s.base) % 16 == 0 &&
           (count * bytes) % 16 == 0;
  };
  // A plain packed descriptor proves every expert address on the host.
  // CSR and discrete inputs retain device-side shape and bounds validation.
  const int m = TA ? setup.a.common_last : setup.a.common_first;
  const int k = TA ? setup.a.common_first : setup.a.common_last;
  const int n = TB ? setup.b.common_first : setup.b.common_last;
  const int mt = (m + cfg.tile_m - 1) / cfg.tile_m;
  const int columns = cfg.n_expand ? 4 : 1;
  const int nt = ((n + cfg.tile_n - 1) / cfg.tile_n) / columns;
  const uint64_t group_tiles = uint64_t(mt) * nt;
  std::optional<WorkspaceGuard> guard;
  if (group_tiles > 0 && group_tiles <= INT_MAX / setup.groups && packed(setup.a, 2) &&
      packed(setup.b, 2) && packed(setup.c, setup.output_bytes) &&
      packed(setup.d, setup.output_bytes) &&
      k == (TB ? setup.b.common_last : setup.b.common_first) && setup.c.common_first == m &&
      setup.c.common_last == n && setup.d.common_first == m && setup.d.common_last == n &&
      setup.a.common_last % 8 == 0 && setup.b.common_last % 8 == 0 &&
      (!cfg.n_expand || ne4_shape(m, n, k))) {
    params.uniform = {reinterpret_cast<uint64_t>(setup.a.base),
                      reinterpret_cast<uint64_t>(setup.b.base),
                      reinterpret_cast<uint64_t>(setup.c.base),
                      reinterpret_cast<uint64_t>(setup.d.base),
                      group_tiles * setup.groups,
                      m,
                      n,
                      k,
                      columns};
    params.alpha = setup.alpha;
    params.beta = setup.beta;
    params.scalar_stride = setup.scalar_count != 1;
    params.tiles_per_group = cutlass::FastDivmod(int(group_tiles));
    params.tiles_per_row = cutlass::FastDivmod(nt);
  } else {
    guard.emplace(cuda::current_device(), &workspace, 1);
    if (!lists.host.empty()) {
      NVTE_CHECK_CUDA(cudaMemcpyAsync(lists.device, lists.host.data(),
                                      lists.host.size() * sizeof(NativeStorage),
                                      cudaMemcpyHostToDevice, stream));
    }
    if (setup.packed_in_place) {
      prepare_native_grouped_metadata<true><<<1, kMetadataThreads, 0, stream>>>(setup);
    } else {
      prepare_native_grouped_metadata<false><<<1, kMetadataThreads, 0, stream>>>(setup);
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
    params.experts = setup.experts;
    params.control = setup.control;
  }
  int blocks_per_sm = 1;
  if (cfg.tile_m <= 128) {
    NVTE_CALL_CHECK_CUDA_DRIVER(cuOccupancyMaxActiveBlocksPerMultiprocessor, &blocks_per_sm,
                                kernel.function, threads, kernel.smem_size);
  }
  void* args[] = {&params};
  NVTE_CALL_CHECK_CUDA_DRIVER(cuLaunchKernel, kernel.function,
                              setup.sms * std::max(1, blocks_per_sm), 1, 1, threads, 1, 1,
                              kernel.smem_size, static_cast<CUstream>(stream), args, nullptr);
}
}  // namespace
}  // namespace grouped_gemm_rtc

bool native_grouped_gemm_supported_device() { return cuda::sm_arch() == 89; }

size_t native_grouped_gemm_workspace_size(size_t groups) {
  using namespace grouped_gemm_rtc;
  NVTE_CHECK(
      groups <= size_t(INT_MAX) && groups <= (std::numeric_limits<size_t>::max() - 512) /
                                                 (sizeof(NativeExpert) + 3 * sizeof(NativeStorage)),
      "native grouped GEMM workspace size overflow");
  return DIVUP_TO_MULTIPLE(sizeof(NativeControl), size_t(128)) +
         DIVUP_TO_MULTIPLE(groups * sizeof(NativeExpert), size_t(128)) +
         3 * DIVUP_TO_MULTIPLE(groups * sizeof(NativeStorage), size_t(128));
}

bool try_native_grouped_gemm(const NVTEGroupedTensor A, int transa, const NVTEGroupedTensor B,
                             int transb, const NVTEGroupedTensor C, NVTEGroupedTensor D,
                             const NVTETensor alpha, const NVTETensor beta,
                             NVTETensor workspace_setup, NVTEGroupedMatmulConfig config,
                             cudaStream_t stream, const NVTETensor* A_list, size_t num_a,
                             const NVTETensor* C_list, NVTETensor* D_list, size_t num_d) {
  using namespace grouped_gemm_rtc;
  if (!native_grouped_gemm_supported_device()) return false;
  const auto* b = convertNVTEGroupedTensorCheck(B);
  if (b->dtype() != DType::kBFloat16 || b->scale_inv.dptr || b->columnwise_scale_inv.dptr)
    return false;
  NVTE_CHECK(transa == 0 || transa == 1, "native grouped GEMM invalid transa");
  NVTE_CHECK(transb == 0 || transb == 1, "native grouped GEMM invalid transb");
  NVTE_CHECK(!(transa && transb), "native grouped GEMM TT layout is unsupported");
  const auto* a = A ? convertNVTEGroupedTensorCheck(A) : nullptr;
  const auto* c = C ? convertNVTEGroupedTensorCheck(C) : nullptr;
  const auto* d = D ? convertNVTEGroupedTensorCheck(D) : nullptr;
  const size_t groups = b->num_tensors;
  NVTE_CHECK(groups > 0 && groups <= size_t(INT_MAX), "native grouped GEMM invalid group count");
  NVTE_CHECK((a && !A_list) || (!a && A_list && num_a == groups),
             "native grouped GEMM invalid A descriptor");
  NVTE_CHECK((d && !D_list) || (!d && D_list && num_d == groups),
             "native grouped GEMM invalid D descriptor");
  const auto output_dtype = d ? d->dtype() : convertNVTETensorCheck(D_list[0])->dtype();
  if (output_dtype != DType::kBFloat16 && output_dtype != DType::kFloat32) return false;
  const int device = cuda::current_device();
  NativeSetup setup{};
  setup.a = grouped_storage(b, DType::kBFloat16, groups, device);
  setup.b = grouped_storage(a, DType::kBFloat16, groups, device);
  setup.d = grouped_storage(d, output_dtype, groups, device);
  setup.c = c == d ? setup.d : grouped_storage(c, output_dtype, groups, device);
  setup.has_c = c != nullptr || C_list != nullptr;
  // The Python packed interface passes the same descriptor for C and D.
  setup.packed_in_place = !A_list && !C_list && !D_list && c && c == d;
  setup.groups = int(groups);
  setup.ta = transb;
  setup.tb = transa;
  setup.output_bytes = output_dtype == DType::kFloat32 ? 4 : 2;
  const auto* at = convertNVTETensorCheck(alpha);
  const auto* bt = convertNVTETensorCheck(beta);
  NVTE_CHECK(at->dtype() == DType::kFloat32 && bt->dtype() == DType::kFloat32 &&
                 (at->numel() == 1 || at->numel() == groups) && at->numel() == bt->numel(),
             "native grouped GEMM alpha/beta must be float32 scalars or per-group arrays");
  check_device_pointer(at->data.dptr, device);
  check_device_pointer(bt->data.dptr, device);
  NVTE_CHECK(at->data.dptr && bt->data.dptr, "native grouped GEMM null alpha/beta");
  NVTE_CHECK(reinterpret_cast<uintptr_t>(at->data.dptr) % alignof(float) == 0 &&
                 reinterpret_cast<uintptr_t>(bt->data.dptr) % alignof(float) == 0,
             "native grouped GEMM alpha/beta requires 4-byte alignment");
  setup.alpha = static_cast<const float*>(at->data.dptr);
  setup.beta = static_cast<const float*>(bt->data.dptr);
  setup.scalar_count = int(at->numel());
  auto* ws = convertNVTETensorCheck(workspace_setup);
  const size_t bytes = native_grouped_gemm_workspace_size(groups);
  NVTE_CHECK(ws->dtype() == DType::kByte && ws->data.shape.size() == 1 && ws->data.dptr &&
                 ws->numel() >= bytes && reinterpret_cast<uintptr_t>(ws->data.dptr) % 128 == 0,
             "native grouped GEMM requires sufficient 128-byte aligned setup workspace");
  check_device_pointer(ws->data.dptr, device);
  auto range = [](const NativeStorage& storage, int element_bytes) {
    return byte_range(storage.base, storage.capacity, element_bytes);
  };
  auto wrange = byte_range(ws->data.dptr, bytes, 1);
  for (auto r : {range(setup.a, 2), range(setup.b, 2), range(setup.c, setup.output_bytes),
                 range(setup.d, setup.output_bytes)})
    NVTE_CHECK(disjoint(wrange, r), "native grouped GEMM workspace overlaps storage");
  auto drange = range(setup.d, setup.output_bytes);
  NVTE_CHECK(disjoint(drange, range(setup.a, 2)) && disjoint(drange, range(setup.b, 2)),
             "native grouped GEMM D overlaps input storage");
  auto crange = range(setup.c, setup.output_bytes);
  NVTE_CHECK(crange == drange || disjoint(crange, drange),
             "native grouped GEMM partially overlapping C/D storage");
  auto check_metadata_disjoint = [&](auto buffer) {
    auto check = [&](const SimpleTensor& t, int element_bytes) {
      auto r = byte_range(t.dptr, t.numel(), element_bytes);
      NVTE_CHECK(disjoint(buffer, r), "native grouped GEMM output or workspace overlaps metadata");
    };
    for (const auto* t : {a, b, c, d}) {
      if (t) {
        check(t->first_dims, 8);
        check(t->last_dims, 8);
        check(t->tensor_offsets, 8);
      }
    }
    check(at->data, 4);
    check(bt->data, 4);
  };
  check_metadata_disjoint(wrange);
  check_metadata_disjoint(drange);
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  NVTE_CHECK(capture == cudaStreamCaptureStatusNone,
             "native grouped GEMM does not support CUDA graph capture");
  const auto* policy = reinterpret_cast<const GroupedMatmulConfig*>(config);
  setup.sms = policy && policy->sm_count > 0 ? policy->sm_count : cuda::sm_count(device);
  if (setup.sms <= 0) setup.sms = 39;

  char* memory = static_cast<char*>(ws->data.dptr);
  setup.control = reinterpret_cast<NativeControl*>(memory);
  memory += DIVUP_TO_MULTIPLE(sizeof(NativeControl), size_t(128));
  setup.experts = reinterpret_cast<NativeExpert*>(memory);
  memory += DIVUP_TO_MULTIPLE(groups * sizeof(NativeExpert), size_t(128));
  DiscreteOperands lists{reinterpret_cast<NativeStorage*>(memory), {}};
  lists.host.reserve(groups * ((A_list != nullptr) + (C_list != nullptr) + (D_list != nullptr)));
  int64_t discrete_rows = 0, discrete_cols = 0;
  auto describe_discrete = [&](const NVTETensor* tensors, DType dtype) -> const NativeStorage* {
    if (!tensors) return nullptr;
    const auto* result = lists.device + lists.host.size();
    for (size_t i = 0; i < groups; ++i) {
      const auto* t = convertNVTETensorCheck(tensors[i]);
      NVTE_CHECK(t->dtype() == dtype && t->data.shape.size() == 2 &&
                     t->data.shape[0] <= size_t(INT_MAX - 256) &&
                     t->data.shape[1] <= size_t(INT_MAX - 256),
                 "native grouped GEMM discrete tensors require uniform dtype and rank 2");
      check_device_pointer(t->data.dptr, device);
      NativeStorage s{t->data.dptr,
                      t->numel(),
                      nullptr,
                      nullptr,
                      nullptr,
                      int64_t(t->data.shape[0]),
                      int64_t(t->data.shape[1])};
      auto r = range(s, dtype == DType::kBFloat16 ? 2 : setup.output_bytes);
      NVTE_CHECK(disjoint(wrange, r), "native grouped GEMM workspace overlaps discrete storage");
      if (tensors == A_list) {
        NVTE_CHECK(disjoint(drange, r), "native grouped GEMM D overlaps discrete input");
        discrete_rows += s.common_first;
        discrete_cols += s.common_last;
      }
      if (tensors == D_list) check_metadata_disjoint(r);
      lists.host.push_back(s);
    }
    return result;
  };
  setup.b_list = describe_discrete(A_list, DType::kBFloat16);
  setup.c_list = describe_discrete(C_list, output_dtype);
  setup.d_list = describe_discrete(D_list, output_dtype);
  // All tuning inputs are host-visible. In particular, do not read per-expert
  // CUDA shape tensors back to the host to choose a tile.
  const auto [b_rows, b_cols] = average_shape(*b);
  const auto [a_rows, a_cols] =
      a ? average_shape(*a)
        : std::make_pair(discrete_rows / int64_t(groups), discrete_cols / int64_t(groups));
  auto dimension = [](int64_t value) { return int(std::clamp<int64_t>(value, 1, INT_MAX - 256)); };
  const int m = dimension(transb ? b_cols : b_rows);
  const int n = dimension(transa ? a_rows : a_cols);
  const int k = dimension(transa ? a_cols : a_rows);
  TileConfig chosen = select_grouped_tile(m, n, k, int(groups), transa, transb, 89, setup.sms);
  if (!transb && ne4_shape(m, n, k)) {
    select_n_expand(chosen, uint64_t(groups) * (m / 256) * (n / 1024), setup.sms);
  }
  auto layout = [&](auto fp32) {
    if (transb)
      launch_native<true, false, decltype(fp32)::value>(setup, chosen, stream, lists,
                                                        workspace_setup);
    else if (transa)
      launch_native<false, true, decltype(fp32)::value>(setup, chosen, stream, lists,
                                                        workspace_setup);
    else
      launch_native<false, false, decltype(fp32)::value>(setup, chosen, stream, lists,
                                                         workspace_setup);
  };
  if (output_dtype == DType::kFloat32)
    layout(std::true_type{});
  else
    layout(std::false_type{});
  return true;
}
namespace grouped_gemm_rtc {
namespace {
struct NativeBiasStorage {
  const int64_t *first, *last, *offsets;
  int64_t common_first, common_last;
  uint64_t capacity;
};

struct NativeBiasSlice {
  uint64_t rows, cols, offset, count;
};

__device__ void native_bias_require(bool condition, const char* message) {
  if (!condition) {
    printf("%s\n", message);
    __assert_fail(message, __FILE__, __LINE__, __func__);
  }
}

__device__ NativeBiasSlice native_bias_slice(const NativeBiasStorage& s, size_t group,
                                             uint64_t& running) {
  const int64_t rows = s.first ? s.first[group] : s.common_first;
  const int64_t cols = s.last ? s.last[group] : s.common_last;
  native_bias_require(rows >= 0 && cols >= 0, "native grouped bias add: negative dimension");
  native_bias_require(rows == 0 || uint64_t(cols) <= uint64_t(INT64_MAX) / uint64_t(rows),
                      "native grouped bias add: dimension product overflow");
  const uint64_t count = uint64_t(rows) * uint64_t(cols);
  uint64_t offset = running;
  if (s.offsets) {
    const int64_t start = s.offsets[group], end = s.offsets[group + 1];
    native_bias_require(
        start >= 0 && end >= start && uint64_t(start) == running && count <= uint64_t(end - start),
        "native grouped bias add: invalid CSR offsets or overlapping groups");
    offset = uint64_t(start);
    running = uint64_t(end);
  } else {
    native_bias_require(count <= uint64_t(INT64_MAX) - running,
                        "native grouped bias add: cumulative offset overflow");
    running += count;
  }
  native_bias_require(offset <= s.capacity && count <= s.capacity - offset && running <= s.capacity,
                      "native grouped bias add: group exceeds storage capacity");
  return {uint64_t(rows), uint64_t(cols), offset, count};
}

template <typename T, bool UseScale>
__global__ void native_grouped_bias_add_kernel(T* output, const T* bias, const float* scale,
                                               NativeBiasStorage d, NativeBiasStorage b,
                                               size_t groups, uint64_t scale_count) {
  if (threadIdx.x == 0) {
    uint64_t d_end = d.offsets ? uint64_t(d.offsets[0]) : 0;
    uint64_t b_end = b.offsets ? uint64_t(b.offsets[0]) : 0;
    uint64_t total_rows = 0, total_elements = 0;
    for (size_t g = 0; g < groups; ++g) {
      const auto ds = native_bias_slice(d, g, d_end);
      const auto bs = native_bias_slice(b, g, b_end);
      native_bias_require(bs.rows == 1 && bs.cols == ds.cols,
                          "native grouped bias add: bias shape must be (1, output width)");
      native_bias_require(ds.rows <= uint64_t(INT64_MAX) - total_rows &&
                              ds.count <= uint64_t(INT64_MAX) - total_elements,
                          "native grouped bias add: total rows or elements overflow");
      total_rows += ds.rows;
      total_elements += ds.count;
    }
    if constexpr (UseScale) {
      native_bias_require(scale_count == total_rows,
                          "native grouped bias add: scale size must equal total output rows");
    }
  }
  // Every CTA completes the same full validation before touching output.
  __syncthreads();

  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t element_start = 0, row_start = 0;
  uint64_t d_end = d.offsets ? uint64_t(d.offsets[0]) : 0;
  uint64_t b_end = b.offsets ? uint64_t(b.offsets[0]) : 0;
  for (size_t g = 0; g < groups; ++g) {
    const auto ds = native_bias_slice(d, g, d_end);
    const auto bs = native_bias_slice(b, g, b_end);
    const uint64_t element_end = element_start + ds.count;
    while (index < element_end) {
      const uint64_t local = index - element_start;
      const float value = static_cast<float>(output[ds.offset + local]);
      const float bias_value = static_cast<float>(bias[bs.offset + local % ds.cols]);
      if constexpr (UseScale) {
        output[ds.offset + local] =
            static_cast<T>(fmaf(bias_value, scale[row_start + local / ds.cols], value));
      } else {
        output[ds.offset + local] = static_cast<T>(value + bias_value);
      }
      index += stride;
    }
    element_start = element_end;
    row_start += ds.rows;
  }
}

}  // namespace
}  // namespace grouped_gemm_rtc

bool try_native_grouped_bias_add(const NVTEGroupedTensor output, const NVTEGroupedTensor bias,
                                 const NVTETensor scale, bool use_scale, cudaStream_t stream) {
  using namespace grouped_gemm_rtc;
  if (!native_grouped_gemm_supported_device()) return false;
  const auto* d = convertNVTEGroupedTensorCheck(output);
  const auto* b = convertNVTEGroupedTensorCheck(bias);
  if ((d->dtype() != DType::kBFloat16 && d->dtype() != DType::kFloat32) ||
      (b->dtype() != DType::kBFloat16 && b->dtype() != DType::kFloat32) ||
      d->scaling_mode != NVTE_DELAYED_TENSOR_SCALING ||
      b->scaling_mode != NVTE_DELAYED_TENSOR_SCALING || d->scale_inv.dptr ||
      d->columnwise_scale_inv.dptr || b->scale_inv.dptr || b->columnwise_scale_inv.dptr) {
    return false;
  }
  NVTE_CHECK(d->dtype() == b->dtype(), "native grouped bias add: output/bias dtypes must match");
  const size_t groups = d->num_tensors;
  NVTE_CHECK(groups > 0 && groups < size_t(INT64_MAX) && groups == b->num_tensors,
             "native grouped bias add: output/bias group counts must be equal and nonzero");
  const int device = cuda::current_device();
  const size_t bytes = typeToSize(d->dtype());
  auto storage = [&](const GroupedTensor* t) {
    NVTE_CHECK(t->logical_shape.ndim == 2 && t->data.shape.size() == 1 &&
                   t->data.dtype == d->dtype() && t->data.shape[0] <= size_t(INT64_MAX) &&
                   t->logical_shape.data[0] <= size_t(INT64_MAX) &&
                   t->logical_shape.data[1] <= size_t(INT64_MAX),
               "native grouped bias add: requires packed rowwise storage and a 2D logical shape");
    NativeBiasStorage s{};
    s.capacity = t->data.shape[0];
    NVTE_CHECK(s.capacity == 0 || t->data.dptr,
               "native grouped bias add: null nonempty rowwise storage");
    check_device_pointer(t->data.dptr, device, bytes);
    for (const auto* metadata : {&t->first_dims, &t->last_dims, &t->tensor_offsets}) {
      NVTE_CHECK(metadata->shape.size() == 1, "native grouped bias add: metadata must be rank 1");
    }
    s.first = checked_metadata(t->first_dims, groups, device);
    s.last = checked_metadata(t->last_dims, groups, device);
    s.offsets = checked_metadata(t->tensor_offsets, groups + 1, device);
    if (!s.first) {
      NVTE_CHECK(s.last || t->logical_shape.data[0] % groups == 0,
                 "native grouped bias add: uniform rows must be divisible by group count");
      s.common_first = t->get_common_first_dim();
    }
    if (!s.last) s.common_last = t->logical_shape.data[1];
    return s;
  };
  const auto ds = storage(d), bs = storage(b);
  const auto d_range = byte_range(d->data.dptr, ds.capacity, bytes);
  NVTE_CHECK(disjoint(d_range, byte_range(b->data.dptr, bs.capacity, bytes)),
             "native grouped bias add: bias must not overlap output");
  for (const auto* t : {d, b}) {
    for (const auto* m : {&t->first_dims, &t->last_dims, &t->tensor_offsets}) {
      NVTE_CHECK(disjoint(d_range, byte_range(m->dptr, m->shape[0], sizeof(int64_t))),
                 "native grouped bias add: output must not overlap immutable metadata");
    }
  }
  const float* scale_ptr = nullptr;
  uint64_t scale_count = 0;
  if (use_scale) {
    const auto* s = convertNVTETensorCheck(scale);
    NVTE_CHECK(s->data.dtype == DType::kFloat32 && s->data.shape.size() == 1 &&
                   s->data.shape[0] <= size_t(INT64_MAX) && (s->data.shape[0] == 0 || s->data.dptr),
               "native grouped bias add: scale must be a float32 row vector");
    check_device_pointer(s->data.dptr, device, sizeof(float));
    scale_ptr = static_cast<const float*>(s->data.dptr);
    scale_count = s->data.shape[0];
    NVTE_CHECK(disjoint(d_range, byte_range(scale_ptr, scale_count, sizeof(float))),
               "native grouped bias add: scale must not overlap output");
  }
  const int blocks = std::max(1, cuda::sm_count(device));
  auto launch = [&](auto tag) {
    constexpr bool UseScale = decltype(tag)::value;
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(d->dtype(), T, {
      native_grouped_bias_add_kernel<T, UseScale><<<blocks, 256, 0, stream>>>(
          static_cast<T*>(d->data.dptr), static_cast<const T*>(b->data.dptr), scale_ptr, ds, bs,
          groups, scale_count);
    });
  };
  if (use_scale)
    launch(std::true_type{});
  else
    launch(std::false_type{});
  NVTE_CHECK_CUDA(cudaGetLastError());
  return true;
}

}  // namespace transformer_engine
