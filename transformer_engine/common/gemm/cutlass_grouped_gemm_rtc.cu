/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#include "cutlass_grouped_gemm_rtc.h"
#include "cutlass_grouped_gemm_rtc_kernel.h"
#include "ppu_adaptive_tile.hpp"
#include "te_profiling_interface.h"

namespace transformer_engine::grouped_gemm_rtc {
namespace {
constexpr size_t kCPUWorkSpaceSize = 4 * 1024 * 1024;

// A slot stays reserved while the host reads its shapes. Its event also
// protects the pinned memory until the asynchronous H2D copy has consumed it.
constexpr unsigned kHostWsSlots = 8;
constexpr size_t kHostWsSlotSize = kCPUWorkSpaceSize / kHostWsSlots;

struct StagingSlot {
  std::mutex mutex;
  cudaEvent_t copied = nullptr;
  bool copy_pending = false;
};
}  // namespace

struct DeviceWorkspace {
  std::once_flag init;
  std::atomic<unsigned> next{0};
  StagingSlot slots[kHostWsSlots];
  char* pinned = nullptr;

  std::mutex cache_mutex;
  std::condition_variable available;
  WorkspaceGuard* active = nullptr;
  size_t waiters = 0;
  char* last_workspace = nullptr;
  cudaStream_t last_stream = nullptr;
  std::vector<char> last_params;

  void init_staging() {
    if (cudaHostAlloc(&pinned, kCPUWorkSpaceSize, cudaHostAllocDefault) != cudaSuccess) {
      (void)cudaGetLastError();
      return;  // Fall back to pageable memory when the locked-memory quota is exhausted.
    }
    for (unsigned i = 0; i < kHostWsSlots; ++i) {
      if (cudaEventCreateWithFlags(&slots[i].copied, cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaGetLastError();
        for (unsigned j = 0; j < i; ++j) cudaEventDestroy(slots[j].copied);
        cudaFreeHost(pinned);
        pinned = nullptr;
        return;
      }
    }
  }
};

namespace {
DeviceWorkspace& device_workspace(int device) {
  static std::mutex mutex;
  static std::map<int, DeviceWorkspace> workspaces;
  std::lock_guard<std::mutex> lock(mutex);
  return workspaces[device];
}

class HostStaging {
 public:
  HostStaging(int device, size_t param_bytes)
      : device_(device_workspace(device)), param_bytes_(param_bytes) {
    std::call_once(device_.init, [&] { device_.init_staging(); });
    if (device_.pinned && param_bytes <= kHostWsSlotSize) {
      const unsigned index = device_.next.fetch_add(1, std::memory_order_relaxed) % kHostWsSlots;
      auto& slot = device_.slots[index];
      lock_ = std::unique_lock<std::mutex>(slot.mutex, std::try_to_lock);
      if (lock_.owns_lock()) {
        slot_ = &slot;
        if (slot_->copy_pending) {
          NVTE_CHECK_CUDA(cudaEventSynchronize(slot_->copied));
          slot_->copy_pending = false;
        }
        data_ = device_.pinned + index * kHostWsSlotSize;
        return;
      }
    }
    heap_.resize(param_bytes);
    data_ = heap_.data();
  }

  char* data() const { return data_; }

  void upload(char* dest, size_t total, bool scratch_unused, cudaStream_t stream) {
    if (slot_ && total > kHostWsSlotSize) {
      heap_.assign(data_, data_ + param_bytes_);
      slot_ = nullptr;
      lock_.unlock();
    }
    if (!slot_) {
      heap_.resize(total);
      data_ = heap_.data();
    }

    {
      std::lock_guard<std::mutex> lock(device_.cache_mutex);
      if (scratch_unused && device_.last_workspace == dest && device_.last_stream == stream &&
          device_.last_params.size() == param_bytes_ &&
          std::memcmp(device_.last_params.data(), data_, param_bytes_) == 0)
        return;
      device_.last_workspace = nullptr;
    }
    std::memset(data_ + param_bytes_, 0, total - param_bytes_);
    NVTE_CHECK_CUDA(cudaMemcpyAsync(dest, data_, total, cudaMemcpyHostToDevice, stream));
    if (slot_) {
      auto status = cudaEventRecord(slot_->copied, stream);
      if (status != cudaSuccess) {
        // The H2D was already submitted. Finish it before releasing the slot.
        NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
        NVTE_CHECK_CUDA(status);
      }
      slot_->copy_pending = true;
    }
    if (scratch_unused) {
      std::lock_guard<std::mutex> lock(device_.cache_mutex);
      device_.last_params.assign(data_, data_ + param_bytes_);
      device_.last_stream = stream;
      device_.last_workspace = dest;
    }
  }

 private:
  DeviceWorkspace& device_;
  size_t param_bytes_;
  StagingSlot* slot_ = nullptr;
  std::unique_lock<std::mutex> lock_;
  std::vector<char> heap_;
  char* data_ = nullptr;
};

using ExpertShape = cute::Shape<int, int, int>;

// Both the host staging buffer and the device workspace use this layout.
struct DescriptorLayout {
  explicit DescriptorLayout(size_t groups)
      : shapes(DIVUP_TO_MULTIPLE(groups * sizeof(ExpertShape), size_t(128))),
        pointers(DIVUP_TO_MULTIPLE(groups * sizeof(void*), size_t(128))),
        strides(DIVUP_TO_MULTIPLE(groups * sizeof(int64_t), size_t(128))) {}

  size_t shapes, pointers, strides;
  size_t bytes() const { return shapes + 3 * (pointers + strides); }

  struct View {
    ExpertShape* shapes;
    void *a, *b, *d;
    int64_t *lda, *ldb, *ldd;
  };

  View view(char* base) const {
    char* ptr = base + shapes;
    char* ld = ptr + 3 * pointers;
    return {reinterpret_cast<ExpertShape*>(base),
            ptr,
            ptr + pointers,
            ptr + 2 * pointers,
            reinterpret_cast<int64_t*>(ld),
            reinterpret_cast<int64_t*>(ld + strides),
            reinterpret_cast<int64_t*>(ld + 2 * strides)};
  }
};

// All tile, architecture and element-type specializations consume the same list
// descriptors. Only CUTLASS argument conversion and kernel geometry are typed.
struct ListGemm {
  KernelConfig kernel;
  int groups, device, sms;
  bool accumulate;
  cudaStream_t stream;
  Tensor& workspace;
  DescriptorLayout layout{size_t(groups)};
  HostStaging staging{device, layout.bytes()};
  TileConfig tile{};

  char* device_data() const { return static_cast<char*>(workspace.data.dptr); }
  ExpertShape* host_shapes() const { return reinterpret_cast<ExpertShape*>(staging.data()); }

  void prepare(const NVTETensor* A, const NVTETensor* B, const NVTETensor* D) {
    const auto h = layout.view(staging.data());
    int64_t sum_m = 0, sum_n = 0, sum_k = 0;
    const bool overwrite_bf16 =
        kernel.sm_arch == 89 && kernel.input_dtype == DType::kBFloat16 && !accumulate;
    bool ne4 = overwrite_bf16 && !kernel.trans_a;
    bool shared_weights_shape = overwrite_bf16 && kernel.output_dtype == DType::kBFloat16;
    uint64_t expanded = 0;
    for (int i = 0; i < groups; ++i) {
      const auto& a = convertNVTETensorCheck(A[i])->data;
      const auto& b = convertNVTETensorCheck(B[i])->data;
      const auto& d = convertNVTETensorCheck(D[i])->data;
      const int m = a.shape[kernel.trans_a ? 1 : 0];
      const int k = a.shape[kernel.trans_a ? 0 : 1];
      const int n = b.shape[kernel.trans_b ? 0 : 1];
      h.shapes[i] = ExpertShape(m, n, k);
      static_cast<void**>(h.a)[i] = a.dptr;
      static_cast<void**>(h.b)[i] = b.dptr;
      static_cast<void**>(h.d)[i] = d.dptr;
      // A and B are packed in their physical row-major tensor shapes. The
      // transpose flags change their GEMM axes, not their leading dimensions.
      h.lda[i] = a.shape[1];
      h.ldb[i] = b.shape[1];
      h.ldd[i] = n;
      sum_m += m;
      sum_n += n;
      sum_k += k;
      shared_weights_shape = shared_weights_shape && n == cute::get<1>(h.shapes[0]) &&
                             k == cute::get<2>(h.shapes[0]);
      if (ne4) {
        ne4 = ne4_shape(m, n, k) &&
              ((reinterpret_cast<uintptr_t>(a.dptr) | reinterpret_cast<uintptr_t>(b.dptr) |
                reinterpret_cast<uintptr_t>(d.dptr)) %
                   16 ==
               0);
        expanded += uint64_t(m / 256) * (n / 1024);
      }
    }
    // The tuning table uses TE layout order, which swaps the CUTLASS operands.
    tile = select_grouped_tile(sum_m / groups, sum_n / groups, sum_k / groups, groups,
                               kernel.trans_b, kernel.trans_a, kernel.sm_arch, sms);
    if (ne4) select_n_expand(tile, expanded, sms);

    // Specialize shared N/K only when full-width tiles can fill the device.
    // Short aligned reductions profit from the four-column pipeline. For
    // row-major B, deeper reductions amortize a separate loop per output tile.
    const int n = cute::get<1>(h.shapes[0]), k = cute::get<2>(h.shapes[0]);
    const bool prefer_expanded = tile.n_expand && (kernel.trans_b || k < 16 * 64);
    if (!prefer_expanded && shared_weights_shape && n > 0 && n % 256 == 0 &&
        k >= 3 * 64 && k <= 32 * 64 && k % 64 == 0 &&
        tile.tile_m == 256 && tile.tile_n == 256 && (sum_m / 256) * (n / 256) >= sms &&
        (!forced_tile() || tile.matches(256, 256, 64, 3))) {
      tile = kDefaultTile;
      kernel.common_n = n;
      kernel.common_k = k;
    }
  }

  void* upload(size_t scratch) {
    // Keep the scheduler's scratch pointer valid even when its size is zero.
    const size_t total =
        layout.bytes() + DIVUP_TO_MULTIPLE(std::max(size_t(128), scratch), size_t(128));
    NVTE_CHECK(total <= kCPUWorkSpaceSize && total <= workspace.numel(),
               "Insufficient workspace for RTC grouped GEMM (required=", total,
               ", host capacity=", kCPUWorkSpaceSize, ", device capacity=", workspace.numel(),
               ").");
    staging.upload(device_data(), total, scratch == 0, stream);
    return device_data() + layout.bytes();
  }

  dim3 launch_grid(dim3 grid, bool along_n, const CompiledKernel& compiled, int threads) const {
    const ExpertShape* shapes = host_shapes();
    int workers = 0;
    if (kernel.sm_arch == 80 && sms == 64 && groups == 16 && tile.workers_per_cu > 0) {
      // A tuned row allows ragged K for wgrad and ragged M for fprop/dgrad.
      workers = tile.workers_per_cu;
      for (int g = 1; g < groups; ++g) {
        const bool uniform = kernel.trans_a ? cute::get<0>(shapes[g]) == cute::get<0>(shapes[0])
                                            : cute::get<2>(shapes[g]) == cute::get<2>(shapes[0]);
        if (!uniform || cute::get<1>(shapes[g]) != cute::get<1>(shapes[0])) {
          workers = 0;
          break;
        }
      }
    }
    if (!workers && tile.tile_m <= 128) {
      if (cuOccupancyMaxActiveBlocksPerMultiprocessor(&workers, compiled.function, threads,
                                                      compiled.smem_size) != CUDA_SUCCESS)
        return grid;
    }
    if (workers <= 0) return grid;
    uint64_t tiles = 0;
    for (int g = 0; g < groups; ++g) {
      const auto m = uint64_t(cute::get<0>(shapes[g]));
      const auto n = uint64_t(cute::get<1>(shapes[g]));
      tiles += ((m + tile.tile_m - 1) / tile.tile_m) * ((n + tile.tile_n - 1) / tile.tile_n);
    }
    uint64_t blocks = uint64_t(sms) * workers;
    if (tiles) blocks = std::min(blocks, tiles);
    // Persistent workers advance along the scheduler's raster direction.
    return along_n ? dim3(1, blocks, 1) : dim3(blocks, 1, 1);
  }

  void launch(void* params, dim3 default_grid, bool along_n, int threads,
              const CompiledKernel& compiled) const {
    const auto grid = launch_grid(default_grid, along_n, compiled, threads);
    te_profiling::TeProfParam tag;
    if (te_profiling::TeProfilingInterface::Instance().enabled()) {
      std::vector<int> m(groups), n(groups), k(groups);
      const auto* shapes = host_shapes();
      for (int g = 0; g < groups; ++g) {
        m[g] = cute::get<0>(shapes[g]);
        n[g] = cute::get<1>(shapes[g]);
        k[g] = cute::get<2>(shapes[g]);
      }
      const int dims[] = {tile.tile_m, tile.tile_n, tile.tile_k, tile.stages};
      tag.set_grouped_gemm("GroupedGemmRtc",
                           kernel.input_dtype == DType::kFloat16 ? "fp16" : "bf16", groups,
                           kernel.trans_a ? "TN" : (kernel.trans_b ? "NT" : "NN"), accumulate,
                           m.data(), n.data(), k.data(), dims, device, stream);
    }
    void* args[] = {params};
    te_profiling::TeHgtxGuard guard(tag);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuLaunchKernel, compiled.function, grid.x, grid.y, grid.z,
                                threads, 1, 1, compiled.smem_size, static_cast<CUstream>(stream),
                                args, nullptr);
  }

  template <class Traits>
  void run() {
    using Kernel = typename Traits::GemmKernel;
    const auto d = layout.view(device_data());
    typename Kernel::Arguments args{};
    args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_shape = {groups, d.shapes, host_shapes()};
    args.hw_info.cu_count = sms;
    auto& main = args.mainloop;
    main.ptr_A = static_cast<decltype(main.ptr_A)>(d.a);
    main.ptr_B = static_cast<decltype(main.ptr_B)>(d.b);
    main.dA = reinterpret_cast<decltype(main.dA)>(d.lda);
    main.dB = reinterpret_cast<decltype(main.dB)>(d.ldb);
    auto& ep = args.epilogue;
    ep.thread.alpha = 1.0f;
    // The direct FP32 epilogue takes beta from the presence of C.
    ep.thread.beta = accumulate && kernel.output_dtype != DType::kFloat32 ? 1.0f : 0.0f;
    ep.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};
    ep.thread.dBeta = {cute::_0{}, cute::_0{}, 0};
    ep.ptr_C = accumulate ? static_cast<decltype(ep.ptr_C)>(d.d) : nullptr;
    ep.ptr_D = static_cast<decltype(ep.ptr_D)>(d.d);
    ep.dC = reinterpret_cast<decltype(ep.dC)>(d.ldd);
    ep.dD = reinterpret_cast<decltype(ep.dD)>(d.ldd);
    NVTE_CHECK(Kernel::can_implement(args), "CUTLASS cannot implement RTC grouped GEMM for ",
               groups, " groups.");
    const auto& compiled = get_or_compile(device, kernel, tile, Kernel::SharedStorageSize);
    WorkspaceGuard guard(device, &workspace.nvte_tensor, 1, /*reuse_list_descriptors=*/true);
    void* scratch = upload(Kernel::get_workspace_size(args));
    // A pageable staging buffer can move when scratch is appended.
    args.problem_shape.host_problem_shapes = host_shapes();
    NVTE_CHECK(Kernel::initialize_workspace(args, scratch, stream) == cutlass::Status::kSuccess,
               "Failed to initialize RTC grouped GEMM workspace.");
    auto params = Kernel::to_underlying_arguments(args, scratch);
    const bool along_n =
        params.scheduler.raster_order_ == decltype(params.scheduler)::RasterOrder::AlongN;
    launch(&params, Kernel::get_grid_shape(params), along_n, Kernel::MaxThreadsPerBlock, compiled);
  }
};

template <bool TA, bool TB, int Arch, bool FP32>
void launch_tile(ListGemm& gemm) {
#define NVTE_GG_LAUNCH(M, N, K, S)                                                 \
  if (gemm.tile.matches(M, N, K, S)) {                                             \
    if (gemm.kernel.input_dtype == DType::kFloat16)                                \
      gemm.run<GemmTraits<TA, TB, Arch, cutlass::half_t, M, N, K, S, FP32>>();     \
    else                                                                           \
      gemm.run<GemmTraits<TA, TB, Arch, cutlass::bfloat16_t, M, N, K, S, FP32>>(); \
    return;                                                                        \
  }
  NVTE_GG_TILE_TABLE(NVTE_GG_LAUNCH)
#undef NVTE_GG_LAUNCH
  NVTE_ERROR("No RTC grouped GEMM host specialization for tile ", gemm.tile.tile_m, "x",
             gemm.tile.tile_n, "x", gemm.tile.tile_k, " stages=", gemm.tile.stages);
}

template <int Arch, bool FP32>
void launch_layout(ListGemm& gemm) {
  if (gemm.kernel.trans_a)
    launch_tile<true, false, Arch, FP32>(gemm);
  else if (gemm.kernel.trans_b)
    launch_tile<false, true, Arch, FP32>(gemm);
  else
    launch_tile<false, false, Arch, FP32>(gemm);
}

bool dispatch(const NVTETensor* A, const NVTETensor* B, NVTETensor* D, int groups, bool transa,
              bool transb, NVTETensor* workspace, bool accumulate, int device, int sms,
              cudaStream_t stream) {
  // nvte_multi_tensor_gemm validates group dtypes, ranks and vector alignment.
  if (transa && transb) return false;
  const int arch = cuda::sm_arch(device);
  if (arch != 80 && arch != 89) return false;
  if (sms <= 0) sms = cuda::sm_count(device);
  if (sms <= 0) sms = 39;
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone) return false;
  const auto input = convertNVTETensorCheck(A[0])->data.dtype;
  const auto output = convertNVTETensorCheck(D[0])->data.dtype;
  ListGemm gemm{
      {arch, transb, transa, input, output, false}, groups, device, sms, accumulate, stream,
      *convertNVTETensorCheck(workspace[0])};
  gemm.prepare(B, A, D);
  const bool fp32 = output == DType::kFloat32;
  if (arch == 80)
    fp32 ? launch_layout<80, true>(gemm) : launch_layout<80, false>(gemm);
  else
    fp32 ? launch_layout<89, true>(gemm) : launch_layout<89, false>(gemm);
  return true;
}

}  // namespace

bool WorkspaceGuard::overlaps(uintptr_t address, size_t bytes) const {
  for (size_t i = 0; i < count_; ++i) {
    const auto& data = convertNVTETensorCheck(workspaces_[i])->data;
    const auto start = reinterpret_cast<uintptr_t>(data.dptr);
    const auto size = data.buffer_size_bytes();
    if (size && bytes && (start <= address ? address - start < size : start - address < bytes))
      return true;
  }
  return false;
}

WorkspaceGuard::WorkspaceGuard(int device, const NVTETensor* workspaces, size_t count,
                               bool reuse_list_descriptors)
    : device_(device_workspace(device)), workspaces_(workspaces), count_(count) {
  std::unique_lock<std::mutex> lock(device_.cache_mutex);
  auto available = [&] {
    for (auto* other = device_.active; other; other = other->next_) {
      for (size_t i = 0; i < count_; ++i) {
        const auto& data = convertNVTETensorCheck(workspaces_[i])->data;
        if (other->overlaps(reinterpret_cast<uintptr_t>(data.dptr), data.buffer_size_bytes()))
          return false;
      }
    }
    return true;
  };
  if (!available()) {
    ++device_.waiters;
    device_.available.wait(lock, available);
    --device_.waiters;
  }
  if (!reuse_list_descriptors && device_.last_workspace &&
      overlaps(reinterpret_cast<uintptr_t>(device_.last_workspace), device_.last_params.size())) {
    device_.last_workspace = nullptr;
  }
  next_ = std::exchange(device_.active, this);
}

WorkspaceGuard::~WorkspaceGuard() {
  std::lock_guard<std::mutex> lock(device_.cache_mutex);
  auto** entry = &device_.active;
  while (*entry != this) entry = &(*entry)->next_;
  *entry = next_;
  if (device_.waiters) device_.available.notify_all();
}

}  // namespace transformer_engine::grouped_gemm_rtc

bool cutlass_grouped_gemm_rtc(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                              int num_gemms, bool transa, bool transb, NVTETensor* workspace,
                              bool accumulate, int device, int math_sm_count, cudaStream_t stream) {
  try {
    return transformer_engine::grouped_gemm_rtc::dispatch(
        A, B, D, num_gemms, transa, transb, workspace, accumulate, device, math_sm_count, stream);
  } catch (const std::exception& e) {
    if (transformer_engine::getenv<bool>("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", false)) {
      NVTE_WARN("CUTLASS RTC grouped GEMM failed: ", e.what());
    }
    return false;
  }
}
