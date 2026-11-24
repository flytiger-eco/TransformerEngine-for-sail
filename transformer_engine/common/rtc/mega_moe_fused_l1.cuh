/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#include "cutlass/gemm/kernel/nvte_grouped_tensor.hpp"
#include "mega_moe_fused_args.h"
#include "mega_moe_fused_protocol.cuh"

namespace cute {

template <int Intermediate>
struct MegaMoEPairedBLoad {};
template <int Intermediate>
struct Copy_Traits<MegaMoEPairedBLoad<Intermediate>>
    : Copy_Traits<PPU0015_AIU_LOAD<Int<256 * 64 * 16>, cutlass::bfloat16_t, false, 256, 64, true>> {
  // The normal B atom loads 256 consecutive rows of one projection. The AIU
  // descriptor below instead views the original W1 allocation as two H×128
  // planes at the gate/up row stride. Both land in the ordinary swizzled
  // 256×64 shared tile and the BF16 matrix pipeline is unchanged.
  template <class TS, class LS, class TD, class LD>
  CUTE_DEVICE friend void copy_unpack(Copy_Traits const& traits, Tensor<TS, LS> const& src,
                                      Tensor<TD, LD>& dst) {
    auto* out = raw_pointer_cast(dst.data());
    auto const* input = src.data().ptr_.get();
    auto c = src.data().coord_;
    int k = get<0>(c);
    int n = get<1>(c) / 2;
    auto* base = static_cast<const cutlass::bfloat16_t*>(input) + size_t(n) * NVTE_MOE_FUSED_HIDDEN;
    constexpr uint64_t row = uint64_t(sizeof(cutlass::bfloat16_t)) * NVTE_MOE_FUSED_HIDDEN;
    constexpr uint64_t part = uint64_t(Intermediate) * row;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b16 "
        "[%0], [%1], {%2,%3,%4}, {%5,%6,%7}, {%8,%9}, {%10,%11,%12}, %13;" ::"r"(out),
        "l"(base), "r"(NVTE_MOE_FUSED_HIDDEN), "r"(128), "r"(2), "r"(64), "r"(128), "r"(2),
        "l"(row), "l"(part), "r"(k), "r"(0), "r"(0), "r"(0));
  }
};
}  // namespace cute

namespace transformer_engine::rtc::fused_moe::l1 {

using Bf16 = cutlass::bfloat16_t;
constexpr int kRows = 256, kCols = 256;
constexpr int kHidden = NVTE_MOE_FUSED_HIDDEN;
constexpr int kIntermediate = NVTE_MOE_FUSED_INTERMEDIATE;
static_assert(kHidden % 256 == 0 && kIntermediate % kCols == 0);
using Traits = grouped_gemm_rtc::GemmTraits<false, true, 89, Bf16, kRows, kCols, 64, 3>;
using Gemm =
    cutlass::gemm::kernel::NvteNativeGrouped<Traits::CollectiveMainloop, Traits::CollectiveEpilogue,
                                             false, 2 * kIntermediate, kHidden>;
static_assert(Gemm::MaxThreadsPerBlock == 512 && Gemm::SharedStorageSize == 196608);

// The scalar used by the training activation is sigmoid(float(BF16_gate)).
// Its FP32 result is exactly 0.5 for |gate| <= 2^-24, 1 for gate >= 128,
// and 0 for gate <= -128. These cutoffs leave fewer than 4096 BF16 values
// per sign. The original gate still participates in multiplication, retaining
// NaN/Inf behavior. Initialize once per persistent CTA, outside GEMM storage.
constexpr unsigned kSmallGate = 0x3380;  // BF16 bits for 2^-24
constexpr unsigned kLargeGate = 0x4300;  // BF16 bits for 128
constexpr unsigned kValuesPerSign = 4096;
static_assert(kLargeGate - kSmallGate < kValuesPerSign);

__device__ __forceinline__ float* sigmoid_table(char* smem) {
  return reinterpret_cast<float*>(smem + Gemm::SharedStorageSize);
}

__device__ __forceinline__ void initialize_sigmoid_table(char* smem) {
  auto* table = sigmoid_table(smem);
  for (unsigned i = threadIdx.x; i < 2 * kValuesPerSign; i += Gemm::MaxThreadsPerBlock) {
    unsigned bits = kSmallGate + (i % kValuesPerSign) + (i >= kValuesPerSign ? 0x8000u : 0u);
    float value = float(Bf16::bitcast(static_cast<uint16_t>(bits)));
    table[i] = 1.f / (1.f + expf(-value));
  }
  __syncthreads();
}

__device__ __forceinline__ float sigmoid(Bf16 gate, const float* table) {
  unsigned bits = gate.raw();
  unsigned magnitude = min(max(bits & 0x7fffu, kSmallGate), kLargeGate);
  return table[magnitude - kSmallGate + ((bits & 0x8000u) >> 3)];
}

struct Tile {
  int expert, valid_rows, first_expert_row;
};

// The expert tables fit in one warp. A thread's inclusive prefix gives its
// expert's past-the-end tile number; the ballot maps a tile ID to an expert.
__device__ __forceinline__ Tile find_tile(int tile, int prefix, int total) {
  unsigned end_after = __ballot_sync(0xffffffffu, prefix > tile);
  int expert = __ffs(end_after) - 1;
  int first = __shfl_sync(0xffffffffu, prefix, expert > 0 ? expert - 1 : 0);
  if (!expert) first = 0;
  int in_expert = (tile - first) * kRows;
  return {expert, min(kRows, __shfl_sync(0xffffffffu, total, expert) - in_expert), in_expert};
}

__device__ __forceinline__ void copy_tile(const MegaMoEFusedArgs& p, int tile, Tile t) {
  int warp = threadIdx.x / 32;
  int global_expert = p.rank * p.experts + t.expert;
  auto* pool = static_cast<Bf16*>(p.packed_x) + int64_t(tile) * kRows * kHidden;
  // Coalesce the peer's route metadata before streaming any token payloads.
  // Issuing one dependent fabric load per warp would serialize route lookup.
  extern __shared__ char smem[];
  auto* slot_by_row = reinterpret_cast<int*>(smem);
  auto* source_by_row = slot_by_row + kRows;
  if (threadIdx.x < t.valid_rows) {
    int row = threadIdx.x;
    int expert_row = t.first_expert_row + row;
    int src = 0, first = 0, prefix = 0;
    // All lanes must execute the same count loads. A divergent early exit in
    // this search can make a mixed-source warp retain the next source's route
    // base while retaining the current source's row ordinal on the PPU.
#pragma unroll
    for (int r = 0; r < NVTE_MOE_FUSED_WORLD; ++r) {
      int count = __ldbl(p.counts[r] + global_expert);
      bool belongs = expert_row >= prefix && expert_row < prefix + count;
      src = belongs ? r : src;
      first = belongs ? prefix : first;
      prefix += count;
    }
    int idx = expert_row - first;
    const auto rec = __ldbl(p.source_routes[src] +
                            static_cast<long long>(global_expert) * p.topk * p.capacity + idx);
    slot_by_row[row] = int(rec);
    source_by_row[row] = src;
    p.packed_probs[tile * kRows + row] = __uint_as_float(static_cast<unsigned>(rec >> 32));
  }
  if (threadIdx.x >= t.valid_rows && threadIdx.x < kRows)
    p.packed_probs[tile * kRows + threadIdx.x] = 0.f;
  __syncthreads();
  for (int row = warp; row < t.valid_rows; row += Gemm::MaxThreadsPerBlock / 32) {
    int src = source_by_row[row];
    int token = slot_by_row[row] / p.topk;
    const void* input = src == p.rank && can_read_local_input(p) ? p.x : p.source_x[src];
    const Bf16* x = static_cast<const Bf16*>(input) + int64_t(token) * kHidden;
    Bf16* dst = pool + int64_t(row) * kHidden;
    // One warp issues up to eight independent 512-byte segments before waiting
    // on a load. The intrinsic takes a common base and moves 16 bytes per lane.
    constexpr int group = kHidden / 256 < 8 ? kHidden / 256 : 8;
    for (int first = 0; first < kHidden / 256; first += group) {
      uint4 data[group];
      if (src == p.rank) {
#pragma unroll
        for (int j = 0; j < group; ++j)
          if (first + j < kHidden / 256)
            data[j] = __ppu_global_ldcg_bulk_b32x4(x + (first + j) * 256);
      } else {
#pragma unroll
        for (int j = 0; j < group; ++j)
          if (first + j < kHidden / 256)
            data[j] = __ppu_remote_load_bulk_b32x4(x + (first + j) * 256);
      }
#pragma unroll
      for (int j = 0; j < group; ++j)
        if (first + j < kHidden / 256)
          __ppu_global_store_bulk_volatile_b32x4(dst + (first + j) * 256, data[j]);
    }
  }
  // Every expert occupies complete 256-row tiles. Initialize the tail explicitly
  // so both AIU GEMMs can use full-tile descriptors, including ragged experts.
  for (int row = t.valid_rows + warp; row < kRows; row += Gemm::MaxThreadsPerBlock / 32) {
    Bf16* dst = pool + int64_t(row) * kHidden;
    for (int col = 0; col < kHidden; col += 256)
      __ppu_global_store_bulk_volatile_b32x4(dst + col, make_uint4(0, 0, 0, 0));
  }
  release_warp_writes();
  __syncthreads();
  if (threadIdx.x == 0) publish(p.tile_ready + tile, p.epoch);
  __syncthreads();
}

__device__ __forceinline__ void project(const MegaMoEFusedArgs& p, int tile, int column, Tile t,
                                        char* smem) {
  using namespace cute;
  using Normal = Traits::CollectiveMainloop;
  using BCopy = typename Normal::GmemTiledCopyB;
  using CopyB = cute::TiledCopy<cute::Copy_Atom<cute::MegaMoEPairedBLoad<kIntermediate>, Bf16>,
                                typename BCopy::TiledLayout_TV, typename BCopy::Tiler_MN>;
  using Mainloop = cutlass::gemm::collective::CollectiveMma<
      typename Normal::ArchTag, typename Normal::DispatchPolicy, typename Normal::TileShape,
      typename Normal::ElementA, typename Normal::StrideA, typename Normal::ElementB,
      typename Normal::StrideB, typename Normal::TiledMma, typename Normal::GmemTiledCopyA,
      typename Normal::SmemLayoutAtomA, typename Normal::SmemCopyAtomA, typename Normal::TransformA,
      CopyB, typename Normal::SmemLayoutAtomB, typename Normal::SmemCopyAtomB,
      typename Normal::TransformB>;

  using Mma = Mainloop::TiledMma;
  using Shape = Mainloop::TileShape;
  const int start = tile * kRows;
  auto const* x = static_cast<const Bf16*>(p.packed_x) + int64_t(start) * kHidden;
  auto const* w = static_cast<const Bf16*>(p.w1[t.expert]);
  const auto da = Gemm::template packed_stride<typename Mainloop::InternalStrideA>(kRows, kHidden);
  const auto db =
      Gemm::template packed_stride<typename Mainloop::InternalStrideB>(2 * kIntermediate, kHidden);
  auto a = make_tensor(make_gmem_ptr(x), make_shape(Int<kRows>{}, Int<kHidden>{}), da);
  auto b = make_tensor(make_gmem_ptr(w), make_shape(Int<2 * kIntermediate>{}, Int<kHidden>{}), db);
  auto coord = make_coord(0, column, _);
  auto ga = local_tile(make_mix_tensor_like(a), Shape{}, coord, Step<_1, Underscore, _1>{});
  auto gb = local_tile(make_mix_tensor_like(b), Shape{}, coord, Step<Underscore, _1, _1>{});
  Mma mma;
  auto acc = partition_fragment_C(mma, take<0, 2>(Shape{}));
  clear(acc);
  Mainloop mainloop(make_shape(kRows, 2 * kIntermediate, kHidden), da, db);
  mainloop(acc, ga, gb, acc, make_coord_iterator(shape<2>(ga)), Int<kHidden / 64>{},
           make_coord(kRows, 2 * kIntermediate - column * kCols, 0), int(threadIdx.x), smem);
  cp_async_wait<0>();
  __syncthreads();
  // For this fixed MMA layout each lane owns 128 FP32 accumulators: the
  // upper 64 correspond to the same rows and columns shifted by 128.
  // Consecutive register pairs produce adjacent BF16 output columns.
  static_assert(decltype(size(acc))::value == 128);
  auto ident = make_identity_tensor(take<0, 2>(Shape{}));
  auto coords = mma.get_thread_slice(int(threadIdx.x)).partition_C(ident);
  Bf16* result =
      static_cast<Bf16*>(p.activation) + int64_t(start) * kIntermediate + column * (kCols / 2);
  // Each thread owns eight rows and four column pairs per row. Iterating by
  // row reuses one coherent probability load and one tail check for all four
  // pairs. The register offsets follow the fixed 256×256 TC02 MMA layout.
  CUTLASS_PRAGMA_UNROLL
  for (int row = 0; row < 8; ++row) {
    int first = (row / 2) * 8 + (row % 2) * 2;
    int r = get<0>(coords(first));
    if (r < kRows) {
      float prob = __ldbl(p.packed_probs + start + r);
      CUTLASS_PRAGMA_UNROLL
      for (int pair = 0; pair < 4; ++pair) {
        int j = first + (pair / 2) * 32 + (pair % 2) * 4;
        int c = get<1>(coords(j));
        cutlass::AlignedArray<Bf16, 2> v;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < 2; ++k) {
          Bf16 gate(acc(j + k));
          float up = float(Bf16(acc(j + 64 + k)));
          v[k] = Bf16((float(gate) * sigmoid(gate, sigmoid_table(smem)) * up) * prob);
        }
        *reinterpret_cast<cutlass::AlignedArray<Bf16, 2>*>(result + int64_t(r) * kIntermediate +
                                                           c) = v;
      }
    }
  }
  __syncthreads();
}

}  // namespace transformer_engine::rtc::fused_moe::l1

extern "C" __global__ __launch_bounds__(512) void nvte_mega_moe_fused_l1(
    transformer_engine::rtc::MegaMoEFusedArgs p) {
  using namespace transformer_engine::rtc::fused_moe;
  using namespace transformer_engine::rtc::fused_moe::l1;
  if (*p.status) return;
  extern __shared__ char smem[];
  int lane = threadIdx.x % 32;
  int expert = p.rank * p.experts + lane;
  int total = 0;
  if (lane < p.experts) {
    for (int src = 0; src < p.world; ++src) total += __ldbl(p.counts[src] + expert);
  }
  int prefix = (total + kRows - 1) / kRows;
#pragma unroll
  for (int delta = 1; delta < 32; delta *= 2) {
    int other = __shfl_up_sync(0xffffffffu, prefix, delta);
    if (lane >= delta) prefix += other;
  }
  int tiles = __shfl_sync(0xffffffffu, prefix, 31);
  // At most one CTA per physical CU. Copy CTAs can always make forward
  // progress while consumers await individual M tiles. They join the same
  // compute queue when all of their assigned copies have completed.
  int producers = min(NVTE_MOE_FUSED_COPY_SMS, int(gridDim.x) - 1);
  if (int(blockIdx.x) < producers) {
    for (int tile = blockIdx.x; tile < tiles; tile += producers)
      copy_tile(p, tile, find_tile(tile, prefix, total));
  }
  initialize_sigmoid_table(smem);
  constexpr int columns = 2 * kIntermediate / kCols;
  // The queue only occupies smem while the CTA is between two matrix tiles.
  auto* work = reinterpret_cast<int*>(smem);
  while (true) {
    if (threadIdx.x == 0) *work = atomicAdd(p.work_count, 1);
    __syncthreads();
    int id = *work;
    __syncthreads();
    if (id >= tiles * columns) break;
    int tile = id / columns, column = id % columns;
    auto t = find_tile(tile, prefix, total);
    if (threadIdx.x == 0) wait_epoch(p.tile_ready + tile, p.epoch);
    __syncthreads();
    project(p, tile, column, t, smem);
  }
}
