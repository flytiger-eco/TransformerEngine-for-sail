/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#include "cutlass/gemm/kernel/nvte_grouped_tensor.hpp"
#include "mega_moe_grad_args.h"

#ifndef NVTE_MOE_WGRAD_M128
#define NVTE_MOE_WGRAD_M128 0
#endif

namespace {
using namespace transformer_engine;
using Args = rtc::MegaMoEGradArgs;
using Element = cute::conditional_t<NVTE_MOE_FP16, cutlass::half_t, cutlass::bfloat16_t>;
constexpr bool kWgrad = NVTE_GG_TRANS_A, kRecompute = NVTE_GG_TRANS_B;
constexpr int kTileM = NVTE_MOE_WGRAD_M128 ? 128 : 256;
constexpr int kStages = NVTE_MOE_WGRAD_M128 ? 2 : 3;
using Traits = grouped_gemm_rtc::GemmTraits<kWgrad, kRecompute, 89, Element, kTileM, 256, 64,
                                            kStages, NVTE_GG_OUT_FP32>;
using DefaultEpilogue = Traits::CollectiveEpilogue;

// Exchange one packed word between neighboring lanes to store each 8-byte span.
struct WgradRegisterEpilogue : DefaultEpilogue {
  template <class Shape, class Coord, class Acc, class Mma>
  CUTLASS_DEVICE static void store_full_tile(Element *dst, int m, int n, Shape shape, Coord coord,
                                             const Acc &acc, Mma mma, int tid) {
    using namespace cute;
    // The dense host admission verifies every expert destination and the full N tile.
#if !NVTE_MOE_DENSE_WGRAD
    if (reinterpret_cast<uintptr_t>(dst) % 16 || n % 8) {
      DefaultEpilogue::store_full_tile(dst, m, n, shape, coord, acc, mma, tid);
      return;
    }
#endif
    auto copy = make_tiled_copy_C(typename DefaultEpilogue::CopyAtomR2S{}, mma);
    // The C copy tile covers 32x64 elements for M128 and 64x64 for M256.
    // Both layouts give neighboring lanes the pairs needed for the same 8-byte store.
    using Copy = decltype(copy);
    using CopyM = Int<kTileM / 4>;
    using ThreadMap = Layout<cute::Shape<_4, _8, Int<kTileM / 64>, _4>,
                             Stride<Int<kTileM / 2>, _1, _16, Int<kTileM * 4>>>;
    using ValueMap = Layout<cute::Shape<_2, _2, _2>, Stride<CopyM, _8, Int<kTileM * 2>>>;
    constexpr auto tv = Copy::get_layoutD_TV();
    static_assert(Copy::AtomNumThr::value == 1 && Copy::AtomNumVal::value == 1 &&
                      is_same_v<typename Copy::Tiler_MN, cute::Shape<CopyM, _64>> &&
                      is_same_v<decltype(coalesce(get<0>(tv))), ThreadMap> &&
                      is_same_v<decltype(coalesce(get<1>(tv))), ValueMap>,
                  "Wgrad register exchange requires the native C-fragment layout");
    auto thread = copy.get_thread_slice(tid);
    auto regs = thread.retile_S(acc);
    auto index = thread.partition_D(make_identity_tensor(take<0, 2>(shape)));
    static_assert(decltype(size(regs))::value == 128 && decltype(size<0>(shape))::value == kTileM &&
                  decltype(size<1>(shape))::value == 256);
    auto *tile = dst + static_cast<int64_t>(get<0>(coord)) * kTileM * n +
                 static_cast<int64_t>(get<1>(coord)) * 256;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(regs); i += 8) {
      CUTLASS_PRAGMA_UNROLL
      for (int mrow = 0; mrow != 2; ++mrow) {
        uint32_t words[2];
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j != 2; ++j) {
          const int value = i + mrow * 2 + j * 4;
          cutlass::Array<float, 2> values;
          values[0] = regs(value);
          values[1] = regs(value + 1);
          auto pair = cutlass::NumericArrayConverter<Element, float, 2>{}(values);
          words[j] = __builtin_bit_cast(uint32_t, pair);
        }
        const bool odd = tid & 1;
        const uint32_t neighbor = __shfl_xor_sync(0xffffffffu, odd ? words[0] : words[1], 1);
        const auto addr = index(i + mrow * 2);
        const int row = get<0>(addr);
        const int col = get<1>(addr) + (tid & 1) * 6;
        *reinterpret_cast<uint2 *>(tile + static_cast<int64_t>(row) * n + col) =
            make_uint2(odd ? neighbor : words[0], odd ? words[1] : neighbor);
      }
    }
  }
};
using WgradEpilogue = cute::conditional_t<kWgrad && !NVTE_GG_OUT_FP32 && !NVTE_MOE_FP16,
                                          WgradRegisterEpilogue, DefaultEpilogue>;
using Gemm =
    cutlass::gemm::kernel::NvteNativeGrouped<typename Traits::CollectiveMainloop, WgradEpilogue,
                                             false, NVTE_MOE_COMMON_N, NVTE_MOE_COMMON_K>;
using RaggedGemm = cutlass::gemm::kernel::NvteNativeGrouped<typename Traits::CollectiveMainloop,
                                                            WgradEpilogue, false>;
constexpr int kWeightRows = NVTE_MOE_WEIGHT_ROWS;
static_assert(Gemm::MaxThreadsPerBlock == 2 * kTileM);
static_assert(Gemm::SharedStorageSize == (NVTE_MOE_WGRAD_M128 ? 98304 : 196608));

#if NVTE_MOE_WGRAD_M128
// Every expert owns a complete, identically shaped dW matrix. A work item
// always owns the same columns, even when a ragged K uses individual GEMMs.
__device__ void compact_wgrad(const Args &p, char *smem) {
  constexpr int n = NVTE_MOE_COMMON_N, m = NVTE_MOE_WEIGHT_ROWS;
  constexpr int ntiles = n / RaggedGemm::BN, groups_of_four = ntiles / 4;
  constexpr int items_per_row = groups_of_four + ntiles % 4;
  constexpr int per_group = (m / RaggedGemm::BM) * items_per_row;
  static_assert(n % RaggedGemm::BN == 0 && m % RaggedGemm::BM == 0);
  for (int work = blockIdx.x; work < p.groups * per_group; work += gridDim.x) {
    const int group = work / per_group;
    const int row = work % per_group / items_per_row;
    int item = work % items_per_row;
    if constexpr (m == 5120 && n == 2304) {
      // Nine column tiles form two four-column groups and a one-column tail.
      // Rotate the tail between CTAs when the persistent grid is divisible by three.
      if (gridDim.x % items_per_row == 0) {
        const int phase = work / items_per_row % items_per_row;
        item += phase;
        if (item >= items_per_row) item -= items_per_row;
      }
    }
    const int first_col = item < groups_of_four ? 4 * item : item + 3 * groups_of_four;
    const int begin = min(max(0, p.offsets[group]), p.rows);
    const int end = min(max(begin, p.offsets[group + 1]), p.rows);
    cutlass::gemm::kernel::NativeExpert e{};
    e.a = reinterpret_cast<uint64_t>(p.a) + int64_t(begin) * m * sizeof(Element);
    e.b = reinterpret_cast<uint64_t>(p.b) + int64_t(begin) * n * sizeof(Element);
    e.d = reinterpret_cast<uint64_t>(p.expert[group]);
    e.m = m;
    e.n = n;
    e.k = min(end - begin, max(0, p.reduction_rows[group]));
    e.alpha = 1.f;
    if (item < groups_of_four && e.k > 0 && e.k <= 512 && e.k % 64 == 0) {
      RaggedGemm::run_nexpand(e, row, first_col, smem);
    } else {
      const int columns = item < groups_of_four ? 4 : 1;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int col = 0; col < columns; ++col)
        RaggedGemm::run_tile<true>(e, row, first_col + col, smem);
    }
  }
}
#endif

#if NVTE_MOE_DENSE_WGRAD
// Dense wgrad has fixed, complete M/N tiles. The expert's true K still comes
// from its row range and can be empty, short, or not divisible by BK.
__device__ __forceinline__ void dense_wgrad(const Args &p, char *smem) {
  using namespace cute;
  static_assert(kWgrad && !NVTE_GG_OUT_FP32 && !NVTE_MOE_FP16);
  using Loop = Traits::CollectiveMainloop;
  using Shape = Loop::TileShape;
  constexpr int m = NVTE_MOE_WEIGHT_ROWS, n = NVTE_MOE_COMMON_N;
  constexpr int mt = m / 256, nt = n / 256, band = 8;
  static_assert(m > 0 && n > 0 && m % 256 == 0 && n % 256 == 0);
  static_assert(sizeof(Loop::SharedStorage) == Gemm::SharedStorageSize);
  static_assert(decltype(size(Loop::TiledMma{}))::value == 512);
  int64_t work = blockIdx.x;
  const int mean =
      (min(max(0, p.offsets[p.groups]), p.rows) - min(max(0, p.offsets[0]), p.rows)) / p.groups;
  for (int pass = 0; pass < 2; ++pass) {
    for (int g = 0; g < p.groups; ++g) {
      const int begin = min(max(0, p.offsets[g]), p.rows);
      const int end = min(max(begin, p.offsets[g + 1]), p.rows);
      const int k = min(end - begin, max(0, p.reduction_rows[g]));
      if ((k < mean) != (pass == 1)) continue;
      const auto *a = static_cast<const Element *>(p.a) + static_cast<int64_t>(begin) * m;
      const auto *b = static_cast<const Element *>(p.b) + static_cast<int64_t>(begin) * n;
      auto *d = static_cast<Element *>(p.expert[g]);
      for (; work < static_cast<int64_t>(mt) * nt; work += gridDim.x) {
        int row, col;
        const int strip = static_cast<int>(work / (static_cast<int64_t>(band) * nt)) * band;
        const int local = static_cast<int>(work - static_cast<int64_t>(strip) * nt);
        if (strip + band <= mt) {
          row = strip + local % band;
          col = local / band;
        } else {
          const int left = mt - strip;
          row = strip + local % left;
          col = local / left;
        }
        auto coord = make_coord(row, col, _, 0);
        Loop::TiledMma mma;
        auto acc = partition_fragment_C(mma, take<0, 2>(Shape{}));
        clear(acc);
        if (k > 0) {
          auto da = Gemm::template packed_stride<Loop::InternalStrideA>(m, k);
          auto db = Gemm::template packed_stride<Loop::InternalStrideB>(n, k);
          auto ta = make_tensor(make_gmem_ptr(a), make_shape(Int<m>{}, k), da);
          auto tb = make_tensor(make_gmem_ptr(b), make_shape(Int<n>{}, k), db);
          auto ga = local_tile(make_mix_tensor_like(ta), Shape{}, take<0, 3>(coord),
                               Step<_1, Underscore, _1>{});
          auto gb = local_tile(make_mix_tensor_like(tb), Shape{}, take<0, 3>(coord),
                               Step<Underscore, _1, _1>{});
          const int kt = size<2>(ga);
          Loop loop(make_shape(Int<m>{}, Int<n>{}, k), da, db);
          loop(acc, ga, gb, acc, make_coord_iterator(shape<2>(ga)), kt,
               make_coord(256, 256, k - static_cast<int64_t>(kt) * 64), threadIdx.x, smem);
        }
        cp_async_wait<0>();
        __syncthreads();
        // No warp reads shared memory after this barrier. The next tile may
        // fill it while slower warps finish their register-to-global stores.
        WgradEpilogue::store_full_tile(d, m, n, Shape{}, coord, acc, mma, threadIdx.x);
      }
      work -= static_cast<int64_t>(mt) * nt;
    }
  }
}
#endif

__device__ void compute(const Args &p, char *smem) {
  using Expert = cutlass::gemm::kernel::NativeExpert;
  // For NN and TN, A is dY [R,O]. Recompute receives X [R,I] and W [O,I].
  const int row_a = kWeightRows ? kWeightRows : (kRecompute ? p.input_width : p.output_width);
  const int n = kWeightRows ? NVTE_MOE_COMMON_N : (kRecompute ? p.output_width : p.input_width);
  const int ntiles = (n + Gemm::BN - 1) / Gemm::BN;
  const auto *a = static_cast<const Element *>(p.a);
  int64_t work = blockIdx.x;
  // A weight tile reduces all rows for one expert. Schedule the long
  // reductions first to avoid leaving a single CU busy at the end.
  const int passes = kWgrad && p.groups > 1 ? 2 : 1;
  const int mean_rows =
      kWgrad ? (min(max(0, p.offsets[p.groups]), p.rows) - min(max(0, p.offsets[0]), p.rows)) /
                   p.groups
             : 0;
  for (int pass = 0; pass < passes; ++pass) {
    for (int g = 0; g < p.groups; ++g) {
      const int begin = min(max(0, p.offsets[g]), p.rows);
      const int end = min(max(begin, p.offsets[g + 1]), p.rows);
      const int count =
          kWgrad && p.reduction_rows ? min(end - begin, max(0, p.reduction_rows[g])) : end - begin;
      if (passes == 2 && (count < mean_rows) != (pass == 1)) continue;
      Expert e{};
      e.m = kWeightRows ? kWeightRows : (kWgrad ? p.output_width : count);
      e.n = n;
      e.k = kWgrad ? count : row_a;
      e.alpha = 1;
      e.a = reinterpret_cast<uint64_t>(a) + int64_t(begin) * row_a * sizeof(Element);
      if constexpr (kWgrad) {
        // dW = dY^T X (one writer per output tile, FP32 reduction). K=0
        // writes zero or leaves an accumulation destination untouched.
        e.b = reinterpret_cast<uint64_t>(p.b) + int64_t(begin) * n * sizeof(Element);
        e.d = reinterpret_cast<uint64_t>(p.expert[g]);
        e.c = e.d;
        e.beta = p.accumulate ? 1.f : 0.f;
        if (!count && p.accumulate) continue;
      } else {
        e.b = reinterpret_cast<uint64_t>(p.expert[g]);
        e.d = reinterpret_cast<uint64_t>(p.d) + int64_t(begin) * n * sizeof(Element);
      }
      int mtiles = int((int64_t(e.m) + Gemm::BM - 1) / Gemm::BM);
      // With four complete N tiles per work item, keep the MMA pipeline
      // live across columns instead of draining it after each epilogue.
      const bool four_columns = NVTE_MOE_FOUR_COLUMNS >= 2
                                    ? count > 0 && count <= 512 && count % 64 == 0
                                    : NVTE_MOE_FOUR_COLUMNS == 1 && count == NVTE_MOE_COMMON_K;
      int groups_of_four = four_columns ? ntiles / 4 : 0;
      if constexpr (NVTE_MOE_FOUR_COLUMNS == 3) {
        if (count >= 960 && count <= 1152 && count % 64 == 0) groups_of_four = ntiles / 4;
      }
      const int per_row = ntiles - 3 * groups_of_four;
      int64_t tiles = int64_t(mtiles) * per_row;
      for (; work < tiles; work += gridDim.x) {
        const int mi = int(work / per_row), item = int(work % per_row);
        if constexpr (NVTE_MOE_FOUR_COLUMNS) {
          if (item < groups_of_four) {
            if constexpr (NVTE_MOE_FOUR_COLUMNS >= 2)
              RaggedGemm::run_nexpand(e, mi, 4 * item, smem);
            else
              Gemm::run_nexpand(e, mi, 4 * item, smem);
            continue;
          }
        }
        const int ni = item + 3 * groups_of_four;
        if constexpr (NVTE_MOE_FOUR_COLUMNS >= 2) {
          RaggedGemm::run_tile(e, mi, ni, smem);
          continue;
        }
        if constexpr (kWeightRows != 0) {
          if (count != NVTE_MOE_COMMON_K) {
            RaggedGemm::run_tile(e, mi, ni, smem);
            continue;
          }
        }
        Gemm::run_tile(e, mi, ni, smem);
      }
      work -= tiles;
    }
  }
}
}  // namespace

#if NVTE_GG_TRANS_A
#define NVTE_MOE_GRAD_KERNEL nvte_mega_moe_tensor_wgrad
#elif NVTE_GG_TRANS_B
#define NVTE_MOE_GRAD_KERNEL nvte_mega_moe_tensor_recompute
#else
#define NVTE_MOE_GRAD_KERNEL nvte_mega_moe_tensor_dgrad
#endif
extern "C" __global__ __launch_bounds__(2 * kTileM, 1) void NVTE_MOE_GRAD_KERNEL(Args p) {
  extern __shared__ char smem[];
#if NVTE_MOE_WGRAD_M128
  compact_wgrad(p, smem);
#elif NVTE_MOE_DENSE_WGRAD
  dense_wgrad(p, smem);
#else
  compute(p, smem);
#endif
}
