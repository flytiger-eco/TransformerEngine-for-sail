/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#include "cutlass/gemm/kernel/nvte_grouped_tensor.hpp"
#include "mega_moe_grad_args.h"

namespace {
using namespace transformer_engine;
using Args = rtc::MegaMoEGradArgs;
using Element = cute::conditional_t<NVTE_MOE_FP16, cutlass::half_t, cutlass::bfloat16_t>;
constexpr bool kWgrad = NVTE_GG_TRANS_A, kRecompute = NVTE_GG_TRANS_B;
using Traits = grouped_gemm_rtc::GemmTraits<kWgrad, kRecompute, 89, Element, 256, 256, 64, 3,
                                            NVTE_GG_OUT_FP32>;
using Gemm = cutlass::gemm::kernel::NvteNativeGrouped<typename Traits::CollectiveMainloop,
                                                      typename Traits::CollectiveEpilogue, false,
                                                      NVTE_MOE_COMMON_N, NVTE_MOE_COMMON_K>;
using RaggedGemm =
    cutlass::gemm::kernel::NvteNativeGrouped<typename Traits::CollectiveMainloop,
                                             typename Traits::CollectiveEpilogue, false>;
constexpr int kWeightRows = NVTE_MOE_WEIGHT_ROWS;
static_assert(Gemm::MaxThreadsPerBlock == 512 && Gemm::SharedStorageSize == 196608);

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
      const bool four_columns = NVTE_MOE_FOUR_COLUMNS == 2
                                    ? count > 0 && count <= 512 && count % 64 == 0
                                    : NVTE_MOE_FOUR_COLUMNS == 1 && count == NVTE_MOE_COMMON_K;
      const int groups_of_four = four_columns ? ntiles / 4 : 0;
      const int per_row = ntiles - 3 * groups_of_four;
      int64_t tiles = int64_t(mtiles) * per_row;
      for (; work < tiles; work += gridDim.x) {
        const int mi = int(work / per_row), item = int(work % per_row);
        if constexpr (NVTE_MOE_FOUR_COLUMNS) {
          if (item < groups_of_four) {
            if constexpr (NVTE_MOE_FOUR_COLUMNS == 2)
              RaggedGemm::run_nexpand(e, mi, 4 * item, smem);
            else
              Gemm::run_nexpand(e, mi, 4 * item, smem);
            continue;
          }
        }
        const int ni = item + 3 * groups_of_four;
        if constexpr (NVTE_MOE_FOUR_COLUMNS == 2) {
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
extern "C" __global__ __launch_bounds__(512, 1) void NVTE_MOE_GRAD_KERNEL(Args p) {
  extern __shared__ char smem[];
  compute(p, smem);
}
