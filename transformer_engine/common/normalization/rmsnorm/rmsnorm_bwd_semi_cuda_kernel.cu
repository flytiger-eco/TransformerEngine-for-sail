/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"
#include "../kernel_traits.h"
#include "../rtc_dispatch.h"
#include "rmsnorm_bwd_kernels.cuh"

using namespace transformer_engine::normalization;

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int CTAS_PER_ROW, int WARPS_M, int WARPS_N,
          int BYTES_PER_LDG_MAIN, int BYTES_PER_LDG_FINAL, bool FUSED_ADD = false>
void launch_rmsnorm_bwd_tuned_(LaunchParams<BackwardKernelParams> &launch_params,
                               const bool configure_params) {  // NOLINT(*)
  using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t, HIDDEN_SIZE,
                                      CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG_MAIN>;
  auto kernel = &rmsnorm_bwd_tuned_kernel<Kernel_traits, FUSED_ADD>;

  if (configure_params) {
    int ctas_per_sm = 0;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES));
    launch_params.params.ctas_per_row = CTAS_PER_ROW;
    launch_params.params.ctas_per_col =
        launch_params.multiprocessorCount * ctas_per_sm / launch_params.params.ctas_per_row;
    if (Kernel_traits::CTAS_PER_ROW > 1) {
      launch_params.barrier_bytes = 2 * launch_params.params.ctas_per_col * sizeof(index_t);
      launch_params.workspace_bytes = launch_params.params.ctas_per_col * Kernel_traits::WARPS_M *
                                      Kernel_traits::CTAS_PER_ROW *
                                      sizeof(typename Kernel_traits::reduce_t) * 2;
    }
    launch_params.dgamma_part_bytes =
        launch_params.params.ctas_per_col * launch_params.params.cols * sizeof(compute_t);
    return;
  }

  if (Kernel_traits::SMEM_BYTES >= 48 * 1024) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         Kernel_traits::SMEM_BYTES));
  }
  auto stream = launch_params.stream;
  auto ctas_per_col = launch_params.params.ctas_per_col;
  auto ctas_per_row = launch_params.params.ctas_per_row;

  if (ctas_per_row == 1) {
    kernel<<<ctas_per_col, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES, stream>>>(
        launch_params.params);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    dim3 grid(ctas_per_row * ctas_per_col);
    dim3 block(Kernel_traits::THREADS_PER_CTA);
    void *params_ = reinterpret_cast<void *>(&launch_params.params);
    NVTE_CHECK_CUDA(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel), grid, block,
                                                reinterpret_cast<void **>(&params_),
                                                Kernel_traits::SMEM_BYTES, stream));
  }

  using Kernel_traits_f =
      Kernel_traits_finalize<HIDDEN_SIZE, weight_t, input_t, output_t, compute_t, index_t,
                             32 * 32,  // THREADS_PER_CTA
                             BYTES_PER_LDG_FINAL>;

  auto kernel_f = &rmsnorm_bwd_finalize_tuned_kernel<Kernel_traits_f>;
  kernel_f<<<Kernel_traits_f::CTAS, Kernel_traits_f::THREADS_PER_CTA, 0, stream>>>(
      launch_params.params);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int WARPS_M, int WARPS_N, int BYTES_PER_LDG_MAIN,
          int BYTES_PER_LDG_FINAL, bool FUSED_ADD = false>
void launch_rmsnorm_bwd_general_(LaunchParams<BackwardKernelParams> &launch_params,
                                 const bool configure_params) {  // NOLINT(*)
  auto ceil_div = [](int x, int y) -> int { return (x + y - 1) / y; };

  // Instantiate kernel
  using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t, HIDDEN_SIZE,
                                      1, WARPS_M, WARPS_N, BYTES_PER_LDG_MAIN>;
  auto kernel = &rmsnorm_bwd_general_kernel<Kernel_traits, FUSED_ADD>;

  // Configure kernel params
  const int rows = launch_params.params.rows;
  const int cols = launch_params.params.cols;
  int ctas_per_col = launch_params.params.ctas_per_col;
  int ctas_per_row = launch_params.params.ctas_per_row;
  if (configure_params) {
    int ctas_per_sm = 0;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, 0));
    const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
    ctas_per_row = ceil_div(cols, HIDDEN_SIZE);
    ctas_per_col = std::min(ceil_div(rows, WARPS_M), max_ctas / ctas_per_row);
    launch_params.params.ctas_per_row = ctas_per_row;
    launch_params.params.ctas_per_col = ctas_per_col;

    if (launch_params.params.ctas_per_row > 1) {
      launch_params.barrier_bytes = 2 * ctas_per_col * sizeof(index_t);
      launch_params.workspace_bytes =
          (ctas_per_col * WARPS_M * ctas_per_row * sizeof(typename Kernel_traits::reduce_t) * 2);
    }
    launch_params.dgamma_part_bytes =
        launch_params.params.ctas_per_col * launch_params.params.cols * sizeof(compute_t);
    return;
  }

  // Launch kernel
  auto stream = launch_params.stream;
  dim3 grid(ctas_per_row * ctas_per_col);
  dim3 block(Kernel_traits::THREADS_PER_CTA);
  if (ctas_per_row == 1) {
    kernel<<<grid, block, 0, stream>>>(launch_params.params);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    void *params_ = reinterpret_cast<void *>(&launch_params.params);
    NVTE_CHECK_CUDA(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel), grid, block,
                                                reinterpret_cast<void **>(&params_), 0, stream));
  }

  // Launch finalization kernel
  constexpr uint32_t WARPS_M_FINAL = 4;
  constexpr uint32_t WARPS_N_FINAL = 1;
  constexpr uint32_t ELTS_N_PER_CTA_FINAL =
      (Kernel_traits::THREADS_PER_WARP * WARPS_N_FINAL * BYTES_PER_LDG_FINAL / sizeof(compute_t));
  auto kernel_final =
      &rmsnorm_bwd_finalize_general_kernel<weight_t, compute_t, WARPS_M_FINAL, WARPS_N_FINAL,
                                           BYTES_PER_LDG_FINAL, Kernel_traits::THREADS_PER_WARP>;
  dim3 block_final(Kernel_traits::THREADS_PER_WARP * WARPS_N_FINAL, WARPS_M_FINAL);
  dim3 grid_final(ceil_div(cols, ELTS_N_PER_CTA_FINAL), 1);
  kernel_final<<<grid_final, block_final, 0, stream>>>(launch_params.params);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Two-level dispatch: stage (Backward / BackwardAdd) and launch type
// (tuned / general). The optional static fallback is selected only when
// NVTE_DISABLE_NVRTC=1.
#define REGISTER_NORM_LAUNCHER_RMSN_BWD_tuned_Backward(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE,                                                 \
                                                       CTAS_PER_ROW, WARPS_M, WARPS_N, BL_MAIN,                                                 \
                                                       BL_FINAL, STATIC_FALLBACK)                                                               \
  [[maybe_unused]] static const int                                                                                                             \
      _rmsn_bwd_tuned_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE##_##CTAS_PER_ROW##_##WARPS_M##_##WARPS_N##_##BL_MAIN##_##BL_FINAL = \
          ([] {                                                                                                                                 \
            ::transformer_engine::normalization::rtc_norm::register_rmsnorm_bwd_tuned(                                                          \
                TypeToDType<WTYPE>::value, TypeToDType<ITYPE>::value, TypeToDType<OTYPE>::value,                                                \
                TypeToDType<CTYPE>::value, HIDDEN_SIZE, CTAS_PER_ROW, WARPS_M, WARPS_N, BL_MAIN,                                                \
                BL_FINAL, false, STATIC_FALLBACK);                                                                                              \
            return 0;                                                                                                                           \
          })()
#define REGISTER_NORM_LAUNCHER_RMSN_BWD_tuned_BackwardAdd(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE,                                                  \
                                                          CTAS_PER_ROW, WARPS_M, WARPS_N, BL_MAIN,                                                  \
                                                          BL_FINAL, ADD_FLAG, STATIC_FALLBACK)                                                      \
  static_assert(ADD_FLAG, "RMSNorm BackwardAdd registrations require ADD_FLAG=true");                                                               \
  [[maybe_unused]] static const int                                                                                                                 \
      _rmsn_bwd_tuned_add_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE##_##CTAS_PER_ROW##_##WARPS_M##_##WARPS_N##_##BL_MAIN##_##BL_FINAL = \
          ([] {                                                                                                                                     \
            ::transformer_engine::normalization::rtc_norm::register_rmsnorm_bwd_tuned(                                                              \
                TypeToDType<WTYPE>::value, TypeToDType<ITYPE>::value, TypeToDType<OTYPE>::value,                                                    \
                TypeToDType<CTYPE>::value, HIDDEN_SIZE, CTAS_PER_ROW, WARPS_M, WARPS_N, BL_MAIN,                                                    \
                BL_FINAL, ADD_FLAG, STATIC_FALLBACK);                                                                                               \
            return 0;                                                                                                                               \
          })()
#define REGISTER_NORM_LAUNCHER_RMSN_BWD_general_Backward(                                                                        \
    HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, WARPS_M, WARPS_N, BL_MAIN, BL_FINAL, STATIC_FALLBACK)                               \
  [[maybe_unused]] static const int                                                                                              \
      _rmsn_bwd_general_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE##_##WARPS_M##_##WARPS_N##_##BL_MAIN##_##BL_FINAL = \
          ([] {                                                                                                                  \
            ::transformer_engine::normalization::rtc_norm::register_rmsnorm_bwd_general(                                         \
                TypeToDType<WTYPE>::value, TypeToDType<ITYPE>::value, TypeToDType<OTYPE>::value,                                 \
                TypeToDType<CTYPE>::value, HIDDEN_SIZE, WARPS_M, WARPS_N, BL_MAIN, BL_FINAL,                                     \
                false, STATIC_FALLBACK);                                                                                         \
            return 0;                                                                                                            \
          })()
#define REGISTER_NORM_LAUNCHER_RMSN_BWD_general_BackwardAdd(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE,                                        \
                                                            CTYPE, WARPS_M, WARPS_N, BL_MAIN,                                        \
                                                            BL_FINAL, ADD_FLAG, STATIC_FALLBACK)                                     \
  static_assert(ADD_FLAG, "RMSNorm BackwardAdd registrations require ADD_FLAG=true");                                                \
  [[maybe_unused]] static const int                                                                                                  \
      _rmsn_bwd_general_add_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE##_##WARPS_M##_##WARPS_N##_##BL_MAIN##_##BL_FINAL = \
          ([] {                                                                                                                      \
            ::transformer_engine::normalization::rtc_norm::register_rmsnorm_bwd_general(                                             \
                TypeToDType<WTYPE>::value, TypeToDType<ITYPE>::value, TypeToDType<OTYPE>::value,                                     \
                TypeToDType<CTYPE>::value, HIDDEN_SIZE, WARPS_M, WARPS_N, BL_MAIN, BL_FINAL,                                         \
                ADD_FLAG, STATIC_FALLBACK);                                                                                          \
            return 0;                                                                                                                \
          })()

#if NVTE_BUILD_LEGACY_STATIC_NORM
#define REGISTER_NORM_LAUNCHER(NORM_TYPE, NORM_STAGE, LAUNCH_TYPE, HIDDEN_SIZE, WTYPE, ITYPE,                   \
                               OTYPE, CTYPE, ...)                                                               \
  namespace {                                                                                                   \
  void                                                                                                          \
      norm_##NORM_TYPE##_##NORM_STAGE##_##LAUNCH_TYPE##_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(  \
          LaunchParams<NORM_STAGE##KernelParams> &launch_params, const bool configure_params) {                 \
    launch_rmsnorm_bwd_##LAUNCH_TYPE##_<WTYPE, ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE,                      \
                                        __VA_ARGS__>(launch_params, configure_params);                          \
  }                                                                                                             \
  REGISTER_NORM_LAUNCHER_RMSN_BWD_##LAUNCH_TYPE##_##NORM_STAGE(                                                 \
      HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, __VA_ARGS__,                                                     \
      norm_##NORM_TYPE##_##NORM_STAGE##_##LAUNCH_TYPE##_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE); \
  }  // namespace
#else
#define REGISTER_NORM_LAUNCHER(NORM_TYPE, NORM_STAGE, LAUNCH_TYPE, HIDDEN_SIZE, WTYPE, ITYPE,    \
                               OTYPE, CTYPE, ...)                                                \
  REGISTER_NORM_LAUNCHER_RMSN_BWD_##LAUNCH_TYPE##_##NORM_STAGE(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, \
                                                               CTYPE, __VA_ARGS__, nullptr)
#endif  // NVTE_BUILD_LEGACY_STATIC_NORM

// Create rmsnorm bwd tuned launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, ...
//                             WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 512, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 512, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 512, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 768, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 768, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 768, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);

// Backward tuned launch geometry, retuned on ZW-M890P (39 SM, 131072 regs/SM).
//
// The backward kernel keeps dy[], y[], dzy_sum[] and gamma[] live per thread, so
// WARPS_N fixes the register footprint through
//   LDGS = h * sizeof(input) / BYTES_PER_LDG / (WARPS_N * 32).
// At h=8192 the factory (1,4) needs LDGS=8 and spills: 256 regs + 224 B/thread
// under hgrtc, 256 regs + 348 B/thread under hgcc. Widening WARPS_N removes the
// spill entirely ((1,8): 184 regs / 0 B, (1,16): 104 regs / 0 B).
//
// Every ratio below is END-TO-END through the public C API (main + finalize),
// which matters: WARPS_M/WARPS_N set the persistent grid, the grid sets the
// number of dgamma_part partials, and that sets the finalize kernel's work. A
// main-kernel-only measurement ranks h=1024 backwards -- it prefers (1,2) by
// +11%, but (1,2) runs 64 threads/CTA, which raises occupancy ~5x, which makes
// finalize ~2x more expensive and the pair 13% slower overall.
//
// Measured with cudaGraph replay (adaptive L so a replay is >=400us, median of
// 15), 2 passes in opposite candidate order, rows=4096 and 16384, at 72-95% of
// the 2395 GB/s D2D roofline after the change:
//   h=8192 bf16  (1,4)->(1,16)  2.47x / 2.51x
//   h=8192 fp16  (1,4)->(1,16)  3.81x / 4.28x
//   h=8192 fp32  (1,4)->(1,8)   1.62x / 1.54x
//   h=4096 bf16  (1,4)->(1,8)   +2.9% / +2.2%
//   h=4096 fp16  (1,4)->(1,8)   +4.1% / +3.2%
//   h=1024 fp32  (4,1)->(4,4)   +5.3% / +5.7%
// Rejected because they did not survive the end-to-end measurement: h=1024
// bf16/fp16 (every candidate <= factory; (1,2) is 0.87x), h=4096 fp32 (best
// candidate +0.5%, inside noise), and h<=768 / h=2048 (left at factory).
// WARPS_M stays <=4 everywhere, matching the factory maximum, so the existing
// rows%4 tuned gate still bounds every row-loop trip count.
//
// Numerics: changing WARPS_N reassociates the fp32 row sums, so dx differs from
// the factory tree at the ulp level and ~0.006% of bf16/fp16 dx elements flip by
// 1 ulp. The final dgamma is bit-identical for bf16/fp16 (the output rounding
// absorbs it) and ulp-level for fp32.
// Evidence: TE_v218/rmsnorm_optimization.md, bench_rmsnorm/lib_ab/sweep_out/.
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 1024, fp32, fp32, fp32, fp32, 1, 4, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 1024, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 1024, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 2048, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 2048, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 2048, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 4096, fp16, fp16, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 4096, bf16, bf16, bf16, fp32, 1, 1, 8, 16, 4);

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 8192, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 8192, fp16, fp16, fp16, fp32, 1, 1, 16, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, tuned, 8192, bf16, bf16, bf16, fp32, 1, 1, 16, 16, 4);

// Create rmsnorm bwd general launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, ...
//                             WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 128, fp32, fp32, fp32, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 128, fp16, fp16, fp16, fp32, 4, 1, 8, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 128, fp16, fp32, fp16, fp32, 4, 1, 8, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 128, bf16, bf16, bf16, fp32, 4, 1, 8, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 128, bf16, fp32, bf16, fp32, 4, 1, 8, 4);

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 512, fp32, fp32, fp32, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 512, fp16, fp16, fp16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 512, fp16, fp32, fp16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 512, bf16, bf16, bf16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 512, bf16, fp32, bf16, fp32, 4, 1, 16, 4);

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 1024, fp32, fp32, fp32, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 1024, fp16, fp16, fp16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 1024, fp16, fp32, fp16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 1024, bf16, bf16, bf16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 1024, bf16, fp32, bf16, fp32, 4, 1, 16, 4);

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 2048, fp32, fp32, fp32, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 2048, fp16, fp16, fp16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 2048, fp16, fp32, fp16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 2048, bf16, bf16, bf16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 2048, bf16, fp32, bf16, fp32, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 4096, fp32, fp32, fp32, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 4096, fp16, fp16, fp16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 4096, fp16, fp32, fp16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 4096, bf16, bf16, bf16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(RMSNorm, Backward, general, 4096, bf16, fp32, bf16, fp32, 1, 4, 16, 4);

// Create fused rmsnorm bwd + add tuned launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, ...
//                             WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 512, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 512, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 512, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4,
                       true);

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 768, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 768, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 768, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4,
                       true);

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 1024, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 1024, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 1024, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4,
                       true);

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 2048, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 2048, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 2048, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4,
                       true);

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 4096, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 4096, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4,
                       true);

// BackwardAdd h=8192: the fused `add` operand keeps a second full row live, so
// the factory (1,4) spills harder here than in plain backward (1935us vs 1040us
// at rows=16384 bf16). (1,8) removes the spill and is the best worst-case pick
// at every dtype, end-to-end through the public API:
//   bf16 (1,4)->(1,8)  3.68x / 3.73x   (rows=4096 / 16384)
//   fp16 (1,4)->(1,8)  4.31x / 4.38x
//   fp32 (1,4)->(1,8)  2.94x / 2.80x
// (1,16) is within 0.7% of (1,8) at fp32 and behind it at bf16/fp16, so (1,8)
// is used for all three. h<=4096 BackwardAdd stays at the factory geometry: no
// candidate showed an end-to-end win there. Same ulp-level numerics as above.
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 8192, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 8192, fp16, fp16, fp16, fp32, 1, 1, 8, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, tuned, 8192, bf16, bf16, bf16, fp32, 1, 1, 8, 16, 4,
                       true);

// Create fused rmsnorm bwd + add general launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, ...
//                             WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 128, fp32, fp32, fp32, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 128, fp16, fp16, fp16, fp32, 4, 1, 8, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 128, fp16, fp32, fp16, fp32, 4, 1, 8, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 128, bf16, bf16, bf16, fp32, 4, 1, 8, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 128, bf16, fp32, bf16, fp32, 4, 1, 8, 4,
                       true);

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 512, fp32, fp32, fp32, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 512, fp16, fp16, fp16, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 512, fp16, fp32, fp16, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 512, bf16, bf16, bf16, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 512, bf16, fp32, bf16, fp32, 4, 1, 16, 4,
                       true);

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 1024, fp32, fp32, fp32, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 1024, fp16, fp16, fp16, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 1024, fp16, fp32, fp16, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 1024, bf16, bf16, bf16, fp32, 4, 1, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 1024, bf16, fp32, bf16, fp32, 4, 1, 16, 4,
                       true);

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 2048, fp32, fp32, fp32, fp32, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 2048, fp16, fp16, fp16, fp32, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 2048, fp16, fp32, fp16, fp32, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 2048, bf16, bf16, bf16, fp32, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 2048, bf16, fp32, bf16, fp32, 1, 4, 16, 4,
                       true);

REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 4096, fp32, fp32, fp32, fp32, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 4096, fp16, fp16, fp16, fp32, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 4096, fp16, fp32, fp16, fp32, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 4096, bf16, bf16, bf16, fp32, 1, 4, 16, 4,
                       true);
REGISTER_NORM_LAUNCHER(RMSNorm, BackwardAdd, general, 4096, bf16, fp32, bf16, fp32, 1, 4, 16, 4,
                       true);
