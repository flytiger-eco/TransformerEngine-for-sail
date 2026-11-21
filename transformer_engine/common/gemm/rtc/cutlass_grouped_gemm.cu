/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Compiled by NVRTC with the same tile, layouts and output type as the host launcher.
#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#if NVTE_GG_GROUPED_TENSOR || NVTE_GG_N_EXPAND == 4 || NVTE_GG_COMMON_K
#include "cutlass/gemm/kernel/nvte_grouped_tensor.hpp"
#endif

namespace {
using Element = cute::conditional_t<NVTE_GG_DTYPE == 1, cutlass::half_t, cutlass::bfloat16_t>;
using Traits = transformer_engine::grouped_gemm_rtc::GemmTraits<
    NVTE_GG_TRANS_A, NVTE_GG_TRANS_B, NVTE_GG_ARCH, Element, NVTE_GG_TILE_M, NVTE_GG_TILE_N,
    NVTE_GG_TILE_K, NVTE_GG_STAGES, NVTE_GG_OUT_FP32>;

#if NVTE_GG_GROUPED_TENSOR
using GemmKernel =
    cutlass::gemm::kernel::NvteNativeGrouped<Traits::CollectiveMainloop, Traits::CollectiveEpilogue,
                                             NVTE_GG_N_EXPAND == 4>;
#elif NVTE_GG_N_EXPAND == 4 || NVTE_GG_COMMON_K
using GemmKernel =
    cutlass::gemm::kernel::NvteListGrouped<Traits::GemmKernel, Traits::CollectiveMainloop,
                                           Traits::CollectiveEpilogue, NVTE_GG_COMMON_N,
                                          NVTE_GG_COMMON_K>;
#else
using GemmKernel = Traits::GemmKernel;
#endif

}  // namespace

// The kernel function name is rewritten per-instantiation by the host TU,
// which prepends an `#undef`/`#define NVTE_GG_KERNEL_NAME <ident>` pair to
// the source string before calling NVRTC. This is done at the source level
// (not via -D) because PPU's hgrtc silently drops macro values passed via
// -D when they contain long identifiers (verified against cubin dumps).
#ifndef NVTE_GG_KERNEL_NAME
#define NVTE_GG_KERNEL_NAME nvte_cutlass_grouped_gemm_rtc_kernel
#endif

extern "C" __global__ void NVTE_GG_KERNEL_NAME(typename GemmKernel::Params params) {
  extern __shared__ char smem_buf[];
  GemmKernel op;
  op(params, smem_buf);
}
