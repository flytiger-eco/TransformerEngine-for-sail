/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <memory>

#include "../util/rtc.h"
#include "../util/system.h"

namespace transformer_engine::grouped_gemm_rtc {

struct TileConfig;

struct KernelConfig {
  int sm_arch;
  bool trans_a, trans_b;  // CUTLASS layouts, after the TE operand swap.
  DType input_dtype, output_dtype;
  bool grouped_tensor;
};

// Keep the module alive through launch, including when caching is disabled.
struct CompiledKernel {
  rtc::Kernel module;
  CUfunction function;
  int smem_size;

  CompiledKernel(int device, const std::string& symbol, std::string cubin, int host_smem);
};

std::shared_ptr<const CompiledKernel> get_or_compile(int device, const KernelConfig& kernel,
                                                     const TileConfig& tile, int host_smem);

inline bool verbose_tile_debug() {
  static const bool verbose = getenv<bool>("NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE", false);
  return verbose;
}

}  // namespace transformer_engine::grouped_gemm_rtc
