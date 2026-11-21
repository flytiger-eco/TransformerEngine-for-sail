/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include "../util/rtc.h"

namespace transformer_engine::grouped_gemm_rtc {

struct TileConfig;

struct KernelConfig {
  int sm_arch;
  bool trans_a, trans_b;  // CUTLASS layouts, after the TE operand swap.
  DType input_dtype, output_dtype;
  bool grouped_tensor;
};

// The process-wide cache owns each module for the lifetime of its CUDA function.
struct CompiledKernel {
  rtc::Kernel module;
  CUfunction function;
  int smem_size;

  CompiledKernel(int device, const std::string& symbol, std::string cubin, int host_smem);
};

const CompiledKernel& get_or_compile(int device, const KernelConfig& kernel, const TileConfig& tile,
                                     int host_smem);

}  // namespace transformer_engine::grouped_gemm_rtc
