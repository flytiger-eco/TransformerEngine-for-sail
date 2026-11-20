/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <transformer_engine/gemm.h>

// RTC backend for the list and GroupedTensor grouped GEMM APIs.
bool cutlass_grouped_gemm_rtc(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                              int num_gemms, bool transa, bool transb, NVTETensor* workspace,
                              bool accumulate, int device, int math_sm_count, cudaStream_t stream);

namespace transformer_engine {
bool try_native_grouped_gemm(const NVTEGroupedTensor A, int transa, const NVTEGroupedTensor B,
                             int transb, const NVTEGroupedTensor C, NVTEGroupedTensor D,
                             const NVTETensor alpha, const NVTETensor beta,
                             NVTETensor workspace_setup, NVTEGroupedMatmulConfig config,
                             cudaStream_t stream, const NVTETensor* A_list = nullptr,
                             size_t num_a = 0, const NVTETensor* C_list = nullptr,
                             NVTETensor* D_list = nullptr, size_t num_d = 0);
bool try_native_grouped_bias_add(NVTEGroupedTensor output, NVTEGroupedTensor bias, NVTETensor scale,
                                 bool use_scale, cudaStream_t stream);
size_t native_grouped_gemm_workspace_size(size_t groups);
bool native_grouped_gemm_supported_device();

namespace grouped_gemm_rtc {
// Invalidate cached list descriptors overlapping a workspace write. Backends
// with opaque scratch usage can invalidate the entire device address range.
void invalidate_workspace(int device, uintptr_t begin = 0, uintptr_t end = UINTPTR_MAX);
}  // namespace grouped_gemm_rtc
}  // namespace transformer_engine
