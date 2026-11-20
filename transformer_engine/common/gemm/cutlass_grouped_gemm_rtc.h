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
struct DeviceWorkspace;

// Keep writes to overlapping workspace views and their consuming GEMM together
// on the caller's stream. Acquire after compilation, before the first GPU write.
class WorkspaceGuard {
 public:
  WorkspaceGuard(int device, const NVTETensor* workspaces, size_t count,
                 bool reuse_list_descriptors = false);
  ~WorkspaceGuard();
  WorkspaceGuard(const WorkspaceGuard&) = delete;
  WorkspaceGuard& operator=(const WorkspaceGuard&) = delete;

 private:
  bool overlaps(uintptr_t address, size_t bytes) const;
  DeviceWorkspace& device_;
  const NVTETensor* workspaces_;
  size_t count_;
  WorkspaceGuard* next_;
};
}  // namespace grouped_gemm_rtc
}  // namespace transformer_engine
