/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_RTC_MEGA_MOE_DEVICE_CUH_
#define TRANSFORMER_ENGINE_COMMON_RTC_MEGA_MOE_DEVICE_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
// HGRTC exposes the CUDA half types above. Select the matching WMMA wrapper
// so its implementation does not include the native HGGC types a second time.
#if defined(__HGGCCC__) && !defined(HGGC_COMPILER_WRAPPER_MODE)
#define HGGC_COMPILER_WRAPPER_MODE
#endif
#include <mma.h>

namespace transformer_engine::rtc::mega_moe {

namespace wmma = nvcuda::wmma;
#if NVTE_MOE_FP16
using Element = __half;
__device__ inline Element round_element(float v) { return __float2half_rn(v); }
__device__ inline float promote(Element v) { return __half2float(v); }
#else
using Element = __nv_bfloat16;
__device__ inline Element round_element(float v) { return __float2bfloat16_rn(v); }
__device__ inline float promote(Element v) { return __bfloat162float(v); }
#endif

constexpr int kWarpThreads = 32;
constexpr int kWarpTile = 16;
constexpr int kRows = kWarpTile, kColumns = 4 * kWarpTile, kDepth = 32;
constexpr int kThreads = (kColumns / kWarpTile) * kWarpThreads;

using Accumulator = wmma::fragment<wmma::accumulator, kWarpTile, kWarpTile, kWarpTile, float>;
using MatrixA =
    wmma::fragment<wmma::matrix_a, kWarpTile, kWarpTile, kWarpTile, Element, wmma::row_major>;
template <typename Layout>
using MatrixB = wmma::fragment<wmma::matrix_b, kWarpTile, kWarpTile, kWarpTile, Element, Layout>;

__device__ inline int route_offset(const int* offsets, int expert, int rows) {
  return max(0, min(offsets[expert], rows));
}

// For expert e the tile IDs start at floor(offset[e] / kRows) + e. The
// extra slot per expert covers any unaligned first row and keeps the keys
// strictly increasing even when experts receive zero routes.
__device__ inline int expert_for_tile(long long tile, const int* offsets, int rows, int experts) {
  int low = 0, high = experts;
  while (low + 1 < high) {
    int mid = low + (high - low) / 2;
    if (static_cast<long long>(route_offset(offsets, mid, rows) / kRows) + mid <= tile)
      low = mid;
    else
      high = mid;
  }
  return low;
}

__device__ inline long long first_route_in_tile(long long tile, int expert, int first) {
  return first + (tile - first / kRows - expert) * kRows;
}

}  // namespace transformer_engine::rtc::mega_moe

#endif  // TRANSFORMER_ENGINE_COMMON_RTC_MEGA_MOE_DEVICE_CUH_
