/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "mega_moe_device.cuh"

namespace {

using namespace transformer_engine::rtc::mega_moe;

union alignas(32) Tiles {
  struct {
    unsigned short lhs[kRows * kDepth];
    unsigned short rhs[kDepth * kColumns];
  } operands;
  float result[kRows * kColumns];
};

// Dgrad multiplies dY [routes,O] by W [O,I]. Wgrad uses the same MMA
// shape, with dY transposed while staging, and reduces over one expert's routes.
template <bool Wgrad>
__device__ void grouped_grad(const Element* dy, const Element* x, const int* offsets,
                             const long long* table, Element* dx, int rows, int out_width,
                             int in_width, int experts, int table_group, Tiles& shared) {
  const long long out_tiles = (out_width + kRows - 1LL) / kRows;
  const long long row_tiles = Wgrad ? out_tiles * experts : (rows + kRows - 1LL) / kRows + experts;
  const long long col_tiles = (in_width + kColumns - 1LL) / kColumns;
  const int warp_col = (threadIdx.x / kWarpThreads) * kWarpTile;
  for (long long tile = blockIdx.x; tile < row_tiles * col_tiles; tile += gridDim.x) {
    const long long row_tile = tile / col_tiles;
    const long long col = (tile % col_tiles) * kColumns;
    const int expert = Wgrad ? static_cast<int>(row_tile / out_tiles)
                             : expert_for_tile(row_tile, offsets, rows, experts);
    const int begin = route_offset(offsets, expert, rows);
    const int end = route_offset(offsets, expert + 1, rows);
    const long long row =
        Wgrad ? (row_tile % out_tiles) * kRows : first_route_in_tile(row_tile, expert, begin);
    const int row_end = Wgrad ? out_width : end;
    if (row < 0 || row >= row_end) continue;
    const int k_begin = Wgrad ? begin : 0;
    const int k_end = Wgrad ? end : out_width;
    auto* expert_ptr =
        reinterpret_cast<Element*>(table[static_cast<long long>(table_group) * experts + expert]);
    const Element* rhs = Wgrad ? x : expert_ptr;
    Element* dst = Wgrad ? expert_ptr : dx;
    auto* a = reinterpret_cast<Element*>(shared.operands.lhs);
    auto* b = reinterpret_cast<Element*>(shared.operands.rhs);
    Accumulator acc;
    wmma::fill_fragment(acc, 0.f);
    for (long long k = k_begin; k < k_end; k += kDepth) {
      for (int i = threadIdx.x; i < kRows * kDepth; i += kThreads) {
        if constexpr (Wgrad) {
          const int rk = i / kRows, rm = i % kRows;
          a[rm * kDepth + rk] = k + rk < k_end && row + rm < row_end
                                    ? dy[(k + rk) * out_width + row + rm]
                                    : round_element(0.f);
        } else {
          const int rm = i / kDepth, rk = i % kDepth;
          a[i] = row + rm < row_end && k + rk < k_end ? dy[(row + rm) * out_width + k + rk]
                                                      : round_element(0.f);
        }
      }
      for (int i = threadIdx.x; i < kDepth * kColumns; i += kThreads) {
        const long long rk = k + i / kColumns, rn = col + i % kColumns;
        b[i] = rk < k_end && rn < in_width ? rhs[rk * in_width + rn] : round_element(0.f);
      }
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < kDepth; kk += kWarpTile) {
        MatrixA af;
        MatrixB<wmma::row_major> bf;
        wmma::load_matrix_sync(af, a + kk, kDepth);
        wmma::load_matrix_sync(bf, b + kk * kColumns + warp_col, kColumns);
        wmma::mma_sync(acc, af, bf, acc);
      }
      __syncthreads();
    }
    // Each weight tile has one owner, including the zero result of an empty expert.
    wmma::store_matrix_sync(shared.result + warp_col, acc, kColumns, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kRows * kColumns; i += kThreads) {
      const long long rm = row + i / kColumns, rn = col + i % kColumns;
      if (rm < row_end && rn < in_width) dst[rm * in_width + rn] = round_element(shared.result[i]);
    }
    __syncthreads();
  }
}

__device__ float warp_sum(float val) {
#pragma unroll
  for (int step = kWarpThreads / 2; step; step /= 2) val += __shfl_down_sync(0xffffffff, val, step);
  return val;
}

}  // namespace

extern "C" __global__ __launch_bounds__(kThreads) void nvte_mega_moe_dgrad(
    const void* dy, const int* offsets, const long long* table, void* dx, int rows, int out_width,
    int in_width, int experts, int table_group) {
  __shared__ Tiles shared;
  grouped_grad<false>(static_cast<const Element*>(dy), nullptr, offsets, table,
                      static_cast<Element*>(dx), rows, out_width, in_width, experts, table_group,
                      shared);
}

extern "C" __global__ __launch_bounds__(kThreads) void nvte_mega_moe_wgrad(
    const void* dy, const void* x, const int* offsets, const long long* table, int rows,
    int out_width, int in_width, int experts, int table_group) {
  __shared__ Tiles shared;
  grouped_grad<true>(static_cast<const Element*>(dy), static_cast<const Element*>(x), offsets,
                     table, nullptr, rows, out_width, in_width, experts, table_group, shared);
}

extern "C" __global__ __launch_bounds__(kThreads) void nvte_mega_moe_swiglu_backward(
    const void* dact_ptr, const void* projection_ptr, const float* probs, void* dprojection_ptr,
    float* dp, void* activation_ptr, int rows, int intermediate) {
  const auto* dact = static_cast<const Element*>(dact_ptr);
  const auto* projection = static_cast<const Element*>(projection_ptr);
  auto* dprojection = static_cast<Element*>(dprojection_ptr);
  __shared__ float partial[kThreads / kWarpThreads];
  for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
    float sum = 0.f;
    const float prob = probs[row];
    const long long base = row * intermediate;
    for (long long i = threadIdx.x; i < intermediate; i += kThreads) {
      const long long gate_pos = 2 * base + i, up_pos = gate_pos + intermediate;
      const float gate = promote(projection[gate_pos]), up = promote(projection[up_pos]);
      const float dh = promote(dact[base + i]);
      const float sigmoid = 1.f / (1.f + expf(-gate));
      const float silu = gate * sigmoid;
      if (activation_ptr != nullptr)
        static_cast<Element*>(activation_ptr)[base + i] =
            round_element((gate / (1.f + expf(-gate))) * up * prob);
      sum += dh * (silu * up);
      const float scaled = dh * prob;
      dprojection[gate_pos] = round_element(scaled * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
      dprojection[up_pos] = round_element(scaled * silu);
    }
    sum = warp_sum(sum);
    const int lane = threadIdx.x % kWarpThreads;
    if (lane == 0) partial[threadIdx.x / kWarpThreads] = sum;
    __syncthreads();
    if (threadIdx.x < kWarpThreads) {
      sum = warp_sum(lane < kThreads / kWarpThreads ? partial[lane] : 0.f);
      if (lane == 0) dp[row] = sum;
    }
    __syncthreads();
  }
}
