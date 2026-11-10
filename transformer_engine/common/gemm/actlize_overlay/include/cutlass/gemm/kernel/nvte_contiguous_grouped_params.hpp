// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE for license information.
#pragma once
#if defined(__CUDACC_RTC__)
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#else
#include <cstddef>
#include <cstdint>
#include <type_traits>
#endif

namespace cutlass::gemm::kernel::nvte_contiguous {
#if defined(__CUDACC_RTC__)
namespace stl = cuda::std;
#else
namespace stl = std;
#endif
// host 和 RTC 共用；偏移以元素为单位，不包含第三方模板类型。
struct Expert {
  int32_t m, k;
  uint64_t a_offset, b_offset, d_offset;
};
struct MBlock {
  int32_t expert, row, valid_rows, rows;
  uint64_t a_offset, b_offset, d_offset;
};
struct Params {
  const void* a;
  const void* b;
  void* d;
  const Expert* experts;
  const MBlock* m_blocks;
  uint64_t work_count;
  int32_t groups, m, n, k;
  int32_t n_tiles, m_tiles, ragged_k, beta;
};
static_assert(stl::is_standard_layout_v<Params> && stl::is_trivially_copyable_v<Params>);
static_assert(sizeof(Expert) == 32 && __builtin_offsetof(Expert, a_offset) == 8);
static_assert(sizeof(MBlock) == 40 && __builtin_offsetof(MBlock, a_offset) == 16);
static_assert(sizeof(Params) == 80 && __builtin_offsetof(Params, work_count) == 40 &&
              __builtin_offsetof(Params, beta) == 76);
}  // namespace cutlass::gemm::kernel::nvte_contiguous
