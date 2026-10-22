/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ROPE_FUSED_ROPE_LAUNCH_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ROPE_FUSED_ROPE_LAUNCH_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../utils.cuh"

namespace transformer_engine {

// Launch-shape selection for the fused_rope kernels.
//
// Every fused_rope kernel launches one block per (s, b) pair -- or one per packed token on the THD
// linear-grid path -- with a `dim3(THREADS_PER_WARP, warps_per_block)` block whose threads stride
// over heads and over rotary lanes. Both loops are strided by the block dimensions, so any warp
// count covers exactly the same elements and each thread lane computes the same values in the same
// order; the choice is purely a performance decision and cannot change the result.
//
// The first-order runtime is flat in warps_per_block. These kernels are register-limited rather
// than shared-memory-limited and their register count does not depend on warps_per_block, so the
// resident-blocks-per-SM limit scales exactly as 1/warps_per_block and the device-wide resident
// warp capacity R is a constant for a given kernel. Writing E(w) for the elements one thread walks
// and waves(w) = ceil(grid_blocks * w / R),
//
//   waves(w) * E(w) ~= (grid_blocks * w / R) * (ceil(h / w) * lanes_per_head)
//
// is independent of w: doubling w halves the work per thread and doubles the warp count. The
// previous fixed `h < 16 ? 4 : 8` therefore sat on a plateau rather than at an optimum, which is
// why it was rarely badly wrong and why beating it needs a second-order term.
//
// Exactly one per-block cost grows with w. The sincos prelude is shared by all warps, but the
// __syncthreads closing it cannot retire until every warp has arrived, so a block's fixed cost is
// proportional to its warp count:
//
//   t_block(w) = ceil(d2 / (16 w)) * t_sincos + E(w) * t_elem + w * t_barrier
//
// Summed over the resident blocks of a wave the first two terms are w-invariant -- that is the
// plateau -- and only the third survives. The measured runtime is therefore modelled as
//
//   cost(w) = waves(w) * (K1 * w + E(w)),   K1 = t_barrier / t_elem
//
// with a single fitted constant. The obvious alternative, a fixed cost per *wave* rather than per
// warp -- waves(w) * (K0 + E(w)) -- is not just weaker, it cannot represent the data at all: of
// 324 measured shape groups there are 25 for which no K0 in [0, 512] selects the fastest measured
// warp count. Wave-count rounding is not the quantity that matters here.
//
// K1 was fitted rather than derived. Every candidate was scored against three disjoint measured
// sweeps totalling 324 shape groups: a broad sweep, a rotary_percent 0.5 sweep, and a gap-fill
// sweep added after a library-level A/B found block counts neither of the first two had measured.
// The score is the number of cells that come out *provably* slower than the old fixed heuristic,
// meaning slower than all three independent measurements of it taken inside that same cell. That
// count is flat and minimal over K1 in [1/6, 1/4] and rises sharply on both sides, so the weight
// below is the middle of the plateau expressed as 1/K1, which makes the argmin exact integer
// arithmetic. At the chosen value the rule scores geomean 1.2279 against the old heuristic, with a
// worst cell of 0.9845 and three provable regressions; the per-shape oracle, which no rule can
// beat, scores 1.2384 and one. Two of the three measure exactly 1.0000, i.e. inside the resolution
// of the timing harness, and the third is a near-tie the model separates by under 1%. The two-term
// rule this replaced scored 1.2079 with a worst cell of 0.7692 and 34. Measurements and the
// rejected alternatives: docs/common_op_optimization.md, Op 8.
constexpr int kFusedRopeMaxWarpsPerBlock = 16;
// 1/K1. Multiplying a cost by a positive constant cannot move its argmin, so
// cost(w) = waves(w) * (w + kFusedRopeElemWeight * E(w)) is the exact integer form of the model.
constexpr int kFusedRopeElemWeight = 5;
// blockDim.y is only ever a power of two, so the candidate list is 1, 2, 4, 8, 16.
constexpr int kFusedRopeWarpCandidates = 5;

/* \brief Rotary lanes one warp covers per head per output tensor
 *
 * The rotary loop covers `d_id < d2` striding by blockDim.x, and the `if (d > d2)` copy-the-rest
 * loop covers `[d2, d)` with the same stride. Shared by both rope kernel families.
 *
 * \param[in] d Head dimension
 * \param[in] d2 Rotary dimension, d2 <= d
 */
inline int fused_rope_lanes_per_head(const int d, const int d2) {
  return DIVUP(d2, static_cast<int>(THREADS_PER_WARP)) +
         (d > d2 ? DIVUP(d - d2, static_cast<int>(THREADS_PER_WARP)) : 0);
}

/* \brief Elements one thread of a rope kernel reads and writes
 *
 * Both loops above are strided over heads by blockDim.y.
 *
 * \param[in] h Number of heads
 * \param[in] d Head dimension
 * \param[in] d2 Rotary dimension, d2 <= d
 * \param[in] warps_per_block Candidate blockDim.y
 */
inline int fused_rope_elems_per_thread(const int h, const int d, const int d2,
                                       const int warps_per_block) {
  return DIVUP(h, warps_per_block) * fused_rope_lanes_per_head(d, d2);
}

/* \brief Chooses blockDim.y by minimising waves(w) * (warp_weight*w + elem_weight*E(w))
 *
 * Pure arithmetic, no device query, so both rope rules can be exercised directly by
 * tests/cpp/operator/test_fused_rope_launch_shape.cu. The candidate list must be ascending so that
 * the strict `<` below resolves a tie towards fewer warps: equal modelled cost means the barrier
 * term and the per-thread work term cancelled, and fewer warps is the shorter barrier.
 *
 * Both weights are integers because scaling a cost by a positive constant cannot move its argmin,
 * so a rational WEIGHT == elem_weight / warp_weight is representable exactly.
 *
 * \param[in] n_cand Number of candidates
 * \param[in] cand_warps Candidate blockDim.y values, ascending
 * \param[in] cand_capacity cand_capacity[i] is the device-wide resident warp count at
 *                          cand_warps[i], or 0 if that candidate could not be queried
 * \param[in] cand_elems cand_elems[i] is E(cand_warps[i])
 * \param[in] warp_weight Coefficient of w
 * \param[in] elem_weight Coefficient of E(w)
 * \param[in] grid_blocks Number of blocks in the chosen grid
 * \param[in] fallback Returned unchanged if any capacity is 0, in which case there is no wave count
 *                     and hence no cost to compare
 * \param[out] cand_cost Optional n_cand entries; the modelled cost of each candidate, or UINT64_MAX
 *                       throughout when the fallback was taken
 *
 * \return One of cand_warps, or fallback
 */
inline int fused_rope_select_from_candidates(const int n_cand, const int *cand_warps,
                                             const size_t *cand_capacity, const int *cand_elems,
                                             const int warp_weight, const int elem_weight,
                                             const size_t grid_blocks, const int fallback,
                                             uint64_t *cand_cost) {
  if (cand_cost != nullptr) {
    for (int i = 0; i < n_cand; ++i) cand_cost[i] = UINT64_MAX;
  }
  int best_warps = fallback;
  uint64_t best_cost = UINT64_MAX;
  for (int i = 0; i < n_cand; ++i) {
    const uint64_t capacity = static_cast<uint64_t>(cand_capacity[i]);
    // kernel_blocks_per_sm reports 0 when the occupancy query is unavailable or the kernel spills
    // to local memory. Without a capacity there is no wave count, hence no cost to compare.
    if (capacity == 0) return fallback;
    const uint64_t warps = static_cast<uint64_t>(cand_warps[i]);
    const uint64_t waves = std::max<uint64_t>(1, (grid_blocks * warps + capacity - 1) / capacity);
    const uint64_t cost = waves * (static_cast<uint64_t>(warp_weight) * warps +
                                   static_cast<uint64_t>(elem_weight) * cand_elems[i]);
    if (cand_cost != nullptr) cand_cost[i] = cost;
    if (cost < best_cost) {
      best_warps = cand_warps[i];
      best_cost = cost;
    }
  }
  return best_warps;
}

/* \brief Device-wide resident warp capacity of a rope kernel at a given block shape
 *
 * \param[in] kernel_func Kernel that is about to be launched
 * \param[in] warps_per_block Candidate blockDim.y
 * \param[in] shared_mem_size Dynamic shared memory per block (the cos/sin scratch)
 *
 * \return Resident warps across the whole device, or 0 if the query is unavailable
 */
inline size_t fused_rope_warp_capacity(const void *kernel_func, const int warps_per_block,
                                       const size_t shared_mem_size) {
  const cuda::DeviceProps &props = cuda::device_props();
  const int blocks_per_sm =
      cuda::kernel_blocks_per_sm(kernel_func, static_cast<int>(THREADS_PER_WARP) * warps_per_block);
  if (blocks_per_sm <= 0) return 0;
  // The occupancy query reports the register- and static-shared-memory-limited block count; the
  // cos/sin scratch these kernels allocate dynamically is accounted for here.
  size_t resident_blocks = static_cast<size_t>(blocks_per_sm);
  if (shared_mem_size > 0) {
    resident_blocks = std::min(resident_blocks, props.shared_mem_per_sm / shared_mem_size);
  }
  return static_cast<size_t>(props.sm_count) * resident_blocks *
         static_cast<size_t>(warps_per_block);
}

/* \brief Chooses blockDim.y by minimising the modelled runtime over the candidate block shapes
 *
 * Pure arithmetic over `warp_capacity`, with no device query of its own, so the decision can be
 * exercised directly by tests/cpp/operator/test_fused_rope_launch_shape.cu. `cuda::device_props`
 * and `cuda::kernel_blocks_per_sm` are internal to libtransformer_engine and are not in its version
 * script, so a test cannot link the wrapper below.
 *
 * \param[in] h Number of heads
 * \param[in] d Head dimension
 * \param[in] d2 Rotary dimension, d2 <= d
 * \param[in] grid_blocks Number of blocks in the chosen grid
 * \param[in] warp_capacity kFusedRopeWarpCandidates entries; entry i is the device-wide resident
 *                          warp count at blockDim.y == (1 << i), or 0 if that candidate could not
 *                          be queried, in which case the model cannot be evaluated at all and the
 *                          fixed shape this replaced is returned
 *
 * \return warps_per_block in [1, kFusedRopeMaxWarpsPerBlock]
 */
inline int fused_rope_select_warps(const int h, const int d, const int d2, const size_t grid_blocks,
                                   const size_t *warp_capacity) {
  // ceil(h / w) saturates at 1 once w >= h, so warps beyond pow2_ceil(h) remove no work and only
  // lengthen the sincos prelude and the __syncthreads barrier that follows it.
  int warp_cap = 1;
  while (warp_cap < h && warp_cap < kFusedRopeMaxWarpsPerBlock) warp_cap <<= 1;

  int cand_warps[kFusedRopeWarpCandidates];
  int cand_elems[kFusedRopeWarpCandidates];
  size_t cand_capacity[kFusedRopeWarpCandidates];
  int n_cand = 0;
  for (int w = 1; w <= warp_cap; w <<= 1) {
    cand_warps[n_cand] = w;
    cand_elems[n_cand] = fused_rope_elems_per_thread(h, d, d2, w);
    // kernel_blocks_per_sm reports 0 when the occupancy query is unavailable or the kernel spills
    // to local memory; fused_rope_select_from_candidates turns that into the fallback.
    cand_capacity[n_cand] = warp_capacity[n_cand];
    ++n_cand;
  }
  return fused_rope_select_from_candidates(n_cand, cand_warps, cand_capacity, cand_elems, 1,
                                           kFusedRopeElemWeight, grid_blocks, h < 16 ? 4 : 8,
                                           nullptr);
}

/* \brief Chooses blockDim.y for a fused_rope launch
 *
 * \param[in] kernel_func Kernel that is about to be launched
 * \param[in] grid_blocks Number of blocks in the chosen grid
 * \param[in] h Number of heads
 * \param[in] d Head dimension
 * \param[in] d2 Rotary dimension, d2 <= d
 * \param[in] shared_mem_size Dynamic shared memory per block
 *
 * \return warps_per_block in [1, kFusedRopeMaxWarpsPerBlock]
 */
inline int fused_rope_warps_per_block(const void *kernel_func, const size_t grid_blocks,
                                      const int h, const int d, const int d2,
                                      const size_t shared_mem_size) {
  size_t warp_capacity[kFusedRopeWarpCandidates];
  for (int i = 0; i < kFusedRopeWarpCandidates; ++i) {
    warp_capacity[i] = fused_rope_warp_capacity(kernel_func, 1 << i, shared_mem_size);
  }
  return fused_rope_select_warps(h, d, d2, grid_blocks, warp_capacity);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ROPE_FUSED_ROPE_LAUNCH_H_
