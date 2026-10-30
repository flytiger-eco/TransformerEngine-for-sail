/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "utils.cuh"

using namespace transformer_engine;

namespace {

// Parameters
using Type = __TYPE__;
constexpr size_t load_size = __LOAD_SIZE__;
constexpr size_t store_size = __STORE_SIZE__;
constexpr size_t warps_per_tile = __WARPS_PER_TILE__;
constexpr size_t block_size = __BLOCK_SIZE__;
constexpr size_t stages = __STAGES__;

/* cp.async (LDGSTS) helpers via inline PTX.
 *
 * NVRTC on this toolchain accepts these without any pipeline header (the
 * static include set is only utils.cuh + util/math.h). Sizes are fixed at
 * 16 B, which maps to a single vmem.ld.tsm.b32x4 on sm_89-class devices:
 * no destination register, no scoreboard stall on the load side.
 */
__device__ __forceinline__ void cp_async16(void *smem_dst,
                                           const void *gmem_src) {
  const unsigned dst =
      static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst),
               "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;");
}

}  // namespace

/* Persistent pipelined transpose.
 *
 * Structure (see docs/transpose_cpasync_persistent.md for the ACU evidence):
 *  - Front-end: the whole input tile is staged global->shared by 16 B
 *    cp.async chunks into a row-major buffer with +8-element row padding
 *    (16 B-aligned chunk destinations, conflict-free 8 B wide reads).
 *  - Back-end: identical ownership transpose as the optimized kernel --
 *    wide row-major smem reads, register transpose, [32][33] staging with
 *    the j1/i1 index swap, vectorized coalesced stores.
 *  - Pipeline: `stages`-deep ring of input tiles. The block issues the
 *    cp.async group for tile t+(stages-1)*G right after the barrier that
 *    proves tile t-G's reads of that slot are done, then processes tile t
 *    whose group has just been awaited. Grid is capped at
 *    sm_count * blocks_per_sm (persistent, grid-stride over tiles).
 *  - 32-bit addressing throughout; the launcher guarantees numel < 2^31
 *    and exact tile divisibility (falls back to the general kernel
 *    otherwise).
 */
__global__ void __launch_bounds__(block_size)
    transpose_pipeline_kernel(const Type *__restrict__ const input,
                              const float *__restrict__ const noop,
                              Type *__restrict__ const output,
                              const unsigned row_length,
                              const unsigned num_rows,
                              const unsigned num_tiles_m,
                              const unsigned total_tiles) {
  if (noop != nullptr && noop[0] == 1.0f) return;

  constexpr size_t nvec_in = load_size / sizeof(Type);
  constexpr size_t nvec_out = store_size / sizeof(Type);
  using IVec = Vec<Type, nvec_in>;
  using OVec = Vec<Type, nvec_out>;

  constexpr size_t bdimx = THREADS_PER_WARP;
  constexpr size_t bdimy = warps_per_tile;
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;
  constexpr size_t num_iterations = THREADS_PER_WARP / warps_per_tile;

  constexpr size_t kPad = 8;  // elements -> 16 B-aligned row stride
  constexpr size_t kRowElts = tile_dim_n + kPad;
  constexpr size_t kStageElts = tile_dim_m * kRowElts;

  constexpr size_t kChunkElts = 16 / sizeof(Type);
  constexpr size_t kChunksPerRow = tile_dim_n / kChunkElts;
  constexpr size_t kNumChunks = tile_dim_m * kChunksPerRow;
  constexpr size_t kChunksPerThread = kNumChunks / block_size;

  extern __shared__ char smem_raw[];
  Type *stage_base = reinterpret_cast<Type *>(smem_raw);
  OVec *staging = reinterpret_cast<OVec *>(smem_raw +
                                           stages * kStageElts * sizeof(Type));

  const size_t tid = threadIdx.x;
  const size_t tidx = tid % bdimx;
  const size_t tidy = tid / bdimx;

  auto issue_tile = [&](unsigned t, Type *stage) {
    const unsigned id_m = t % num_tiles_m;
    const unsigned id_n = t / num_tiles_m;
    const Type *src =
        input + static_cast<size_t>(id_m * tile_dim_m) * row_length +
        id_n * tile_dim_n;
#pragma unroll
    for (size_t c = 0; c < kChunksPerThread; c++) {
      const unsigned chunk = tid + c * block_size;
      const unsigned r = chunk / kChunksPerRow;
      const unsigned k = chunk % kChunksPerRow;
      cp_async16(stage + r * kRowElts + k * kChunkElts,
                 src + static_cast<size_t>(r) * row_length + k * kChunkElts);
    }
  };

  auto process_tile = [&](unsigned t, const Type *stage) {
    const unsigned id_m = t % num_tiles_m;
    const unsigned id_n = t / num_tiles_m;
    const unsigned tile_row = id_m * tile_dim_m;
    const unsigned tile_col = id_n * tile_dim_n;

    // Wide row-major reads from the stage buffer + register transpose.
    OVec local_output[nvec_in][num_iterations];
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * bdimy;
#pragma unroll
      for (size_t i2 = 0; i2 < nvec_out; ++i2) {
        const size_t r = i1 * nvec_out + i2;
        IVec local_input;
        local_input.load_from(stage + r * kRowElts + tidx * nvec_in);
#pragma unroll
        for (size_t j2 = 0; j2 < nvec_in; ++j2) {
          local_output[j2][iter].data.elt[i2] = local_input.data.elt[j2];
        }
      }
    }

    // Ownership transpose through the [32][33] staging array (identical
    // store pattern to transpose_optimized_kernel).
#pragma unroll
    for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        const size_t i1 = tidy + iter * bdimy;
        const size_t j1 = tidx;
        staging[j1 * (THREADS_PER_WARP + 1) + i1] = local_output[j2][iter];
      }
      __syncthreads();
#pragma unroll
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        const size_t i1 = tidx;
        const size_t j1 = tidy + iter * bdimy;
        const unsigned row = tile_row + i1 * nvec_out;
        const unsigned col = tile_col + j1 * nvec_in + j2;
        staging[j1 * (THREADS_PER_WARP + 1) + i1]
            .store_to(&output[static_cast<size_t>(col) * num_rows + row]);
      }
      __syncthreads();
    }
  };

  const unsigned G = gridDim.x;
  unsigned t = blockIdx.x;
  if (t >= total_tiles) return;

  if (stages == 1) {
    issue_tile(t, stage_base);
    cp_async_commit();
    while (t < total_tiles) {
      cp_async_wait_all();
      __syncthreads();
      process_tile(t, stage_base);
      t += G;
      if (t < total_tiles) {
        issue_tile(t, stage_base);
        cp_async_commit();
      }
    }
    return;
  }

  // Prologue: fill stages-1 pipeline slots.
#pragma unroll
  for (size_t s = 0; s < stages - 1; s++) {
    const unsigned tp = t + s * G;
    if (tp < total_tiles) issue_tile(tp, stage_base + s * kStageElts);
    cp_async_commit();
  }

  size_t stage = 0;
  unsigned t_next = t + (stages - 1) * G;
  while (t < total_tiles) {
    // Wait until only the stages-2 newest groups remain in flight, i.e. the
    // group carrying tile t has landed.
    asm volatile("cp.async.wait_group %0;" ::"n"(stages - 2));
    __syncthreads();
    // Every thread is past the barrier, so tile t-G's reads of slot
    // (stage-1)%stages are complete and that slot can be refilled while
    // tile t is processed.
    if (t_next < total_tiles) {
      const size_t ns = (stage + stages - 1) % stages;
      issue_tile(t_next, stage_base + ns * kStageElts);
    }
    cp_async_commit();
    t_next += G;
    process_tile(t, stage_base + stage * kStageElts);
    t += G;
    stage = (stage + 1) % stages;
  }
}
