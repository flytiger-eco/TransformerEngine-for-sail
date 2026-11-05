/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "utils.cuh"

using namespace transformer_engine;

namespace {

using CType = float;
using IType = __ITYPE__;
using OType = __OTYPE__;
using IndexType = __INDEX_TYPE__;
constexpr bool combine_components = __COMBINE_COMPONENTS__;
constexpr size_t load_size = __LOAD_SIZE__;
constexpr size_t store_size = __STORE_SIZE__;
constexpr size_t warps_per_tile = __WARPS_PER_TILE__;
constexpr size_t block_size = __BLOCK_SIZE__;
constexpr size_t stages = __STAGES__;

static_assert(stages >= 1 && stages <= 3, "Pipeline supports one to three stages");
static_assert(warps_per_tile > 0 && warps_per_tile <= THREADS_PER_WARP &&
                  THREADS_PER_WARP % warps_per_tile == 0,
              "Warps must evenly cover the tile");
static_assert(block_size == THREADS_PER_WARP * warps_per_tile,
              "Block size must match the tile's warp geometry");
static_assert(load_size > 0 && load_size % sizeof(IType) == 0,
              "Load vector must contain whole input elements");
static_assert(store_size > 0 && store_size % sizeof(OType) == 0,
              "Store vector must contain whole output elements");
static_assert(16 % sizeof(IType) == 0, "Input elements must divide cp.async chunks");

__device__ __forceinline__ void cp_async16(void *smem_dst, const void *gmem_src) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;");
}

}

__global__ void __launch_bounds__(block_size) cast_transpose_pipeline_kernel(
    const IType *__restrict__ const input, const CType *__restrict__ const noop,
    OType *__restrict__ const output_c, OType *__restrict__ const output_t,
    const CType *__restrict__ const scale_ptr, CType *__restrict__ const amax_ptr,
    CType *__restrict__ const scale_inv_ptr, const unsigned row_length,
    const unsigned num_rows, const unsigned num_tiles_m, const unsigned total_tiles) {
  if (noop != nullptr && noop[0] == 1.0f) return;

  constexpr size_t nvec_in = load_size / sizeof(IType);
  constexpr size_t nvec_out = store_size / sizeof(OType);
  using IVec = Vec<IType, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  constexpr size_t bdimx = THREADS_PER_WARP;
  constexpr size_t bdimy = warps_per_tile;
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;
  constexpr size_t num_iterations = THREADS_PER_WARP / warps_per_tile;

  constexpr size_t kPad = 16 / sizeof(IType);
  constexpr size_t kRowElts = tile_dim_n + kPad;
  constexpr size_t kStageElts = tile_dim_m * kRowElts;
  constexpr size_t kStageBytes = kStageElts * sizeof(IType);
  constexpr size_t kChunkElts = 16 / sizeof(IType);
  constexpr size_t kChunksPerRow = tile_dim_n / kChunkElts;
  constexpr size_t kNumChunks = tile_dim_m * kChunksPerRow;
  constexpr size_t kChunksPerThread = kNumChunks / block_size;

  static_assert(tile_dim_n % kChunkElts == 0 && kChunksPerRow > 0,
                "cp.async chunks must cover each input row exactly");
  static_assert(kNumChunks % block_size == 0 && kChunksPerThread > 0,
                "Threads must cover all cp.async chunks exactly");
  static_assert(kRowElts * sizeof(IType) % 16 == 0 && kStageBytes % 16 == 0,
                "cp.async rows and ring slots must be 16-byte aligned");
  static_assert(sizeof(IVec) == load_size && sizeof(OVecT) == store_size &&
                    sizeof(OVecC) == nvec_in * sizeof(OType),
                "Vector storage must match the requested element counts");
  static_assert(16 % alignof(IVec) == 0 &&
                    kRowElts * sizeof(IType) % alignof(IVec) == 0 &&
                    load_size % alignof(IVec) == 0,
                "Input ring reads must be vector aligned");
  static_assert(16 % alignof(OVecT) == 0 &&
                    stages * kStageBytes % alignof(OVecT) == 0,
                "Transpose scratch must be vector aligned");
  static_assert(tile_dim_n * sizeof(OType) % alignof(OVecC) == 0 &&
                    tile_dim_m * sizeof(OType) % alignof(OVecT) == 0,
                "Full-tile output strides must be vector aligned");

  extern __shared__ __align__(16) char smem_raw[];
  IType *stage_base = reinterpret_cast<IType *>(smem_raw);
  OVecT *staging = reinterpret_cast<OVecT *>(smem_raw + stages * kStageBytes);

  const size_t tid = threadIdx.x;
  const size_t tidx = tid % bdimx;
  const size_t tidy = tid / bdimx;
  const CType scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  CType amax = 0;

  auto issue_tile = [&](unsigned t, IType *stage) {
    const unsigned id_m = t % num_tiles_m;
    const unsigned id_n = t / num_tiles_m;
    // 元素索引缩窄不改变指针宽度；dispatch 已检查 2^31 元素上限。
    const IType *src = input + static_cast<IndexType>(id_m * tile_dim_m) * row_length +
                       id_n * tile_dim_n;
#pragma unroll
    for (size_t c = 0; c < kChunksPerThread; ++c) {
      const unsigned chunk = tid + c * block_size;
      const unsigned r = chunk / kChunksPerRow;
      const unsigned k = chunk % kChunksPerRow;
      cp_async16(stage + r * kRowElts + k * kChunkElts,
                 src + static_cast<IndexType>(r) * row_length + k * kChunkElts);
    }
  };

  auto process_tile = [&](unsigned t, const IType *stage) {
    const unsigned id_m = t % num_tiles_m;
    const unsigned id_n = t / num_tiles_m;
    const unsigned tile_row = id_m * tile_dim_m;
    const unsigned tile_col = id_n * tile_dim_n;

    OVecT local_output_t[nvec_in][num_iterations];
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * bdimy;
#pragma unroll
      for (size_t i2 = 0; i2 < nvec_out; ++i2) {
        const size_t r = i1 * nvec_out + i2;
        const unsigned row = tile_row + r;
        const unsigned col = tile_col + tidx * nvec_in;
        IVec local_input;
        OVecC local_output_c;
        local_input.load_from(stage + r * kRowElts + tidx * nvec_in);
#pragma unroll
        for (size_t j2 = 0; j2 < nvec_in; ++j2) {
          const CType in = static_cast<CType>(local_input.data.elt[j2]);
          const OType out = OType(in * scale);
          __builtin_assume(amax >= 0);
          amax = fmaxf(fabsf(in), amax);
          local_output_c.data.elt[j2] = out;
          local_output_t[j2][iter].data.elt[i2] = out;
        }
        local_output_c.store_to(&output_c[static_cast<IndexType>(row) * row_length + col]);
      }
    }

    if constexpr (combine_components) {
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
        for (size_t iter = 0; iter < num_iterations; ++iter) {
          const size_t i1 = tidy + iter * bdimy;
          const size_t j1 = tidx;
          staging[(j2 * THREADS_PER_WARP + j1) * (THREADS_PER_WARP + 1) + i1] =
              local_output_t[j2][iter];
        }
      }
      // 一次填充所有分量，共享内存换取更少同步及独立读取窗口。
      __syncthreads();
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
        for (size_t iter = 0; iter < num_iterations; ++iter) {
          const size_t i1 = tidx;
          const size_t j1 = tidy + iter * bdimy;
          const unsigned row = tile_row + i1 * nvec_out;
          const unsigned col = tile_col + j1 * nvec_in + j2;
          staging[(j2 * THREADS_PER_WARP + j1) * (THREADS_PER_WARP + 1) + i1]
              .store_to(&output_t[static_cast<IndexType>(col) * num_rows + row]);
        }
      }
      __syncthreads();
    } else {
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
        for (size_t iter = 0; iter < num_iterations; ++iter) {
          const size_t i1 = tidy + iter * bdimy;
          const size_t j1 = tidx;
          staging[j1 * (THREADS_PER_WARP + 1) + i1] = local_output_t[j2][iter];
        }
        __syncthreads();
#pragma unroll
        for (size_t iter = 0; iter < num_iterations; ++iter) {
          const size_t i1 = tidx;
          const size_t j1 = tidy + iter * bdimy;
          const unsigned row = tile_row + i1 * nvec_out;
          const unsigned col = tile_col + j1 * nvec_in + j2;
          staging[j1 * (THREADS_PER_WARP + 1) + i1]
              .store_to(&output_t[static_cast<IndexType>(col) * num_rows + row]);
        }
        __syncthreads();
      }
    }
  };

  const unsigned G = gridDim.x;
  unsigned t = blockIdx.x;
  if (t >= total_tiles) return;

  if constexpr (stages == 1) {
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
  } else {
#pragma unroll
    for (size_t s = 0; s < stages - 1; ++s) {
      const unsigned tp = t + s * G;
      if (tp < total_tiles) issue_tile(tp, stage_base + s * kStageElts);
      cp_async_commit();
    }

    size_t stage = 0;
    unsigned t_next = t + (stages - 1) * G;
    while (t < total_tiles) {
      asm volatile("cp.async.wait_group %0;" :: "n"(stages - 2));
      __syncthreads();
      // The barrier also proves all readers of the refill slot are finished.
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

  // Drain all groups before the shared-memory lifetime ends, including empty tails.
  cp_async_wait_all();

  if (amax_ptr != nullptr) {
    amax = reduce_max<warps_per_tile>(amax, tidy);
    if (threadIdx.x == 0) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 && scale_inv_ptr != nullptr) {
    reciprocal<CType>(scale_inv_ptr, scale);
  }
}
