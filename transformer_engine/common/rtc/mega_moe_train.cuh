/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#include "cutlass/gemm/kernel/nvte_grouped_tensor.hpp"
#include "mega_moe_fused_args.h"
#include "mega_moe_fused_protocol.cuh"

namespace {
using namespace transformer_engine::rtc;
using B = cutlass::bfloat16_t;
using Traits = transformer_engine::grouped_gemm_rtc::GemmTraits<false, NVTE_GG_TRANS_B, 89, B, 256,
                                                                256, 64, 3>;
using Gemm =
    cutlass::gemm::kernel::NvteNativeGrouped<Traits::CollectiveMainloop, Traits::CollectiveEpilogue,
                                             false, TRAIN_N, TRAIN_K>;
static_assert(Gemm::MaxThreadsPerBlock == 512 && Gemm::SharedStorageSize == 196608);

#ifndef TRAIN_SCATTER_ONLY
#define TRAIN_SCATTER_ONLY 0
#endif

constexpr bool kBatchReturn =
    TRAIN_SCATTER_ONLY && NVTE_GG_TRANS_B && TRAIN_K == 3072 && TRAIN_N == 3584;
constexpr int kReturnStride = kBatchReturn ? 264 : 256;

// Reuse the staged GEMM mainloop storage to transpose the accumulator fragments
// into complete rows before sending them to the originating top-k slots.
struct ScatterEpilogue : Traits::CollectiveEpilogue {
  template <class Shape, class Coord, class Acc, class Mma>
  CUTLASS_DEVICE static void store_full_tile(B *staging, int, int, Shape shape, Coord,
                                             const Acc &acc, Mma mma, int tid) {
    using namespace cute;
    auto tile = make_tensor(make_smem_ptr(staging), take<0, 2>(shape),
                            make_stride(Int<kReturnStride>{}, _1{}));
    auto thread = make_tiled_copy_C(CopyAtomR2S{}, mma).get_thread_slice(tid);
    auto regs = thread.retile_S(acc);
    auto dest = thread.partition_D(tile);
    constexpr int frag = decltype(size(regs(_, 0, 0)))::value;
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(regs); ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<2>(regs); ++n) {
        auto src = regs(_, m, n);
        cutlass::Array<float, frag> values;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < frag; ++i) values[i] = src(i);
        auto bf16 = cutlass::NumericArrayConverter<B, float, frag>{}(values);
        auto result = make_tensor(bf16.data(), src.layout());
        auto output = dest(_, m, n);
        copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, result, output);
      }
    }
  }
};
using ScatterGemm =
    cutlass::gemm::kernel::NvteNativeGrouped<Traits::CollectiveMainloop, ScatterEpilogue, false,
                                             TRAIN_N, TRAIN_K>;

// A warp-bulk x4 store sends four consecutive 128-byte stripes. Each lane
// contributes the word at the same position in each stripe.
__device__ __forceinline__ uint4 shared_row(const B *staging, int row) {
  const auto *words =
      reinterpret_cast<const uint32_t *>(staging + row * kReturnStride) + (threadIdx.x & 31);
  return make_uint4(words[0], words[32], words[64], words[96]);
}

__device__ __forceinline__ void scatter_shared(const MegaMoETrainArgs &p, int tile, int col,
                                               const B *staging) {
  const auto *route = static_cast<const int64_t *>(p.route) + tile * 256;
  const int warp = threadIdx.x / 32;
  if constexpr (kBatchReturn) {
    constexpr int kRowBatch = 2;
    // One warp owns sixteen adjacent rows. Its lanes fetch the route descriptors
    // together, then broadcast two row destinations at a time to the senders.
    const int64_t key = route[warp * 16 + (threadIdx.x & 15)];
    const int peer = static_cast<int>(key >> 32);
    const auto dest =
        key < 0 ? 0ull
                : reinterpret_cast<uint64_t>(
                      static_cast<B *>(p.peer[peer]) +
                      (int64_t(p.capacity) + static_cast<unsigned>(key)) * TRAIN_N + col * 256);
    for (int first = 0; first < 16; first += kRowBatch) {
      uint4 values[kRowBatch];
      uint64_t targets[kRowBatch];
      bool local[kRowBatch];
#pragma unroll
      for (int r = 0; r < kRowBatch; ++r) {
        values[r] = shared_row(staging, warp * 16 + first + r);
        targets[r] = __shfl_sync(0xffffffffu, dest, first + r);
        local[r] = __shfl_sync(0xffffffffu, peer, first + r) == p.rank;
      }
#pragma unroll
      for (int r = 0; r < kRowBatch; ++r) {
        if (!targets[r]) continue;
        auto *dst = reinterpret_cast<void *>(targets[r]);
        if (local[r])
          __ppu_global_store_bulk_volatile_b32x4(dst, values[r]);
        else
          __ppu_remote_store_bulk_volatile_b32x4(dst, values[r]);
      }
    }
  } else {
    for (int row = warp; row < 256; row += 16) {
      const int64_t key = route[row];
      if (key < 0) continue;
      const int peer = static_cast<int>(key >> 32);
      auto *dst = static_cast<B *>(p.peer[peer]) +
                  (int64_t(p.capacity) + static_cast<unsigned>(key)) * TRAIN_N + col * 256;
      const uint4 value = shared_row(staging, row);
      if (peer == p.rank)
        __ppu_global_store_bulk_volatile_b32x4(dst, value);
      else
        __ppu_remote_store_bulk_volatile_b32x4(dst, value);
    }
  }
  // All warps must finish reading their rows before the next GEMM uses smem.
  __syncthreads();
}

// Each warp transfers one row, keeping independent 512-byte segments in flight.
template <int Batch>
__device__ __forceinline__ void pull_row(const MegaMoETrainArgs &p, int row) {
  constexpr int segments = TRAIN_K / 256;
  const int64_t route = static_cast<const int64_t *>(p.route)[row];
  const int src = route < 0 ? 0 : static_cast<int>(route >> 32);
  const auto token = route < 0 ? 0u : static_cast<unsigned>(route) / 8;
  const auto *from = static_cast<const B *>(p.peer[src]) + int64_t(token) * TRAIN_K;
  auto *to = static_cast<B *>(p.a) + int64_t(row) * TRAIN_K;
  for (int first = 0; first < segments; first += Batch) {
    uint4 values[Batch];
    if (route < 0) {
#pragma unroll
      for (int j = 0; j < Batch; ++j) values[j] = make_uint4(0, 0, 0, 0);
    } else if (src == p.rank) {
#pragma unroll
      for (int j = 0; j < Batch; ++j)
        if (first + j < segments)
          values[j] = __ppu_global_ldcg_bulk_b32x4(from + (first + j) * 256);
    } else {
#pragma unroll
      for (int j = 0; j < Batch; ++j)
        if (first + j < segments)
          values[j] = __ppu_remote_load_bulk_b32x4(from + (first + j) * 256);
    }
#pragma unroll
    for (int j = 0; j < Batch; ++j)
      if (first + j < segments)
        __ppu_global_store_bulk_volatile_b32x4(to + (first + j) * 256, values[j]);
  }
}

__device__ void transfer(const MegaMoETrainArgs &p, int tile, int column) {
  for (int r = threadIdx.x / 32; r < 256; r += 16) {
    const int row = tile * 256 + r;
    if constexpr (TRAIN_PULL) {
      constexpr int segments = TRAIN_K / 256;
      pull_row<(segments < 8 ? segments : 8)>(p, row);
    } else {
      const int64_t route = static_cast<const int64_t *>(p.route)[row];
      if (route < 0) continue;
      const int src = static_cast<int>(route >> 32);
      const unsigned slot = static_cast<unsigned>(route);
      auto *from = static_cast<B *>(p.d) + int64_t(row) * TRAIN_N + column * 256;
      auto *dst =
          static_cast<B *>(p.peer[src]) + (int64_t(p.capacity) + slot) * TRAIN_N + column * 256;
      auto value = __ppu_global_ldcg_bulk_b32x4(from);
      if (src == p.rank)
        __ppu_global_store_bulk_volatile_b32x4(dst, value);
      else
        __ppu_remote_store_bulk_volatile_b32x4(dst, value);
    }
  }
  fused_moe::release_warp_writes();
  __syncthreads();
}

// The router pads all owner blocks to 256 rows. Every output tile is full and
// aligned, so the matrix loop can use the native full-tile BF16 store directly.
template <class Epilogue = Traits::CollectiveEpilogue>
__device__ __forceinline__ void project_full_tile(const B *x, const B *weight, B *out, int column,
                                                  char *smem) {
  using namespace cute;
  using Mainloop = Traits::CollectiveMainloop;
  using Shape = Mainloop::TileShape;
  auto da = Gemm::template packed_stride<Mainloop::InternalStrideA>(256, TRAIN_K);
  auto db = Gemm::template packed_stride<Mainloop::InternalStrideB>(TRAIN_N, TRAIN_K);
  auto a = make_tensor(make_gmem_ptr(x), make_shape(Int<256>{}, Int<TRAIN_K>{}), da);
  auto b = make_tensor(make_gmem_ptr(weight), make_shape(Int<TRAIN_N>{}, Int<TRAIN_K>{}), db);
  auto coordinate = make_coord(0, column, _, 0);
  auto ga = local_tile(make_mix_tensor_like(a), Shape{}, take<0, 3>(coordinate),
                       Step<_1, Underscore, _1>{});
  auto gb = local_tile(make_mix_tensor_like(b), Shape{}, take<0, 3>(coordinate),
                       Step<Underscore, _1, _1>{});
  Mainloop::TiledMma mma;
  auto acc = partition_fragment_C(mma, take<0, 2>(Shape{}));
  clear(acc);
  Mainloop mainloop(make_shape(256, TRAIN_N, TRAIN_K), da, db);
  mainloop(acc, ga, gb, acc, make_coord_iterator(shape<2>(ga)), Int<TRAIN_K / 64>{},
           make_coord(256, TRAIN_N - column * 256, 0), static_cast<int>(threadIdx.x), smem);
  cp_async_wait<0>();
  __syncthreads();
  Epilogue::store_full_tile(out, 256, TRAIN_N, Shape{}, coordinate, acc, mma,
                            static_cast<int>(threadIdx.x));
  __syncthreads();
}

__device__ __forceinline__ void pull_rows(const MegaMoETrainArgs &p, int cta, int ctas, int first,
                                          int end) {
  constexpr int segments = TRAIN_K / 256;
  for (int row =
           first * 256 + (cta * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x)) / 32;
       row < end * 256; row += ctas * static_cast<int>(blockDim.x) / 32) {
    pull_row<segments>(p, row);
  }
  fused_moe::release_warp_writes();
  __syncthreads();
}

__device__ __forceinline__ void grid_barrier(int *counter) {
  // Every CTA is resident in this cooperative launch. A separate zeroed
  // counter per transition ensures that a fast CTA cannot skip a generation.
  if (!threadIdx.x) {
    atomicAdd(counter, 1);
    fused_moe::wait_epoch(counter, gridDim.x);
  }
  __syncthreads();
}

__device__ __forceinline__ void project_rows(const MegaMoETrainArgs &p, int cta, int ctas,
                                             int first, int end, char *smem) {
  constexpr int columns = TRAIN_N / 256;
  for (int task = first * columns + cta; task < end * columns; task += ctas) {
    int tile = task / columns, column = task % columns;
    if constexpr (NVTE_GG_TRANS_B && TRAIN_N == 6144) {
      // Interleave two adjacent N tiles across eight M tiles. At MPE=2048
      // these CTAs reuse the same expert's W1 tiles in the last-level cache.
      constexpr int m_group = 8, n_group = 2;
      const int local = task - first * columns;
      const int base = first + (local / (m_group * columns)) * m_group;
      if (base + m_group <= end) {
        const int index = local % (m_group * columns);
        tile = base + (index / n_group) % m_group;
        column = (index / (m_group * n_group)) * n_group + index % n_group;
      }
    }
    const auto *x = static_cast<const B *>(p.a) + int64_t(tile) * 256 * TRAIN_K;
    const auto *weight = static_cast<const B *>(p.weight[p.expert[tile]]);
    auto *out = static_cast<B *>(p.d) + int64_t(tile) * 256 * TRAIN_N;
    project_full_tile(x, weight, out, column, smem);
  }
}

__device__ __forceinline__ void pull_then_project(const MegaMoETrainArgs &p, char *smem) {
  const int cta = blockIdx.x, ctas = gridDim.x;
  if constexpr ((NVTE_GG_TRANS_B && TRAIN_N == 6144) || (!NVTE_GG_TRANS_B && TRAIN_N == 3072)) {
    // Warm up a prefix using all CTAs, then overlap the remaining peer reads
    // with FC1 or dgrad2 on the rows that are already local. Dgrad2 has half
    // as many output tiles, so it uses more producers and a longer prefix.
    constexpr int copy_ctas = NVTE_GG_TRANS_B ? 7 : 10;
    constexpr int prefix_divisor = NVTE_GG_TRANS_B ? 4 : 3;
    const int producers = min(copy_ctas, ctas - 1);
    const int first_tiles = producers ? max(1, p.tiles / prefix_divisor) : p.tiles;
    pull_rows(p, cta, ctas, 0, first_tiles);
    grid_barrier(p.ready);
    if (cta < producers)
      pull_rows(p, cta, producers, first_tiles, p.tiles);
    else
      project_rows(p, cta - producers, ctas - producers, 0, first_tiles, smem);
    grid_barrier(p.ready + 1);
    project_rows(p, cta, ctas, first_tiles, p.tiles, smem);
  } else {
    pull_rows(p, cta, ctas, 0, p.tiles);
    grid_barrier(p.ready);
    project_rows(p, cta, ctas, 0, p.tiles, smem);
  }
}
}  // namespace

extern "C" __global__ __launch_bounds__(512, 1) void TRAIN_KERNEL(MegaMoETrainArgs p) {
  extern __shared__ char smem[];
  if constexpr (TRAIN_PULL && TRAIN_K == 3584 && (TRAIN_N == 6144 || TRAIN_N == 3072)) {
    pull_then_project(p, smem);
    return;
  }
  // Cooperative launch makes the copy producers resident alongside the consumers.
  const int copies = min(8, static_cast<int>(gridDim.x) - 1);
  if constexpr (TRAIN_PULL) {
    if (blockIdx.x < copies) {
      for (int tile = blockIdx.x; tile < p.tiles; tile += copies) {
        transfer(p, tile, 0);
        if (!threadIdx.x) fused_moe::publish(p.ready + tile, 1);
      }
    }
  }
  constexpr int columns = TRAIN_N / 256;
  // Copy producers rejoin the compute queue. Scatter retains its static tile assignment.
  int task = blockIdx.x;
  while (true) {
    if constexpr (TRAIN_PULL) {
      auto *next = reinterpret_cast<int *>(smem);
      if (!threadIdx.x) *next = atomicAdd(p.ready + p.tiles, 1);
      __syncthreads();
      task = *next;
      __syncthreads();
    }
    if (task >= p.tiles * columns) break;
    const int tile = task / columns, col = task % columns;
    if constexpr (TRAIN_PULL) {
      if (!threadIdx.x) fused_moe::wait_epoch(p.ready + tile, 1);
      __syncthreads();
    }
    cutlass::gemm::kernel::NativeExpert e{};
    e.m = 256;
    e.n = TRAIN_N;
    e.k = TRAIN_K;
    e.alpha = 1.f;
    e.a = reinterpret_cast<uint64_t>(p.a) + int64_t(tile) * 256 * TRAIN_K * sizeof(B);
    e.b = reinterpret_cast<uint64_t>(p.weight[p.expert[tile]]);
    e.d = reinterpret_cast<uint64_t>(p.d) + int64_t(tile) * 256 * TRAIN_N * sizeof(B);
    if constexpr (TRAIN_SCATTER_ONLY) {
      static_assert(!TRAIN_SCATTER_ONLY || !TRAIN_PULL);
      const int column = kBatchReturn && (tile & 1) ? columns - 1 - col : col;
      auto *staging = reinterpret_cast<B *>(smem);
      if constexpr (kBatchReturn) {
        // Adjacent M tiles visit W2's column tiles in opposite orders.
        project_full_tile<ScatterEpilogue>(reinterpret_cast<const B *>(e.a),
                                           reinterpret_cast<const B *>(e.b), staging, column, smem);
      } else {
        e.d = reinterpret_cast<uint64_t>(smem);
        ScatterGemm::run_tile(e, 0, column, smem);
      }
      scatter_shared(p, tile, column, staging);
    } else {
      Gemm::run_tile(e, 0, col, smem);
      if constexpr (!TRAIN_PULL) {
        fused_moe::release_warp_writes();
        __syncthreads();
        transfer(p, tile, col);
      }
    }
    if constexpr (!TRAIN_PULL) task += gridDim.x;
  }
  // The following stream-ordered peer barrier publishes the completed slots.
  if constexpr (TRAIN_SCATTER_ONLY) fused_moe::release_warp_writes();
}
