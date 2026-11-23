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

// All lanes in a warp operate on the same row and the same peer.
__device__ void transfer(const MegaMoETrainArgs &p, int tile, int column) {
  constexpr int width = TRAIN_PULL ? TRAIN_K : TRAIN_N;
  for (int r = threadIdx.x / 32; r < 256; r += 16) {
    const int row = tile * 256 + r;
    const int64_t route = static_cast<const int64_t *>(p.route)[row];
    const int src = route < 0 ? 0 : static_cast<int>(route >> 32);
    const unsigned slot = static_cast<unsigned>(route);
    B *base = static_cast<B *>(p.peer[src]);
    if constexpr (TRAIN_PULL) {
      auto *dst = static_cast<B *>(p.a) + int64_t(row) * width;
      for (int c = 0; c < width; c += 256) {
        uint4 value = make_uint4(0, 0, 0, 0);
        if (route >= 0) {
          auto *from = base + int64_t(slot / 8) * width + c;
          value = src == p.rank ? __ppu_global_ldcg_bulk_b32x4(from)
                                : __ppu_remote_load_bulk_b32x4(from);
        }
        __ppu_global_store_bulk_volatile_b32x4(dst + c, value);
      }
    } else if (route >= 0) {
      auto *from = static_cast<B *>(p.d) + int64_t(row) * width + column * 256;
      auto *dst = base + (int64_t(p.capacity) + slot) * width + column * 256;
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
}  // namespace

extern "C" __global__ __launch_bounds__(512, 1) void TRAIN_KERNEL(MegaMoETrainArgs p) {
  extern __shared__ char smem[];
  // Cooperative launch makes the copy producers resident alongside the consumers.
  const int copies = min(8, static_cast<int>(gridDim.x) - 1);
  if constexpr (TRAIN_PULL) {
    if (blockIdx.x < copies) {
      for (int tile = blockIdx.x; tile < p.tiles; tile += copies) {
        transfer(p, tile, 0);
        if (!threadIdx.x) fused_moe::publish(p.ready + tile, 1);
      }
      return;
    }
  }
  constexpr int columns = TRAIN_N / 256;
  const int first = TRAIN_PULL ? copies : 0;
  for (int task = static_cast<int>(blockIdx.x) - first; task < p.tiles * columns;
       task += gridDim.x - first) {
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
    Gemm::run_tile(e, 0, col, smem);
    if constexpr (!TRAIN_PULL) {
      fused_moe::release_warp_writes();
      __syncthreads();
      transfer(p, tile, col);
    }
  }
}
