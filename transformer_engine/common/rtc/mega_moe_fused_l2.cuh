/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#include "cutlass/gemm/kernel/nvte_grouped_tensor.hpp"
#include "mega_moe_fused_args.h"
#include "mega_moe_fused_protocol.cuh"

namespace transformer_engine::rtc::fused_moe::l2 {

using Bf16 = cutlass::bfloat16_t;
constexpr int kTile = 256;
constexpr int kHidden = NVTE_MOE_FUSED_HIDDEN, kIntermediate = NVTE_MOE_FUSED_INTERMEDIATE;
constexpr int kTopK = NVTE_MOE_FUSED_TOPK;
constexpr int kRouteBatch = NVTE_MOE_FUSED_ROUTE_BATCH;
constexpr int kColumnTiles = kHidden / kTile;
constexpr int kParts = kColumnTiles > 1 ? 2 : 1;
// Give the first part the extra column tile when the tile count is odd.
constexpr int kFirstPartTiles = (kColumnTiles + 1) / 2;

__device__ constexpr int part_tiles(int part) {
  return part == 0 ? kFirstPartTiles : kColumnTiles / 2;
}

constexpr int kRetire = 32,
              kArrived = 34;  // Two local counters and one mailbox per part and sender.
static_assert(kHidden % kTile == 0 && kIntermediate % 64 == 0);

using Traits = grouped_gemm_rtc::GemmTraits<false, true, 89, Bf16, kTile, kTile, 64, 3>;
using Gemm =
    cutlass::gemm::kernel::NvteNativeGrouped<Traits::CollectiveMainloop, Traits::CollectiveEpilogue,
                                             false, kHidden, kIntermediate>;
static_assert(Gemm::MaxThreadsPerBlock == 512 && Gemm::SharedStorageSize == 196608);

// Each FC2 group writes its result directly to an expert-major symmetric buffer.
// A part becomes visible to all ranks only after every producing warp has drained
// its own writes. Each rank publishes into the peer's local mailbox.
__device__ __forceinline__ void project_part(const MegaMoEFusedArgs& p, char* smem, int part,
                                             int blocks) {
  const int tiles_n = part_tiles(part);
  int work = blockIdx.x, first = 0;
  for (int expert = 0; expert < p.experts; ++expert) {
    const int global = p.rank * p.experts + expert;
    int rows = 0;
    for (int src = 0; src < p.world; ++src) rows += __ldbl(p.counts[src] + global);
    const int tiles_m = (rows + kTile - 1) / kTile;
    cutlass::gemm::kernel::NativeExpert e{};
    e.m = tiles_m * kTile;
    e.alpha = 1.f;
    e.a = reinterpret_cast<uint64_t>(static_cast<const Bf16*>(p.activation) +
                                     static_cast<int64_t>(first) * kIntermediate);
    e.b = reinterpret_cast<uint64_t>(p.w2[expert]);
    e.d = reinterpret_cast<uint64_t>(static_cast<Bf16*>(p.return_slots[p.rank]) +
                                     static_cast<int64_t>(first) * kHidden);
    for (; work < tiles_m * tiles_n; work += blocks) {
      const int mi = work / tiles_n;
      const int ni = part * kFirstPartTiles + work % tiles_n;
      // run_tile reverses the column index for odd M tiles internally; each part
      // must produce the same output columns for every M tile.
      Gemm::run_tile(e, mi, (mi & 1) ? kColumnTiles - 1 - ni : ni, smem);
    }
    work -= tiles_m * tiles_n;
    first += tiles_m * kTile;
  }

  // Every writer drains its warp before the grid barrier. CTA 0 acquires
  // the other producers' cacheable ACTLIZE stores before publishing to peers.
  release_warp_writes();
  __syncthreads();
  if (threadIdx.x == 0) {
    // Each participating CTA increments once. CTA 0 adjusts the increment
    // so the high bit changes precisely when the last producer arrives.
    constexpr unsigned phase_bit = 0x80000000u;
    const unsigned increment = blockIdx.x == 0 ? phase_bit - (blocks - 1) : 1u;
    auto* retired = p.control[p.rank] + kRetire + part;
    unsigned before, after;
    asm volatile("ppu.atom.add.release.gpu.u32 %0, [%1], %2;"
                 : "=r"(before)
                 : "l"(retired), "r"(increment)
                 : "memory");
    do {
      asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];" : "=r"(after) : "l"(retired) : "memory");
    } while (((before ^ after) & phase_bit) == 0);
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int mailbox = kArrived + MegaMoEFusedArgs::kMaxRanks * part + p.rank;
    for (int dst = 0; dst < p.world; ++dst) publish(p.control[dst] + mailbox, p.epoch);
  }
}

__device__ __forceinline__ uint4 sum_routes(uint4 left, uint4 right) {
  uint4 result;
  const auto* a = reinterpret_cast<const __nv_bfloat162*>(&left);
  const auto* b = reinterpret_cast<const __nv_bfloat162*>(&right);
  auto* dst = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __bfloat1622float2(a[i]), y = __bfloat1622float2(b[i]);
    dst[i] = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
  }
  return result;
}

// Turn the per-source ordinal recorded by pre_dispatch into an owner row. Rows
// are padded per expert to 256, with source contributions in group-rank order.
__device__ __forceinline__ void source_bases(const MegaMoEFusedArgs& p, int* starts) {
  int global = threadIdx.x;
  int before_source = 0;
  if (global < p.world * p.experts) {
    int total = 0;
    for (int src = 0; src < p.world; ++src) {
      int count = __ldbl(p.counts[src] + global);
      total += count;
      if (src < p.rank) before_source += count;
    }
    starts[global] = ((total + kTile - 1) / kTile) * kTile;
  }
  __syncthreads();
  int base = before_source;
  if (global < p.world * p.experts) {
    for (int e = global / p.experts * p.experts; e < global; ++e) base += starts[e];
  }
  __syncthreads();
  if (global < p.world * p.experts) starts[global] = base;
}

struct SourceRow {
  const Bf16* ptr;
  bool remote;
  __device__ __forceinline__ uint4 load(int offset) const {
    if (ptr == nullptr) return make_uint4(0, 0, 0, 0);
    return remote ? __ppu_remote_load_bulk_b32x4(ptr + offset)
                  : __ppu_global_ldbl_bulk_b32x4(ptr + offset);
  }
};

__device__ __forceinline__ SourceRow lookup(const MegaMoEFusedArgs& p, const int* starts,
                                            int expert, int slot, int column) {
  if (expert < 0) return {nullptr, false};
  const int owner = expert / p.experts;
  const int row = starts[expert] + p.slot_rows[slot];
  return {static_cast<const Bf16*>(p.return_slots[owner]) + int64_t(row) * kHidden + column,
          owner != p.rank};
}

__device__ __forceinline__ void combine_part(const MegaMoEFusedArgs& p, const int* starts,
                                             int block, int blocks, int part) {
  if (threadIdx.x < p.world) {
    wait_epoch(p.control[p.rank] + kArrived + MegaMoEFusedArgs::kMaxRanks * part + threadIdx.x,
               p.epoch);
  }
  __syncthreads();

  constexpr int warps = Gemm::MaxThreadsPerBlock / 32;
  constexpr int batch = kFirstPartTiles < 4 ? kFirstPartTiles : 4;
  const int tiles_n = part_tiles(part);
  const int column = part * kFirstPartTiles * kTile;
  for (int token = block * warps + (threadIdx.x >> 5); token < p.tokens; token += blocks * warps) {
    const auto* ids = p.ids + int64_t(token) * p.id_stride0;
    SourceRow routes[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
      routes[k] = lookup(p, starts, ids[k * p.id_stride1], kTopK * token + k, column);
    auto* out = static_cast<Bf16*>(p.output) + int64_t(token) * kHidden + column;
    for (int first = 0; first < tiles_n; first += batch) {
      float2 acc[batch][4];
#pragma unroll
      for (int i = 0; i < batch; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = make_float2(0.f, 0.f);
      }
      // Accumulate all selected routes in FP32 and round only once, preserving
      // the public top-k contract instead of chaining pairwise BF16 sums.
#pragma unroll
      for (int first_route = 0; first_route < kTopK; first_route += kRouteBatch) {
        uint4 values[kRouteBatch][batch];
#pragma unroll
        for (int r = 0; r < kRouteBatch; ++r)
#pragma unroll
          for (int i = 0; i < batch; ++i)
            if (first_route + r < kTopK && first + i < tiles_n)
              values[r][i] = routes[first_route + r].load((first + i) * kTile);
#pragma unroll
        for (int r = 0; r < kRouteBatch; ++r)
#pragma unroll
          for (int i = 0; i < batch; ++i) {
            if (first_route + r < kTopK && first + i < tiles_n) {
              auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&values[r][i]);
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                float2 v = __bfloat1622float2(pairs[j]);
                acc[i][j].x += v.x;
                acc[i][j].y += v.y;
              }
            }
          }
      }
#pragma unroll
      for (int i = 0; i < batch; ++i) {
        if (first + i < tiles_n) {
          uint4 value;
          auto* pairs = reinterpret_cast<__nv_bfloat162*>(&value);
#pragma unroll
          for (int j = 0; j < 4; ++j) pairs[j] = __floats2bfloat162_rn(acc[i][j].x, acc[i][j].y);
          __ppu_global_store_bulk_volatile_b32x4(out + (first + i) * kTile, value);
        }
      }
    }
  }
}

__device__ void run(const MegaMoEFusedArgs& p, char* smem) {
  __shared__ int starts[MegaMoEFusedArgs::kMaxRanks * MegaMoEFusedArgs::kMaxExperts];
  source_bases(p, starts);
  __syncthreads();
  project_part(p, smem, 0, gridDim.x);
  if constexpr (kParts == 2) {
    // Once part 0 is published, independent CTAs fetch/reduce it over the
    // peer link while the compute CTAs evaluate the remaining columns.
    const int copy = min(NVTE_MOE_FUSED_COMBINE_SMS, int(gridDim.x) - 1);
    const int compute = gridDim.x - copy;
    if (copy == 0) combine_part(p, starts, blockIdx.x, gridDim.x, 0);
    if (blockIdx.x < compute) {
      project_part(p, smem, 1, compute);
    } else {
      combine_part(p, starts, blockIdx.x - compute, copy, 0);
    }
  }
  combine_part(p, starts, blockIdx.x, gridDim.x, kParts - 1);
}

}  // namespace transformer_engine::rtc::fused_moe::l2

extern "C" __global__ __launch_bounds__(512, 1) void nvte_mega_moe_fused_l2(
    transformer_engine::rtc::MegaMoEFusedArgs p) {
  if (*p.status) return;
  extern __shared__ char smem[];
  transformer_engine::rtc::fused_moe::l2::run(p, smem);
}
