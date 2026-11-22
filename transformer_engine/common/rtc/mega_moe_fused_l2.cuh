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
// A part becomes visible to both ranks only after every producing warp has drained
// its own writes. Each rank publishes into the peer's local mailbox.
__device__ __forceinline__ void project_part(const MegaMoEFusedArgs& p, char* smem, int part,
                                             int blocks) {
  const int tiles_n = part_tiles(part);
  int work = blockIdx.x, first = 0;
  for (int expert = 0; expert < p.experts; ++expert) {
    const int global = p.rank * p.experts + expert;
    const int rows = p.counts[0][global] + p.counts[1][global];
    const int tiles_m = (rows + kTile - 1) / kTile;
    cutlass::gemm::kernel::NativeExpert e{};
    e.m = rows;
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
    const int mailbox = kArrived + 2 * part + p.rank;
    publish(p.control[0] + mailbox, p.epoch);
    publish(p.control[1] + mailbox, p.epoch);
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
// are padded per expert to 256, with rank 0's contribution preceding rank 1's.
__device__ __forceinline__ void source_bases(const MegaMoEFusedArgs& p, int* starts) {
  int lane = threadIdx.x & 31;
  if (threadIdx.x < 32) {
    const bool active = lane < 2 * p.experts;
    const int a = active ? p.counts[0][lane] : 0, b = active ? p.counts[1][lane] : 0;
    const int padded = ((a + b + kTile - 1) / kTile) * kTile;
    int end = padded;
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
      const int previous = __shfl_up_sync(0xffffffffu, end, stride);
      if (lane >= stride) end += previous;
    }
    const int split = __shfl_sync(0xffffffffu, end, p.experts - 1);
    if (active) starts[lane] = end - padded - (lane >= p.experts ? split : 0) + (p.rank ? a : 0);
  }
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
  const int owner = int(expert >= p.experts);
  const int row = starts[expert] + p.slot_rows[slot];
  return {static_cast<const Bf16*>(p.return_slots[owner]) + int64_t(row) * kHidden + column,
          owner != p.rank};
}

__device__ __forceinline__ void combine_part(const MegaMoEFusedArgs& p, const int* starts,
                                             int block, int blocks, int part) {
  if (threadIdx.x < 2) {
    wait_epoch(p.control[p.rank] + kArrived + 2 * part + threadIdx.x, p.epoch);
  }
  __syncthreads();

  constexpr int warps = Gemm::MaxThreadsPerBlock / 32;
  constexpr int batch = kFirstPartTiles < 4 ? kFirstPartTiles : 4;
  const int tiles_n = part_tiles(part);
  const int column = part * kFirstPartTiles * kTile;
  for (int token = block * warps + (threadIdx.x >> 5); token < p.tokens; token += blocks * warps) {
    const auto* ids = p.ids + int64_t(token) * p.id_stride0;
    const auto x = lookup(p, starts, ids[0], 2 * token, column);
    const auto y = lookup(p, starts, ids[p.id_stride1], 2 * token + 1, column);
    auto* out = static_cast<Bf16*>(p.output) + int64_t(token) * kHidden + column;
    for (int first = 0; first < tiles_n; first += batch) {
      uint4 a[batch], b[batch];
#pragma unroll
      for (int i = 0; i < batch; ++i) {
        if (first + i < tiles_n) {
          a[i] = x.load((first + i) * kTile);
          b[i] = y.load((first + i) * kTile);
        }
      }
#pragma unroll
      for (int i = 0; i < batch; ++i) {
        if (first + i < tiles_n)
          __ppu_global_store_bulk_volatile_b32x4(out + (first + i) * kTile, sum_routes(a[i], b[i]));
      }
    }
  }
}

__device__ void run(const MegaMoEFusedArgs& p, char* smem) {
  __shared__ int starts[2 * MegaMoEFusedArgs::kMaxExperts];
  source_bases(p, starts);
  __syncthreads();
  project_part(p, smem, 0, gridDim.x);
  if constexpr (kParts == 2) {
    // Once part 0 is published, independent CTAs fetch/reduce it over the
    // peer link while the compute CTAs evaluate the remaining columns.
    const int copy = gridDim.x * 2 / 5;
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
