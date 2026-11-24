/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include "mega_moe_fused_args.h"
#include "mega_moe_fused_protocol.cuh"

namespace transformer_engine::rtc::fused_moe {

// Each sender owns one mailbox in every rank's control page, so publishing a
// generation needs no remote atomic. The host zeroes the pages at rendezvous.
enum RouteControl {
  kEntryClaim = 0,
  kRetiring = 1,
  kInvalid = 2,
  kEntered = 8,        // one mailbox per sender
  kRouted = 16,        // one mailbox per sender
  kPeerHeader = 64,    // seven fields per sender
  kClaimCounts = 256,  // world * kMaxExperts, local scratch
};

__device__ inline void stage_inputs(const MegaMoEFusedArgs &p) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long threads = static_cast<long long>(gridDim.x) * blockDim.x;
  const auto *src = static_cast<const unsigned short *>(p.x);
  auto *dst = static_cast<unsigned short *>(p.source_x[p.rank]);
  const long long size = static_cast<long long>(p.tokens) * p.hidden;
  if (p.x_stride1 == 1 && p.hidden % 8 == 0 && p.x_stride0 % 8 == 0 &&
      reinterpret_cast<unsigned long long>(src) % alignof(uint4) == 0) {
    const auto *src_vec = reinterpret_cast<const uint4 *>(src);
    auto *dst_vec = reinterpret_cast<uint4 *>(dst);
    if (can_read_local_input(p)) {
      // Only stage tokens that the peer needs. Local copy CTAs can read aligned X directly.
      const int warp = int(i) / 32, stride = int(threads) / 32, lane = threadIdx.x % 32;
      int vecs = p.hidden / 8;
      for (int tok = warp; tok < p.tokens; tok += stride) {
        const int *ids = p.ids + static_cast<long long>(tok) * p.id_stride0;
        bool remote = false;
        for (int k = 0; k < p.topk; ++k) {
          const int e = ids[k * p.id_stride1];
          remote |= e >= 0 && e < p.world * p.experts && e / p.experts != p.rank;
        }
        if (remote)
          for (int v = lane; v < vecs; v += 32) {
            auto off = static_cast<long long>(tok) * vecs + v;
            dst_vec[off] = src_vec[off];
          }
      }
    } else {
      const int vec_width = p.hidden / 8;
      for (long long j = i; j < size / 8; j += threads)
        dst_vec[j] = src_vec[(j / vec_width) * (p.x_stride0 / 8) + j % vec_width];
    }
  } else {
    for (long long j = i; j < size; j += threads)
      dst[j] = src[(j / p.hidden) * p.x_stride0 + (j % p.hidden) * p.x_stride1];
  }
}

// Claim one contiguous interval per nonempty (CTA, expert). Slot identities
// make duplicate experts distinguishable even when two probabilities differ.
__device__ inline void stage_routes(const MegaMoEFusedArgs &p, int *histogram) {
  const int experts = p.world * p.experts;
  const long long slots = static_cast<long long>(p.topk) * p.tokens;
  auto *control = p.control[p.rank];
  for (long long first = static_cast<long long>(blockIdx.x) * blockDim.x; first < slots;
       first += static_cast<long long>(gridDim.x) * blockDim.x) {
    if (threadIdx.x < experts) histogram[threadIdx.x] = 0;
    __syncthreads();
    const long long slot = first + threadIdx.x;
    int id = -1;
    int position = 0;
    float weight = 0;
    if (slot < slots) {
      const long long row = slot / p.topk, k = slot % p.topk;
      id = p.ids[row * p.id_stride0 + k * p.id_stride1];
      if (id >= 0 && id < experts) {
        position = atomicAdd(histogram + id, 1);
        weight = p.probs[row * p.prob_stride0 + k * p.prob_stride1];
      } else if (id != -1) {
        atomicExch(control + kInvalid, 1);
        id = -1;
      }
    }
    __syncthreads();
    if (threadIdx.x < experts) {
      const int count = histogram[threadIdx.x];
      if (count) histogram[threadIdx.x] = atomicAdd(control + kClaimCounts + threadIdx.x, count);
    }
    __syncthreads();
    if (id >= 0) {
      const long long start = static_cast<long long>(id) * p.topk * p.capacity;
      const int ordinal = histogram[id] + position;
      p.source_routes[p.rank][start + ordinal] =
          (static_cast<unsigned long long>(__float_as_uint(weight)) << 32) |
          static_cast<unsigned>(slot);
      p.slot_rows[slot] = ordinal;
    }
    __syncthreads();
  }
}

}  // namespace transformer_engine::rtc::fused_moe

extern "C" __global__ void nvte_mega_moe_fused_pre_dispatch(
    transformer_engine::rtc::MegaMoEFusedArgs p) {
  using namespace transformer_engine::rtc::fused_moe;
  int *local = p.control[p.rank];
  // A rank can reach the next call while its peer still combines the previous
  // one. Enter on every stream before overwriting any peer-visible storage.
  // The first scheduled CTA posts the arrival even on oversubscribed grids.
  if (threadIdx.x == 0) {
    if (atomicExch(local + kEntryClaim, p.epoch) != p.epoch) {
      for (int dst = 0; dst < p.world; ++dst) publish(p.control[dst] + kEntered + p.rank, p.epoch);
    }
    for (int src = 0; src < p.world; ++src) wait_epoch(local + kEntered + src, p.epoch);
  }
  __syncthreads();

  __shared__ int histogram[transformer_engine::rtc::MegaMoEFusedArgs::kMaxRanks *
                           transformer_engine::rtc::MegaMoEFusedArgs::kMaxExperts];
  if (p.eligible == p.kReady) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *p.work_count = 0;
    stage_routes(p, histogram);
    stage_inputs(p);
  }
  release_warp_writes();
  __syncthreads();

  __shared__ bool last;
  if (threadIdx.x == 0) {
    // The grid can be oversubscribed. Retire without waiting; the last
    // acquire/release RMW joins each CTA's data before notifying the peer.
    unsigned ticket;
    asm volatile("ppu.atom.add.acq_rel.gpu.u32 %0, [%1], %2;"
                 : "=r"(ticket)
                 : "l"(local + kRetiring), "r"(1u)
                 : "memory");
    last = ticket == gridDim.x - 1;
    if (last) local[kRetiring] = 0;
  }
  __syncthreads();
  if (!last) return;

  if (p.eligible == p.kReady && threadIdx.x < p.world * p.experts) {
    p.counts[p.rank][threadIdx.x] = local[kClaimCounts + threadIdx.x];
    local[kClaimCounts + threadIdx.x] = 0;
  }
  if (threadIdx.x == 0) {
    for (int dst = 0; dst < p.world; ++dst) {
      int *header = p.control[dst] + kPeerHeader + 7 * p.rank;
      __stbl(header, p.eligible);
      __stbl(header + 1, p.hidden);
      __stbl(header + 2, p.intermediate);
      __stbl(header + 3, p.experts);
      __stbl(header + 4, p.tokens);
      __stbl(header + 5, local[kInvalid]);
      __stbl(header + 6, p.topk);
    }
  }
  release_warp_writes();
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int dst = 0; dst < p.world; ++dst) publish(p.control[dst] + kRouted + p.rank, p.epoch);
    for (int src = 0; src < p.world; ++src) wait_epoch(local + kRouted + src, p.epoch);
    int max_tokens = 0;
    bool allocation_failed = false, unsupported = false, needs_capacity = false, invalid = false;
    for (int src = 0; src < p.world; ++src) {
      const int *header = local + kPeerHeader + 7 * src;
      int mode = __ldbl(header);
      allocation_failed |= mode == p.kOutputAllocationFailed;
      unsupported |= mode == p.kUnsupported || __ldbl(header + 1) != p.hidden ||
                     __ldbl(header + 2) != p.intermediate || __ldbl(header + 3) != p.experts ||
                     __ldbl(header + 6) != p.topk;
      needs_capacity |= mode == p.kNeedsCapacity;
      invalid |= __ldbl(header + 5) != 0;
      max_tokens = max(max_tokens, __ldbl(header + 4));
    }
    int status = 0;
    if (allocation_failed) {
      status = 4;
    } else if (unsupported || max_tokens < 2048) {
      status = 1;
    } else if (needs_capacity) {
      status = 3;
    } else if (invalid) {
      status = 2;
    }
    local[kInvalid] = 0;
    p.status[0] = status;
    p.status[1] = max_tokens;
    if (p.host_status) {
      p.host_status[0] = status;
      p.host_status[1] = max_tokens;
      __threadfence_system();
    }
  }
}
