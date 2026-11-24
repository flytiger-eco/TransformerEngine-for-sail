/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

namespace transformer_engine::rtc::fused_moe {

// A warp-bulk read consumes one aligned 512-byte segment across all lanes.
__device__ __forceinline__ bool can_read_local_input(const MegaMoEFusedArgs &p) {
  return p.x_stride0 == p.hidden && p.x_stride1 == 1 &&
         reinterpret_cast<unsigned long long>(p.x) % 512 == 0;
}

// Poll a local mapping. Its sole producer can be a CTA on the peer PPU.
__device__ __forceinline__ void wait_epoch(const int *signal, int epoch) {
  int value;
  do {
    asm volatile("ppu.ld.acquire.sys.b32 %0, [%1];" : "=r"(value) : "l"(signal) : "memory");
  } while (value != epoch);
}

__device__ __forceinline__ void publish(int *signal, int epoch) {
  asm volatile("ppu.st.release.sys.b32 [%0], %1;" : : "l"(signal), "r"(epoch) : "memory");
}

// A PPU fence drains the calling warp's stores. The local retirement counter
// joins the releases from every writer warp before one CTA notifies the peer.
__device__ __forceinline__ void release_warp_writes() {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) __ppu_threadfence_system_uncache();
}

}  // namespace transformer_engine::rtc::fused_moe
