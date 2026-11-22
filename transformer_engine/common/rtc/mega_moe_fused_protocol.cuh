/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

namespace transformer_engine::rtc::fused_moe {

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
