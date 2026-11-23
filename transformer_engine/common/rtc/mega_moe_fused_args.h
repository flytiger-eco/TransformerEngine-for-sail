/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

namespace transformer_engine::rtc {

// Passed by value to all three RTC kernels. Peer-visible arrays use group-rank order.
struct MegaMoEFusedArgs {
  static constexpr int kMaxExperts = 16;
  static constexpr int kMaxRanks = 8;
  static constexpr int kControlWords = 1024;
  enum Admission { kUnsupported, kReady, kNeedsCapacity, kOutputAllocationFailed };
  const void *x;
  const int *ids;
  const float *probs;
  void *output;
  const void *w1[kMaxExperts], *w2[kMaxExperts];
  int *control[kMaxRanks];  // Dedicated peer-visible protocol state, persists across resizes.
  int *counts[kMaxRanks];   // [world * experts], global expert ID
  unsigned long long *source_routes[kMaxRanks];  // [global expert, ordinal]
  void *source_x[kMaxRanks];                     // BF16 [capacity, hidden]
  void *return_slots[kMaxRanks];                 // BF16 padded owner rows
  void *packed_x;    // Aliases this rank's return buffer until FC1 finishes
  int *host_status;  // Mapped pinned [decision, max_tokens], written by pre-dispatch
  void *activation;  // BF16 [padded owner rows, intermediate]
  float *packed_probs;
  int *tile_ready;  // One int per owner tile, generation-tagged
  int *work_count;  // FC1 work queue cursor
  int *status;      // Device [decision, max_tokens] used by both speculative layers
  int tokens, hidden, intermediate, experts, rank, world, topk, capacity, epoch, eligible;
  long long x_stride0, x_stride1, id_stride0, id_stride1, prob_stride0, prob_stride1;
  int *slot_rows;  // Local source slot -> ordinal within the source expert list
};
}  // namespace transformer_engine::rtc
