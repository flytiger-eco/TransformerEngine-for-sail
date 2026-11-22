/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_RTC_MEGA_MOE_H_
#define TRANSFORMER_ENGINE_COMMON_RTC_MEGA_MOE_H_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace transformer_engine::rtc {

// Contiguous expert-major routes. offsets contains experts+1 row boundaries.
// The device pointer table holds all FC1 pointers then all FC2 pointers. For
// backward the two corresponding gradient pointer arrays follow them.
struct MegaMoEProblem {
  const void *x;
  const float *probs;
  const int *offsets;
  const void *weights;
  int rows;
  int hidden;
  int intermediate;
  int experts;
  bool fp16;
};

// Expert-major symmetric row buffers addressed by the source token's route positions.
struct MegaMoEBulkTransport {
  const int *ids, *positions;
  void *peer0, *peer1;
  int64_t tokens, capacity, row_stride, col_stride, id_row_stride, id_slot_stride;
  int hidden, topk, experts, rank;
};

struct MegaMoEFusedArgs;
void mega_moe_fused_pre_dispatch(const MegaMoEFusedArgs &p, cudaStream_t stream);
void mega_moe_fused_layers(const MegaMoEFusedArgs &p, cudaStream_t stream);

void mega_moe_bulk_transport(const MegaMoEBulkTransport &p, void *local, bool pull,
                             cudaStream_t stream);

void mega_moe_combine(const void *slots, void *output, int64_t tokens, int topk, int hidden,
                      bool fp16, cudaStream_t stream);
bool mega_moe_grouped_forward_supported();
void mega_moe_forward(const MegaMoEProblem &p, void *activation, void *out, cudaStream_t stream);
void mega_moe_peer_forward(const MegaMoEProblem &p, const void *peers, const void *routes,
                           void *activation, int world, int max_tokens, int max_slots,
                           cudaStream_t stream);
void mega_moe_backward(const MegaMoEProblem &p, const void *dy, void *projection, void *activation,
                       void *dactivation, void *dprojection, void *dx, float *dp,
                       cudaStream_t stream);

}  // namespace transformer_engine::rtc

#endif  // TRANSFORMER_ENGINE_COMMON_RTC_MEGA_MOE_H_
