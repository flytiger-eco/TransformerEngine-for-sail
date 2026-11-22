/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "mega_moe.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "../util/rtc.h"
#include "string_code_mega_moe_backward_cu.h"
#include "string_code_mega_moe_cu.h"
#include "string_code_mega_moe_device_cuh.h"

namespace transformer_engine::rtc {
namespace {

int64_t tiles(int value, int tile) { return (int64_t(value) + tile - 1) / tile; }

int64_t route_tiles(const MegaMoEProblem &p, int width) {
  // Each expert can contribute one partial row tile. The device maps this
  // upper bound onto the unpadded ranges using the device-resident offsets.
  return (tiles(p.rows, 16) + p.experts) * tiles(width, 64);
}

template <typename... Args>
void launch(bool fp16, const char *name, bool backward, int64_t blocks, cudaStream_t stream,
            const Args &...args) {
  auto &manager = KernelManager::instance();
  std::string label = std::string(name) + (fp16 ? "_fp16" : "_bf16");
  if (!manager.is_compiled(label)) {
    manager.compile(label, name,
                    backward ? string_code_mega_moe_backward_cu : string_code_mega_moe_cu,
                    backward ? "mega_moe_backward.cu" : "mega_moe.cu",
                    {fp16 ? "-DNVTE_MOE_FP16=1" : "-DNVTE_MOE_FP16=0"},
                    {{string_code_mega_moe_device_cuh, "mega_moe_device.cuh"}});
  }
  manager.launch(label, static_cast<unsigned>(std::min<int64_t>(blocks, 65535)), 128, 0, stream,
                 args...);
}

void fc1(const MegaMoEProblem &p, void *projection, void *activation, cudaStream_t stream) {
  launch(p.fp16, "nvte_mega_moe_fc1", false, route_tiles(p, p.intermediate), stream, p.x, p.probs,
         p.offsets, p.weights, projection, activation, p.rows, p.hidden, p.intermediate, p.experts);
}

}  // namespace

void mega_moe_bulk_transport(const MegaMoEBulkTransport &p, void *local, bool pull,
                             cudaStream_t stream) {
  if (!p.tokens || !p.hidden) return;
  launch(false, pull ? "nvte_mega_moe_bulk_pull" : "nvte_mega_moe_bulk_push", false, p.tokens,
         stream, local, p.ids, p.positions, p.peer0, p.peer1, p.tokens, p.capacity, p.hidden,
         p.topk, p.experts, p.rank, p.row_stride, p.col_stride, p.id_row_stride, p.id_slot_stride);
}

void mega_moe_combine(const void *slots, void *output, int64_t tokens, int topk, int hidden,
                      bool fp16, cudaStream_t stream) {
  const int64_t blocks = (tokens * hidden - 1) / 128 + 1;
  launch(fp16, "nvte_mega_moe_combine", false, blocks, stream, slots, output, tokens, topk, hidden);
}

void mega_moe_forward(const MegaMoEProblem &p, void *activation, void *out, cudaStream_t stream) {
  fc1(p, nullptr, activation, stream);
  launch(p.fp16, "nvte_mega_moe_fc2", false, route_tiles(p, p.hidden), stream, activation,
         p.offsets, p.weights, out, p.rows, p.hidden, p.intermediate, p.experts);
}

void mega_moe_peer_forward(const MegaMoEProblem &p, const void *peers, const void *routes,
                           void *activation, int world, int max_tokens, int max_slots,
                           cudaStream_t stream) {
  launch(p.fp16, "nvte_mega_moe_peer_fc1", false, route_tiles(p, p.intermediate), stream, p.probs,
         p.offsets, p.weights, peers, routes, activation, p.rows, p.hidden, p.intermediate,
         p.experts, world, max_tokens, max_slots);
  launch(p.fp16, "nvte_mega_moe_peer_fc2", false, route_tiles(p, p.hidden), stream, activation,
         p.offsets, p.weights, peers, routes, p.rows, p.hidden, p.intermediate, p.experts, world,
         max_tokens, max_slots);
}

void mega_moe_backward(const MegaMoEProblem &p, const void *dy, void *projection, void *activation,
                       void *dactivation, void *dprojection, void *dx, float *dp,
                       cudaStream_t stream) {
  // Recompute FC1 after dispatch, so neither its 2I projections nor the
  // expanded top-k activations are retained across the autograd boundary.
  fc1(p, projection, activation, stream);
  auto dgrad = [&](const void *grad, void *out, int o, int i, int weights) {
    launch(p.fp16, "nvte_mega_moe_dgrad", true, route_tiles(p, i), stream, grad, p.offsets,
           p.weights, out, p.rows, o, i, p.experts, weights);
  };
  auto wgrad = [&](const void *grad, const void *input, int o, int i, int weights) {
    launch(p.fp16, "nvte_mega_moe_wgrad", true, tiles(o, 16) * tiles(i, 64) * p.experts, stream,
           grad, input, p.offsets, p.weights, p.rows, o, i, p.experts, weights);
  };
  dgrad(dy, dactivation, p.hidden, p.intermediate, 1);
  wgrad(dy, activation, p.hidden, p.intermediate, 3);
  launch(p.fp16, "nvte_mega_moe_swiglu_backward", true, p.rows, stream, dactivation, projection,
         p.probs, dprojection, dp, p.rows, p.intermediate);
  dgrad(dprojection, dx, 2 * p.intermediate, p.hidden, 0);
  wgrad(dprojection, p.x, 2 * p.intermediate, p.hidden, 2);
}

}  // namespace transformer_engine::rtc
