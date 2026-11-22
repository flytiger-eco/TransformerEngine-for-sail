/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>

#include "common/rtc/mega_moe.h"
#include "common/rtc/mega_moe_fused_args.h"

namespace transformer_engine::pytorch {
namespace {
using at::Tensor;
using Weights = std::vector<Tensor>;
using rtc::MegaMoEFusedArgs;

// The two peers use identical byte offsets in their IPC-mapped storage. The local
// scratch covers the worst case where every valid top-2 slot selects this rank.
struct FusedLayout {
  int64_t counts, routes, inputs, returns, peer_bytes;
  int64_t inverse, activated, weights, ready, work_count, local_bytes;

  FusedLayout(int64_t capacity, int64_t h, int64_t i, int64_t e) {
    TORCH_CHECK(capacity > 0 && capacity <= (INT_MAX - 256 * e) / 4 && h > 0 && h <= INT_MAX &&
                    i > 0 && i <= INT_MAX / 2 && e > 0 && e <= MegaMoEFusedArgs::kMaxExperts,
                "Invalid fused MegaMoE workspace shape");
    int64_t cursor = 0;
    auto reserve = [&](int64_t n, int64_t size) {
      const int64_t offset = cursor;
      TORCH_CHECK(n <= (INT64_MAX - cursor - 511) / size, "MegaMoE workspace size overflow");
      cursor = (cursor + n * size + 511) / 512 * 512;
      return offset;
    };
    const int64_t slots = 2 * capacity;
    const int64_t rows = 2 * slots + e * 256;
    counts = reserve(2 * e, 4);
    routes = reserve(2 * e * slots, 8);
    inputs = reserve(capacity, 2 * h);
    returns = reserve(rows, 2 * h);
    peer_bytes = cursor;
    cursor = 0;
    inverse = reserve(slots, 4);
    activated = reserve(rows, 2 * i);
    weights = reserve(rows, 4);
    ready = reserve((rows + 255) / 256, 4);
    work_count = reserve(1, 4);
    local_bytes = cursor;
  }
};

bool fused_available() {
  if (!rtc::mega_moe_grouped_forward_supported()) return false;
  const auto *device = at::cuda::getCurrentDeviceProperties();
  return device->canMapHostMemory && device->cooperativeLaunch &&
         device->multiProcessorCount >= 2 && device->sharedMemPerBlockOptin >= 229376 + 256;
}

std::tuple<int64_t, int64_t, int64_t, int64_t> workspace_sizes(int capacity, int hidden,
                                                               int intermediate, int experts) {
  FusedLayout layout(capacity, hidden, intermediate, experts);
  return {layout.peer_bytes, layout.local_bytes, layout.ready, layout.local_bytes - layout.ready};
}

// Keep the CPU mailbox and its completion event for the lifetime of a workspace.
struct FusedStatus {
  Tensor host;
  int *mapped;
  at::cuda::CUDAEvent ready;

  explicit FusedStatus(const Tensor &control)
      : host(at::empty({2},
                       at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true))) {
    c10::cuda::CUDAGuard guard(control.device());
    void *address = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&address, host.data_ptr(), 0));
    mapped = static_cast<int *>(address);
  }
};

std::tuple<Tensor, int, int> fused_forward(const Tensor &x, const Tensor &ids, const Tensor &probs,
                                           const Weights &fc1, const Weights &fc2,
                                           const Weights &control, const Weights &payload,
                                           const std::optional<Tensor> &scratch,
                                           FusedStatus &mailbox, int rank, int capacity,
                                           int epoch) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && ids.dim() == 2 && ids.device() == x.device() &&
                  ids.scalar_type() == at::kInt && ids.size(0) == x.size(0) && ids.size(1) > 0 &&
                  probs.sizes() == ids.sizes() && probs.device() == x.device() &&
                  probs.scalar_type() == at::kFloat && !fc2.empty() && fc1.size() == fc2.size() &&
                  fc2[0].dim() == 2 && x.size(0) <= INT_MAX && x.size(1) <= INT_MAX &&
                  fc2[0].size(1) <= INT_MAX / 2 && fc1.size() <= INT_MAX / 2,
              "Invalid fused MegaMoE inputs");
  TORCH_CHECK(control.size() == 2 && (rank == 0 || rank == 1) && epoch > 0,
              "Invalid fused MegaMoE control workspace");
  TORCH_CHECK(control[rank].is_cuda(), "Expected a device control buffer");
  c10::cuda::CUDAGuard guard(control[rank].device());
  auto stream = at::cuda::getCurrentCUDAStream();
  MegaMoEFusedArgs p{};
  p.tokens = x.size(0);
  p.hidden = x.size(1);
  p.intermediate = fc2[0].size(1);
  p.experts = fc2.size();
  p.rank = rank;
  p.epoch = epoch;
  p.capacity = capacity;
  p.x = x.data_ptr();
  p.ids = ids.data_ptr<int>();
  p.probs = probs.data_ptr<float>();
  p.x_stride0 = x.stride(0);
  p.x_stride1 = x.stride(1);
  p.id_stride0 = ids.stride(0);
  p.id_stride1 = ids.stride(1);
  p.prob_stride0 = probs.stride(0);
  p.prob_stride1 = probs.stride(1);
  for (int r = 0; r < 2; ++r) {
    TORCH_CHECK(control[r].device() == control[rank].device() &&
                    control[r].scalar_type() == at::kInt && control[r].is_contiguous() &&
                    control[r].numel() >= 256,
                "Invalid peer control buffer");
    p.control[r] = control[r].data_ptr<int>();
  }
  p.status = p.control[rank] + 252;
  p.host_status = mailbox.mapped;

  bool eligible = x.device() == control[rank].device() && x.scalar_type() == at::kBFloat16 &&
                  ids.size(1) == 2 && p.hidden > 0 && p.intermediate > 0 && p.hidden % 256 == 0 &&
                  p.intermediate % 256 == 0 && p.experts <= p.kMaxExperts &&
                  p.tokens < (INT_MAX - 8192) / 4;
  for (int e = 0; eligible && e < p.experts; ++e) {
    auto usable = [&](const Tensor &w, int rows, int cols) {
      return w.sizes() == at::IntArrayRef({rows, cols}) && w.device() == x.device() &&
             w.scalar_type() == at::kBFloat16 && w.is_contiguous() &&
             reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0;
    };
    eligible =
        usable(fc1[e], 2 * p.intermediate, p.hidden) && usable(fc2[e], p.hidden, p.intermediate);
    p.w1[e] = fc1[e].data_ptr();
    p.w2[e] = fc2[e].data_ptr();
  }
  // An admissible rank with insufficient capacity still enters pre_dispatch. Both
  // ranks see the same resize/fallback decision even when only one input changed.
  Tensor out;
  p.eligible = eligible ? p.kNeedsCapacity : p.kUnsupported;
  if (eligible && capacity > 0 && p.tokens <= capacity) {
    FusedLayout layout(capacity, p.hidden, p.intermediate, p.experts);
    auto bytes = [&](const Tensor &t, int64_t n) {
      TORCH_CHECK(t.device() == x.device() && t.scalar_type() == at::kByte && t.is_contiguous() &&
                      t.numel() >= n,
                  "Invalid fused MegaMoE payload workspace");
      return t.data_ptr<uint8_t>();
    };
    TORCH_CHECK(payload.size() == 2 && scratch.has_value(), "Missing fused MegaMoE workspace");
    for (int r = 0; r < 2; ++r) {
      auto base = bytes(payload[r], layout.peer_bytes);
      p.counts[r] = reinterpret_cast<int *>(base + layout.counts);
      p.source_routes[r] = reinterpret_cast<unsigned long long *>(base + layout.routes);
      p.source_x[r] = base + layout.inputs;
      p.return_slots[r] = base + layout.returns;
    }
    auto local = bytes(*scratch, layout.local_bytes);
    p.slot_rows = reinterpret_cast<int *>(local + layout.inverse);
    p.packed_x = p.return_slots[rank];
    p.activation = local + layout.activated;
    p.packed_probs = reinterpret_cast<float *>(local + layout.weights);
    p.tile_ready = reinterpret_cast<int *>(local + layout.ready);
    p.work_count = reinterpret_cast<int *>(local + layout.work_count);
    try {
      out = at::empty({p.tokens, p.hidden}, x.options());
      p.output = out.data_ptr();
      p.eligible = p.kReady;
    } catch (const c10::OutOfMemoryError &) {
      // Let the peer retire its speculative kernels without waiting for this rank.
      p.eligible = p.kOutputAllocationFailed;
    }
  }

  rtc::mega_moe_fused_pre_dispatch(p, stream);
  // The public operator reports invalid IDs on both ranks. Only two status words
  // cross to the host; expert counts, offsets and routing tables stay on device.
  mailbox.ready.record(stream);
  if (p.eligible == p.kReady) {
    // Enqueue both layers immediately. A failed collective admission makes both
    // kernels exit before touching payloads or entering their peer protocol.
    rtc::mega_moe_fused_layers(p, stream);
  }
  mailbox.ready.synchronize();
  auto status = mailbox.host.data_ptr<int>();
  return {out, status[0], status[1]};
}
}  // namespace

void init_mega_moe_fused_bindings(pybind11::module &m) {
  pybind11::class_<FusedStatus>(m, "MegaMoEFusedStatus").def(pybind11::init<const Tensor &>());
  m.def("mega_moe_fused_available", &fused_available);
  m.def("mega_moe_fused_workspace_sizes", &workspace_sizes);
  m.def("mega_moe_fused_forward", &fused_forward,
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
}  // namespace transformer_engine::pytorch
