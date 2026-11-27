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

// Peers use identical byte offsets in their IPC-mapped storage. Local scratch
// covers the worst case where every valid slot on every source selects this rank.
struct FusedLayout {
  int64_t counts, routes, inputs, returns, peer_bytes;
  int64_t inverse, activated, weights, ready, work_count, local_bytes;

  FusedLayout(int64_t capacity, int64_t h, int64_t i, int64_t e, int64_t world, int64_t topk) {
    TORCH_CHECK(world > 0 && world <= MegaMoEFusedArgs::kMaxRanks && topk > 0 && topk <= 8,
                "Invalid fused MegaMoE EP/top-k");
    TORCH_CHECK(capacity > 0 && capacity <= (INT_MAX - 256 * e) / (world * topk) && h > 0 &&
                    h <= INT_MAX && i > 0 && i <= INT_MAX / 2 && e > 0 &&
                    e <= MegaMoEFusedArgs::kMaxExperts,
                "Invalid fused MegaMoE workspace shape");
    int64_t cursor = 0;
    auto reserve = [&](int64_t n, int64_t size) {
      const int64_t offset = cursor;
      TORCH_CHECK(n <= (INT64_MAX - cursor - 511) / size, "MegaMoE workspace size overflow");
      cursor = (cursor + n * size + 511) / 512 * 512;
      return offset;
    };
    const int64_t slots = topk * capacity;
    const int64_t rows = world * slots + e * 256;
    counts = reserve(world * e, 4);
    routes = reserve(world * e * slots, 8);
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
                                                               int intermediate, int experts,
                                                               int world, int topk) {
  FusedLayout layout(capacity, hidden, intermediate, experts, world, topk);
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
                  ids.size(1) <= INT_MAX && probs.sizes() == ids.sizes() &&
                  probs.device() == x.device() && probs.scalar_type() == at::kFloat &&
                  !fc2.empty() && fc1.size() == fc2.size() && fc2[0].dim() == 2 &&
                  x.size(0) <= INT_MAX && x.size(1) <= INT_MAX && fc2[0].size(1) <= INT_MAX / 2 &&
                  fc1.size() <= INT_MAX / 2,
              "Invalid fused MegaMoE inputs");
  TORCH_CHECK(control.size() > 0 && control.size() <= MegaMoEFusedArgs::kMaxRanks && rank >= 0 &&
                  rank < control.size() && epoch > 0,
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
  p.world = control.size();
  p.topk = ids.size(1);
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
  for (int r = 0; r < p.world; ++r) {
    TORCH_CHECK(control[r].device() == control[rank].device() &&
                    control[r].scalar_type() == at::kInt && control[r].is_contiguous() &&
                    control[r].numel() >= MegaMoEFusedArgs::kControlWords,
                "Invalid peer control buffer");
    p.control[r] = control[r].data_ptr<int>();
  }
  p.status = p.control[rank] + MegaMoEFusedArgs::kControlWords - 4;
  p.host_status = mailbox.mapped;

  bool eligible = x.device() == control[rank].device() && x.scalar_type() == at::kBFloat16 &&
                  p.topk <= 8 && p.hidden > 0 && p.intermediate > 0 && p.hidden % 256 == 0 &&
                  p.intermediate % 256 == 0 && p.experts <= p.kMaxExperts &&
                  p.tokens < (INT_MAX - 8192) / (p.world * p.topk);
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
  // An admissible rank with insufficient capacity still enters pre_dispatch. All
  // ranks see the same resize/fallback decision even when only one input changed.
  Tensor out;
  p.eligible = eligible ? p.kNeedsCapacity : p.kUnsupported;
  if (eligible && capacity > 0 && p.tokens <= capacity) {
    FusedLayout layout(capacity, p.hidden, p.intermediate, p.experts, p.world, p.topk);
    auto bytes = [&](const Tensor &t, int64_t n) {
      TORCH_CHECK(t.device() == x.device() && t.scalar_type() == at::kByte && t.is_contiguous() &&
                      t.numel() >= n,
                  "Invalid fused MegaMoE payload workspace");
      return t.data_ptr<uint8_t>();
    };
    TORCH_CHECK(payload.size() == p.world && scratch.has_value(),
                "Missing fused MegaMoE workspace");
    for (int r = 0; r < p.world; ++r) {
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
  // The public operator reports invalid IDs on all ranks. Only two status words
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

// Internal entry point. route/expert come from the collectively validated router.
static Tensor train_linear(const Tensor &a, const Weights &w, const Tensor &route,
                           const Tensor &expert, const Weights &peers, int rank, int capacity,
                           bool pull, bool transpose, bool scatter_only, int topk,
                           const std::optional<Tensor> &links, const std::optional<Tensor> &heads) {
  TORCH_CHECK(topk == 6 || topk == 8, "Unsupported MegaMoE training top-k");
  TORCH_CHECK(!scatter_only || !pull, "Scatter-only mode needs a return operation");
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 &&
                  a.scalar_type() == at::kBFloat16 && !w.empty() &&
                  w.size() <= rtc::MegaMoETrainArgs::kMaxExperts && rank >= 0 &&
                  static_cast<size_t>(rank) < peers.size() &&
                  (peers.size() == 8 || peers.size() == rtc::MegaMoETrainArgs::kMaxPeers) &&
                  capacity > 0 && a.size(0) <= INT_MAX && a.size(1) > 0 && a.size(1) <= INT_MAX,
              "Invalid MegaMoE training operands");
  c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(route.device() == a.device() && route.is_contiguous() &&
                  route.scalar_type() == at::kLong && expert.device() == a.device() &&
                  expert.is_contiguous() && expert.scalar_type() == at::kInt &&
                  expert.numel() * 256 == a.size(0) && route.numel() == a.size(0),
              "Invalid owner tile plan");
  rtc::MegaMoETrainArgs p{};
  p.rank = rank;
  p.capacity = capacity;
  p.topk = topk;
  p.tiles = expert.numel();
  p.k = a.size(1);
  TORCH_CHECK(w.front().dim() == 2 && w.front().size(transpose ? 0 : 1) <= INT_MAX);
  p.n = w.front().size(transpose ? 0 : 1);
  TORCH_CHECK(
      p.n > 0 && p.n % 256 == 0 && p.k % 256 == 0 && int64_t(p.tiles) * (p.n / 256) < INT_MAX,
      "Unsupported training tile shape");
  for (size_t i = 0; i < w.size(); ++i) {
    TORCH_CHECK(w[i].device() == a.device() && w[i].is_contiguous() &&
                w[i].scalar_type() == a.scalar_type() && w[i].sizes() == w.front().sizes() &&
                w[i].size(transpose ? 1 : 0) == p.k);
    p.weight[i] = w[i].data_ptr();
  }
  for (size_t r = 0; r < peers.size(); ++r) {
    TORCH_CHECK(peers[r].device() == a.device() && peers[r].is_contiguous() &&
                peers[r].scalar_type() == a.scalar_type() && peers[r].dim() == 2 &&
                peers[r].size(0) == (1LL + topk) * capacity &&
                peers[r].size(1) == (pull ? p.k : p.n));
    p.peer[r] = peers[r].data_ptr();
  }
  auto d = scatter_only ? Tensor() : at::empty({a.size(0), p.n}, a.options());
  auto flags = at::zeros({pull ? 2 : 0}, expert.options());
  p.a = a.data_ptr();
  p.d = scatter_only ? nullptr : d.data_ptr();
  p.route = route.data_ptr<int64_t>();
  p.expert = expert.data_ptr<int>();
  if (links) {
    TORCH_CHECK(pull && links->device() == a.device() && links->is_contiguous() &&
                    links->scalar_type() == at::kInt && links->dim() == 1 &&
                    links->numel() == a.size(0),
                "Invalid source-token fanout");
    p.links = links->data_ptr<int>();
  }
  if (heads) {
    const int64_t entries = ((static_cast<int64_t>(capacity) + 63) / 64) * 64 * 8;
    TORCH_CHECK(links && peers.size() == 8 && topk == 8 && p.n == (transpose ? 1024 : 512) &&
                    entries <= INT_MAX && heads->device() == a.device() && heads->is_contiguous() &&
                    heads->scalar_type() == at::kInt && heads->dim() == 1 &&
                    heads->numel() == entries,
                "Invalid source-token head table");
    p.heads = heads->data_ptr<int>();
  }
  p.ready = flags.data_ptr<int>();
  rtc::mega_moe_train_linear(p, pull, transpose, at::cuda::getCurrentCUDAStream());
  return d;
}

void init_mega_moe_fused_bindings(pybind11::module &m) {
  m.def("_mega_moe_train_linear", &train_linear, pybind11::arg("a"), pybind11::arg("w"),
        pybind11::arg("route"), pybind11::arg("expert"), pybind11::arg("peers"),
        pybind11::arg("rank"), pybind11::arg("capacity"), pybind11::arg("pull"),
        pybind11::arg("transpose"), pybind11::arg("scatter_only") = false,
        pybind11::arg("topk") = 8, pybind11::arg("links") = pybind11::none(),
        pybind11::arg("heads") = pybind11::none(),
        pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<FusedStatus>(m, "MegaMoEFusedStatus").def(pybind11::init<const Tensor &>());
  m.def("mega_moe_fused_available", &fused_available);
  m.def("mega_moe_fused_workspace_sizes", &workspace_sizes, pybind11::arg("capacity"),
        pybind11::arg("hidden"), pybind11::arg("intermediate"), pybind11::arg("experts"),
        pybind11::arg("world") = 2, pybind11::arg("topk") = 2);
  m.def("mega_moe_fused_forward", &fused_forward,
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
}  // namespace transformer_engine::pytorch
