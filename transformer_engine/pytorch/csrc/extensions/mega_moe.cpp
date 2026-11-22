/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/rtc/mega_moe.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>

namespace transformer_engine::pytorch {
namespace {
using at::Tensor;
using Weights = std::vector<Tensor>;

struct LocalExperts {
  rtc::MegaMoEProblem problem{};
  Tensor table;
  Weights w1, w2, dw1, dw2;

  LocalExperts(const Tensor &x, int64_t rows, const Tensor &probs, const Tensor &offsets,
               const Weights &fc1, const Weights &fc2, bool backward, const Weights &peers = {}) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                    (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16),
                "MegaMoE expects contiguous FP16 or BF16 device routes [R,H]");
    TORCH_CHECK(
        !fc1.empty() && fc1.size() == fc2.size() && fc1.size() <= INT_MAX / 4 && fc2[0].dim() == 2,
        "Invalid MegaMoE expert list");
    const auto r = rows, h = x.size(1), i = fc2[0].size(1);
    const auto e = static_cast<int64_t>(fc1.size());
    TORCH_CHECK(r <= INT_MAX && h <= INT_MAX && i <= INT_MAX / 2, "MegaMoE shape is too large");
    TORCH_CHECK(probs.device() == x.device() && probs.scalar_type() == at::kFloat &&
                    probs.is_contiguous() && probs.dim() == 1 && probs.size(0) == r,
                "Expected contiguous FP32 probabilities [R]");
    TORCH_CHECK(offsets.device() == x.device() && offsets.scalar_type() == at::kInt &&
                    offsets.is_contiguous() && offsets.dim() == 1 && offsets.size(0) == e + 1,
                "Expected contiguous int32 expert offsets [E+1]");
    const bool empty = r == 0 || h == 0 || i == 0;
    for (int64_t g = 0; g < e; ++g) {
      auto check = [&](const Tensor &w, int64_t rows, int64_t cols) {
        TORCH_CHECK(w.device() == x.device() && w.scalar_type() == x.scalar_type() &&
                        w.dim() == 2 && w.size(0) == rows && w.size(1) == cols,
                    "Incompatible MegaMoE expert weight ", g);
      };
      check(fc1[g], 2 * i, h);
      check(fc2[g], h, i);
      if (!empty) {
        w1.push_back(fc1[g].contiguous());
        w2.push_back(fc2[g].contiguous());
      }
      if (backward) {
        dw1.push_back(empty ? at::zeros({2 * i, h}, x.options())
                            : at::empty({2 * i, h}, x.options()));
        dw2.push_back(empty ? at::zeros({h, i}, x.options()) : at::empty({h, i}, x.options()));
      }
    }
    if (empty) return;
    // Copy only the addresses of the independent expert Parameters. The pinned
    // allocator tracks the asynchronous copy on the current device stream.
    auto host =
        at::empty({(backward ? 4 : 2) * e + static_cast<int64_t>(peers.size())},
                  at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
    auto addresses = host.data_ptr<int64_t>();
    for (const auto *group : {&w1, &w2, &dw1, &dw2}) {
      for (const auto &w : *group) *addresses++ = reinterpret_cast<int64_t>(w.data_ptr());
    }
    for (const auto &peer : peers) *addresses++ = reinterpret_cast<int64_t>(peer.data_ptr());
    table = host.to(x.device(), at::kLong, /*non_blocking=*/true);
    problem = {x.data_ptr(),
               probs.data_ptr<float>(),
               offsets.data_ptr<int>(),
               table.data_ptr(),
               int(r),
               int(h),
               int(i),
               int(e),
               x.scalar_type() == at::kHalf};
  }
};

Tensor local_forward(const Tensor &x, const Tensor &probs, const Tensor &offsets,
                     const Weights &fc1, const Weights &fc2) {
  TORCH_CHECK(x.is_cuda(), "MegaMoE requires device routes");
  c10::cuda::CUDAGuard guard(x.device());
  LocalExperts local(x, x.size(0), probs, offsets, fc1, fc2, false);
  const auto &p = local.problem;
  if (!p.rows) return at::zeros_like(x);
  auto out = at::empty_like(x);
  auto activation = at::empty({p.rows, p.intermediate}, x.options());
  rtc::mega_moe_forward(p, activation.data_ptr(), out.data_ptr(), at::cuda::getCurrentCUDAStream());
  return out;
}

// The group has already rendezvoused the symmetric buffers. Each expert
// gathers input rows and writes results directly to their unique source slots.
void peer_forward(const Weights &peers, const Tensor &routes, const Tensor &probs,
                  const Tensor &offsets, const Weights &fc1, const Weights &fc2,
                  int64_t max_tokens) {
  TORCH_CHECK(!peers.empty() && peers.size() <= INT_MAX, "Invalid MegaMoE peer group");
  const auto &base = peers[0];
  TORCH_CHECK(base.is_cuda() && base.dim() == 2 && max_tokens >= 0 && max_tokens <= INT_MAX &&
                  base.size(0) >= max_tokens && base.size(0) - max_tokens <= INT_MAX,
              "Invalid MegaMoE peer buffer geometry");
  c10::cuda::CUDAGuard guard(base.device());
  for (const auto &peer : peers) {
    TORCH_CHECK(peer.sizes() == base.sizes() && peer.device() == base.device() &&
                    peer.scalar_type() == base.scalar_type() && peer.is_contiguous() &&
                    reinterpret_cast<uintptr_t>(peer.data_ptr()) % 16 == 0,
                "Incompatible MegaMoE peer buffer");
  }
  TORCH_CHECK(routes.device() == base.device() && routes.scalar_type() == at::kLong &&
                  routes.is_contiguous() && routes.dim() == 2 && routes.size(1) == 3,
              "Expected MegaMoE source routes [R,3]");
  LocalExperts local(base, routes.size(0), probs, offsets, fc1, fc2, false, peers);
  const auto &p = local.problem;
  if (!p.rows) return;
  auto activation = at::empty({p.rows, p.intermediate}, base.options());
  rtc::mega_moe_peer_forward(p, local.table.data_ptr<int64_t>() + 2 * p.experts, routes.data_ptr(),
                             activation.data_ptr(), static_cast<int>(peers.size()), int(max_tokens),
                             int(base.size(0) - max_tokens), at::cuda::getCurrentCUDAStream());
}

std::tuple<Tensor, Tensor, Weights, Weights> local_backward(const Tensor &dy, const Tensor &x,
                                                            const Tensor &probs,
                                                            const Tensor &offsets,
                                                            const Weights &fc1,
                                                            const Weights &fc2) {
  TORCH_CHECK(x.is_cuda(), "MegaMoE requires device routes");
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(dy.device() == x.device() && dy.scalar_type() == x.scalar_type() &&
                  dy.sizes() == x.sizes() && dy.is_contiguous(),
              "Invalid MegaMoE output gradient");
  LocalExperts local(x, x.size(0), probs, offsets, fc1, fc2, true);
  const auto &p = local.problem;
  if (!p.rows) return {at::zeros_like(x), at::zeros_like(probs), local.dw1, local.dw2};
  auto dx = at::empty_like(x), dp = at::empty_like(probs);
  auto projection = at::empty({p.rows, 2 * p.intermediate}, x.options());
  auto activation = at::empty({p.rows, p.intermediate}, x.options());
  auto dprojection = at::empty_like(projection), dactivation = at::empty_like(activation);
  rtc::mega_moe_backward(p, dy.data_ptr(), projection.data_ptr(), activation.data_ptr(),
                         dactivation.data_ptr(), dprojection.data_ptr(), dx.data_ptr(),
                         dp.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
  return {dx, dp, local.dw1, local.dw2};
}
Tensor combine(const Tensor &slots) {
  TORCH_CHECK(slots.is_cuda() && slots.dim() == 3 && slots.is_contiguous() &&
                  (slots.scalar_type() == at::kHalf || slots.scalar_type() == at::kBFloat16) &&
                  slots.size(1) > 0 && slots.size(1) <= INT_MAX && slots.size(2) <= INT_MAX,
              "Expected contiguous BF16 or FP16 MegaMoE slots [T,K,H] with K > 0");
  c10::cuda::CUDAGuard guard(slots.device());
  auto out = at::empty({slots.size(0), slots.size(2)}, slots.options());
  if (out.numel() > 0) {
    rtc::mega_moe_combine(slots.data_ptr(), out.data_ptr(), slots.size(0), int(slots.size(1)),
                          int(slots.size(2)), slots.scalar_type() == at::kHalf,
                          at::cuda::getCurrentCUDAStream());
  }
  return out;
}
}  // namespace

void init_mega_moe_bindings(pybind11::module &m) {
  m.def("mega_moe_combine", &combine, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_local_forward", &local_forward,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_peer_forward", &peer_forward,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_local_backward", &local_backward,
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
}  // namespace transformer_engine::pytorch
