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
#include <optional>

#include "transformer_engine/activation.h"
#include "transformer_engine/gemm.h"

namespace transformer_engine::pytorch {
namespace {
using at::Tensor;
using Weights = std::vector<Tensor>;

struct LocalExperts {
  rtc::MegaMoEProblem problem{};
  Tensor table;
  Weights w1, w2, dw1, dw2;

  LocalExperts(const Tensor &x, int64_t rows, const Tensor &probs, const Tensor &offsets,
               const Weights &fc1, const Weights &fc2, bool backward) {
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
    problem = {x.data_ptr(),
               probs.data_ptr<float>(),
               offsets.data_ptr<int>(),
               nullptr,
               int(r),
               int(h),
               int(i),
               int(e),
               x.scalar_type() == at::kHalf};
  }

  void prepare_table(const Tensor &x, const Weights &peers = {}) {
    // Copy only the addresses of the independent expert Parameters. The pinned
    // allocator tracks the asynchronous copy on the current device stream.
    auto host = at::empty(
        {static_cast<int64_t>(w1.size() + w2.size() + dw1.size() + dw2.size() + peers.size())},
        at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
    auto addresses = host.data_ptr<int64_t>();
    for (const auto *group : {&w1, &w2, &dw1, &dw2}) {
      for (const auto &w : *group) *addresses++ = reinterpret_cast<int64_t>(w.data_ptr());
    }
    for (const auto &peer : peers) *addresses++ = reinterpret_cast<int64_t>(peer.data_ptr());
    table = host.to(x.device(), at::kLong, /*non_blocking=*/true);
    problem.weights = table.data_ptr();
  }
};

// Large expert batches amortize the setup of the SM89 grouped tensor-core pipeline.
// The metadata stays on device even when every expert has a different route count.
bool use_grouped_forward(const LocalExperts &local, cudaStream_t stream) {
  const auto &p = local.problem;
  if (p.fp16 || p.rows < 128LL * p.experts || p.hidden < 256 || p.intermediate < 128 ||
      p.hidden % 8 || p.intermediate % 8 || reinterpret_cast<uintptr_t>(p.x) % 16)
    return false;
  // On SM89 the fused kernel is faster at 2048 routes, H=256, I=128; the
  // larger tile amortizes its descriptor setup at 4096 routes with these widths.
  constexpr int64_t min_route_hidden_intermediate = 1LL << 27;
  if (int64_t(p.rows) * p.hidden <= (min_route_hidden_intermediate - 1) / p.intermediate)
    return false;
  if (!rtc::mega_moe_grouped_forward_supported()) return false;
  for (const auto *weights : {&local.w1, &local.w2}) {
    for (const auto &weight : *weights) {
      if (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16) return false;
    }
  }
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  return capture == cudaStreamCaptureStatusNone;
}

TensorWrapper describe(const Tensor &tensor, DType type) {
  return TensorWrapper(tensor.data_ptr(),
                       std::vector<size_t>(tensor.sizes().begin(), tensor.sizes().end()), type);
}

void grouped_forward(const Tensor &x, const Tensor &probs, const Tensor &offsets,
                     const Tensor &output, const LocalExperts &local, cudaStream_t stream) {
  const auto &p = local.problem;
  const size_t groups = p.experts;
  // Discrete expert weights have independent allocations. The grouped inputs
  // and outputs share device row counts and have fixed, aligned row strides.
  auto rows = (offsets.slice(0, 1) - offsets.slice(0, 0, -1)).to(at::kLong);
  auto alpha = at::ones({1}, probs.options()), beta = at::zeros({1}, probs.options());
  auto ws = at::empty({static_cast<int64_t>(nvte_get_grouped_gemm_setup_workspace_size(groups))},
                      x.options().dtype(at::kByte));
  auto one = describe(alpha, DType::kFloat32), zero = describe(beta, DType::kFloat32);
  auto workspace = describe(ws, DType::kByte);
  auto packed = [&](const Tensor &t) {
    GroupedTensorWrapper result(groups, std::vector<size_t>{size_t(t.size(0)), size_t(t.size(1))});
    result.set_rowwise_data(t.data_ptr(), DType::kBFloat16, std::vector<size_t>{size_t(t.numel())});
    result.set_first_dims(rows.data_ptr(), DType::kInt64, std::vector<size_t>{groups});
    return result;
  };
  auto linear = [&](const Tensor &input, const Weights &weights, const Tensor &output) {
    auto b = packed(input), d = packed(output);
    std::vector<TensorWrapper> descriptors;
    std::vector<NVTETensor> pointers;
    descriptors.reserve(groups);
    pointers.reserve(groups);
    for (const auto &w : weights) {
      descriptors.emplace_back(describe(w, DType::kBFloat16));
      pointers.push_back(descriptors.back().data());
    }
    nvte_grouped_gemm_with_discrete_inputA(pointers.data(), groups, 1, b.data(), 0, nullptr,
                                           d.data(), one.data(), zero.data(), workspace.data(),
                                           nullptr, nullptr, stream);
  };
  auto projection = at::empty({p.rows, 2 * p.intermediate}, x.options());
  auto activation = at::empty({p.rows, p.intermediate}, x.options());
  linear(x, local.w1, projection);
  auto z = describe(projection, DType::kBFloat16), h = describe(activation, DType::kBFloat16);
  auto scales = describe(probs, DType::kFloat32);
  nvte_scaled_swiglu(z.data(), scales.data(), h.data(), 0, stream);
  linear(activation, local.w2, output);
}

// The symmetric-memory caller supplies contiguous owner-local storage. An
// exact alias with x is safe: FC2 starts after FC1 has consumed all input.
void check_output(const Tensor &out, const Tensor &x, const Tensor &probs, const Tensor &offsets,
                  const Weights &w1, const Weights &w2) {
  TORCH_CHECK(out.device() == x.device() && out.scalar_type() == x.scalar_type() &&
                  out.sizes() == x.sizes() && out.is_contiguous(),
              "MegaMoE out must match the packed input shape, dtype and device and be contiguous");
  auto disjoint = [&](const Tensor &input) {
    if (!out.numel() || !input.numel()) return;
    size_t span = 1;
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
      span += (input.size(dim) - 1) * input.stride(dim);
    }
    const auto a = reinterpret_cast<uintptr_t>(input.data_ptr());
    const auto b = reinterpret_cast<uintptr_t>(out.data_ptr());
    TORCH_CHECK(a < b ? b - a >= span * input.element_size() : a - b >= out.nbytes(),
                "MegaMoE out overlaps an input");
  };
  if (out.data_ptr() != x.data_ptr()) disjoint(x);
  disjoint(probs);
  disjoint(offsets);
  for (const auto &w : w1) disjoint(w);
  for (const auto &w : w2) disjoint(w);
}

Tensor local_forward(const Tensor &x, const Tensor &probs, const Tensor &offsets,
                     const Weights &fc1, const Weights &fc2, const std::optional<Tensor> &output) {
  TORCH_CHECK(x.is_cuda(), "MegaMoE requires device routes");
  c10::cuda::CUDAGuard guard(x.device());
  LocalExperts local(x, x.size(0), probs, offsets, fc1, fc2, false);
  const auto &p = local.problem;
  if (output) check_output(*output, x, probs, offsets, fc1, fc2);
  auto out = output ? *output : at::empty_like(x);
  if (!p.rows) return out.zero_();
  if (use_grouped_forward(local, at::cuda::getCurrentCUDAStream()) &&
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0) {
    grouped_forward(x, probs, offsets, out, local, at::cuda::getCurrentCUDAStream());
    return out;
  }
  local.prepare_table(x);
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
  LocalExperts local(base, routes.size(0), probs, offsets, fc1, fc2, false);
  const auto &p = local.problem;
  if (!p.rows) return;
  local.prepare_table(base, peers);
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
  local.prepare_table(x);
  auto dx = at::empty_like(x), dp = at::empty_like(probs);
  auto projection = at::empty({p.rows, 2 * p.intermediate}, x.options());
  auto activation = at::empty({p.rows, p.intermediate}, x.options());
  auto dprojection = at::empty_like(projection), dactivation = at::empty_like(activation);
  rtc::mega_moe_backward(p, dy.data_ptr(), projection.data_ptr(), activation.data_ptr(),
                         dactivation.data_ptr(), dprojection.data_ptr(), dx.data_ptr(),
                         dp.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
  return {dx, dp, local.dw1, local.dw2};
}
Tensor bulk_transport(const Tensor *input, const Tensor &ids, const Tensor &positions,
                      const Tensor &peer0, const Tensor &peer1, int experts, int rank) {
  TORCH_CHECK(peer0.is_cuda() && peer0.dim() == 2 && peer0.scalar_type() == at::kBFloat16 &&
                  peer0.is_contiguous() && peer0.size(1) <= INT_MAX &&
                  peer1.device() == peer0.device() && peer1.sizes() == peer0.sizes() &&
                  peer1.scalar_type() == peer0.scalar_type() && peer1.is_contiguous(),
              "Expected two contiguous BF16 symmetric row buffers");
  TORCH_CHECK(ids.device() == peer0.device() && ids.scalar_type() == at::kInt && ids.dim() == 2 &&
                  ids.size(1) > 0 && ids.size(1) <= 8 && positions.device() == ids.device() &&
                  positions.scalar_type() == at::kInt && positions.is_contiguous() &&
                  positions.numel() == ids.numel() && experts > 0 && experts <= INT_MAX / 2 &&
                  (rank == 0 || rank == 1),
              "Invalid two-rank MegaMoE route geometry");
  c10::cuda::CUDAGuard guard(peer0.device());
  const int64_t tokens = ids.size(0), hidden = peer0.size(1);
  if (input != nullptr) {
    TORCH_CHECK(input->device() == peer0.device() && input->scalar_type() == peer0.scalar_type() &&
                    input->dim() == 2 && input->size(0) == tokens && input->size(1) == hidden,
                "Invalid MegaMoE source activation");
  }
  Tensor local = input ? *input : at::empty({tokens, hidden}, peer0.options());
  rtc::MegaMoEBulkTransport layout{ids.data_ptr<int>(),
                                   positions.data_ptr<int>(),
                                   peer0.data_ptr(),
                                   peer1.data_ptr(),
                                   tokens,
                                   peer0.size(0),
                                   local.stride(0),
                                   local.stride(1),
                                   ids.stride(0),
                                   ids.stride(1),
                                   static_cast<int>(hidden),
                                   static_cast<int>(ids.size(1)),
                                   experts,
                                   rank};
  rtc::mega_moe_bulk_transport(layout, local.data_ptr(), !input, at::cuda::getCurrentCUDAStream());
  return local;
}

void bulk_push(const Tensor &x, const Tensor &ids, const Tensor &positions, const Tensor &peer0,
               const Tensor &peer1, int experts, int rank) {
  bulk_transport(&x, ids, positions, peer0, peer1, experts, rank);
}

Tensor bulk_pull(const Tensor &ids, const Tensor &positions, const Tensor &peer0,
                 const Tensor &peer1, int experts, int rank) {
  return bulk_transport(nullptr, ids, positions, peer0, peer1, experts, rank);
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
  m.def("mega_moe_bulk_push", &bulk_push, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_bulk_pull", &bulk_pull, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_combine", &combine, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_local_forward", &local_forward, pybind11::arg("x"), pybind11::arg("probs"),
        pybind11::arg("offsets"), pybind11::arg("fc1"), pybind11::arg("fc2"),
        pybind11::arg("out") = pybind11::none(),
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_peer_forward", &peer_forward,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_local_backward", &local_backward,
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
}  // namespace transformer_engine::pytorch
