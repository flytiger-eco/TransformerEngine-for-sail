/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/rtc/mega_moe.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <optional>

#include "common/rtc/mega_moe_grad_args.h"
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

// Wrapping a device pointer through DLPack or symmetric memory may give the
// same allocation multiple StorageImpl identities. Check the actual addresses.
// The output is contiguous and a strided input can cover a larger span.
void check_disjoint(const Tensor &out, const Tensor &input) {
  if (!out.numel() || !input.numel()) return;
  const auto a = reinterpret_cast<uintptr_t>(input.data_ptr());
  const auto b = reinterpret_cast<uintptr_t>(out.data_ptr());
  size_t bytes = input.nbytes();
  if (a < b && !input.is_non_overlapping_and_dense()) {
    size_t span = 1;
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
      span += (input.size(dim) - 1) * input.stride(dim);
    }
    bytes = span * input.element_size();
  }
  TORCH_CHECK(a < b ? b - a >= bytes : a - b >= out.nbytes(), "MegaMoE out overlaps an input");
}

// The symmetric-memory caller supplies contiguous owner-local storage. An
// exact alias with x is safe: FC2 starts after FC1 has consumed all input.
void check_output(const Tensor &out, const Tensor &x, const Tensor &probs, const Tensor &offsets,
                  const Weights &w1, const Weights &w2) {
  TORCH_CHECK(out.device() == x.device() && out.scalar_type() == x.scalar_type() &&
                  out.sizes() == x.sizes() && out.is_contiguous(),
              "MegaMoE out must match the packed input shape, dtype and device and be contiguous");
  if (out.data_ptr() != x.data_ptr()) check_disjoint(out, x);
  check_disjoint(out, probs);
  check_disjoint(out, offsets);
  for (const auto &w : w1) check_disjoint(out, w);
  for (const auto &w : w2) check_disjoint(out, w);
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

bool aligned(const Tensor &tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0;
}

void expert_gemm(rtc::MegaMoEGradGemm op, const Tensor &a, const Tensor &b, const Tensor &d,
                 const Tensor &offsets, const Weights &expert, bool accumulate, cudaStream_t stream,
                 int reserve_sms = 0) {
  rtc::MegaMoEGradArgs p{};
  p.rows = a.size(0);
  p.output_width = expert.front().size(0);
  p.input_width = expert.front().size(1);
  p.a = a.data_ptr();
  p.b = b.defined() ? b.data_ptr() : nullptr;
  p.d = d.defined() ? d.data_ptr() : nullptr;
  p.accumulate = accumulate;
  p.reserve_sms = reserve_sms;
  for (size_t first = 0; first < expert.size(); first += p.kMaxExperts) {
    p.offsets = offsets.data_ptr<int>() + first;
    p.groups = std::min<size_t>(p.kMaxExperts, expert.size() - first);
    for (int g = 0; g < p.groups; ++g) p.expert[g] = expert[first + g].data_ptr();
    rtc::mega_moe_grad_gemm(op, p, a.scalar_type() == at::kHalf,
                            expert.front().scalar_type() == at::kFloat, stream);
  }
}

// Single expert-layer entry points accept the same CSR row boundaries and
// use exactly the same NN / TN kernels as the public two-projection backward.
void check_grad_gemm(const Tensor &dy, const Tensor &offsets, const Weights &w, bool wgrad) {
  TORCH_CHECK(dy.is_cuda() && dy.dim() == 2 && dy.is_contiguous() && aligned(dy) &&
                  (dy.scalar_type() == at::kHalf || dy.scalar_type() == at::kBFloat16) &&
                  dy.size(0) <= INT_MAX && !w.empty() && w.size() < INT_MAX &&
                  w.front().dim() == 2 && w.front().size(0) == dy.size(1),
              "Invalid packed MegaMoE gradient operands");
  TORCH_CHECK(offsets.device() == dy.device() && offsets.scalar_type() == at::kInt &&
                  offsets.dim() == 1 && offsets.is_contiguous() &&
                  offsets.numel() == int64_t(w.size()) + 1,
              "Expected device int32 expert offsets [E+1]");
  TORCH_CHECK(rtc::mega_moe_grouped_forward_supported(),
              "MegaMoE tensor gradients require SM89 ACTLIZE");
  const auto &first = w.front();
  for (auto n : first.sizes()) {
    TORCH_CHECK(n > 0 && n <= INT_MAX - 256 && n % 8 == 0,
                "Expected aligned MegaMoE gradient widths");
  }
  TORCH_CHECK(
      first.scalar_type() == dy.scalar_type() || (wgrad && first.scalar_type() == at::kFloat),
      "Invalid expert gradient dtype");
  for (const auto &tensor : w) {
    TORCH_CHECK(tensor.device() == dy.device() && tensor.scalar_type() == first.scalar_type() &&
                    tensor.sizes() == first.sizes() && tensor.is_contiguous() && aligned(tensor),
                "Incompatible expert tensor");
  }
}

Tensor dgrad(const Tensor &dy, const Tensor &offsets, const Weights &weights) {
  TORCH_CHECK(dy.is_cuda(), "MegaMoE requires device routes");
  c10::cuda::CUDAGuard guard(dy.device());
  check_grad_gemm(dy, offsets, weights, false);
  auto out = at::empty({dy.size(0), weights.front().size(1)}, dy.options());
  if (dy.size(0)) {
    expert_gemm(rtc::MegaMoEGradGemm::kDgrad, dy, {}, out, offsets, weights, false,
                at::cuda::getCurrentCUDAStream());
  }
  return out;
}

void wgrad(const Tensor &dy, const Tensor &x, const Tensor &offsets, const Weights &grads,
           bool accumulate) {
  TORCH_CHECK(dy.is_cuda(), "MegaMoE requires device routes");
  c10::cuda::CUDAGuard guard(dy.device());
  check_grad_gemm(dy, offsets, grads, true);
  TORCH_CHECK(x.device() == dy.device() && x.scalar_type() == dy.scalar_type() && x.dim() == 2 &&
                  x.is_contiguous() && aligned(x) && x.size(0) == dy.size(0) &&
                  x.size(1) == grads.front().size(1),
              "Invalid packed MegaMoE wgrad input");
  // Gradients are independent output matrices. Reject overlap with an input
  // or another output before concurrent tile stores can clobber a live range.
  std::vector<std::pair<uintptr_t, uintptr_t>> regions;
  for (const auto &matrix : grads) {
    uintptr_t begin = reinterpret_cast<uintptr_t>(matrix.data_ptr());
    uintptr_t end = begin + matrix.nbytes();
    for (const auto *input : {&x, &dy, &offsets}) check_disjoint(matrix, *input);
    regions.emplace_back(begin, end);
  }
  std::sort(regions.begin(), regions.end());
  for (size_t j = 1; j < regions.size(); ++j) {
    TORCH_CHECK(regions[j - 1].second <= regions[j].first, "Overlapping MegaMoE wgrad outputs");
  }
  expert_gemm(rtc::MegaMoEGradGemm::kWgrad, dy, x, {}, offsets, grads, accumulate,
              at::cuda::getCurrentCUDAStream());
}

// The transport stream waits only for source-row gradients. Weight gradients
// and their workspace stay on the caller's compute stream.
struct GradReady {
  explicit GradReady(int reserve_sms) : reserve_sms(reserve_sms) {
    TORCH_CHECK(reserve_sms >= 0, "Expected nonnegative MegaMoE return SM reservation");
  }
  int reserve_sms;
  at::cuda::CUDAEvent event;
  void wait() { event.block(at::cuda::getCurrentCUDAStream()); }
};

std::tuple<Tensor, Tensor, Weights, Weights> local_backward(
    const Tensor &dy, const Tensor &x, const Tensor &probs, const Tensor &offsets,
    const Weights &fc1, const Weights &fc2, const std::optional<Tensor> &out_dx,
    const std::optional<Tensor> &out_dp, GradReady *ready) {
  TORCH_CHECK(x.is_cuda(), "MegaMoE requires device routes");
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(dy.device() == x.device() && dy.scalar_type() == x.scalar_type() &&
                  dy.sizes() == x.sizes() && dy.is_contiguous(),
              "Invalid MegaMoE output gradient");
  LocalExperts local(x, x.size(0), probs, offsets, fc1, fc2, true);
  const auto &p = local.problem;
  if (out_dx) {
    check_output(*out_dx, x, probs, offsets, fc1, fc2);
    TORCH_CHECK(!x.numel() || out_dx->data_ptr() != x.data_ptr(), "MegaMoE dx must not alias x");
    check_disjoint(*out_dx, dy);
  }
  if (out_dp) {
    TORCH_CHECK(out_dp->sizes() == probs.sizes() && out_dp->scalar_type() == probs.scalar_type() &&
                    out_dp->device() == probs.device() && out_dp->is_contiguous(),
                "Invalid MegaMoE probability gradient output");
    for (const auto *v : {&probs, &x, &dy, &offsets}) check_disjoint(*out_dp, *v);
    if (out_dx) check_disjoint(*out_dp, *out_dx);
    for (const auto *weights : {&fc1, &fc2}) {
      for (const auto &w : *weights) check_disjoint(*out_dp, w);
    }
  }
  auto dx = out_dx ? *out_dx : at::empty_like(x);
  auto dp = out_dp ? *out_dp : at::empty_like(probs);
  if (!p.rows) {
    dx.zero_();
    dp.zero_();
    if (ready) ready->event.record(at::cuda::getCurrentCUDAStream());
    return {dx, dp, local.dw1, local.dw2};
  }
  auto projection = at::empty({p.rows, 2 * p.intermediate}, x.options());
  auto activation = at::empty({p.rows, p.intermediate}, x.options());
  auto dprojection = at::empty_like(projection), dactivation = at::empty_like(activation);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool native = p.rows >= 32 && p.hidden >= 128 && p.intermediate >= 64 &&
                      p.hidden <= INT_MAX - 256 && 2 * p.intermediate <= INT_MAX - 256 &&
                      p.hidden % 8 == 0 && p.intermediate % 8 == 0 && aligned(x) && aligned(dy) &&
                      aligned(dx) && std::all_of(local.w1.begin(), local.w1.end(), aligned) &&
                      std::all_of(local.w2.begin(), local.w2.end(), aligned) &&
                      rtc::mega_moe_grouped_forward_supported();
  if (native) {
    expert_gemm(rtc::MegaMoEGradGemm::kRecompute, x, {}, projection, offsets, local.w1, false,
                stream);
    expert_gemm(rtc::MegaMoEGradGemm::kDgrad, dy, {}, dactivation, offsets, local.w2, false,
                stream);
    rtc::mega_moe_activation_backward(p, projection.data_ptr(), dactivation.data_ptr(),
                                      activation.data_ptr(), dprojection.data_ptr(),
                                      dp.data_ptr<float>(), stream);
    expert_gemm(rtc::MegaMoEGradGemm::kDgrad, dprojection, {}, dx, offsets, local.w1, false,
                stream);
    if (ready) ready->event.record(stream);
    const int reserve_sms = ready ? ready->reserve_sms : 0;
    expert_gemm(rtc::MegaMoEGradGemm::kWgrad, dy, activation, {}, offsets, local.dw2, false, stream,
                reserve_sms);
    expert_gemm(rtc::MegaMoEGradGemm::kWgrad, dprojection, x, {}, offsets, local.dw1, false, stream,
                reserve_sms);
  } else {
    local.prepare_table(x);
    rtc::mega_moe_backward(p, dy.data_ptr(), projection.data_ptr(), activation.data_ptr(),
                           dactivation.data_ptr(), dprojection.data_ptr(), dx.data_ptr(),
                           dp.data_ptr<float>(), stream);
    if (ready) ready->event.record(stream);
  }
  return {dx, dp, local.dw1, local.dw2};
}
Tensor bulk_transport(const Tensor *input, const Tensor &ids, const Tensor &positions,
                      const Tensor &peer0, const Tensor &peer1, int experts, int rank,
                      const std::optional<Tensor> &output = std::nullopt) {
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
  if (output) {
    TORCH_CHECK(output->device() == peer0.device() &&
                    output->scalar_type() == peer0.scalar_type() && output->dim() == 2 &&
                    output->size(0) == tokens && output->size(1) == hidden &&
                    output->is_contiguous(),
                "Expected a contiguous BF16 MegaMoE return buffer [T,H]");
    for (const auto *v : {&ids, &positions, &peer0, &peer1}) check_disjoint(*output, *v);
  }
  Tensor local = input ? *input : (output ? *output : at::empty({tokens, hidden}, peer0.options()));
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
                 const Tensor &peer1, int experts, int rank, const std::optional<Tensor> &output) {
  return bulk_transport(nullptr, ids, positions, peer0, peer1, experts, rank, output);
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
  m.def("mega_moe_bulk_pull", &bulk_pull, pybind11::arg("ids"), pybind11::arg("positions"),
        pybind11::arg("peer0"), pybind11::arg("peer1"), pybind11::arg("experts"),
        pybind11::arg("rank"), pybind11::arg("out") = pybind11::none(),
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_combine", &combine, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_local_forward", &local_forward, pybind11::arg("x"), pybind11::arg("probs"),
        pybind11::arg("offsets"), pybind11::arg("fc1"), pybind11::arg("fc2"),
        pybind11::arg("out") = pybind11::none(),
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_peer_forward", &peer_forward,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<GradReady>(m, "MegaMoEGradReady")
      .def(pybind11::init<int>(), pybind11::arg("reserve_sms") = 0)
      .def("wait", &GradReady::wait);
  m.def("mega_moe_dgrad", &dgrad, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_wgrad", &wgrad, pybind11::arg("dy"), pybind11::arg("x"), pybind11::arg("offsets"),
        pybind11::arg("grads"), pybind11::arg("accumulate") = false,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("mega_moe_local_backward", &local_backward, pybind11::arg("dy"), pybind11::arg("x"),
        pybind11::arg("probs"), pybind11::arg("offsets"), pybind11::arg("fc1"),
        pybind11::arg("fc2"), pybind11::arg("out_dx") = pybind11::none(),
        pybind11::arg("out_dp") = pybind11::none(), pybind11::arg("ready") = pybind11::none(),
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
}  // namespace transformer_engine::pytorch
