/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "mega_moe.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>

#include "../gemm/cutlass_grouped_gemm_rtc_kernel.h"
#include "../util/rtc.h"
#include "mega_moe_grad_args.h"
#include "string_code_mega_moe_backward_cu.h"
#include "string_code_mega_moe_backward_gemm_cu.h"
#include "string_code_mega_moe_cu.h"
#include "string_code_mega_moe_device_cuh.h"
#include "string_code_mega_moe_grad_args_h.h"

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
  // combine_slots processes four adjacent BF16/FP16 elements per lane when
  // its row stride and both buffers satisfy the vector alignment.
  const auto addresses = reinterpret_cast<uintptr_t>(slots) | reinterpret_cast<uintptr_t>(output);
  const int width = hidden % 4 == 0 && addresses % 8 == 0 ? 4 : 1;
  const int64_t blocks = (tokens * (hidden / width) - 1) / 128 + 1;
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

void mega_moe_grad_gemm(MegaMoEGradGemm op, const MegaMoEGradArgs &p, bool fp16, bool fp32_out,
                        cudaStream_t stream) {
  auto &manager = KernelManager::instance();
  const bool wgrad = op == MegaMoEGradGemm::kWgrad, recompute = op == MegaMoEGradGemm::kRecompute;
  const char *name =
      wgrad ? "nvte_mega_moe_tensor_wgrad"
            : (recompute ? "nvte_mega_moe_tensor_recompute" : "nvte_mega_moe_tensor_dgrad");
  const int n = recompute ? p.output_width : p.input_width;
  const bool fixed = !wgrad && n % 256 == 0;
  // Uniform short reductions benefit from constant operand strides and
  // pipeline offsets. The kernel checks each device-side count before using
  // this candidate, so the same launch accepts an arbitrary CSR partition.
  const int mean = p.rows / p.groups;
  const bool fixed_wgrad = wgrad && !p.accumulate && !fp32_out && p.groups > 1 && n >= 2048 &&
                           n % 256 == 0 && p.output_width % 256 == 0 && p.rows % p.groups == 0 &&
                           mean >= 512 && mean <= 1536 && mean % 256 == 0;
  // The training router supplies aligned K bounds. Four consecutive N tiles can share
  // one pipeline even if adjacent experts have different reduction lengths.
  const bool ragged_columns =
      wgrad && p.reduction_rows && !p.accumulate && !fp32_out && n >= 1024 && n % 256 == 0 &&
      p.output_width % 256 == 0 &&
      int64_t(p.output_width / 256) * (n / 1024) * p.groups >= 2 * cuda::sm_count();
  // In these full pointer windows the two Qwen gradient shapes have a fixed
  // number of tasks per expert. Index their tiles directly for the observed
  // 1024..1280 padded rows per expert, while reading each expert's live K.
  const bool direct_tiles =
      wgrad && !fp16 && !fp32_out && !p.accumulate && !p.reserve_sms && p.reduction_rows &&
      p.groups == p.kMaxExperts && mean >= 1024 && mean <= 1280 &&
      ((p.output_width == 1024 && n == 2048) || (p.output_width == 2048 && n == 512));
  const int common_n = fixed || fixed_wgrad || ragged_columns || direct_tiles ? n : 0;
  const int common_k =
      fixed_wgrad ? mean : (fixed ? (recompute ? p.input_width : p.output_width) : 0);
  const int weight_rows = fixed_wgrad || ragged_columns || direct_tiles ? p.output_width : 0;
  const bool four_columns =
      fixed_wgrad && mean == 512 && n % 1024 == 0 &&
      int64_t(p.output_width / 256) * (n / 1024) * p.groups >= cuda::sm_count() * 2;
  // For 16 experts with a 1024x2048 weight gradient, long reductions run faster
  // with the regular ragged tiles. Keep the fixed-K case at mean == 1536.
  const bool prefer_ragged = p.groups == 16 && p.output_width == 1024 && n == 2048 && mean > 1536;
  bool dense = ragged_columns && !fp16 && mean >= 1536 && !prefer_ragged;
  bool compact_wgrad =
      ragged_columns && !fp16 && !p.reserve_sms && n >= 2048 && p.output_width >= 2048 &&
      int64_t(p.rows) <= int64_t(1024) * p.groups &&
      int64_t(p.output_width / 128) * (n / 256) * p.groups <= std::numeric_limits<int>::max();
  if (dense || compact_wgrad) {
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(p.a) | reinterpret_cast<uintptr_t>(p.b)) % 16 == 0) &&
        std::all_of(p.expert, p.expert + p.groups,
                    [](const void *ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; });
    dense = dense && aligned;
    compact_wgrad = compact_wgrad && aligned;
  }
  // Give the 32-expert long ring its own RTC cache key. Smaller groups keep mode 2.
  const bool long_w1_columns =
      ragged_columns && !fp16 && !dense && p.groups == 32 && p.output_width == 1024 && n == 2048;
  const int column_mode =
      long_w1_columns ? 3 : (ragged_columns ? 2 : static_cast<int>(four_columns));
  const int threads = compact_wgrad ? 256 : 512;
  const int smem = compact_wgrad ? 98304 : 196608;
  const auto label = concat_strings(name, "_", fp16, "_", fp32_out, "_", common_n, "_", common_k,
                                    "_", weight_rows, "_", column_mode, "_", dense,
                                    compact_wgrad ? "_m128" : "", direct_tiles ? "_direct" : "");
  // Function attributes belong to a device even if its architecture shares a compiled image.
  static std::mutex mutex;
  static std::map<std::pair<int, std::string>, int> residency;
  int blocks_per_sm;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto key = std::make_pair(cuda::current_device(), label);
    auto it = residency.find(key);
    if (it == residency.end()) {
      auto options = grouped_gemm_rtc::actlize_compiler_options();
      options.insert(options.end(),
                     {"-DNVTE_GG_ARCH=89", "-DNVTE_GG_N_EXPAND=0",
                      concat_strings("-DNVTE_GG_TRANS_A=", int(wgrad)),
                      concat_strings("-DNVTE_GG_TRANS_B=", int(recompute)),
                      concat_strings("-DNVTE_GG_OUT_FP32=", int(fp32_out)),
                      concat_strings("-DNVTE_MOE_FP16=", int(fp16)),
                      concat_strings("-DNVTE_GG_COMMON_K=", recompute ? 0 : common_k),
                      concat_strings("-DNVTE_MOE_WEIGHT_ROWS=", weight_rows),
                      concat_strings("-DNVTE_MOE_FOUR_COLUMNS=", column_mode),
                      concat_strings("-DNVTE_MOE_DENSE_WGRAD=", static_cast<int>(dense)),
                      concat_strings("-DNVTE_MOE_WGRAD_M128=", static_cast<int>(compact_wgrad)),
                      concat_strings("-DNVTE_MOE_COMMON_N=", common_n),
                      concat_strings("-DNVTE_MOE_COMMON_K=", common_k)});
      if (direct_tiles) options.emplace_back("-DNVTE_MOE_DIRECT_WGRAD=1");
      manager.compile(label, name, string_code_mega_moe_backward_gemm_cu,
                      "mega_moe_backward_gemm.cu", options,
                      {{string_code_mega_moe_grad_args_h, "mega_moe_grad_args.h"}});
      manager.set_function_attribute(label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);
      const int active =
          compact_wgrad ? manager.occupancy_max_active_blocks_per_sm(label, threads, smem) : 1;
      NVTE_CHECK(active > 0, "MegaMoE gradient kernel has no resident CTA");
      it = residency.emplace(key, std::min(active, 2)).first;
    }
    blocks_per_sm = it->second;
  }
  // Compact weight gradients can run two persistent CTAs per CU.
  const int sm = cuda::sm_count();
  int blocks = sm;
  if (wgrad) {
    const int64_t count = tiles(p.output_width, 256) * tiles(p.input_width, 256) * p.groups;
    // Leave the requested CUs available for the concurrent source-gradient return.
    blocks = int(std::min<int64_t>(std::max(1, sm - p.reserve_sms), count));
  }
  manager.launch(label, blocks * blocks_per_sm, threads, smem, stream, p);
}

void mega_moe_activation_backward(const MegaMoEProblem &p, const void *projection,
                                  const void *dactivation, void *activation, void *dprojection,
                                  float *dp, cudaStream_t stream) {
  launch(p.fp16, "nvte_mega_moe_swiglu_backward", true, p.rows, stream, dactivation, projection,
         p.probs, dprojection, dp, activation, p.rows, p.intermediate);
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
  mega_moe_activation_backward(p, projection, dactivation, nullptr, dprojection, dp, stream);
  dgrad(dprojection, dx, 2 * p.intermediate, p.hidden, 0);
  wgrad(dprojection, p.x, 2 * p.intermediate, p.hidden, 2);
}

}  // namespace transformer_engine::rtc
