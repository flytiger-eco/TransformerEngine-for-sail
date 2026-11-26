/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#include <cstdlib>
#include <mutex>
#include <set>

#include "../gemm/cutlass_grouped_gemm_rtc_kernel.h"
#include "../util/rtc.h"
#include "mega_moe.h"
#include "mega_moe_fused_args.h"
#include "string_code_mega_moe_fused_args_h.h"
#include "string_code_mega_moe_fused_l1_cuh.h"
#include "string_code_mega_moe_fused_l2_cuh.h"
#include "string_code_mega_moe_fused_protocol_cuh.h"
#include "string_code_mega_moe_fused_route_cuh.h"
#include "string_code_mega_moe_train_cuh.h"

namespace transformer_engine::rtc {
namespace {
int partition_sms(const char *name, int fallback, int minimum, int maximum) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) return fallback < maximum ? fallback : maximum;
  char *end = nullptr;
  const auto value = std::strtol(raw, &end, 10);
  NVTE_CHECK(end != raw && *end == '\0' && value >= minimum && value <= maximum, name,
             " must be an integer between ", minimum, " and ", maximum);
  return static_cast<int>(value);
}

std::vector<Header> headers() {
  return {{string_code_mega_moe_fused_args_h, "mega_moe_fused_args.h"},
          {string_code_mega_moe_fused_protocol_cuh, "mega_moe_fused_protocol.cuh"}};
}

void layer(const MegaMoEFusedArgs &p, bool first, cudaStream_t stream) {
  auto &manager = KernelManager::instance();
  const char *name = first ? "nvte_mega_moe_fused_l1" : "nvte_mega_moe_fused_l2";
  const int copy_sms = partition_sms("NVTE_MOE_FUSED_COPY_SMS", 8, 1, cuda::sm_count() - 1);
  const int combine_sms = partition_sms("NVTE_MOE_FUSED_COMBINE_SMS", 15, 0, cuda::sm_count() - 1);
  const int route_batch = partition_sms("NVTE_MOE_FUSED_ROUTE_BATCH", 1, 1, 4);
  const auto partition = first ? concat_strings("_copy", copy_sms)
                               : concat_strings("_combine", combine_sms, "_rb", route_batch);
  const auto label = concat_strings(name, "_h", p.hidden, "_i", p.intermediate, "_ep", p.world,
                                    "_topk", p.topk, partition);
  constexpr int threads = 512;
  const int smem = first ? 229376 : 196608;
  // The driver attribute belongs to a device function, even when identical
  // architectures share a compiled RTC image in KernelManager.
  static std::mutex mutex;
  static std::set<std::pair<int, std::string>> configured;
  {
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(cuda::current_device(), label);
    if (!configured.count(key)) {
      auto opts = grouped_gemm_rtc::actlize_compiler_options();
      opts.insert(opts.end(),
                  {"-DNVTE_GG_ARCH=89", "-DNVTE_GG_TRANS_A=0", "-DNVTE_GG_TRANS_B=1",
                   first ? "-DNVTE_GG_COMMON_K=0" : "-DNVTE_GG_COMMON_K=1", "-DNVTE_GG_N_EXPAND=1",
                   concat_strings("-DNVTE_MOE_FUSED_HIDDEN=", p.hidden),
                   concat_strings("-DNVTE_MOE_FUSED_INTERMEDIATE=", p.intermediate),
                   concat_strings("-DNVTE_MOE_FUSED_WORLD=", p.world),
                   concat_strings("-DNVTE_MOE_FUSED_TOPK=", p.topk),
                   concat_strings("-DNVTE_MOE_FUSED_COPY_SMS=", copy_sms),
                   concat_strings("-DNVTE_MOE_FUSED_COMBINE_SMS=", combine_sms),
                   concat_strings("-DNVTE_MOE_FUSED_ROUTE_BATCH=", route_batch)});
      manager.compile(label, name,
                      first ? string_code_mega_moe_fused_l1_cuh : string_code_mega_moe_fused_l2_cuh,
                      first ? "mega_moe_fused_l1.cu" : "mega_moe_fused_l2.cu", opts, headers());
      manager.set_function_attribute(label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);
      configured.insert(key);
    }
  }
  // Producers, consumers and the final peer rendezvous must all be runnable.
  // The cooperative launch guarantees residency of this single wave of CTAs.
  manager.launch_cooperative(label, cuda::sm_count(), threads, smem, stream, p);
}
}  // namespace

void mega_moe_fused_pre_dispatch(const MegaMoEFusedArgs &p, cudaStream_t stream) {
  auto &manager = KernelManager::instance();
  constexpr auto name = "nvte_mega_moe_fused_pre_dispatch";
  if (!manager.is_compiled(name)) {
    manager.compile(name, name, string_code_mega_moe_fused_route_cuh, "mega_moe_fused_route.cu", {},
                    headers());
  }
  manager.launch(name, p.eligible == p.kReady ? 8 * cuda::sm_count() : 1, 256, 0, stream, p);
}

void mega_moe_fused_layers(const MegaMoEFusedArgs &p, cudaStream_t stream) {
  layer(p, true, stream);
  layer(p, false, stream);
}

void mega_moe_train_linear(const MegaMoETrainArgs &p, bool pull, bool transpose,
                           cudaStream_t stream) {
  if (!p.tiles) return;
  auto &manager = KernelManager::instance();
  const auto name =
      concat_strings("nvte_mega_moe_", transpose ? "fwd_" : "dgrad_", pull ? "pull" : "scatter");
  const bool scatter_only = !pull && p.d == nullptr;
  const auto label =
      concat_strings(name, "_k", p.k, "_n", p.n, "_topk", p.topk, scatter_only ? "_direct" : "");
  static std::mutex mutex;
  static std::set<std::pair<int, std::string>> configured;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!configured.count({cuda::current_device(), label})) {
      auto opts = grouped_gemm_rtc::actlize_compiler_options();
      opts.insert(
          opts.end(),
          {"-DNVTE_GG_ARCH=89", "-DNVTE_GG_TRANS_A=0",
           concat_strings("-DNVTE_GG_TRANS_B=", static_cast<int>(transpose)),
           "-DNVTE_GG_COMMON_K=1", "-DNVTE_GG_N_EXPAND=1", concat_strings("-DTRAIN_KERNEL=", name),
           concat_strings("-DTRAIN_PULL=", static_cast<int>(pull)),
           concat_strings("-DTRAIN_SCATTER_ONLY=", static_cast<int>(scatter_only)),
           concat_strings("-DTRAIN_TOPK=", p.topk), concat_strings("-DTRAIN_K=", p.k),
           concat_strings("-DTRAIN_N=", p.n)});
      manager.compile(label, name, string_code_mega_moe_train_cuh, "mega_moe_train.cu", opts,
                      headers());
      manager.set_function_attribute(label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                     196608);
      configured.insert({cuda::current_device(), label});
    }
  }
  manager.launch_cooperative(label, cuda::sm_count(), 512, 196608, stream, p);
}
}  // namespace transformer_engine::rtc
