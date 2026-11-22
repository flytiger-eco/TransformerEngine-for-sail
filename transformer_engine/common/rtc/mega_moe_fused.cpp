/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
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

namespace transformer_engine::rtc {
namespace {
std::vector<Header> headers() {
  return {{string_code_mega_moe_fused_args_h, "mega_moe_fused_args.h"},
          {string_code_mega_moe_fused_protocol_cuh, "mega_moe_fused_protocol.cuh"}};
}

void layer(const MegaMoEFusedArgs &p, bool first, cudaStream_t stream) {
  auto &manager = KernelManager::instance();
  const char *name = first ? "nvte_mega_moe_fused_l1" : "nvte_mega_moe_fused_l2";
  const auto label = concat_strings(name, "_h", p.hidden, "_i", p.intermediate);
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
                   concat_strings("-DNVTE_MOE_FUSED_INTERMEDIATE=", p.intermediate)});
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
}  // namespace transformer_engine::rtc
