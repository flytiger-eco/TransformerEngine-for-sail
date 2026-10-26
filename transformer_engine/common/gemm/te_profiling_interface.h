/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

// HGTX (PPU's NVTX equivalent) per-launch tag instrumentation for the TE
// grouped GEMM backends (RTC / static PPU CUTLASS / cuBLAS fallback).
//
// Port of DeepGemm's profiling_interface.cuh three-piece set (tag push/pop
// around every kernel launch + SHOW_PARAMS stdout dump), with the two-layer
// tag design from DeepGemm/nvtx_tag_replay_guide.md section 4.4:
//
//   HGTX range message (ACU --hgtx-include 'tengine@<name>/' matches this
//   exact full name; no escaping needed since it is all [A-Za-z0-9_]):
//     TE_GroupedGemmRtc_bf16_g16_TN_m2048_n1024_k4096_gpu0
//   SHOW_PARAMS stdout line (human / caselist / replay-script form):
//     [TE] --format=GroupedGemmRtc,data_type:bf16,groups:16,layout:TN,
//          m:2048,n:1024,k:4096,tile:256x128x64s2,distribution:[...],
//          accumulate:0,gpu:0
//
// Environment switches (read once at first use, so export them before the
// process starts -- same contract as DeepGemm):
//   PPU_LIB_PERF_INSTRUMENT=1  HGTX range push/pop around every launch
//   PPU_LIB_SHOW_PARAMS=1      print the full tag to stdout
//   PPU_LIB_TFLOPS=1           CUDA-event timing per launch; right after
//                              the pop a short [TE-TFLOPS] HGTX range (plus a
//                              stdout line under SHOW_PARAMS) carries flops /
//                              kernel time / achieved TFLOPS / tensor-core
//                              utilization against the device's dense peak
//                              (139.2 TFLOPS for SM80 810E, 415.3 TFLOPS for
//                              SM89 890M).
//                              Note: end_timing() synchronizes the stream, so
//                              this is a profiling-mode switch, not for
//                              production throughput runs.
//
// TE needs no D2H distribution dump: grouped GEMM extents are host-side
// tensor shapes, so the real per-group distribution rides in the tag for
// free and there is no sync to gate behind PPU_LIB_DUMP_DEVICE.
//
// Compile-time gate: CMake defines NVTE_ENABLE_HGTX when hgtx3/hgToolsExt.h
// (header-only) is found in the PPU SDK. Without it every entry point here
// compiles to a no-op, so non-PPU builds are unaffected.
//
// Usage (three-piece around one kernel launch, exception-safe via guard):
//   te_profiling::TeProfParam prof;
//   if (te_profiling::TeProfilingInterface::Instance().enabled()) {
//     prof.set_grouped_gemm(...);  // host-side shapes, no sync
//   }
//   {
//     te_profiling::TeHgtxGuard guard(prof);  // push
//     ... kernel launch (cuLaunchKernel / gemm.run / cublasLt calls) ...
//   }                                          // pop on scope exit

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#ifdef NVTE_ENABLE_HGTX
// Included as <hgToolsExt.h> with the include path pointing at the SDK's
// .../include/hgtx3 directory itself (see CMakeLists.txt). Pointing the path
// at .../include instead (the DeepGemm style `#include <hgtx3/hgToolsExt.h>`)
// would shadow same-named CUDA SDK headers (vector_types.h, host_defines.h,
// driver_types.h, cub/, ...) and break the surrounding TU's compilation.
// All of hgtx3's internal includes are quoted-relative or system headers, so
// the narrower path is sufficient.
#include <hgToolsExt.h>
#endif

namespace transformer_engine {
namespace te_profiling {

// ---------------------------------------------------------------------------
// TeProfParam: assembles the per-launch tag of one grouped GEMM call.
//
// Group extents are given as parallel host-side arrays (one entry per group,
// CUTLASS problem-shape order M, N, K). Uniform extents serialize as scalars;
// varying extents serialize as bracketed lists, so the `m` list doubles as
// DeepGemm's "distribution" field (per-group true M) and keeps a collected
// tag replayable without extra information.
// ---------------------------------------------------------------------------
class TeProfParam {
 public:
  // op_name: backend selector inside the tag, one of
  //   GroupedGemmRtc    - NVRTC-JIT CUTLASS path (tile/stages known)
  //   GroupedGemmPpu    - statically compiled PPU CUTLASS path
  //   GroupedGemmCublas - cuBLAS per-group fallback
  // layout: "NN"/"TN"/"NT"/"TT" (trans_a, trans_b of the backend's own
  //   operand convention -- the RTC/PPU paths swap A/B before this point,
  //   the cuBLAS path uses the caller's transa/transb).
  // tile: {tile_m, tile_n, tile_k, stages} when the backend picked a tile,
  //   nullptr otherwise.
  // stream: launch stream, required for the PPU_LIB_TFLOPS event timing (the
  //   stop event is recorded on it when the guard pops).
  void set_grouped_gemm(const char* op_name, const char* data_type,
                        int num_groups, const char* layout, bool accumulate,
                        const int* shapes_m, const int* shapes_n,
                        const int* shapes_k, const int* tile, int device_id,
                        cudaStream_t stream = nullptr) {
    op_name_ = op_name;
    add("data_type", data_type);
    add("groups", num_groups);
    add("layout", layout);
    add_extent("m", shapes_m, num_groups, &m0_);
    add_extent("n", shapes_n, num_groups, &n0_);
    add_extent("k", shapes_k, num_groups, &k0_);
    if (tile != nullptr) {
      std::stringstream ss;
      ss << tile[0] << "x" << tile[1] << "x" << tile[2] << "s" << tile[3];
      add("tile", ss.str());
    }
    add("accumulate", accumulate ? 1 : 0);
    add("gpu", device_id);
    for (int i = 0; i < num_groups; ++i) {
      flops_ += 2.0 * double(shapes_m[i]) * double(shapes_n[i]) *
                double(shapes_k[i]);
    }
    device_id_ = device_id;
    stream_ = stream;
  }

  const std::string& op_name() const { return op_name_; }
  double flops() const { return flops_; }
  int device_id() const { return device_id_; }
  cudaStream_t stream() const { return stream_; }

  // PPU_LIB_TFLOPS timing: a CUDA-event pair on the launch stream, started
  // when the guard pushes and consumed when it pops. The pop-side sync makes
  // this profiling-only.
  void begin_timing() {
    cudaEventCreateWithFlags(&ev_start_, cudaEventDefault);
    cudaEventCreateWithFlags(&ev_stop_, cudaEventDefault);
    cudaEventRecord(ev_start_, stream_);
  }
  void end_timing() {
    cudaEventRecord(ev_stop_, stream_);
    cudaEventSynchronize(ev_stop_);
    cudaEventElapsedTime(&ms_, ev_start_, ev_stop_);
    cudaEventDestroy(ev_start_);
    cudaEventDestroy(ev_stop_);
    ev_start_ = nullptr;
    ev_stop_ = nullptr;
  }
  bool timing_active() const { return ev_start_ != nullptr; }
  float elapsed_ms() const { return ms_; }

  // Machine-facing range name for the HGTX message. Restricted to
  // [A-Za-z0-9_] so the ACU filter needs no escaping; varying group extents
  // collapse to the first group.
  std::string range_name() const {
    std::stringstream ss;
    ss << "TE_" << op_name_ << "_" << at("data_type") << "_g" << at("groups")
       << "_" << at("layout") << "_m" << m0_ << "_n" << n0_ << "_k" << k0_
       << "_gpu" << at("gpu");
    return ss.str();
  }

  // Human-facing full tag: printed by PPU_LIB_SHOW_PARAMS and directly
  // consumable as the --format descriptor of a TE replay script.
  std::string format() const {
    std::stringstream ss;
    ss << "[TE] --format=" << op_name_;
    for (const auto& key : order_) {
      ss << "," << key << ":" << args_.at(key);
    }
    return ss.str();
  }

 private:
  void add(const std::string& key, const std::string& value) {
    args_[key] = value;
    order_.push_back(key);
  }

  template <typename T>
  void add(const std::string& key, const T& value) {
    std::stringstream ss;
    ss << value;
    add(key, ss.str());
  }

  // Serialize an extent as a scalar when uniform across groups, else as a
  // bracketed list; first_ always receives the first group's value.
  void add_extent(const std::string& key, const int* values, int num_groups,
                  int* first_) {
    *first_ = num_groups > 0 ? values[0] : 0;
    const bool uniform =
        std::all_of(values, values + num_groups,
                    [&](int v) { return v == *first_; });
    if (num_groups == 0) {
      add(key, 0);
    } else if (uniform) {
      add(key, *first_);
    } else {
      std::stringstream ss;
      ss << "[";
      for (int i = 0; i < num_groups; ++i) {
        ss << values[i] << (i + 1 < num_groups ? "," : "");
      }
      ss << "]";
      add(key, ss.str());
    }
  }

  std::string at(const std::string& key) const {
    const auto it = args_.find(key);
    return it != args_.end() ? it->second : "";
  }

  std::string op_name_;
  std::unordered_map<std::string, std::string> args_;
  std::vector<std::string> order_;
  int m0_ = 0;
  int n0_ = 0;
  int k0_ = 0;
  double flops_ = 0.0;
  int device_id_ = 0;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t ev_start_ = nullptr;
  cudaEvent_t ev_stop_ = nullptr;
  float ms_ = 0.0f;
};

// ---------------------------------------------------------------------------
// TeProfilingInterface: per-module singleton owning the "tengine" HGTX domain.
//
// Like DeepGemm, every linked module (main .so, static archives) gets its own
// instance and domain; same-name domains coexist and the ACU filter matches
// on 'domain@range_name' regardless.
// ---------------------------------------------------------------------------
class TeProfilingInterface {
 public:
  TeProfilingInterface(const TeProfilingInterface&) = delete;
  void operator=(const TeProfilingInterface&) = delete;

  static TeProfilingInterface& Instance() {
    static TeProfilingInterface instance;
    return instance;
  }

  // Gate for the tag assembly as well: when false, callers skip building the
  // TeProfParam entirely and the per-launch cost is one inlined bool check.
  bool enabled() const { return use_hgtx_ || show_params_; }
  bool tflops_enabled() const { return tflops_; }

  void instrument(bool start, TeProfParam& params) {
    if (!enabled()) return;

    if (start) {
      const std::string name = params.range_name();
      if (show_params_) {
        std::cout << params.format() << std::endl;
      }
#ifdef NVTE_ENABLE_HGTX
      if (use_hgtx_) {
        hgtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = HGTX_VERSION;
        eventAttrib.size = HGTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.messageType = HGTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name.c_str();
        hgtxDomainRangePushEx(domain_, &eventAttrib);
      }
#endif
      // NOTE: stream 0 (the legacy default stream) is a valid record target,
      // so there is no null check here -- torch's getCurrentCUDAStream()
      // returns 0 outside explicit stream contexts.
      if (tflops_) {
        params.begin_timing();
      }
    } else {
      // Pop the case range before the timing sync so its span is not widened
      // by the cudaEventSynchronize inside end_timing().
#ifdef NVTE_ENABLE_HGTX
      if (use_hgtx_) {
        hgtxDomainRangePop(domain_);
      }
#endif
      if (tflops_ && params.timing_active()) {
        params.end_timing();
        report_tflops(params);
      }
    }
  }

 private:
  // Dense bf16/fp16 tensor-core peak in TFLOPS: 810E is SM80, 890M is SM89.
  // Devices outside the table report peak 0 and the utilization is printed
  // as n/a. The grouped-GEMM binding releases the GIL around the call, so
  // concurrent TFLOPS profiling can reach this cache from several host
  // threads -- protect it.
  static double peak_tflops(int device) {
    static std::mutex cache_mutex;
    static std::unordered_map<int, double> cache;
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto it = cache.find(device);
    if (it != cache.end()) return it->second;
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    const double peak =
        (major == 8 && minor == 0) ? 139.2 :
        (major == 8 && minor == 9) ? 415.3 : 0.0;
    cache.emplace(device, peak);
    return peak;
  }

  // Emit the measured throughput as a short [TE-TFLOPS] push/pop range that
  // follows the case range at the same timeline level -- an instantaneous
  // marker is invisible on the asight timeline, while a PushPop range shows
  // its message text and is addressable by acu --hgtx-include. Also printed
  // to stdout under SHOW_PARAMS.
  void report_tflops(const TeProfParam& params) const {
    const double ms = double(params.elapsed_ms());
    if (ms <= 0.0) return;
    const double tflops = params.flops() / (ms * 1e-3) / 1e12;
    const double peak = peak_tflops(params.device_id());
    std::stringstream ss;
    ss << "[TE-TFLOPS] op=" << params.op_name()
       << std::scientific << std::setprecision(3)
       << " flops=" << params.flops()
       << std::fixed << std::setprecision(1)
       << " time_us=" << ms * 1e3
       << std::setprecision(2)
       << " tflops=" << tflops;
    if (peak > 0.0) {
      ss << " peak=" << peak
         << std::setprecision(1)
         << " util=" << (tflops / peak * 100.0) << "%";
    } else {
      ss << " peak=n/a util=n/a";
    }
    const std::string msg = ss.str();
    if (show_params_) {
      std::cout << msg << std::endl;
    }
#ifdef NVTE_ENABLE_HGTX
    if (use_hgtx_) {
      hgtxEventAttributes_t eventAttrib = {0};
      eventAttrib.version = HGTX_VERSION;
      eventAttrib.size = HGTX_EVENT_ATTRIB_STRUCT_SIZE;
      eventAttrib.messageType = HGTX_MESSAGE_TYPE_ASCII;
      eventAttrib.message.ascii = msg.c_str();
      hgtxDomainRangePushEx(domain_, &eventAttrib);
      hgtxDomainRangePop(domain_);
    }
#endif
  }
  TeProfilingInterface() {
#ifdef NVTE_ENABLE_HGTX
    domain_ = hgtxDomainCreateA("tengine");
    use_hgtx_ = env_flag("PPU_LIB_PERF_INSTRUMENT");
    tflops_ = env_flag("PPU_LIB_TFLOPS");
#else
    // Without the HGTX SDK there is no range to emit, and the TFLOPS path
    // would still add a CUDA event pair plus a per-launch synchronize to
    // every grouped GEMM -- keep the documented compile-time no-op gate.
    use_hgtx_ = false;
    tflops_ = false;
#endif
    // Tag construction and the '[TE] --format=' stdout line are plain text and
    // stay available even without HGTX, so PPU_LIB_SHOW_PARAMS is honoured in
    // both builds (it costs one line of stdout per launch).
    show_params_ = env_flag("PPU_LIB_SHOW_PARAMS");
  }

  ~TeProfilingInterface() {
#ifdef NVTE_ENABLE_HGTX
    hgtxDomainDestroy(domain_);
#endif
  }

  // Same strictness as DeepGemm: only exact "1" turns a switch on.
  static bool env_flag(const char* key) {
    const char* value = std::getenv(key);
    return value != nullptr && *value == '1';
  }

  bool use_hgtx_ = false;
  bool show_params_ = false;
  bool tflops_ = false;
#ifdef NVTE_ENABLE_HGTX
  hgtxDomainHandle_t domain_ = nullptr;
#endif
};

// RAII pairing of instrument(true/false) around one launch: an NVTE_ERROR
// thrown by a failed launch check cannot leave the HGTX range stack
// unbalanced.
class TeHgtxGuard {
 public:
  explicit TeHgtxGuard(TeProfParam& params) : params_(params) {
    TeProfilingInterface::Instance().instrument(true, params_);
  }
  ~TeHgtxGuard() { TeProfilingInterface::Instance().instrument(false, params_); }

  TeHgtxGuard(const TeHgtxGuard&) = delete;
  void operator=(const TeHgtxGuard&) = delete;

 private:
  TeProfParam& params_;
};

}  // namespace te_profiling
}  // namespace transformer_engine
