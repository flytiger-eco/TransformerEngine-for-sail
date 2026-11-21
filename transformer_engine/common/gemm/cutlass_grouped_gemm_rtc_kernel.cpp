/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cutlass_grouped_gemm_rtc_kernel.h"

#include <dlfcn.h>
#include <nvrtc.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <tuple>
#include <vector>

#include "../util/system.h"
#include "ppu_adaptive_tile.hpp"

namespace transformer_engine::grouped_gemm_rtc {
namespace {
#include "string_code_gemm_rtc_cutlass_grouped_gemm_cu.h"
#include "string_path_cutlass3_include.h"

std::string kernel_symbol(const KernelConfig& kernel, const TileConfig& tile) {
  const char* input = kernel.input_dtype == DType::kFloat16 ? "fp16" : "bf16";
  const char* output = kernel.output_dtype == DType::kFloat32 ? "fp32" : input;
  return concat_strings(
      "nvte_grouped_gemm3_", input, "_", input, "_", output, "_acc_fp32_LA_",
      kernel.trans_a ? "CM" : "RM", "_LB_", kernel.trans_b ? "CM" : "RM", "_LC_RM_sm",
      kernel.sm_arch, "_tb", tile.tile_m, "x", tile.tile_n, "x", tile.tile_k, "_s", tile.stages,
      "_C", output, kernel.grouped_tensor ? "_grouped_tensor" : "", tile.n_expand ? "_ne4" : "",
      kernel.common_k ? concat_strings("_n", kernel.common_n, "k", kernel.common_k) : "");
}

std::string resolve_cutlass_include_dir() {
  namespace fs = std::filesystem;
  auto check = [](const fs::path& p) {
    return !p.empty() && fs::exists(p / "ppu_include.hpp") &&
           fs::exists(p / "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp");
  };

  // Explicit include directory overrides.
  for (const char* env : {"NVTE_CUTLASS3_INCLUDE_DIR", "NVTE_CUTLASS_INCLUDE_DIR"}) {
    auto path = getenv<fs::path>(env);
    if (check(path)) return path.string();
  }

  // Wheels put include/cutlass3 at the installation root and the shared
  // library in transformer_engine/ or transformer_engine/wheel_lib/.
  fs::path installed;
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void*>(&resolve_cutlass_include_dir), &dl_info) &&
      dl_info.dli_fname) {
    fs::path dir = fs::path(dl_info.dli_fname).parent_path();
    if (dir.filename() == "wheel_lib") dir = dir.parent_path();
    const bool packaged = dir.filename() == "transformer_engine";
    installed = (packaged ? dir.parent_path() : dir) / "include" / "cutlass3";
    if (packaged && check(installed)) return installed.string();
  }

  // Source builds use the effective include tree materialized by CMake.
  fs::path baked(string_path_cutlass3_include);
  if (check(baked)) return baked.string();

  // CUTLASS_PATH may name a checkout or its include directory.
  fs::path cutlass_path = getenv<fs::path>("CUTLASS_PATH");
  if (!cutlass_path.empty()) {
    fs::path with_include = cutlass_path / "include";
    if (check(with_include)) return with_include.string();
    if (check(cutlass_path)) return cutlass_path.string();
  }

  // Older source trees keep CUTLASS under NVTE_HOME.
  fs::path nvte_home = getenv<fs::path>("NVTE_HOME");
  if (!nvte_home.empty()) {
    fs::path candidate = nvte_home / "3rdparty" / "cutlass3" / "include";
    if (check(candidate)) return candidate.string();
  }

  // Standalone CMake installs keep include/cutlass3 next to the library.
  if (check(installed)) return installed.string();

  // __FILE__-relative fallback for that source layout.
  fs::path here(__FILE__);
  fs::path repo_root = here.parent_path()   // gemm
                           .parent_path()   // common
                           .parent_path()   // transformer_engine
                           .parent_path();  // repo root
  fs::path fallback = repo_root / "3rdparty" / "cutlass3" / "include";
  if (check(fallback)) return fallback.string();

  NVTE_ERROR(
      "Could not locate CUTLASS 3.x + PPU include directory. "
      "Searched: NVTE_CUTLASS3_INCLUDE_DIR, baked=",
      string_path_cutlass3_include,
      ", CUTLASS_PATH, NVTE_HOME, dladdr-relative, __FILE__-relative. "
      "Set NVTE_CUTLASS3_INCLUDE_DIR to the effective PPU include tree with the TE overlay.");
}

const std::string& get_cutlass_include_dir() {
  static std::string cached = resolve_cutlass_include_dir();
  return cached;
}

// The two FNV-1a seeds and the byte order are part of the on-disk cache key.
struct CacheDigest {
  uint64_t h0 = 0xcbf29ce484222325ull;
  uint64_t h1 = 0x9e3779b97f4a7c15ull;

  void update(std::string_view bytes) {
    for (unsigned char byte : bytes) {
      h0 = (h0 ^ byte) * 0x100000001b3ull;
      h1 = (h1 ^ byte) * 0x100000001b3ull;
    }
  }

  std::string hex() const {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h0 << std::setw(16) << h1;
    return oss.str();
  }
};

std::string compute_rtc_source_digest() {
  namespace fs = std::filesystem;
  CacheDigest digest;
  digest.update(string_code_gemm_rtc_cutlass_grouped_gemm_cu);
  const fs::path root = get_cutlass_include_dir();
  std::vector<fs::path> headers;
  if (fs::is_directory(root)) {
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
      const auto ext = entry.path().extension();
      if ((ext == ".h" || ext == ".hpp" || ext == ".inl" || ext == ".cuh") &&
          entry.is_regular_file())
        headers.push_back(entry.path());
    }
  }
  std::sort(headers.begin(), headers.end());
  char chunk[16 * 1024];
  for (const auto& path : headers) {
    digest.update(fs::relative(path, root).string());
    std::ifstream in(path, std::ios::binary);
    while (in) {
      in.read(chunk, sizeof(chunk));
      digest.update({chunk, static_cast<size_t>(in.gcount())});
    }
  }
  return digest.hex();
}

const std::string& get_rtc_source_digest() {
  static std::string cached = compute_rtc_source_digest();
  return cached;
}

std::filesystem::path get_rtc_cache_dir() {
  namespace fs = std::filesystem;
  if (auto dir = getenv<std::string>("NVTE_GG_RTC_CACHE_DIR"); !dir.empty()) {
    return fs::path(dir);
  }
  if (auto cache_home = getenv<std::string>("XDG_CACHE_HOME"); !cache_home.empty()) {
    return fs::path(cache_home) / "nvte_gg_rtc";
  }
  if (auto home = getenv<std::string>("HOME"); !home.empty()) {
    return fs::path(home) / ".cache" / "nvte_gg_rtc";
  }
  return {};
}

std::filesystem::path get_rtc_cache_path(const std::string& cache_key) {
  const auto dir = get_rtc_cache_dir();
  if (dir.empty()) return {};
  CacheDigest digest;
  digest.update(cache_key);
  return dir / (digest.hex() + ".cubin");
}

void save_cached_cubin(const std::filesystem::path& path, const std::string& cubin) {
  namespace fs = std::filesystem;
  if (path.empty()) return;

  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  if (ec) return;

  const auto tmp = path.string() + "." + std::to_string(getpid()) + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return;
    f.write(cubin.data(), cubin.size());
    if (!f) return;
  }
  fs::rename(tmp, path, ec);
  if (ec) {
    fs::remove(tmp, ec);
  }
}

std::string read_cubin(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return {};
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::vector<std::string> compiler_options(const KernelConfig& kernel, const TileConfig& tile,
                                          bool line_info) {
  const auto cuda_include = cuda::include_directory(true);
  std::vector<std::string> opts{
      // hgrtc's disk cache does not account for edits to included headers.
      "--no-cache",
      "--std=c++17",
      "-DNDEBUG",
      "-DUSE_SAIL=1",
      kernel.sm_arch == 80 ? "--gpu-architecture=sm_80a" : "--gpu-architecture=sm_89",
      "-I" + cuda_include,
      "-I" + cuda_include + "/cccl",
      "-I" + get_cutlass_include_dir(),
      "-I/usr/local/PPU_SDK/targets/x86_64-linux/include",
      concat_strings("-DNVTE_GG_TRANS_A=", int(kernel.trans_a)),
      concat_strings("-DNVTE_GG_TRANS_B=", int(kernel.trans_b)),
      concat_strings("-DNVTE_GG_ARCH=", kernel.sm_arch),
      concat_strings("-DNVTE_GG_DTYPE=", int(kernel.input_dtype == DType::kFloat16)),
      concat_strings("-DNVTE_GG_OUT_FP32=", int(kernel.output_dtype == DType::kFloat32)),
      concat_strings("-DNVTE_GG_GROUPED_TENSOR=", int(kernel.grouped_tensor)),
      concat_strings("-DNVTE_GG_N_EXPAND=", tile.n_expand ? 4 : 1),
      concat_strings("-DNVTE_GG_COMMON_N=", kernel.common_n),
      concat_strings("-DNVTE_GG_COMMON_K=", kernel.common_k),
      concat_strings("-DNVTE_GG_TILE_M=", tile.tile_m),
      concat_strings("-DNVTE_GG_TILE_N=", tile.tile_n),
      concat_strings("-DNVTE_GG_TILE_K=", tile.tile_k),
      concat_strings("-DNVTE_GG_STAGES=", tile.stages),
      "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
      "-D__HGGC_FP16_H__=1",
      "-D_HGGC_FP16_H_=1",
      "-DHGGC_FP16_H=1",
      "-D__HGGC_BF16_H__=1",
      "-D__HGGC_BF16_HPP__=1",
      "-D__BF16_DEFINED__=1",
  };
  if (line_info) opts.emplace_back("-lineinfo");
  return opts;
}

std::string compile_cubin(const std::string& symbol, const std::vector<std::string>& options) {
  // hgrtc truncates long kernel identifiers passed via -D. Put the name in the source.
  const auto source = concat_strings("#define NVTE_GG_KERNEL_NAME ", symbol, "\n",
                                     string_code_gemm_rtc_cutlass_grouped_gemm_cu);
  nvrtcProgram program{};
  NVTE_CHECK_NVRTC(nvrtcCreateProgram(&program, source.c_str(), "cutlass_grouped_gemm_rtc.cu", 0,
                                      nullptr, nullptr));
  auto destroy = [](nvrtcProgram p) { nvrtcDestroyProgram(&p); };
  std::unique_ptr<std::remove_pointer_t<nvrtcProgram>, decltype(destroy)> guard(program, destroy);
  std::vector<const char*> args;
  for (const auto& option : options) args.push_back(option.c_str());
  if (nvrtcCompileProgram(program, args.size(), args.data()) != NVRTC_SUCCESS) {
    size_t size = 0;
    NVTE_CHECK_NVRTC(nvrtcGetProgramLogSize(program, &size));
    std::string log(size, '\0');
    NVTE_CHECK_NVRTC(nvrtcGetProgramLog(program, log.data()));
    NVTE_ERROR("NVRTC grouped GEMM compilation failed for ", symbol, ":\n", log);
  }
  size_t size = 0;
  NVTE_CHECK_NVRTC(nvrtcGetCUBINSize(program, &size));
  std::string cubin(size, '\0');
  NVTE_CHECK_NVRTC(nvrtcGetCUBIN(program, cubin.data()));
  return cubin;
}
}  // namespace

CompiledKernel::CompiledKernel(int device, const std::string& symbol, std::string cubin,
                               int host_smem)
    : module(symbol, std::move(cubin)), function(module.get_function(device)) {
  int static_smem = 0;
  NVTE_CALL_CHECK_CUDA_DRIVER(cuFuncGetAttribute, &static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                              function);
  smem_size = std::max(static_smem, host_smem);
  if (smem_size >= (48 << 10)) {
    module.set_function_attribute(device, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  smem_size);
  }
}

const CompiledKernel& get_or_compile(int device, const KernelConfig& kernel, const TileConfig& tile,
                                     int host_smem) {
  const auto key =
      std::make_tuple(device, kernel.sm_arch, kernel.trans_a, kernel.trans_b, kernel.input_dtype,
                      kernel.output_dtype, kernel.grouped_tensor, tile.tile_m, tile.tile_n,
                      tile.tile_k, tile.stages, tile.n_expand, kernel.common_n, kernel.common_k);
  static std::map<std::decay_t<decltype(key)>, CompiledKernel> cache;
  static std::mutex cache_mutex;
  auto lookup = [&]() -> const CompiledKernel* {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    return it == cache.end() ? nullptr : &it->second;
  };
  if (const auto* cached = lookup()) return *cached;

  // Serialize NVRTC and disk writes. Warm launches only take the map lock.
  static std::mutex compile_mutex;
  std::lock_guard<std::mutex> compiling(compile_mutex);
  if (const auto* cached = lookup()) return *cached;

  const auto symbol = kernel_symbol(kernel, tile);
  static const bool line_info = getenv<bool>("NVTE_GG_RTC_LINEINFO", false);
  const auto options = compiler_options(kernel, tile, line_info);
  auto disk_key = symbol + "|" + get_rtc_source_digest();
  for (const auto& option : options) disk_key += "|" + option;
  const auto path = get_rtc_cache_path(disk_key);
  auto cubin = path.empty() ? std::string{} : read_cubin(path);
  const bool cache_miss = cubin.empty();
  if (cache_miss) cubin = compile_cubin(symbol, options);
  CompiledKernel compiled(device, symbol, cubin, host_smem);
  if (cache_miss && !path.empty()) save_cached_cubin(path, cubin);
  std::lock_guard<std::mutex> lock(cache_mutex);
  return cache.emplace(key, std::move(compiled)).first->second;
}

}  // namespace transformer_engine::grouped_gemm_rtc
