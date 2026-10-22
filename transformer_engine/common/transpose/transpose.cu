/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/cast_transpose_noop.h>
#include <transformer_engine/transpose.h>

#include <algorithm>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/rtc.h"
#include "../util/string.h"
#include "../utils.cuh"
#include "./transpose.h"

namespace transformer_engine {
namespace detail {

namespace {

// String with RTC kernel implementation
#include "string_code_transpose_rtc_transpose_cu.h"

// Hard-coded kernel parameters
constexpr size_t warps_per_tile = 4;
constexpr size_t block_size = THREADS_PER_WARP * warps_per_tile;

/* Performance heuristics for optimized kernel parameters
 *
 * Estimated cost of a config is waves * work_per_wave:
 *  - waves = ceil(num_blocks / concurrent_blocks), where concurrent_blocks
 *    is an occupancy estimate limited by registers, shared memory, and
 *    resident threads per SM. This captures wave quantization, which
 *    dominates on devices with few SMs.
 *  - work_per_wave is the per-thread instruction stream: global memory
 *    instructions (loads, transposed stores), shared memory staging
 *    instructions (half weight: cheaper than global), per-element copy
 *    work, and a fixed per-wave overhead (block launch, tile indexing,
 *    barriers, epilogue). The overhead term matters for this memory-bound
 *    kernel, where config-dependent work is small.
 *
 * Register usage is estimated from the transposed-output staging array
 * (nvec_in * num_iterations vectors) plus a fixed base for addressing and
 * loop state. The model constants were calibrated against measured config
 * sweeps (see docs/transpose_optimization.md). Integer arithmetic (weights
 * scaled by 2) ensures stable ordering.
 */
struct KernelConfig {
  /** Vector load size */
  size_t load_size;
  /** Vector store size */
  size_t store_size;
  /** Warps per tile (block size = warp_size * warps_per_tile) */
  size_t warps_per_tile;

  /* Whether config is valid */
  bool valid = false;
  /* Number of CUDA blocks */
  size_t num_blocks = 0;
  /* Threads per block */
  size_t block_size = 0;
  /* Estimated cost (lower is better) */
  size_t cost = 0;

  KernelConfig(size_t row_length, size_t num_rows, size_t type_size, size_t load_size_,
               size_t store_size_, size_t warps, const cuda::DeviceProps &props)
      : load_size{load_size_}, store_size{store_size_}, warps_per_tile{warps} {
    // Check that tiles are correctly aligned
    constexpr size_t cache_line_size = 128;
    if (load_size % type_size != 0 || store_size % type_size != 0 ||
        cache_line_size % type_size != 0 || warps == 0 || THREADS_PER_WARP % warps != 0) {
      return;
    }
    const size_t nvec_in = load_size / type_size;
    const size_t nvec_out = store_size / type_size;
    const size_t row_tile_elements = nvec_in * THREADS_PER_WARP;
    const size_t col_tile_elements = nvec_out * THREADS_PER_WARP;
    valid = (row_length % row_tile_elements == 0 && num_rows % col_tile_elements == 0);
    if (!valid) {
      return;
    }

    // Number of CUDA blocks
    num_blocks = (row_length / row_tile_elements) * (num_rows / col_tile_elements);
    block_size = THREADS_PER_WARP * warps;

    // Occupancy estimate
    const size_t num_iterations = THREADS_PER_WARP / warps;
    constexpr size_t base_regs = 40;  // addressing and loop state
    constexpr size_t max_blocks_per_sm = 32;  // conservative HW limit
    const size_t staging_vecs = nvec_in * num_iterations;
    const size_t staging_vec_regs = std::max<size_t>(1, nvec_out * type_size / 4);
    const size_t est_regs = staging_vecs * staging_vec_regs + base_regs;
    const size_t smem_bytes = THREADS_PER_WARP * (THREADS_PER_WARP + 1) * nvec_out * type_size;
    const size_t reg_limit = static_cast<size_t>(props.regs_per_sm) / (est_regs * block_size);
    const size_t smem_limit = props.shared_mem_per_sm / smem_bytes;
    const size_t thread_limit = static_cast<size_t>(props.max_threads_per_sm) / block_size;
    const size_t blocks_per_sm =
        std::max<size_t>(1, std::min({reg_limit, smem_limit, thread_limit, max_blocks_per_sm}));
    const size_t concurrent_blocks = blocks_per_sm * props.sm_count;
    const size_t waves = DIVUP(num_blocks, concurrent_blocks);

    // Per-thread instruction stream (weights scaled by 2 for integer math)
    const size_t n_load = num_iterations * nvec_out;
    const size_t n_store_t = nvec_in * num_iterations;
    const size_t n_smem = 2 * staging_vecs;
    const size_t elts_per_thread = nvec_in * nvec_out * num_iterations;
    constexpr size_t w_comp = 2;  // ~1 copy op per element
    constexpr size_t w_block = 120;  // fixed per-wave overhead (~60 instrs)
    const size_t work = 2 * (n_load + n_store_t) + n_smem + w_comp * elts_per_thread + w_block;
    cost = waves * work;
  }

  /* Compare by estimated cost */
  bool operator<(const KernelConfig &other) const {
    if (this->valid && other.valid) {
      return this->cost < other.cost;
    } else {
      return this->valid && !other.valid;
    }
  }
};

template <size_t load_size, size_t store_size, typename Type>
__global__ void __launch_bounds__(block_size)
    transpose_general_kernel(const Type *__restrict__ const input, const fp32 *const noop,
                             Type *__restrict__ const output, const size_t row_length,
                             const size_t num_rows) {
  if (noop != nullptr && noop[0] == 1.0f) return;

  // Vectorized load/store sizes
  constexpr size_t nvec_in = load_size / sizeof(Type);
  constexpr size_t nvec_out = store_size / sizeof(Type);
  using IVec = Vec<Type, nvec_in>;
  using OVec = Vec<Type, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr size_t bdimx = THREADS_PER_WARP;
  constexpr size_t bdimy = warps_per_tile;
  const size_t tid = threadIdx.x;
  const size_t tidx = tid % bdimx;
  const size_t tidy = tid / bdimx;
  const size_t bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Position of tile within tensor
  const size_t num_tiles_m = (num_rows + tile_dim_m - 1) / tile_dim_m;
  const size_t tile_id_m = bid % num_tiles_m;
  const size_t tile_id_n = bid / num_tiles_m;
  const size_t tile_row = tile_id_m * tile_dim_m;
  const size_t tile_col = tile_id_n * tile_dim_n;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr size_t num_iterations = THREADS_PER_WARP / warps_per_tile;

  // Load input and store to registers
  // Note: Each thread loads num_iterations subtiles and transposes in
  // registers.
  OVec local_output[nvec_in][num_iterations];
#pragma unroll
  for (size_t iter = 0; iter < num_iterations; ++iter) {
    const size_t i1 = tidy + iter * bdimy;
    const size_t j1 = tidx;
#pragma unroll
    for (size_t i2 = 0; i2 < nvec_out; ++i2) {
      const size_t row = tile_row + i1 * nvec_out + i2;
      const size_t col = tile_col + j1 * nvec_in;
      IVec local_input;
      local_input.clear();
      if (row < num_rows) {
#pragma unroll
        for (size_t j2 = 0; j2 < nvec_in; ++j2) {
          if (col + j2 < row_length) {
            local_input.data.elt[j2] = input[row * row_length + col + j2];
          }
        }
      }
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
        local_output[j2][iter].data.elt[i2] = local_input.data.elt[j2];
      }
    }
  }

  // Copy transposed output from registers to global memory
  __shared__ OVec shared_output[THREADS_PER_WARP][THREADS_PER_WARP + 1];
#pragma unroll
  for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * bdimy;
      const size_t j1 = tidx;
      shared_output[j1][i1] = local_output[j2][iter];
    }
    __syncthreads();
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidx;
      const size_t j1 = tidy + iter * bdimy;
      const size_t row = tile_row + i1 * nvec_out;
      const size_t col = tile_col + j1 * nvec_in + j2;
      if (col < row_length) {
#pragma unroll
        for (size_t i2 = 0; i2 < nvec_out; ++i2) {
          if (row + i2 < num_rows) {
            output[col * num_rows + row + i2] = shared_output[j1][i1].data.elt[i2];
          }
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

void transpose(const Tensor &input, const Tensor &noop, Tensor *output_, cudaStream_t stream) {
  Tensor &output = *output_;
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output.data.shape.size() == 2, "Output must have 2 dimensions.");
  const size_t row_length = input.data.shape[1];
  const size_t num_rows = input.data.shape[0];

  NVTE_CHECK(output.data.shape[0] == row_length, "Wrong dimension of output.");
  NVTE_CHECK(output.data.shape[1] == num_rows, "Wrong dimension of output.");

  NVTE_CHECK(input.data.dptr != nullptr, "Input is not allocated.");
  NVTE_CHECK(output.data.dptr != nullptr, "Output is not allocated.");
  NVTE_CHECK(input.data.dtype == output.data.dtype, "Input (dtype=", to_string(input.data.dtype),
             ") and output (dtype=", to_string(output.data.dtype), ") do not match.");

  if (noop.data.dptr != nullptr) {
    NVTE_CHECK(noop.numel() == 1, "Expected 1 element, ", "but found ", noop.numel(), ".");
    NVTE_CHECK(noop.data.dtype == DType::kFloat32);
    NVTE_CHECK(noop.data.dptr != nullptr);
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
      input.data.dtype, Type, constexpr const char *type_name = TypeInfo<Type>::name;
      constexpr size_t type_size = sizeof(Type);

      // Choose between runtime-compiled or statically-compiled kernel
      const bool aligned = (row_length % THREADS_PER_WARP == 0 && num_rows % THREADS_PER_WARP == 0);
      if (aligned && rtc::is_enabled()) {  // Runtime-compiled tuned kernel
        // Pick kernel config
        // Note: Configs are ordered by preference for tie-breaking
        // (larger vector accesses first, more warps per tile first).
        std::vector<KernelConfig> kernel_configs;
        kernel_configs.reserve(32);
        const auto &props = cuda::device_props();
        auto add_config = [&](size_t load_size, size_t store_size, size_t warps) {
          kernel_configs.emplace_back(row_length, num_rows, type_size, load_size, store_size, warps,
                                      props);
        };
        for (size_t warps : {size_t{8}, size_t{4}}) {
          add_config(8, 8, warps);
          add_config(4, 8, warps);
          add_config(8, 4, warps);
          add_config(4, 4, warps);
          add_config(2, 8, warps);
          add_config(8, 2, warps);
          add_config(2, 4, warps);
          add_config(4, 2, warps);
          add_config(2, 2, warps);
          add_config(1, 8, warps);
          add_config(8, 1, warps);
          add_config(1, 4, warps);
          add_config(4, 1, warps);
          add_config(1, 2, warps);
          add_config(2, 1, warps);
          add_config(1, 1, warps);
        }
        const auto &kernel_config = *std::min_element(kernel_configs.begin(), kernel_configs.end());
        NVTE_CHECK(kernel_config.valid, "invalid kernel config");
        const size_t load_size = kernel_config.load_size;
        const size_t store_size = kernel_config.store_size;
        const size_t num_blocks = kernel_config.num_blocks;
        const size_t cfg_warps_per_tile = kernel_config.warps_per_tile;
        const size_t cfg_block_size = kernel_config.block_size;

        // Compile NVRTC kernel if needed and launch
        auto &rtc_manager = rtc::KernelManager::instance();
        const std::string kernel_label = concat_strings(
            "transpose"
            ",type=",
            type_name, ",load_size=", load_size, ",store_size=", store_size,
            ",warps_per_tile=", cfg_warps_per_tile);
        if (!rtc_manager.is_compiled(kernel_label)) {
          std::string code = string_code_transpose_rtc_transpose_cu;
          code = regex_replace(code, "__TYPE__", type_name);
          code = regex_replace(code, "__LOAD_SIZE__", load_size);
          code = regex_replace(code, "__STORE_SIZE__", store_size);
          code = regex_replace(code, "__WARPS_PER_TILE__", cfg_warps_per_tile);
          code = regex_replace(code, "__BLOCK_SIZE__", cfg_block_size);
          rtc_manager.compile(kernel_label, "transpose_optimized_kernel", code,
                              "transformer_engine/common/transpose/rtc/transpose.cu");
        }
        rtc_manager.launch(kernel_label, num_blocks, cfg_block_size, 0, stream,
                           static_cast<const Type *>(input.data.dptr),
                           static_cast<const fp32 *>(noop.data.dptr),
                           static_cast<Type *>(output.data.dptr), row_length, num_rows);
      } else {  // Statically-compiled general kernel
        constexpr size_t load_size = 4;
        constexpr size_t store_size = 4;
        constexpr size_t row_tile_size = load_size / type_size * THREADS_PER_WARP;
        constexpr size_t col_tile_size = store_size / type_size * THREADS_PER_WARP;
        const int num_blocks = (DIVUP(row_length, row_tile_size) * DIVUP(num_rows, col_tile_size));
        transpose_general_kernel<load_size, store_size, Type>
            <<<num_blocks, block_size, 0, stream>>>(static_cast<const Type *>(input.data.dptr),
                                                    static_cast<const fp32 *>(noop.data.dptr),
                                                    static_cast<Type *>(output.data.dptr),
                                                    row_length, num_rows);
        NVTE_CHECK_CUDA(cudaGetLastError());
      });  // NOLINT(*)
}

}  // namespace detail
}  // namespace transformer_engine

void nvte_transpose(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_transpose);
  using namespace transformer_engine;
  auto noop = Tensor();
  detail::transpose(*convertNVTETensorCheck(input), noop, convertNVTETensor(output), stream);
}

void nvte_transpose_with_noop(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                              cudaStream_t stream) {
  NVTE_API_CALL(nvte_transpose_with_noop);
  using namespace transformer_engine;
  detail::transpose(*convertNVTETensorCheck(input), *convertNVTETensorCheck(noop),
                    convertNVTETensor(output), stream);
}
