/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/transpose.h>

#include <cfloat>
#include <iostream>
#include <limits>
#include <vector>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../utils.cuh"

namespace transformer_engine {

namespace {

// Fallback launch configuration, i.e. the parameters this kernel has always
// used. Reached only when no candidate configuration tiles every input tensor
// exactly, in which case behavior is unchanged.
constexpr int kFallbackWarpsPerTile = 4;
constexpr int kFallbackLoadSize = 8;
constexpr int kFallbackStoreSize = 8;
constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB

// Candidates use 8 warps per tile: a measured sweep of 40 configurations over
// 9 workloads found 8 warps optimal for every workload, and halving it doubles
// the per-thread register staging, which is what limits the fallback
// configuration to 4 resident blocks per SM. The candidate list itself lives in
// NVTE_MCT_CANDIDATE_LIST below, where the launch dispatch can share it.
constexpr int kCandidateWarpsPerTile = 8;

struct MultiCastTransposeArgs {
  // (input) Data buffers for input tensors
  void* input_list[kMaxTensorsPerKernel];
  // (output) Data buffers for cast output tensors
  void* output_c_list[kMaxTensorsPerKernel];
  // (output) Data buffers for transpose output tensors
  void* output_t_list[kMaxTensorsPerKernel];
  // (input) Scaling factor for output tensors
  void* scale_list[kMaxTensorsPerKernel];
  // (output) AMAX's of input tensors
  void* amax_list[kMaxTensorsPerKernel];
  // (output) Inverse of scaling factor for output tensors
  void* scale_inv_list[kMaxTensorsPerKernel];
  // Input matrix heights
  int num_rows_list[kMaxTensorsPerKernel];
  // Input matrix widths
  int row_length_list[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each
  // tensor
  int block_range[kMaxTensorsPerKernel + 1];
  // Number of tensors being processed by kernel
  int num_tensors;
};

template <int nvec_in, int nvec_out, int warps_per_tile, bool aligned, typename CType,
          typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_WARP * warps_per_tile)
    multi_cast_transpose_kernel(MultiCastTransposeArgs args) {
  using IVec = Vec<IType, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = warps_per_tile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr int n_iterations = THREADS_PER_WARP / warps_per_tile;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const IType* input = reinterpret_cast<const IType*>(args.input_list[tensor_id]);
  OType* output_c = reinterpret_cast<OType*>(args.output_c_list[tensor_id]);
  OType* output_t = reinterpret_cast<OType*>(args.output_t_list[tensor_id]);
  const CType* scale_ptr = reinterpret_cast<CType*>(args.scale_list[tensor_id]);
  const CType scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  CType* amax_ptr = reinterpret_cast<CType*>(args.amax_list[tensor_id]);
  CType* scale_inv_ptr = reinterpret_cast<CType*>(args.scale_inv_list[tensor_id]);
  const int num_rows = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];

  // Find position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id = bid - args.block_range[tensor_id];
  const int tile_id_m = tile_id / num_tiles_n;
  const int tile_id_n = tile_id % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Load input and store to registers
  // Note: Each thread loads n_iterations subtiles, casts to output
  // type, and transposes in registers.
  OVecT local_output_t[nvec_in][n_iterations];
  CType local_amax = 0;
#pragma unroll
  for (int iter = 0; iter < n_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
#pragma unroll
    for (int i2 = 0; i2 < nvec_out; ++i2) {
      const int row = tile_row + i1 * nvec_out + i2;
      const int col = tile_col + j1 * nvec_in;
      IVec local_input;
      OVecC local_output_c;
      if constexpr (aligned) {
        local_input.load_from(&input[row * row_length + col]);
      } else {
        local_input.clear();
        if (row < num_rows) {
#pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              local_input.data.elt[j2] = input[row * row_length + col + j2];
            }
          }
        }
      }
#pragma unroll
      for (int j2 = 0; j2 < nvec_in; ++j2) {
        const CType x = CType(local_input.data.elt[j2]);
        const OType y = OType(scale * x);
        local_output_c.data.elt[j2] = y;
        local_output_t[j2][iter].data.elt[i2] = y;
        __builtin_assume(local_amax >= 0);
        local_amax = fmaxf(fabsf(x), local_amax);
      }
      if constexpr (aligned) {
        local_output_c.store_to(&output_c[row * row_length + col]);
      } else {
        if (row < num_rows) {
#pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              output_c[row * row_length + col + j2] = local_output_c.data.elt[j2];
            }
          }
        }
      }
    }
  }

  // Copy transposed output from registers to global memory
  __shared__ OVecT shared_output_t[THREADS_PER_WARP][THREADS_PER_WARP + 1];
#pragma unroll
  for (int j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (int iter = 0; iter < n_iterations; ++iter) {
      const int i1 = tidy + iter * bdimy;
      const int j1 = tidx;
      shared_output_t[j1][i1] = local_output_t[j2][iter];
    }
    __syncthreads();
#pragma unroll
    for (int iter = 0; iter < n_iterations; ++iter) {
      const int i1 = tidx;
      const int j1 = tidy + iter * bdimy;
      const int row = tile_row + i1 * nvec_out;
      const int col = tile_col + j1 * nvec_in + j2;
      if constexpr (aligned) {
        shared_output_t[j1][i1].store_to(&output_t[col * num_rows + row]);
      } else {
        if (col < row_length) {
#pragma unroll
          for (int i2 = 0; i2 < nvec_out; ++i2) {
            if (row + i2 < num_rows) {
              output_t[col * num_rows + row + i2] = shared_output_t[j1][i1].data.elt[i2];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // Finalize fp8 factors
  local_amax = reduce_max<warps_per_tile>(local_amax, tidy);
  if (tid == 0) {
    static_assert(std::is_same<CType, float>::value);
    if (amax_ptr != nullptr) atomicMaxFloat(amax_ptr, local_amax);
  }
  if (tile_id == 0 && tid == 0 && scale_inv_ptr != nullptr) {
    reciprocal<CType>(scale_inv_ptr, scale);
  }
}

/* Launch configuration for the fused kernel */
struct KernelConfig {
  /** Elements per vectorized input load */
  int nvec_in = 0;
  /** Elements per vectorized transposed store */
  int nvec_out = 0;
  /** Warps per tile (block size = THREADS_PER_WARP * warps_per_tile) */
  int warps_per_tile = 0;
};

/* Estimated per-thread work for one tile
 *
 * Counts the instruction stream: vectorized input loads and cast stores
 * (nvec_out of each per iteration), transposed stores (nvec_in per iteration),
 * shared memory staging (write plus read of each staged vector), and
 * per-element arithmetic (cast to compute type, scale multiply, amax update,
 * cast to output type).
 *
 * This kernel is memory-bound, so global memory instructions get the dominant
 * weight: shared memory staging hits on-chip SRAM and consumes no DRAM
 * bandwidth. The weights decide correctly between configurations that move
 * the same number of bytes with different vector widths, which occupancy
 * alone cannot separate. They were validated against the measured sweep: the
 * cost argmin lands on the measured optimum for all 9 workloads, and 223 of
 * the 400 triples in a surrounding weight scan do the same, so this choice
 * sits on a broad plateau rather than a fitted spike. Integer arithmetic keeps
 * the ordering stable. See docs/common_op_optimization.md.
 */
inline size_t kernel_work(int nvec_in, int nvec_out, int warps_per_tile) {
  constexpr size_t w_mem = 4;
  constexpr size_t w_smem = 1;
  constexpr size_t w_comp = 4;
  const size_t num_iterations = THREADS_PER_WARP / warps_per_tile;
  const size_t n_load = num_iterations * nvec_out;
  const size_t n_store_t = nvec_in * num_iterations;
  const size_t n_smem = 2 * nvec_in * num_iterations;
  const size_t elts_per_thread = nvec_in * nvec_out * num_iterations;
  return w_mem * (2 * n_load + n_store_t) + w_smem * n_smem + w_comp * elts_per_thread;
}

/* Candidate launch configurations, as (load size in bytes, transposed store
 * vector width) pairs. Load sizes are in bytes so that the same targets apply
 * to every input dtype; store widths are element counts, since the transposed
 * store width is what drives register pressure. This list is the single source
 * of truth for both the cost-model search and the launch dispatch, so the two
 * cannot drift apart.
 */
#define NVTE_MCT_CANDIDATE_LIST(ACTION)   \
  ACTION(4, 1) ACTION(4, 2) ACTION(4, 4)  \
  ACTION(8, 1) ACTION(8, 2) ACTION(8, 4)  \
  ACTION(16, 1) ACTION(16, 2) ACTION(16, 4)

/* Score one candidate configuration against a batch of tensors
 *
 * Leaves best untouched if the candidate is unusable: because the load size is
 * not a whole number of input elements, because some tensor is not tiled
 * exactly (which would force the bounds-checked kernel), or because the driver
 * reports that it spills registers to local memory.
 */
template <size_t load_size, int store_vec, typename IType, typename OType>
void score_candidate(const std::vector<Tensor *> &input_list, const cuda::DeviceProps &props,
                     KernelConfig &best, size_t &best_cost) {
  if constexpr (load_size % sizeof(IType) != 0) {
    return;
  } else {
    constexpr int nvec_in = static_cast<int>(load_size / sizeof(IType));
    constexpr int nvec_out = store_vec;
    constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
    constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

    size_t num_blocks = 0;
    for (const auto *input : input_list) {
      const int num_rows = input->data.shape[0];
      const int row_length = input->data.shape[1];
      if (num_rows % tile_dim_m != 0 || row_length % tile_dim_n != 0) return;
      num_blocks += (num_rows / tile_dim_m) * (row_length / tile_dim_n);
    }
    if (num_blocks == 0) return;

    const int blocks_per_sm = cuda::kernel_blocks_per_sm(
        reinterpret_cast<const void *>(multi_cast_transpose_kernel<
                                       nvec_in, nvec_out, kCandidateWarpsPerTile, true, fp32, IType,
                                       OType>),
        THREADS_PER_WARP * kCandidateWarpsPerTile);
    if (blocks_per_sm <= 0) return;

    const size_t concurrent_blocks = static_cast<size_t>(blocks_per_sm) * props.sm_count;
    const size_t cost =
        DIVUP(num_blocks, concurrent_blocks) *
        kernel_work(nvec_in, nvec_out, kCandidateWarpsPerTile);
    if (cost < best_cost) {
      best_cost = cost;
      best.nvec_in = nvec_in;
      best.nvec_out = nvec_out;
      best.warps_per_tile = kCandidateWarpsPerTile;
    }
  }
}

/* Choose the launch configuration for a fused batch of tensors
 *
 * Estimated cost is waves * work, where waves is the number of passes the
 * device needs over the aggregate grid. Occupancy comes from the CUDA runtime
 * rather than a register estimate, because these kernels are compiled
 * statically and the driver knows their exact register and shared memory
 * footprint.
 *
 * Only configurations that tile every input tensor exactly are considered, so
 * the selected launch always uses the aligned kernel. If none qualifies the
 * fallback configuration is returned and behavior is unchanged.
 */
template <typename IType, typename OType>
KernelConfig select_kernel_config(const std::vector<Tensor *> &input_list) {
  KernelConfig config;
  config.nvec_in = kFallbackLoadSize / sizeof(IType);
  config.nvec_out = kFallbackStoreSize / sizeof(OType);
  config.warps_per_tile = kFallbackWarpsPerTile;

  const auto &props = cuda::device_props();
  size_t best_cost = std::numeric_limits<size_t>::max();
#define NVTE_MCT_SCORE(LOAD_SIZE, STORE_VEC)                                   \
  score_candidate<(LOAD_SIZE), (STORE_VEC), IType, OType>(input_list, props, config, best_cost);
  NVTE_MCT_CANDIDATE_LIST(NVTE_MCT_SCORE)
#undef NVTE_MCT_SCORE
  return config;
}

template <int nvec_in, int nvec_out, int warps_per_tile, bool aligned, typename IType,
          typename OType>
void launch_kernel(const MultiCastTransposeArgs &args, int num_blocks, cudaStream_t stream) {
  multi_cast_transpose_kernel<nvec_in, nvec_out, warps_per_tile, aligned, fp32, IType, OType>
      <<<num_blocks, THREADS_PER_WARP * warps_per_tile, 0, stream>>>(args);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

/* Turn a runtime KernelConfig into template arguments.
 *
 * Candidates are enumerated over load sizes in bytes, so the vector width that
 * needs instantiating depends on the input dtype and `if constexpr` prunes the
 * widths no candidate load size can produce. These macros expect `config` and
 * `stream` from the enclosing scope, plus the `mct_args` and `mct_num_blocks`
 * bindings that NVTE_MCT_LAUNCH_ALIGNED introduces, and must be expanded
 * inside a template parameterized on IType and OType.
 */
#define NVTE_MCT_LAUNCH_FALLBACK(ALIGNED, ARGS, NUM_BLOCKS)                            \
  launch_kernel<static_cast<int>(kFallbackLoadSize / sizeof(IType)),                   \
                static_cast<int>(kFallbackStoreSize / sizeof(OType)),                  \
                kFallbackWarpsPerTile, ALIGNED, IType, OType>((ARGS), (NUM_BLOCKS), stream);

#define NVTE_MCT_TRY_CANDIDATE(LOAD_SIZE, STORE_VEC)                                    \
  if constexpr ((LOAD_SIZE) % sizeof(IType) == 0) {                                     \
    constexpr int kNvecIn = static_cast<int>((LOAD_SIZE) / sizeof(IType));              \
    if (!launched && config.nvec_in == kNvecIn && config.nvec_out == (STORE_VEC) &&     \
        config.warps_per_tile == kCandidateWarpsPerTile) {                              \
      launch_kernel<kNvecIn, (STORE_VEC), kCandidateWarpsPerTile, true, IType, OType>(  \
          mct_args, mct_num_blocks, stream);                                            \
      launched = true;                                                                  \
    }                                                                                   \
  }

#define NVTE_MCT_LAUNCH_ALIGNED(ARGS, NUM_BLOCKS)                                       \
  {                                                                                     \
    const MultiCastTransposeArgs &mct_args = (ARGS);                                    \
    const int mct_num_blocks = (NUM_BLOCKS);                                            \
    bool launched = false;                                                              \
    NVTE_MCT_CANDIDATE_LIST(NVTE_MCT_TRY_CANDIDATE)                                     \
    if (!launched) {                                                                    \
      NVTE_MCT_LAUNCH_FALLBACK(true, mct_args, mct_num_blocks)                          \
    }                                                                                   \
  }

namespace {

template <typename IType, typename OType>
void launch_multi_cast_transpose(const std::vector<Tensor *> &input_list,
                                 std::vector<Tensor *> &output_list, cudaStream_t stream) {
  const KernelConfig config = select_kernel_config<IType, OType>(input_list);

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  const int tile_dim_m = THREADS_PER_WARP * config.nvec_out;
  const int tile_dim_n = THREADS_PER_WARP * config.nvec_in;

  // Add tensors to kernel argument struct
  MultiCastTransposeArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args_aligned.num_tensors == kMaxTensorsPerKernel) {
      const int n_blocks = kernel_args_aligned.block_range[kernel_args_aligned.num_tensors];
      NVTE_MCT_LAUNCH_ALIGNED(kernel_args_aligned, n_blocks)
      kernel_args_aligned.num_tensors = 0;
    }
    if (kernel_args_unaligned.num_tensors == kMaxTensorsPerKernel) {
      const int n_blocks = kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors];
      NVTE_MCT_LAUNCH_FALLBACK(false, kernel_args_unaligned, n_blocks)
      kernel_args_unaligned.num_tensors = 0;
    }

    // Calculate number of thread blocks needed for tensor
    const int num_rows = input_list[tensor_id]->data.shape[0];
    const int row_length = input_list[tensor_id]->data.shape[1];
    const int num_tiles_m = (num_rows + tile_dim_m - 1) / tile_dim_m;
    const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const int num_tiles = num_tiles_m * num_tiles_n;

    // Figure out whether to use aligned or unaligned kernel
    const bool aligned =
        ((num_tiles_m * tile_dim_m == num_rows) && (num_tiles_n * tile_dim_n == row_length));
    auto &kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = const_cast<void *>(input_list[tensor_id]->data.dptr);
    kernel_args.output_c_list[pos] = output_list[tensor_id]->data.dptr;
    kernel_args.output_t_list[pos] = output_list[tensor_id]->columnwise_data.dptr;
    kernel_args.scale_list[pos] = output_list[tensor_id]->scale.dptr;
    kernel_args.amax_list[pos] = output_list[tensor_id]->amax.dptr;
    kernel_args.scale_inv_list[pos] = output_list[tensor_id]->scale_inv.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }

  // A candidate configuration is only selected after every tensor was verified
  // to tile exactly with these same tile dimensions, so the unaligned bucket,
  // which only has a fallback instantiation, must be empty.
  NVTE_CHECK(config.warps_per_tile == kFallbackWarpsPerTile ||
                 kernel_args_unaligned.num_tensors == 0,
             "Internal error: candidate configuration produced unaligned tensors");

  // Launch kernel
  if (kernel_args_aligned.num_tensors > 0) {
    const int n_blocks = kernel_args_aligned.block_range[kernel_args_aligned.num_tensors];
    NVTE_MCT_LAUNCH_ALIGNED(kernel_args_aligned, n_blocks)
  }
  if (kernel_args_unaligned.num_tensors > 0) {
    const int n_blocks = kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors];
    NVTE_MCT_LAUNCH_FALLBACK(false, kernel_args_unaligned, n_blocks)
  }
}

}  // namespace

#undef NVTE_MCT_TRY_CANDIDATE
#undef NVTE_MCT_LAUNCH_ALIGNED
#undef NVTE_MCT_LAUNCH_FALLBACK

void multi_cast_transpose(const std::vector<Tensor*> input_list, std::vector<Tensor*> output_list,
                          cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_list.size() == input_list.size(),
             "Number of input and output tensors must match");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType itype = input_list[0]->data.dtype;
  DType otype = output_list[0]->dtype();
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& output = *output_list[tensor_id];
    CheckInputTensor(input, "multi_cast_transpose_input_" + std::to_string(tensor_id));
    CheckInputTensor(output, "multi_cast_transpose_output_" + std::to_string(tensor_id));
    NVTE_CHECK(output.has_data() && output.has_columnwise_data(),
               "Both rowwise and columnwise output data needs to be allocated.");

    NVTE_CHECK(input.data.dtype == itype, "Input tensor types do not match.");
    NVTE_CHECK(output.data.dtype == otype, "C output tensor types do not match.");
    NVTE_CHECK(output.data.dtype == otype, "T output tensor types do not match.");

    NVTE_CHECK(input.data.shape.size() == 2, "Input tensor must have 2 dimensions, but shape is ",
               input.data.shape);
    NVTE_CHECK(output.data.shape == input.data.shape, "C output tensor shape ", output.data.shape,
               "does not match input tensor shape ", input.data.shape);
    NVTE_CHECK(output.columnwise_data.shape.size() == 2, "T output tensor shape ",
               output.columnwise_data.shape, "does not match input tensor shape ",
               input.data.shape);
    NVTE_CHECK(output.columnwise_data.shape[0] == input.data.shape[1], "T output tensor shape ",
               output.columnwise_data.shape, "does not match input tensor shape ",
               input.data.shape);
    NVTE_CHECK(output.columnwise_data.shape[1] == input.data.shape[0], "T output tensor shape ",
               output.columnwise_data.shape, "does not match input tensor shape ",
               input.data.shape);
  }

  // Select the launch configuration and dispatch. Both the tile dimensions and
  // the aligned/unaligned split depend on it, so this happens per dtype pair.
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      itype, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          otype, OutputType,
          launch_multi_cast_transpose<InputType, OutputType>(input_list, output_list, stream);));
}

}  // namespace transformer_engine

void nvte_multi_cast_transpose(size_t num_tensors, const NVTETensor* input_list,
                               NVTETensor* output_list, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_cast_transpose);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_, output_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(convertNVTETensorCheck(input_list[i]));
    output_list_.push_back(convertNVTETensorCheck(output_list[i]));
  }
  multi_cast_transpose(input_list_, output_list_, stream);
}
