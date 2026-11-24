#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include <climits>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "async_loader.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

// Backward: grad_probs + intermediate_output + routing_map → grad_logits.
//
// Double-buffered cp.async loads all 3 inputs in original types.  Two-pass
// fused approach (eliminates the comp_buf shmem buffer):
//   Pass 1 (reduction): accumulate warp-level sums needed by normalization/softmax bwd.
//   Pass 2 (element-wise): compute per-element gradient and write to global memory.
//
// Shmem layout (B = num_buffers, W = warps/block):
//   grad_raw:  B × E × W × sizeof(DataType) — async-loaded grad
//   act_buf:   B × E × W × sizeof(CompType) — async-loaded activations
//   mask_buf:  B × E × W × sizeof(bool)     — async-loaded routing mask

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat, int ScoreFunc>
__global__ void fused_topk_with_score_function_backward_kernel(
    const uint8_t *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    DataType *grad_logits, int num_buffers) {
  constexpr bool kIsBitmap = (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8);
  /***
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  extern __shared__ char shmem_bwd[];
  char *shmem_ptr = shmem_bwd;

  DataType *grad_shmem_base = reinterpret_cast<DataType *>(shmem_ptr);
  RawAsyncLoader<DataType> grad_loader(grad_shmem_base, warp_id, num_experts, num_token_per_block,
                                       num_buffers);
  shmem_ptr += RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  CompType *act_shmem_base = reinterpret_cast<CompType *>(shmem_ptr);
  RawAsyncLoader<CompType> act_loader(act_shmem_base, warp_id, num_experts, num_token_per_block,
                                      num_buffers);
  shmem_ptr += RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  uint8_t *mask_shmem_base = reinterpret_cast<uint8_t *>(shmem_ptr);
  RawAsyncLoader<uint8_t> mask_loader(mask_shmem_base, warp_id, num_experts, num_token_per_block,
                                      num_buffers);

  /***
     * Section: Main Loop — persistent grid with double-buffered async load
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  int first_round = blockIdx.x;
  if (first_round >= total_round) return;

  // Kick off first async load
  {
    int first_token = first_round * num_token_per_block + warp_id;
    if (first_token < num_tokens) {
      int pos = first_token * num_experts;
      grad_loader.load_current(grad_probs + pos, num_experts, lane_id);
      act_loader.load_current(intermediate_output + pos, num_experts, lane_id);
      if constexpr (!kIsBitmap) {
        mask_loader.load_current(routing_map + pos, num_experts, lane_id);
      }
    }
  }

  for (int round = first_round; round < total_round; round += gridDim.x) {
    int token_idx = round * num_token_per_block + warp_id;
    if (token_idx >= num_tokens) break;
    int pos = token_idx * num_experts;

    if (num_buffers == 1 && round != first_round) {
      grad_loader.load_current(grad_probs + pos, num_experts, lane_id);
      act_loader.load_current(intermediate_output + pos, num_experts, lane_id);
      if constexpr (!kIsBitmap) {
        mask_loader.load_current(routing_map + pos, num_experts, lane_id);
      }
    }

    /***
         * Section: Wait for async load + prefetch next round
         */
    grad_loader.wait();
    act_loader.wait();
    if constexpr (!kIsBitmap) {
      mask_loader.wait();
    }

    DataType *raw_grad = grad_loader.current_buf();
    CompType *local_act = act_loader.current_buf();
    uint8_t *local_mask = mask_loader.current_buf();
    if constexpr (kIsBitmap) {
      const int bitmap_row_bytes = (num_experts + 7) / 8;
      const uint8_t *bitmap_row = routing_map + static_cast<size_t>(token_idx) * bitmap_row_bytes;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_mask[i] = (bitmap_row[i / 8] >> (i % 8)) & 1u;
      }
      __syncwarp();
    }

    // Prefetch next round only when double-buffered; single-buffer loads above.
    if (num_buffers > 1) {
      int next_round = round + gridDim.x;
      if (next_round < total_round) {
        int next_token = next_round * num_token_per_block + warp_id;
        if (next_token < num_tokens) {
          int next_pos = next_token * num_experts;
          grad_loader.start_load(grad_probs + next_pos, num_experts, lane_id);
          act_loader.start_load(intermediate_output + next_pos, num_experts, lane_id);
          if constexpr (!kIsBitmap) {
            mask_loader.start_load(routing_map + next_pos, num_experts, lane_id);
          }
        }
      }
    }

    /***
         * Section: Pass 1 — Reduction
         * Accumulate warp-level sums needed by the backward passes:
         *   sigmoid/sqrtsoftplus (topk>1): sum_act, sum_grad_act for normalization bwd
         *   softmax:                       sum_output_x_grad = Σ(grad * softmax_output)
         *
         * For sqrtsoftplus, intermediate_output stores original logits, so we
         * recompute sqrtsoftplus(x) on the fly to get the activation value.
         */
    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
    CompType sum_output_x_grad = 0.0f;

    bool need_reduce = ((ScoreFunc == 0 || ScoreFunc == 2) && topk > 1) || (ScoreFunc == 1);
    if (need_reduce) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        CompType g = static_cast<CompType>(raw_grad[i]) * scaling_factor;
        CompType act = local_act[i];
        bool routed = local_mask[i];

        if constexpr (ScoreFunc == 0) {  // Sigmoid
          // act = sigmoid output; accumulate over routed experts only
          if (routed) {
            sum_act += act;
            sum_grad_act += g * act;
          }
        } else if constexpr (ScoreFunc == 2) {  // Sqrtsoftplus
          // act = original logit; recompute sqrtsoftplus to get activation
          if (routed) {
            CompType v = sqrtsoftplus_scalar(act);
            sum_act += v;
            sum_grad_act += g * v;
          }
        } else if constexpr (ScoreFunc == 1) {  // Softmax
          if (!use_pre_softmax) {
            // Post-softmax: act = softmax output (routed positions only)
            if (routed) sum_output_x_grad += g * act;
          } else {
            // Pre-softmax: act = softmax output (all experts)
            sum_output_x_grad += (routed ? g : 0.0f) * act;
          }
        }
      }
      if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
        sum_act = warp_allreduce_sum(sum_act);
        sum_grad_act = warp_allreduce_sum(sum_grad_act);
      }
      if constexpr (ScoreFunc == 1) {
        sum_output_x_grad = warp_allreduce_sum(sum_output_x_grad);
      }
    }

    /***
         * Section: Pass 2 — Element-wise gradient
         * Compute per-element gradient using the warp-level sums from Pass 1.
         * Applies backward ops in reverse of forward order:
         *   1. Backward of scaling_factor (multiply grad by scaling_factor)
         *   2. Backward of normalization (sigmoid/sqrtsoftplus with topk > 1)
         *   3. Backward of post-softmax / topk mask
         *   4. Backward of pre-softmax
         *   5. Backward of activation (sigmoid / sqrtsoftplus)
         * Write the grad_logits to the global mem
         */
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      CompType g = static_cast<CompType>(raw_grad[i]) * scaling_factor;
      CompType act = local_act[i];
      bool routed = local_mask[i];

      // Sigmoid/Sqrtsoftplus Post-processing bwd when topk > 1 (normalization backward)
      if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
        if (topk > 1) {
          g = normalize_bwd_scalar(g, routed, sum_act, sum_grad_act);
        }
      }

      // Softmax bwd if use_pre_softmax is false (routed subset only)
      if constexpr (ScoreFunc == 1) {
        if (!use_pre_softmax) {
          g = routed ? softmax_bwd_scalar(g, act, sum_output_x_grad) : 0.0f;
        }
      }

      // Backward of topk: mask the unselected position in the grad
      if (!routed) g = 0.0f;

      // Pre-softmax bwd (all experts participate)
      if constexpr (ScoreFunc == 1) {
        if (use_pre_softmax) {
          g = softmax_bwd_scalar(g, act, sum_output_x_grad);
        }
      }

      // Sigmoid bwd: dy/dx = y * (1 - y), where y = sigmoid output
      if constexpr (ScoreFunc == 0) {
        g = sigmoid_bwd_scalar(g, act);
        // Sqrtsoftplus bwd: dy/dx = sigmoid(x) / (2 * y), where x = original logit
      } else if constexpr (ScoreFunc == 2) {
        g = sqrtsoftplus_bwd_scalar(g, act, sqrtsoftplus_scalar(act));
      }

      grad_logits[pos + i] = static_cast<DataType>(g);
    }

    grad_loader.flip();
    act_loader.flip();
    mask_loader.flip();
  }
}

template <typename DataType, int Tiles, bool Full>
__global__ void fused_topk_backward_postsoftmax_register_kernel(
    const uint8_t *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, CompType scaling_factor, DataType *grad_logits) {
  int token = blockIdx.x * (blockDim.x / kThreadsPerWarp) + threadIdx.x / kThreadsPerWarp;
  if (token >= num_tokens) return;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  int base = token * (Full ? Tiles * kThreadsPerWarp : num_experts) + lane_id;
  CompType grad[Tiles], act[Tiles];
  bool routed[Tiles];
#pragma unroll
  for (int tile = 0; tile < Tiles; ++tile) {
    int offset = base + tile * kThreadsPerWarp;
    bool valid = Full || lane_id + tile * kThreadsPerWarp < num_experts;
    grad[tile] = valid ? static_cast<CompType>(grad_probs[offset]) : 0.0f;
    act[tile] = valid ? intermediate_output[offset] : 0.0f;
    routed[tile] = valid && routing_map[offset] != 0;
  }
  CompType dot = 0.0f;
#pragma unroll
  for (int tile = 0; tile < Tiles; ++tile) {
    CompType scaled = grad[tile] * scaling_factor;
    if (routed[tile]) dot += scaled * act[tile];
  }
  dot = warp_allreduce_sum(dot);
#pragma unroll
  for (int tile = 0; tile < Tiles; ++tile) {
    if (Full || lane_id + tile * kThreadsPerWarp < num_experts) {
      grad_logits[base + tile * kThreadsPerWarp] =
          static_cast<DataType>(routed[tile] ? act[tile] * fmaf(scaling_factor, grad[tile], -dot) : 0.0f);
    }
  }
}

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat>
void fused_topk_with_score_function_backward_kernel_launcher(
    const uint8_t *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, cudaStream_t stream) {
  NVTE_CHECK(num_experts > 0, "num_experts must be positive, got ", num_experts);
  NVTE_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts], got topk=", topk,
             " num_experts=", num_experts);
  NVTE_CHECK(static_cast<int64_t>(num_tokens) * num_experts <= INT_MAX,
             "num_tokens * num_experts exceeds INT_MAX (kernel uses int offsets), got ",
             static_cast<int64_t>(num_tokens) * num_experts);
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;

#if defined(__HGGC__)
  if constexpr (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BYTEMAP) {
    if (num_experts <= 8 * kThreadsPerWarp && score_function == 1 && !use_pre_softmax) {
      dispatch_router_tiles(num_experts, [&](auto tiles, auto full) {
        fused_topk_backward_postsoftmax_register_kernel<DataType, decltype(tiles)::value, decltype(full)::value>
            <<<total_blocks, kThreadsPerBlock, 0, stream>>>(routing_map, intermediate_output,
                grad_probs, num_tokens, num_experts, scaling_factor, grad_logits);
      });
      NVTE_CHECK_CUDA(cudaGetLastError());
      return;
    }
  }
#endif
  size_t single_buf_shmem =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, 1) +
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, 1) +
      RawAsyncLoader<uint8_t>::shmem_bytes(num_experts, num_token_per_block, 1);
  int num_buffers = choose_num_buffers(single_buf_shmem, 0);
  size_t shmem_bytes =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers) +
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, num_buffers) +
      RawAsyncLoader<uint8_t>::shmem_bytes(num_experts, num_token_per_block, num_buffers);
  check_shared_memory_capacity_num_experts(shmem_bytes, num_experts);

  auto launch = [&](auto kernel) {
    NVTE_CHECK_CUDA(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    size_t grid_size = compute_persistent_grid(kernel, kThreadsPerBlock, shmem_bytes, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shmem_bytes, stream>>>(
        routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
        use_pre_softmax, scaling_factor, grad_logits, num_buffers);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  switch (score_function) {
    case 0:
      launch(fused_topk_with_score_function_backward_kernel<DataType, RoutingMapFormat, 0>);
      break;
    case 1:
      launch(fused_topk_with_score_function_backward_kernel<DataType, RoutingMapFormat, 1>);
      break;
    case 2:
      launch(fused_topk_with_score_function_backward_kernel<DataType, RoutingMapFormat, 2>);
      break;
    default:
      NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
  }
}

void fused_topk_with_score_function_backward(const Tensor &routing_map,
                                             NVTERoutingMapFormat routing_map_format,
                                             const Tensor &intermediate_output,
                                             const Tensor &grad_probs, int num_tokens,
                                             int num_experts, int topk, bool use_pre_softmax,
                                             float scaling_factor, int score_function,
                                             Tensor &grad_logits, cudaStream_t stream) {
  check_routing_map_format(routing_map_format);
  NVTE_CHECK(num_tokens > 0 && num_experts > 0,
             "num_tokens and num_experts must be positive; got num_tokens=", num_tokens,
             ", num_experts=", num_experts);
  const std::vector<size_t> dense_shape{static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  NVTE_CHECK(intermediate_output.data.shape == dense_shape,
             "intermediate_output shape must be [num_tokens, num_experts]=[", num_tokens, ", ",
             num_experts, "], got ", intermediate_output.data.shape);
  NVTE_CHECK(grad_probs.data.shape == dense_shape,
             "grad_probs shape must be [num_tokens, num_experts]=[", num_tokens, ", ", num_experts,
             "], got ", grad_probs.data.shape);
  NVTE_CHECK(grad_logits.data.shape == dense_shape,
             "grad_logits shape must be [num_tokens, num_experts]=[", num_tokens, ", ", num_experts,
             "], got ", grad_logits.data.shape);
  const auto routing_map_shape =
      expected_routing_map_shape(num_tokens, num_experts, routing_map_format);
  NVTE_CHECK(routing_map.data.shape == routing_map_shape, "routing_map shape mismatch for ",
             (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8 ? "BITMAP_U8" : "BYTEMAP"),
             "; expected ", routing_map_shape, ", got ", routing_map.data.shape);
#define ROUTER_BACKWARD_DISPATCH(RoutingMapFormatVal)                                         \
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(                                                            \
      grad_logits.data.dtype, DataType,                                                       \
      fused_topk_with_score_function_backward_kernel_launcher<DataType, RoutingMapFormatVal>( \
          reinterpret_cast<uint8_t *>(routing_map.data.dptr),                                 \
          reinterpret_cast<CompType *>(intermediate_output.data.dptr),                        \
          reinterpret_cast<DataType *>(grad_probs.data.dptr), num_tokens, num_experts, topk,  \
          use_pre_softmax, scaling_factor, score_function,                                    \
          reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
  if (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    ROUTER_BACKWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BITMAP_U8)
  } else {
    ROUTER_BACKWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BYTEMAP)
  }
#undef ROUTER_BACKWARD_DISPATCH
}

template <typename DataType, typename IndexType, int ScoreFunc>
__global__ void fused_topk_backward_selected_indices_kernel(
    const IndexType *topk_indices, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    DataType *grad_logits) {
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;

  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_idx = round * num_token_per_block + warp_id;
    if (token_idx >= num_tokens) break;

    int pos = token_idx * num_experts;
    const IndexType *token_topk_indices = topk_indices + token_idx * topk;

    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
    CompType sum_output_x_grad = 0.0f;

    if (topk > 1 || ScoreFunc == 1) {
      for (int k = lane_id; k < topk; k += kThreadsPerWarp) {
        int expert_idx = static_cast<int>(token_topk_indices[k]);
        CompType g = static_cast<CompType>(grad_probs[pos + expert_idx]) * scaling_factor;
        CompType act = intermediate_output[pos + expert_idx];

        if constexpr (ScoreFunc == 0) {
          sum_act += act;
          sum_grad_act += g * act;
        } else if constexpr (ScoreFunc == 2) {
          CompType v = sqrtsoftplus_scalar(act);
          sum_act += v;
          sum_grad_act += g * v;
        } else if constexpr (ScoreFunc == 1) {
          sum_output_x_grad += g * act;
        }
      }
      if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
        sum_act = warp_allreduce_sum(sum_act);
        sum_grad_act = warp_allreduce_sum(sum_grad_act);
      } else if constexpr (ScoreFunc == 1) {
        sum_output_x_grad = warp_allreduce_sum(sum_output_x_grad);
      }
    }

    if constexpr (ScoreFunc == 1) {
      if (use_pre_softmax) {
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          CompType act = intermediate_output[pos + i];
          grad_logits[pos + i] =
              static_cast<DataType>(softmax_bwd_scalar(0.0f, act, sum_output_x_grad));
        }
      } else {
        vec_fill_global(grad_logits + pos, static_cast<DataType>(0.0f), num_experts, lane_id);
      }
    } else {
      vec_fill_global(grad_logits + pos, static_cast<DataType>(0.0f), num_experts, lane_id);
    }
    __syncwarp();

    for (int k = lane_id; k < topk; k += kThreadsPerWarp) {
      int expert_idx = static_cast<int>(token_topk_indices[k]);
      CompType g = static_cast<CompType>(grad_probs[pos + expert_idx]) * scaling_factor;
      CompType act = intermediate_output[pos + expert_idx];

      if constexpr (ScoreFunc == 0) {
        if (topk > 1) {
          g = normalize_bwd_scalar(g, true, sum_act, sum_grad_act);
        }
        g = sigmoid_bwd_scalar(g, act);
      } else if constexpr (ScoreFunc == 2) {
        CompType v = sqrtsoftplus_scalar(act);
        if (topk > 1) {
          g = normalize_bwd_scalar(g, true, sum_act, sum_grad_act);
        }
        g = sqrtsoftplus_bwd_scalar(g, act, v);
      } else if constexpr (ScoreFunc == 1) {
        g = softmax_bwd_scalar(g, act, sum_output_x_grad);
      }

      grad_logits[pos + expert_idx] = static_cast<DataType>(g);
    }
  }
}

template <typename DataType, typename IndexType>
void fused_topk_with_score_function_backward_with_indices_kernel_launcher(
    const IndexType *topk_indices, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, cudaStream_t stream) {
  NVTE_CHECK(num_experts > 0, "num_experts must be positive, got ", num_experts);
  NVTE_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts], got topk=", topk,
             " num_experts=", num_experts);
  NVTE_CHECK(static_cast<int64_t>(num_tokens) * num_experts <= INT_MAX,
             "num_tokens * num_experts exceeds INT_MAX (kernel uses int offsets), got ",
             static_cast<int64_t>(num_tokens) * num_experts);
  if constexpr (std::is_same_v<IndexType, int16_t>) {
    NVTE_CHECK(num_experts <= INT16_MAX, "int16 topk indices require num_experts <= ", INT16_MAX,
               ", got ", num_experts);
  }
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;

  auto launch_selected_indices = [&](auto kernel) {
    size_t grid_size = compute_persistent_grid(kernel, kThreadsPerBlock, 0, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, 0, stream>>>(
        topk_indices, intermediate_output, grad_probs, num_tokens, num_experts, topk,
        use_pre_softmax, scaling_factor, grad_logits);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  if (score_function == 0) {
    launch_selected_indices(fused_topk_backward_selected_indices_kernel<DataType, IndexType, 0>);
    return;
  }
  if (score_function == 2) {
    launch_selected_indices(fused_topk_backward_selected_indices_kernel<DataType, IndexType, 2>);
    return;
  }
  if (score_function == 1) {
    launch_selected_indices(fused_topk_backward_selected_indices_kernel<DataType, IndexType, 1>);
    return;
  }

  NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
}

void fused_topk_with_score_function_backward_with_indices(
    const Tensor &topk_indices, const Tensor &intermediate_output, const Tensor &grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, Tensor &grad_logits, cudaStream_t stream) {
  NVTE_CHECK(num_tokens > 0 && num_experts > 0,
             "num_tokens and num_experts must be positive; got num_tokens=", num_tokens,
             ", num_experts=", num_experts);
  NVTE_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts], got topk=", topk,
             " num_experts=", num_experts);
  const std::vector<size_t> dense_shape{static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> indices_shape{static_cast<size_t>(num_tokens),
                                          static_cast<size_t>(topk)};
  NVTE_CHECK(topk_indices.data.shape == indices_shape,
             "topk_indices shape must be [num_tokens, "
             "topk]=[",
             num_tokens, ", ", topk, "], got ", topk_indices.data.shape);
  NVTE_CHECK(intermediate_output.data.shape == dense_shape,
             "intermediate_output shape must be [num_tokens, num_experts]=[", num_tokens, ", ",
             num_experts, "], got ", intermediate_output.data.shape);
  NVTE_CHECK(grad_probs.data.shape == dense_shape,
             "grad_probs shape must be [num_tokens, num_experts]=[", num_tokens, ", ", num_experts,
             "], got ", grad_probs.data.shape);
  NVTE_CHECK(grad_logits.data.shape == dense_shape,
             "grad_logits shape must be [num_tokens, num_experts]=[", num_tokens, ", ", num_experts,
             "], got ", grad_logits.data.shape);
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_logits.data.dtype, DataType,
      TE_ROUTER_DENSE_INDEX_TYPE_SWITCH_ALL(
          topk_indices.data.dtype, IndexType,
          fused_topk_with_score_function_backward_with_indices_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<IndexType *>(topk_indices.data.dptr),
              reinterpret_cast<CompType *>(intermediate_output.data.dptr),
              reinterpret_cast<DataType *>(grad_probs.data.dptr), num_tokens, num_experts, topk,
              use_pre_softmax, scaling_factor, score_function,
              reinterpret_cast<DataType *>(grad_logits.data.dptr), stream);););
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_topk_with_score_function_backward_v2(const NVTETensor routing_map,
                                                     NVTERoutingMapFormat routing_map_format,
                                                     const NVTETensor intermediate_output,
                                                     const NVTETensor grad_probs, int num_tokens,
                                                     int num_experts, int topk, int use_pre_softmax,
                                                     float scaling_factor, int score_function,
                                                     NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward_v2);
  using namespace transformer_engine;
  fused_router::fused_topk_with_score_function_backward(
      *convertNVTETensorCheck(routing_map), routing_map_format,
      *convertNVTETensorCheck(intermediate_output), *convertNVTETensorCheck(grad_probs), num_tokens,
      num_experts, topk, static_cast<bool>(use_pre_softmax), scaling_factor, score_function,
      *convertNVTETensorCheck(grad_logits), stream);
}

void nvte_fused_topk_with_score_function_backward(const NVTETensor routing_map,
                                                  const NVTETensor intermediate_output,
                                                  const NVTETensor grad_probs, int num_tokens,
                                                  int num_experts, int topk, int use_pre_softmax,
                                                  float scaling_factor, int score_function,
                                                  NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward);
  nvte_fused_topk_with_score_function_backward_v2(
      routing_map, NVTE_ROUTING_MAP_FORMAT_BYTEMAP, intermediate_output, grad_probs, num_tokens,
      num_experts, topk, use_pre_softmax, scaling_factor, score_function, grad_logits, stream);
}

void nvte_fused_topk_with_score_function_backward_with_indices(
    const NVTETensor topk_indices, const NVTETensor intermediate_output,
    const NVTETensor grad_probs, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    float scaling_factor, int score_function, NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward_with_indices);
  using namespace transformer_engine;
  fused_router::fused_topk_with_score_function_backward_with_indices(
      *convertNVTETensorCheck(topk_indices), *convertNVTETensorCheck(intermediate_output),
      *convertNVTETensorCheck(grad_probs), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), scaling_factor, score_function,
      *convertNVTETensorCheck(grad_logits), stream);
}
