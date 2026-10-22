/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/padding.h>

#include <cfloat>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../common.h"
#include "../utils.cuh"

namespace transformer_engine {

namespace {

// Fixed launch configuration. Padding performs no arithmetic, so the only thing
// that matters is how many bytes each thread moves per memory instruction.
// kVecBytes is the widest single global access the architecture issues; a
// 567-configuration sweep over {4,8,16,32} bytes x {1,2,4,8,16} vectors per
// thread put 16 bytes at the optimum for every dtype and workload measured.
// kVecsPerThread gives each block enough work to amortize the per-block linear
// scan of block_range, which otherwise dominates for inputs with many tensors.
constexpr int kVecBytes = 16;
constexpr int kVecsPerThread = 8;
constexpr int n_warps_per_tile = 4;
constexpr int threads_per_block = THREADS_PER_WARP * n_warps_per_tile;
constexpr int kVecsPerBlock = threads_per_block * kVecsPerThread;
constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB

struct MultiPaddingArgs {
  // (input) Data buffers for input tensors
  void* input_list[kMaxTensorsPerKernel];
  // (output) Data buffers for cast output tensors
  void* output_list[kMaxTensorsPerKernel];
  // Number of rows to copy from each input matrix
  int num_rows_list[kMaxTensorsPerKernel];
  // Number of rows to write into each output matrix
  int total_rows_list[kMaxTensorsPerKernel];
  // Input matrix widths
  int row_length_list[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each
  // tensor
  int block_range[kMaxTensorsPerKernel + 1];
  // Number of tensors being processed by kernel
  int num_tensors;
};

// Row r of the input and row r of the output start at the same flat offset
// r * row_length, so the whole operation is a copy of the first
// num_rows * row_length elements followed by a zero-fill of the rest. That
// lets the block walk a single linear index space instead of a 2D tile, which
// in turn lets every in-bounds access be issued at the full kVecBytes width --
// the per-element bounds checks in a 2D walk force the compiler to emit scalar
// accesses no matter how the vector type is declared.
//
// The op performs no arithmetic at all: a bitwise element copy plus a
// static_cast<Type>(0.f) zero-fill. Access width, kVecsPerThread and the index
// decomposition therefore cannot change a single output byte, for any dtype and
// including NaN/Inf payloads. kPad == false is unpadding, where nothing is
// zero-filled and the copy ends at the output size.
template <int nvec, bool kPad, typename Type>
__global__ void __launch_bounds__(threads_per_block) multi_padding_kernel(MultiPaddingArgs args) {
  using Vec = Vec<Type, nvec>;

  // Find tensor corresponding to block
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const Type* input = reinterpret_cast<const Type*>(args.input_list[tensor_id]);
  Type* output = reinterpret_cast<Type*>(args.output_list[tensor_id]);
  const int row_length = args.row_length_list[tensor_id];
  const int64_t copy_end = static_cast<int64_t>(args.num_rows_list[tensor_id]) * row_length;
  const int64_t total_rows = args.total_rows_list[tensor_id];
  const int64_t total = kPad ? total_rows * row_length : copy_end;

  const Type zero = static_cast<Type>(0.f);

  // Wide accesses require both buffers to be kVecBytes aligned. Callers that
  // split one allocation into per-tensor views -- e.g. the PyTorch multi-split
  // path -- advance the base pointer by rows * row_length * element_size, which
  // need not be a multiple of kVecBytes.
  const uintptr_t misalign =
      (reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) & (kVecBytes - 1);
  const bool aligned = (misalign == 0);

  const int64_t vec0 =
      static_cast<int64_t>(bid - args.block_range[tensor_id]) * kVecsPerBlock + threadIdx.x;

#pragma unroll
  for (int v = 0; v < kVecsPerThread; ++v) {
    const int64_t elt = (vec0 + static_cast<int64_t>(v) * threads_per_block) * nvec;
    if (elt >= total) break;  // elt is monotone in v
    const int64_t elt_end = elt + nvec;
    if (aligned && elt_end <= total) {
      const int64_t vec_id = elt / nvec;
      Vec local;
      if (elt_end <= copy_end) {
        local.load_from(input, vec_id);
      } else if (elt >= copy_end) {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          local.data.elt[j] = zero;
        }
      } else {
        // Vector straddles the copy/zero-fill boundary
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          local.data.elt[j] = (elt + j < copy_end) ? input[elt + j] : zero;
        }
      }
      local.store_to(output, vec_id);
    } else {
      const int64_t hi = (elt_end < total) ? elt_end : total;
      for (int64_t e = elt; e < hi; ++e) {
        output[e] = (e < copy_end) ? input[e] : zero;
      }
    }
  }
}

template <bool kPad>
void launch_multi_padding(const std::vector<Tensor*>& input_list,
                          const std::vector<Tensor*>& output_list, const std::vector<int>& rows_list,
                          DType type, cudaStream_t stream) {
  const size_t elt_size = typeToSize(type);
  NVTE_CHECK(kVecBytes % elt_size == 0, "Unsupported element size in multi_padding.");
  const size_t nvec = kVecBytes / elt_size;

  // Add tensors to kernel argument struct
  MultiPaddingArgs kernel_args;
  kernel_args.num_tensors = 0;
  kernel_args.block_range[0] = 0;

  const auto launch = [&]() {
    const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
        type, Type, multi_padding_kernel<kVecBytes / sizeof(Type), kPad, Type>
        <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args););  // NOLINT(*)
    NVTE_CHECK_CUDA(cudaGetLastError());
    kernel_args.num_tensors = 0;
  };

  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
      launch();
    }

    // Number of rows copied from the input and written to the output
    const int num_rows = kPad ? input_list[tensor_id]->data.shape[0] : rows_list[tensor_id];
    const int total_rows = kPad ? rows_list[tensor_id] : num_rows;
    const int row_length = input_list[tensor_id]->data.shape[1];

    // Calculate number of thread blocks needed for tensor
    const size_t total_elt = static_cast<size_t>(total_rows) * row_length;
    const size_t num_vecs = (total_elt + nvec - 1) / nvec;
    const size_t num_blocks = (num_vecs + kVecsPerBlock - 1) / kVecsPerBlock;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = const_cast<void*>(input_list[tensor_id]->data.dptr);
    kernel_args.output_list[pos] = output_list[tensor_id]->data.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.total_rows_list[pos] = total_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + static_cast<int>(num_blocks);
    kernel_args.num_tensors++;
  }

  // Launch kernel
  if (kernel_args.num_tensors > 0) {
    launch();
  }
}

}  // namespace

void multi_padding(const std::vector<Tensor*> input_list, std::vector<Tensor*> output_list,
                   const std::vector<int> padded_num_rows_list, cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_list.size() == input_list.size(),
             "Number of input and output tensors must match");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType type = input_list[0]->data.dtype;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& output = *output_list[tensor_id];
    CheckInputTensor(input, "multi_padding_input_" + std::to_string(tensor_id));
    CheckInputTensor(output, "multi_padding_output_" + std::to_string(tensor_id));

    NVTE_CHECK(input.data.dtype == type, "Input tensor types do not match.");
    NVTE_CHECK(output.data.dtype == type, "Output tensor types do not match.");

    NVTE_CHECK(input.data.shape.size() == 2, "Input tensor must have 2 dimensions.");
    NVTE_CHECK(output.data.shape[0] == padded_num_rows_list[tensor_id],
               "output tensor shape does not match padded input shape.");
  }

  launch_multi_padding<true>(input_list, output_list, padded_num_rows_list, type, stream);
}

void multi_unpadding(const std::vector<Tensor*> input_list, std::vector<Tensor*> output_list,
                     const std::vector<int> unpadded_num_rows_list, cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_list.size() == input_list.size(),
             "Number of input and output tensors must match");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType type = input_list[0]->data.dtype;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& output = *output_list[tensor_id];
    CheckInputTensor(input, "multi_unpadding_input_" + std::to_string(tensor_id));
    CheckInputTensor(output, "multi_unpadding_output_" + std::to_string(tensor_id));

    NVTE_CHECK(input.data.dtype == type, "Input tensor types do not match.");
    NVTE_CHECK(output.data.dtype == type, "Output tensor types do not match.");

    NVTE_CHECK(input.data.shape.size() == 2, "Input tensor must have 2 dimensions.");
    NVTE_CHECK(output.data.shape[0] == unpadded_num_rows_list[tensor_id],
               "output tensor shape does not match padded input shape.");
  }

  launch_multi_padding<false>(input_list, output_list, unpadded_num_rows_list, type, stream);
}

}  // namespace transformer_engine

void nvte_multi_padding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                        const int* padded_num_rows_list, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_padding);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_, output_list_;
  std::vector<int> padded_num_rows_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(convertNVTETensorCheck(input_list[i]));
    output_list_.push_back(convertNVTETensorCheck(output_list[i]));
    padded_num_rows_list_.push_back(padded_num_rows_list[i]);
  }
  multi_padding(input_list_, output_list_, padded_num_rows_list_, stream);
}

void nvte_multi_unpadding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                          const int* unpadded_num_rows_list, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_unpadding);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_, output_list_;
  std::vector<int> unpadded_num_rows_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(convertNVTETensorCheck(input_list[i]));
    output_list_.push_back(convertNVTETensorCheck(output_list[i]));
    unpadded_num_rows_list_.push_back(unpadded_num_rows_list[i]);
  }
  multi_unpadding(input_list_, output_list_, unpadded_num_rows_list_, stream);
}
