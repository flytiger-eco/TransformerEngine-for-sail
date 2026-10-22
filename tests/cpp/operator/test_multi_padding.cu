/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/padding.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename InputType, typename OutputType>
void compute_ref(const std::vector<std::vector<InputType>>& input_list,
                 std::vector<std::vector<OutputType>>& output_list,
                 const std::vector<size_t>& height_list,
                 const std::vector<size_t>& width_list,
                 const std::vector<int>& padded_height_list) {
  using compute_t = float;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = input_list[tensor_id];
    auto& output = output_list[tensor_id];
    const size_t height = height_list[tensor_id];
    const size_t width = width_list[tensor_id];
    const size_t padded_height = padded_height_list[tensor_id];

    for (size_t i = 0; i < padded_height; ++i) {
      if (i < height) {
        for (size_t j = 0; j < width; ++j) {
          const compute_t x = static_cast<compute_t>(input[i * width + j]);
          const OutputType y = static_cast<OutputType>(x);
          output[i * width + j] = y;
        }
      } else {
        for (size_t j = 0; j < width; ++j) {
          output[i * width + j] = static_cast<OutputType>(0.f);
        }
      }
    }
  }
}

// Runs one padding (or unpadding) problem where every tensor gets its own
// allocation, so all base pointers are 256-byte aligned.
template <typename InputType, typename OutputType>
void performTest(bool unpadding) {
  using namespace test;

  const DType itype = TypeInfo<InputType>::dtype;
  const DType otype = TypeInfo<OutputType>::dtype;
  const std::vector<std::pair<size_t, size_t>> tensor_dims = {{1,1},
                                                              {1,768},
                                                              {768,1},
                                                              {768,768},
                                                              {43,43},
                                                              {43,256},
                                                              {256,43},
                                                              {256,256}};
  const size_t num_tensors = tensor_dims.size();
  constexpr int align = 16;

  // Buffers for Transformer Engine implementation
  std::vector<Tensor> input_list, output_list;

  // Buffers for reference implementation
  std::vector<std::vector<InputType>> ref_input_list;
  std::vector<std::vector<OutputType>> ref_output_list;
  std::vector<size_t> ref_height_list(num_tensors), ref_width_list(num_tensors);
  std::vector<int> ref_rows_list(num_tensors);

  // Initialize buffers
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t height = tensor_dims[tensor_id].first;
    const size_t width = tensor_dims[tensor_id].second;
    const size_t padded_height = (height + align - 1) / align * align;
    // Unpadding reads a padded matrix and writes the first `height` rows of it;
    // padding reads `height` rows and writes a padded matrix.
    const size_t input_height = unpadding ? padded_height : height;
    const size_t output_height = unpadding ? height : padded_height;
    input_list.emplace_back(Tensor("input_" + std::to_string(tensor_id),
                                   std::vector<size_t>{ input_height, width }, itype));
    output_list.emplace_back(Tensor("output_" + std::to_string(tensor_id),
                                    std::vector<size_t>{ output_height, width }, otype));

    auto& input = input_list.back();
    auto& output = output_list.back();
    fillUniform(&input);
    setRandomScale(&output);

    ref_input_list.emplace_back(input_height*width);
    ref_output_list.emplace_back(output_height*width);

    std::copy(input.rowwise_cpu_dptr<InputType>(),
              input.rowwise_cpu_dptr<InputType>() + input_height * width,
              ref_input_list.back().begin());
    ref_height_list[tensor_id] = height;
    ref_width_list[tensor_id] = width;
    ref_rows_list[tensor_id] = output_height;
  }

  // Transformer Engine implementation
  auto make_nvte_vector = [](std::vector<Tensor>& tensor_list)
    -> std::vector<NVTETensor> {
    std::vector<NVTETensor> nvte_tensor_list;
    for (auto& tensor : tensor_list) {
      nvte_tensor_list.emplace_back(tensor.data());
    }
    return nvte_tensor_list;
  };
  if (unpadding) {
    nvte_multi_unpadding(num_tensors,
                         make_nvte_vector(input_list).data(),
                         make_nvte_vector(output_list).data(),
                         ref_rows_list.data(),
                         0);
  } else {
    nvte_multi_padding(num_tensors,
                       make_nvte_vector(input_list).data(),
                       make_nvte_vector(output_list).data(),
                       ref_rows_list.data(),
                       0);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Reference implementation
  compute_ref<InputType, OutputType>(ref_input_list,
                                     ref_output_list,
                                     ref_height_list,
                                     ref_width_list,
                                     ref_rows_list);

  // Check correctness
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("output",
                   output_list[tensor_id],
                   ref_output_list[tensor_id].data(),
                   true,
                   atol, rtol);
  }
}

// Runs one multi-padding problem twice and compares both runs against a host
// reference: first with every tensor in its own allocation (all base pointers
// 256-byte aligned, the wide-access path), then with all tensors packed into two
// shared allocations at byte offsets that are deliberately not multiples of 16
// (the scalar fallback path). Callers reach the fallback whenever they split one
// allocation into per-tensor views, as the PyTorch multi-split path does by
// advancing the base pointer by rows * row_length * element_size.
//
// num_tensors > 64 also exercises the chunked launch path, because the kernel
// argument struct holds at most 64 tensors per launch.
template <typename Type>
void performPackedTest(size_t num_tensors) {
  using namespace test;

  const DType dtype = TypeInfo<Type>::dtype;
  const size_t elt = sizeof(Type);
  constexpr int align = 16;

  // Odd widths make the cumulative byte offsets land on every residue class
  // modulo 16 rather than always on a multiple of it.
  std::vector<size_t> height(num_tensors), width(num_tensors), padded_height(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    height[i] = 1 + (i * 37) % 129;
    width[i] = 1 + (i * 53) % 299;
    padded_height[i] = (height[i] + align - 1) / align * align;
  }

  // --- Reference run: one allocation per tensor, all aligned ---
  std::vector<Tensor> ref_input, ref_output;
  ref_input.reserve(num_tensors);
  ref_output.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    ref_input.emplace_back(Tensor("ref_in_" + std::to_string(i),
                                  std::vector<size_t>{ height[i], width[i] }, dtype));
    ref_output.emplace_back(Tensor("ref_out_" + std::to_string(i),
                                   std::vector<size_t>{ padded_height[i], width[i] }, dtype));
    fillUniform(&ref_input.back());
    setRandomScale(&ref_output.back());
  }
  {
    std::vector<NVTETensor> in, out;
    for (size_t i = 0; i < num_tensors; ++i) {
      in.emplace_back(ref_input[i].data());
      out.emplace_back(ref_output[i].data());
    }
    std::vector<int> rows(padded_height.begin(), padded_height.end());
    nvte_multi_padding(num_tensors, in.data(), out.data(), rows.data(), 0);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  }

  // Snapshot the inputs and build the expected output on the host. Padding
  // copies the input bytes verbatim and zero-fills the extra rows, so the
  // expectation needs no dtype arithmetic beyond the representation of zero.
  // Both runs are compared against this host reference rather than against each
  // other, so a bug that affects every launch -- e.g. mis-assigning blocks to
  // tensors when the argument struct is chunked -- cannot cancel out.
  std::vector<std::vector<uint8_t>> ref_bytes(num_tensors), in_bytes(num_tensors);
  uint8_t zero_bytes[16];
  {
    const Type zero = static_cast<Type>(0.f);
    std::memcpy(zero_bytes, &zero, elt);
  }
  for (size_t i = 0; i < num_tensors; ++i) {
    const size_t ob = padded_height[i] * width[i] * elt;
    const size_t ib = height[i] * width[i] * elt;
    in_bytes[i].resize(ib);
    ASSERT_EQ(cudaMemcpy(in_bytes[i].data(), ref_input[i].rowwise_dptr(), ib,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ref_bytes[i].resize(ob);
    std::memcpy(ref_bytes[i].data(), in_bytes[i].data(), ib);
    for (size_t b = ib; b < ob; b += elt) {
      std::memcpy(ref_bytes[i].data() + b, zero_bytes, elt);
    }
  }
  for (size_t i = 0; i < num_tensors; ++i) {
    std::vector<uint8_t> got(ref_bytes[i].size());
    ASSERT_EQ(cudaMemcpy(got.data(), ref_output[i].rowwise_dptr(), got.size(),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(got, ref_bytes[i]) << "aligned run, tensor " << i << " (" << height[i] << "x"
                                 << width[i] << " -> " << padded_height[i] << "x" << width[i]
                                 << ")";
  }

  // --- Packed run: two shared allocations at misaligned offsets ---
  size_t total_in = 0, total_out = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    total_in += height[i] * width[i] * elt;
    total_out += padded_height[i] * width[i] * elt;
  }
  // Offsetting both base pointers by one element leaves tensor 0 misaligned by
  // elt bytes, which is nonzero for every dtype here (elt <= 8 < 16).
  const size_t shift = elt;
  void *in_alloc = nullptr, *out_alloc = nullptr;
  ASSERT_EQ(cudaMalloc(&in_alloc, total_in + shift), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out_alloc, total_out + shift), cudaSuccess);
  char *in_base = static_cast<char *>(in_alloc) + shift;
  char *out_base = static_cast<char *>(out_alloc) + shift;

  std::vector<size_t> in_off(num_tensors), out_off(num_tensors);
  size_t acc_in = 0, acc_out = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    in_off[i] = acc_in;
    out_off[i] = acc_out;
    acc_in += height[i] * width[i] * elt;
    acc_out += padded_height[i] * width[i] * elt;
    ASSERT_EQ(cudaMemcpy(in_base + in_off[i], in_bytes[i].data(), in_bytes[i].size(),
                         cudaMemcpyHostToDevice), cudaSuccess);
  }
  // Poison the output so unwritten or stale bytes cannot pass as correct.
  ASSERT_EQ(cudaMemset(out_alloc, 0xCD, total_out + shift), cudaSuccess);

  // FP8 tensors must carry a scale_inv; non-FP8 tensors must not.
  const bool fp8 = isFp8Type(dtype);
  float *scale_inv = nullptr;
  if (fp8) {
    ASSERT_EQ(cudaMalloc(&scale_inv, 2 * num_tensors * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemset(scale_inv, 0, 2 * num_tensors * sizeof(float)), cudaSuccess);
  }

  std::vector<TensorWrapper> wrapped_in, wrapped_out;
  wrapped_in.reserve(num_tensors);
  wrapped_out.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    const std::vector<size_t> in_shape{height[i], width[i]};
    const std::vector<size_t> out_shape{padded_height[i], width[i]};
    const std::vector<size_t> scale_inv_shape{1};
    float *in_scale_inv = fp8 ? scale_inv + 2 * i : nullptr;
    float *out_scale_inv = fp8 ? scale_inv + 2 * i + 1 : nullptr;
    wrapped_in.emplace_back(TensorWrapper(in_base + in_off[i], in_shape, dtype, nullptr, nullptr,
                                          in_scale_inv, scale_inv_shape));
    wrapped_out.emplace_back(TensorWrapper(out_base + out_off[i], out_shape, dtype, nullptr, nullptr,
                                           out_scale_inv, scale_inv_shape));
  }
  std::vector<NVTETensor> in, out;
  for (size_t i = 0; i < num_tensors; ++i) {
    in.emplace_back(wrapped_in[i].data());
    out.emplace_back(wrapped_out[i].data());
  }
  std::vector<int> rows(padded_height.begin(), padded_height.end());
  nvte_multi_padding(num_tensors, in.data(), out.data(), rows.data(), 0);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // --- Compare byte for byte ---
  std::vector<uint8_t> got;
  for (size_t i = 0; i < num_tensors; ++i) {
    got.resize(ref_bytes[i].size());
    ASSERT_EQ(cudaMemcpy(got.data(), out_base + out_off[i], got.size(),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(got, ref_bytes[i]) << "packed padding run, tensor " << i << " (" << height[i] << "x"
                                 << width[i] << " -> " << padded_height[i] << "x" << width[i]
                                 << ")";
  }

  // --- Unpad the packed result back and compare against the original inputs ---
  // This exercises the scalar fallback and the chunked launch for unpadding too,
  // which shares the launch helper and the kernel body with padding.
  void *unpad_alloc = nullptr;
  ASSERT_EQ(cudaMalloc(&unpad_alloc, total_in + shift), cudaSuccess);
  char *unpad_base = static_cast<char *>(unpad_alloc) + shift;
  ASSERT_EQ(cudaMemset(unpad_alloc, 0xCD, total_in + shift), cudaSuccess);

  std::vector<TensorWrapper> wrapped_unpad;
  wrapped_unpad.reserve(num_tensors);
  std::vector<NVTETensor> unpad_out;
  for (size_t i = 0; i < num_tensors; ++i) {
    const std::vector<size_t> unpad_shape{height[i], width[i]};
    const std::vector<size_t> scale_inv_shape{1};
    wrapped_unpad.emplace_back(TensorWrapper(unpad_base + in_off[i], unpad_shape, dtype, nullptr,
                                             nullptr, fp8 ? scale_inv + 2 * i : nullptr,
                                             scale_inv_shape));
    unpad_out.emplace_back(wrapped_unpad.back().data());
  }
  std::vector<int> unpadded_rows(height.begin(), height.end());
  nvte_multi_unpadding(num_tensors, out.data(), unpad_out.data(), unpadded_rows.data(), 0);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  for (size_t i = 0; i < num_tensors; ++i) {
    got.resize(in_bytes[i].size());
    ASSERT_EQ(cudaMemcpy(got.data(), unpad_base + in_off[i], got.size(),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(got, in_bytes[i]) << "packed unpadding run, tensor " << i << " (" << padded_height[i]
                                << "x" << width[i] << " -> " << height[i] << "x" << width[i] << ")";
  }

  cudaFree(in_alloc);
  cudaFree(out_alloc);
  cudaFree(unpad_alloc);
  cudaFree(scale_inv);
}

}  // namespace

class MultiPaddingTestSuite
  : public ::testing::TestWithParam<
                                               transformer_engine::DType> {};

TEST_P(MultiPaddingTestSuite, TestMultiPaddingTranspose) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = GetParam();
  const DType output_type = input_type;

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      performTest<InputType, OutputType>(false);
    );
  );
}

TEST_P(MultiPaddingTestSuite, TestMultiUnpadding) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = GetParam();
  const DType output_type = input_type;

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      performTest<InputType, OutputType>(true);
    );
  );
}

// Fewer tensors than one kernel argument struct holds.
TEST_P(MultiPaddingTestSuite, TestMultiPaddingPackedUnaligned) {
  using namespace test;

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(GetParam(), Type,
    performPackedTest<Type>(8);
  );
}

// More tensors than one kernel argument struct holds, so the launch is chunked.
TEST_P(MultiPaddingTestSuite, TestMultiPaddingPackedManyTensors) {
  using namespace test;

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(GetParam(), Type,
    performPackedTest<Type>(129);
  );
}


INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  MultiPaddingTestSuite,
  ::testing::ValuesIn(test::all_fp_types),
  [](const testing::TestParamInfo<MultiPaddingTestSuite::ParamType>& info) {
    std::string name = test::typeName(info.param);
    return name;
  });
