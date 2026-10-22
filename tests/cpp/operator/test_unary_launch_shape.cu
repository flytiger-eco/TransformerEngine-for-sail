/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The unary cast kernels pick their launch shape from the tensor size: a tensor large enough to
// fill the device runs as narrow blocks carrying several vectors per thread, a smaller one keeps
// the shipped one-vector-per-thread shape. Each output element is a function of exactly one input
// element and amax is a pure max reduction, so neither may depend on which shape was chosen.
// These tests pin that invariant for quantize, dequantize and the fused activation backward,
// which is what makes the launch-shape tuning safe.

#include <cmath>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

// Deterministic, varied, finite input. Uniform random data would hide an off-by-one in the
// vector indexing far more often than a pattern whose value depends on the index.
template <typename T>
void fillDeterministic(T *data, const size_t size) {
  uint64_t state = 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i < size; ++i) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    // Spread over roughly [-2, 2] so that a few elements reach the fp8 range boundaries.
    const double u = static_cast<double>(state >> 11) / 9007199254740992.0;
    data[i] = static_cast<T>(4.0 * u - 2.0);
  }
}

constexpr float kScale = 0.375f;
constexpr float kScaleInv = 0.5f;

template <typename InputType, typename OutputType>
void performQuantizePrefixTest(const size_t full_size, const std::vector<size_t> &prefix_sizes) {
  using namespace test;

  std::vector<InputType> host(full_size);
  fillDeterministic(host.data(), full_size);

  Tensor full_input("full_input", std::vector<size_t>{full_size}, TypeInfo<InputType>::dtype);
  Tensor full_output("full_output", std::vector<size_t>{full_size}, TypeInfo<OutputType>::dtype);
  std::memcpy(full_input.rowwise_cpu_dptr<InputType>(), host.data(), full_size * sizeof(InputType));
  full_input.from_cpu();
  full_output.set_scale(kScale);
  full_output.set_amax(0.f);

  nvte_quantize(full_input.data(), full_output.data(), 0);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
  full_output.to_cpu();
  const OutputType *full_ref = full_output.rowwise_cpu_dptr<OutputType>();
  const float full_amax = full_output.amax();

  for (const size_t k : prefix_sizes) {
    ASSERT_LE(k, full_size);
    Tensor input("input", std::vector<size_t>{k}, TypeInfo<InputType>::dtype);
    Tensor output("output", std::vector<size_t>{k}, TypeInfo<OutputType>::dtype);
    std::memcpy(input.rowwise_cpu_dptr<InputType>(), host.data(), k * sizeof(InputType));
    input.from_cpu();
    output.set_scale(kScale);
    output.set_amax(0.f);

    nvte_quantize(input.data(), output.data(), 0);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    output.to_cpu();

    // The first k outputs must be bit-identical to the first k outputs of the full tensor, even
    // though the two launches differ in block size and vectors per thread.
    const OutputType *got = output.rowwise_cpu_dptr<OutputType>();
    EXPECT_EQ(0, std::memcmp(got, full_ref, k * sizeof(OutputType)))
        << "quantize output depends on the tensor size at prefix " << k;

    // amax is a max reduction, so it is order independent and exactly reproducible on the host.
    // A block-size mismatch in the shared-memory reduction would read uninitialized staging here.
    float ref_amax = 0.f;
    for (size_t i = 0; i < k; ++i) {
      ref_amax = fmaxf(ref_amax, fabsf(static_cast<float>(host[i])));
    }
    float got_amax = output.amax();
    EXPECT_EQ(std::memcmp(&got_amax, &ref_amax, sizeof(float)), 0)
        << "amax mismatch at prefix " << k << ": got " << got_amax << " want " << ref_amax;
    EXPECT_LE(got_amax, full_amax);

    // Negative control: the byte comparison above is sensitive to a shift, so it would catch a
    // vector-indexing error rather than passing vacuously.
    if (k + 1 <= full_size) {
      EXPECT_NE(0, std::memcmp(got, full_ref + 1, k * sizeof(OutputType)))
          << "negative control failed at prefix " << k;
    }
  }
}

template <typename InputType, typename OutputType>
void performDequantizePrefixTest(const size_t full_size, const std::vector<size_t> &prefix_sizes) {
  using namespace test;

  std::vector<InputType> host(full_size);
  fillDeterministic(host.data(), full_size);

  Tensor full_input("full_input", std::vector<size_t>{full_size}, TypeInfo<InputType>::dtype);
  Tensor full_output("full_output", std::vector<size_t>{full_size}, TypeInfo<OutputType>::dtype);
  std::memcpy(full_input.rowwise_cpu_dptr<InputType>(), host.data(), full_size * sizeof(InputType));
  full_input.from_cpu();
  // The full and prefix tensors are separate allocations, so the scale they dequantize by has to be
  // pinned explicitly rather than left to whatever the allocator handed back.
  full_input.set_scale_inv(kScaleInv);

  nvte_dequantize(full_input.data(), full_output.data(), 0);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
  full_output.to_cpu();
  const OutputType *full_ref = full_output.rowwise_cpu_dptr<OutputType>();

  for (const size_t k : prefix_sizes) {
    ASSERT_LE(k, full_size);
    Tensor input("input", std::vector<size_t>{k}, TypeInfo<InputType>::dtype);
    Tensor output("output", std::vector<size_t>{k}, TypeInfo<OutputType>::dtype);
    std::memcpy(input.rowwise_cpu_dptr<InputType>(), host.data(), k * sizeof(InputType));
    input.from_cpu();
    input.set_scale_inv(kScaleInv);

    nvte_dequantize(input.data(), output.data(), 0);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    output.to_cpu();

    const OutputType *got = output.rowwise_cpu_dptr<OutputType>();
    EXPECT_EQ(0, std::memcmp(got, full_ref, k * sizeof(OutputType)))
        << "dequantize output depends on the tensor size at prefix " << k;
    if (k + 1 <= full_size) {
      EXPECT_NE(0, std::memcmp(got, full_ref + 1, k * sizeof(OutputType)))
          << "negative control failed at prefix " << k;
    }
  }
}

// nvte_dgelu is the fused activation backward that reaches unary_grad_kernel: it quantizes to fp8
// without dbias, and the dbias variant is rejected outright below compute capability 10.0. The
// reference value of the gelu derivative is not reproduced on the host here, because it would have
// to match the kernel's fp32 operation order exactly; what is checked instead is that the output
// bytes and the amax do not depend on the launch shape.
template <typename InputType, typename OutputType>
void performDgeluPrefixTest(const size_t full_size, const std::vector<size_t> &prefix_sizes) {
  using namespace test;

  std::vector<InputType> host_grad(full_size), host_input(full_size);
  fillDeterministic(host_grad.data(), full_size);
  // A second, different sequence: reusing the first would make dgelu's two operands identical and
  // hide a swapped-argument bug.
  uint64_t state = 0xD1B54A32D192ED03ull;
  for (size_t i = 0; i < full_size; ++i) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const double u = static_cast<double>(state >> 11) / 9007199254740992.0;
    host_input[i] = static_cast<InputType>(4.0 * u - 2.0);
  }

  auto run = [&](const size_t k, const char *tag, float *amax_out) -> std::vector<OutputType> {
    Tensor grad(tag, std::vector<size_t>{k}, TypeInfo<InputType>::dtype);
    Tensor input(tag, std::vector<size_t>{k}, TypeInfo<InputType>::dtype);
    Tensor output(tag, std::vector<size_t>{k}, TypeInfo<OutputType>::dtype);
    std::memcpy(grad.rowwise_cpu_dptr<InputType>(), host_grad.data(), k * sizeof(InputType));
    std::memcpy(input.rowwise_cpu_dptr<InputType>(), host_input.data(), k * sizeof(InputType));
    grad.from_cpu();
    input.from_cpu();
    output.set_scale(kScale);
    output.set_amax(0.f);

    nvte_dgelu(grad.data(), input.data(), output.data(), 0);
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    output.to_cpu();
    *amax_out = output.amax();
    const OutputType *p = output.rowwise_cpu_dptr<OutputType>();
    return std::vector<OutputType>(p, p + k);
  };

  float full_amax = 0.f;
  const std::vector<OutputType> full_ref = run(full_size, "full_dgelu", &full_amax);

  for (const size_t k : prefix_sizes) {
    ASSERT_LE(k, full_size);
    float amax = 0.f;
    const std::vector<OutputType> got = run(k, "dgelu", &amax);

    EXPECT_EQ(0, std::memcmp(got.data(), full_ref.data(), k * sizeof(OutputType)))
        << "dgelu output depends on the tensor size at prefix " << k;

    // amax is a max over a subset, so it can only shrink. A reduce_max instantiated with the wrong
    // warp count reads uninitialized shared staging and typically breaks this, and at k ==
    // full_size it has to come back bit-identical.
    EXPECT_TRUE(std::isfinite(amax) && amax >= 0.f) << "non-finite amax at prefix " << k;
    EXPECT_LE(amax, full_amax) << "amax grew at prefix " << k;
    if (k == full_size) {
      EXPECT_EQ(std::memcmp(&amax, &full_amax, sizeof(float)), 0)
          << "amax is not reproducible at the full size";
    }

    if (k + 1 <= full_size) {
      EXPECT_NE(0, std::memcmp(got.data(), full_ref.data() + 1, k * sizeof(OutputType)))
          << "negative control failed at prefix " << k;
    }
  }
}

// Prefix sizes chosen to fall in every bucket of the launch rule. The bucket boundaries depend on
// the device, so these are derived from a 39 SM part with 2048 threads per SM, where the fill
// target is 312 blocks: a tensor takes the narrow shape at 39936 aligned elements and raises
// vectors-per-thread to 2 at 79872 and to 4 at 159744. On a larger device the same sizes simply
// land in the shipped shape, which the assertions also allow. 65537 is deliberately not a multiple
// of the vector length, so the unaligned arm is exercised as well.
std::vector<size_t> prefixSizes() {
  return {1024,   8192,   32768,  45056,   65537,  100000,
          131072, 300032, 700000, 1048576, 4194304};
}

constexpr size_t kFullSize = 4194304;

}  // namespace

TEST(UnaryLaunchShape, QuantizePrefixIsSizeIndependent) {
  performQuantizePrefixTest<__nv_bfloat16, __nv_fp8_e4m3>(kFullSize, prefixSizes());
  performQuantizePrefixTest<__half, __nv_fp8_e4m3>(kFullSize, prefixSizes());
  performQuantizePrefixTest<float, __nv_fp8_e4m3>(kFullSize, prefixSizes());
  performQuantizePrefixTest<__nv_bfloat16, __nv_fp8_e5m2>(kFullSize, prefixSizes());
  performQuantizePrefixTest<float, __nv_fp8_e5m2>(kFullSize, prefixSizes());
}

TEST(UnaryLaunchShape, DequantizePrefixIsSizeIndependent) {
  performDequantizePrefixTest<__nv_fp8_e4m3, __nv_bfloat16>(kFullSize, prefixSizes());
  performDequantizePrefixTest<__nv_fp8_e4m3, float>(kFullSize, prefixSizes());
  performDequantizePrefixTest<__nv_fp8_e5m2, __nv_bfloat16>(kFullSize, prefixSizes());
}

TEST(UnaryLaunchShape, DgeluPrefixIsSizeIndependent) {
  performDgeluPrefixTest<__nv_bfloat16, __nv_fp8_e4m3>(kFullSize, prefixSizes());
  performDgeluPrefixTest<float, __nv_fp8_e4m3>(kFullSize, prefixSizes());
  performDgeluPrefixTest<__nv_bfloat16, __nv_fp8_e5m2>(kFullSize, prefixSizes());
}
