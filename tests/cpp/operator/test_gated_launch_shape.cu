/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The gated activation kernels size their launch from m * n: a tensor large enough to fill the
// device runs as narrow blocks carrying several vectors per thread, a smaller one keeps the
// shipped one-vector-per-thread shape. Output row r is a function of input row r alone, and amax
// is a pure max reduction, so neither may depend on how many rows the launch covered. These tests
// pin that invariant for both gated directions, which is what makes the launch-shape tuning safe.
// They mirror test_unary_launch_shape.cu, which covers the unary half of the same rule.

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

// Deterministic, varied, finite input whose value depends on the index, so that an off-by-one in
// the row or vector indexing shows up. Uniform random data would hide it far more often.
template <typename T>
void fillDeterministic(T *data, const size_t size, const uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < size; ++i) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    // Spread over roughly [-2, 2] so that a few elements reach the fp8 range boundaries.
    const double u = static_cast<double>(state >> 11) / 9007199254740992.0;
    data[i] = static_cast<T>(4.0 * u - 2.0);
  }
}

constexpr float kScale = 0.375f;

// Checks the two properties the launch rule has to leave alone: the output bytes of the first
// `rows` rows, and amax. `expected` is the full-row-count result, `got` the prefix result.
template <typename OutputType>
void checkPrefixInvariance(const char *tag, const std::vector<OutputType> &got,
                           const std::vector<OutputType> &expected, const float amax,
                           const float expected_amax, const size_t rows, const size_t row_stride,
                           const size_t full_rows) {
  EXPECT_EQ(0, std::memcmp(got.data(), expected.data(), rows * row_stride * sizeof(OutputType)))
      << tag << " output depends on the row count at " << rows;

  // amax is a max over a subset, so it can only shrink. A reduce_max instantiated with the wrong
  // warp count reads uninitialized shared staging and typically breaks this, and at the full row
  // count it has to come back bit-identical.
  EXPECT_TRUE(std::isfinite(amax) && amax >= 0.f) << tag << ": non-finite amax at " << rows;
  EXPECT_LE(amax, expected_amax) << tag << ": amax grew at " << rows;
  if (rows == full_rows) {
    EXPECT_EQ(std::memcmp(&amax, &expected_amax, sizeof(float)), 0)
        << tag << ": amax is not reproducible at the full row count";
  }

  // Negative control: the byte comparison above is sensitive to a shift, so it would catch a row or
  // vector indexing error rather than passing vacuously.
  if (rows < full_rows) {
    EXPECT_NE(0, std::memcmp(got.data(), expected.data() + row_stride,
                             rows * row_stride * sizeof(OutputType)))
        << tag << ": negative control failed at " << rows;
  }
}

// Forward gated activation: input [rows, 2 * cols] -> output [rows, cols].
template <typename InputType, typename OutputType>
void performGatedFwdRowPrefixTest(const size_t cols, const size_t full_rows,
                                  const std::vector<size_t> &prefix_rows,
                                  void (*op)(const NVTETensor, NVTETensor, cudaStream_t),
                                  const char *tag) {
  using namespace test;

  std::vector<InputType> host(full_rows * cols * 2);
  fillDeterministic(host.data(), host.size(), 0x9E3779B97F4A7C15ull);

  auto run = [&](const size_t rows) -> std::pair<std::vector<OutputType>, float> {
    Tensor input(tag, std::vector<size_t>{rows, cols * 2}, TypeInfo<InputType>::dtype);
    Tensor output(tag, std::vector<size_t>{rows, cols}, TypeInfo<OutputType>::dtype);
    std::memcpy(input.rowwise_cpu_dptr<InputType>(), host.data(),
                rows * cols * 2 * sizeof(InputType));
    input.from_cpu();
    output.set_scale(kScale);
    output.set_amax(0.f);

    op(input.data(), output.data(), 0);
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    output.to_cpu();
    const OutputType *p = output.rowwise_cpu_dptr<OutputType>();
    return {std::vector<OutputType>(p, p + rows * cols), output.amax()};
  };

  const auto full = run(full_rows);
  for (const size_t rows : prefix_rows) {
    ASSERT_LE(rows, full_rows);
    const auto got = run(rows);
    checkPrefixInvariance<OutputType>(tag, got.first, full.first, got.second, full.second, rows,
                                      cols, full_rows);
  }
}

// Backward gated activation: grad [rows, cols], input [rows, 2 * cols] -> output [rows, 2 * cols].
template <typename InputType, typename OutputType>
void performGatedBwdRowPrefixTest(const size_t cols, const size_t full_rows,
                                  const std::vector<size_t> &prefix_rows,
                                  void (*op)(const NVTETensor, const NVTETensor, NVTETensor,
                                             cudaStream_t),
                                  const char *tag) {
  using namespace test;

  std::vector<InputType> host_grad(full_rows * cols);
  std::vector<InputType> host_input(full_rows * cols * 2);
  fillDeterministic(host_grad.data(), host_grad.size(), 0x9E3779B97F4A7C15ull);
  // A second, different sequence: reusing the first would make the two operands of the backward
  // identical and hide a swapped-argument bug.
  fillDeterministic(host_input.data(), host_input.size(), 0xD1B54A32D192ED03ull);

  auto run = [&](const size_t rows) -> std::pair<std::vector<OutputType>, float> {
    Tensor grad(tag, std::vector<size_t>{rows, cols}, TypeInfo<InputType>::dtype);
    Tensor input(tag, std::vector<size_t>{rows, cols * 2}, TypeInfo<InputType>::dtype);
    Tensor output(tag, std::vector<size_t>{rows, cols * 2}, TypeInfo<OutputType>::dtype);
    std::memcpy(grad.rowwise_cpu_dptr<InputType>(), host_grad.data(),
                rows * cols * sizeof(InputType));
    std::memcpy(input.rowwise_cpu_dptr<InputType>(), host_input.data(),
                rows * cols * 2 * sizeof(InputType));
    grad.from_cpu();
    input.from_cpu();
    output.set_scale(kScale);
    output.set_amax(0.f);

    op(grad.data(), input.data(), output.data(), 0);
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    output.to_cpu();
    const OutputType *p = output.rowwise_cpu_dptr<OutputType>();
    return {std::vector<OutputType>(p, p + rows * cols * 2), output.amax()};
  };

  const auto full = run(full_rows);
  for (const size_t rows : prefix_rows) {
    ASSERT_LE(rows, full_rows);
    const auto got = run(rows);
    checkPrefixInvariance<OutputType>(tag, got.first, full.first, got.second, full.second, rows,
                                      cols * 2, full_rows);
  }
}

// Row counts chosen to land in every bucket of the launch rule. The bucket boundaries depend on the
// device, so these are derived from a 39 SM part with 2048 threads per SM. With cols = 4096 the
// bf16 vector count per row is 256, so the iteration count is 256 * rows: the gelu forward reaches
// its fill target of 312 blocks at 39936 iterations, raises vectors-per-thread to 2 at 79872 and to
// 4 at 159744, and 624 rows lands exactly on that last boundary. The swiglu backward carries the
// highest register pressure of the four kernels (88 registers, 10 blocks per SM), so its target is
// lower (195) and the same row counts straddle its boundaries differently. On a larger device the
// same sizes simply land in the shipped shape, which the assertions also allow.
std::vector<size_t> rowCounts() {
  return {1, 2, 8, 32, 60, 120, 128, 200, 300, 400, 600, 624, 1024, 2048};
}

constexpr size_t kCols = 4096;
constexpr size_t kFullRows = 2048;

// A column count that is not a multiple of the vector length sends CheckAlignment down its
// DIFFERENT arm, because input and input + cols then disagree on alignment. That arm cannot
// vectorize and keeps the shipped shape, so it is here as a regression guard rather than as
// coverage of the tuning.
constexpr size_t kOddCols = 4097;
constexpr size_t kOddFullRows = 512;

std::vector<size_t> oddRowCounts() { return {1, 7, 64, 120, 200, 512}; }

}  // namespace

TEST(GatedLaunchShape, FwdRowPrefixIsSizeIndependent) {
  performGatedFwdRowPrefixTest<__nv_bfloat16, __nv_fp8_e4m3>(kCols, kFullRows, rowCounts(),
                                                             nvte_geglu, "geglu_bf16_e4m3");
  performGatedFwdRowPrefixTest<__nv_bfloat16, __nv_fp8_e4m3>(kCols, kFullRows, rowCounts(),
                                                             nvte_swiglu, "swiglu_bf16_e4m3");
  performGatedFwdRowPrefixTest<float, __nv_fp8_e4m3>(kCols, kFullRows, rowCounts(), nvte_geglu,
                                                     "geglu_fp32_e4m3");
  performGatedFwdRowPrefixTest<float, __nv_fp8_e4m3>(kCols, kFullRows, rowCounts(), nvte_swiglu,
                                                     "swiglu_fp32_e4m3");
  performGatedFwdRowPrefixTest<__nv_bfloat16, __nv_fp8_e5m2>(kCols, kFullRows, rowCounts(),
                                                             nvte_swiglu, "swiglu_bf16_e5m2");
}

TEST(GatedLaunchShape, BwdRowPrefixIsSizeIndependent) {
  performGatedBwdRowPrefixTest<__nv_bfloat16, __nv_fp8_e4m3>(kCols, kFullRows, rowCounts(),
                                                             nvte_dgeglu, "dgeglu_bf16_e4m3");
  performGatedBwdRowPrefixTest<__nv_bfloat16, __nv_fp8_e4m3>(kCols, kFullRows, rowCounts(),
                                                             nvte_dswiglu, "dswiglu_bf16_e4m3");
  performGatedBwdRowPrefixTest<float, __nv_fp8_e4m3>(kCols, kFullRows, rowCounts(), nvte_dgeglu,
                                                     "dgeglu_fp32_e4m3");
  performGatedBwdRowPrefixTest<float, __nv_fp8_e4m3>(kCols, kFullRows, rowCounts(), nvte_dswiglu,
                                                     "dswiglu_fp32_e4m3");
  performGatedBwdRowPrefixTest<__nv_bfloat16, __nv_fp8_e5m2>(kCols, kFullRows, rowCounts(),
                                                             nvte_dswiglu, "dswiglu_bf16_e5m2");
}

TEST(GatedLaunchShape, NonVectorizedColumnsAreSizeIndependent) {
  performGatedFwdRowPrefixTest<__nv_bfloat16, __nv_fp8_e4m3>(kOddCols, kOddFullRows,
                                                             oddRowCounts(), nvte_geglu,
                                                             "geglu_bf16_oddcols");
  performGatedBwdRowPrefixTest<__nv_bfloat16, __nv_fp8_e4m3>(kOddCols, kOddFullRows,
                                                             oddRowCounts(), nvte_dswiglu,
                                                             "dswiglu_bf16_oddcols");
}
