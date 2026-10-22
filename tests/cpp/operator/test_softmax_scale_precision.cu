/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Precision contract for the scale factor of the fused softmax *forward* ops.
//
// Every nvte_scaled_*softmax_forward() entry point takes `float scale_factor`
// and every device kernel accumulates in `acc_t` (float). The host dispatchers
// used to declare that parameter as `input_t`, so the fp32 scale was narrowed
// to the 16-bit I/O dtype on the way in and widened back to fp32 right before
// the launch. That silently quantizes the user's scale by up to one fp16/bf16
// ulp: 1/sqrt(128) = 0.08838834764831845 becomes 0.08837890625, a relative
// error of 1.07e-4, roughly a thousand times the fp32 rounding the kernels are
// otherwise allowed to introduce.
//
// Three checks pin that contract. The first two fail when the scale is
// quantized and pass when it reaches the kernel as fp32; the third proves they
// are still able to tell the difference.
//
//   AliasedScalesAreDistinguished -- two fp32 scales that the I/O dtype cannot
//   tell apart must still produce different outputs. With a narrowing
//   dispatcher the two runs are bit-identical; without it, about three quarters
//   of the elements move by roughly one output ulp.
//
//   ForwardMatchesDoublePrecisionReference -- the bitwise mismatch rate against
//   a correctly rounded double-precision reference must stay at the level fp32
//   accumulation alone produces (a few elements in a million). Quantizing the
//   scale pushes it into the tens of percent.
//
//   QuantizedScaleIsDetectablyWorse -- feeds the kernel the pre-narrowed scale
//   while holding the reference at the requested one, reproducing the bug
//   without changing the library, and requires the resulting mismatch rate to be
//   far above the threshold the previous test asserts on.
//
// The backward dispatchers already took `acc_t`, so they are not covered here.
//
// These checks cannot go through compareResults(): its mismatch counter waives
// any pair of adjacent representable values, and a quantized scale shows up
// almost exclusively as exactly such one-ulp differences.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <transformer_engine/softmax.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

enum class SoftmaxOp { kScaled, kMasked, kUpperTriang, kAlignedCausal };

const char *opName(SoftmaxOp op) {
  switch (op) {
    case SoftmaxOp::kScaled: return "scaled";
    case SoftmaxOp::kMasked: return "masked";
    case SoftmaxOp::kUpperTriang: return "upper_triang";
    case SoftmaxOp::kAlignedCausal: return "aligned_causal";
  }
  return "?";
}

// Geometry of one test case: `rows` softmax rows of `cols` elements each. The
// upper-triangular layout stores them as attn_batches square [seq, seq] tiles.
struct Geometry {
  std::vector<size_t> shape;
  size_t rows;         // total number of softmax rows in the tensor
  size_t cols;         // elements per row
  size_t rows_per_tile;  // rows before the causal pattern repeats
};

Geometry makeGeometry(SoftmaxOp op) {
  if (op == SoftmaxOp::kUpperTriang) {
    constexpr size_t seq = 64;
    // attn_batches must be a multiple of the kernel's batches_per_block (2 here).
    return Geometry{{8, seq, seq}, 8 * seq, seq, seq};
  }
  return Geometry{{2, 2, 64, 128}, 2 * 2 * 64, 128, 64};
}

// Number of elements row `r` normalizes over; the rest are written as zero by
// the kernels and by the reference alike.
size_t validCount(SoftmaxOp op, const Geometry &g, size_t r) {
  const size_t i = r % g.rows_per_tile;
  switch (op) {
    case SoftmaxOp::kScaled:
    case SoftmaxOp::kMasked:
      return g.cols;
    case SoftmaxOp::kUpperTriang:
      return i + 1;
    case SoftmaxOp::kAlignedCausal:
      // Mirrors masked_elements = i + cols - rows + 1 in the kernel.
      return i + g.cols - g.rows_per_tile + 1;
  }
  return g.cols;
}

template <typename Type>
uint16_t bitsOf(Type v) {
  uint16_t b = 0;
  std::memcpy(&b, &v, sizeof(b));
  return b;
}

// One softmax row in double precision, rounded to Type exactly once.
template <typename Type>
void refRow(Type *out, const Type *in, const uint8_t *mask, size_t valid, size_t cols,
            double scale) {
  std::vector<double> e(cols, 0.0);
  double max_value = -std::numeric_limits<double>::infinity();
  for (size_t j = 0; j < valid; ++j) {
    if (mask != nullptr && mask[j] == 1) continue;
    max_value = std::max(max_value, static_cast<double>(in[j]) * scale);
  }
  double sum = 0.0;
  if (std::isfinite(max_value)) {
    for (size_t j = 0; j < valid; ++j) {
      if (mask != nullptr && mask[j] == 1) continue;
      e[j] = std::exp(static_cast<double>(in[j]) * scale - max_value);
      sum += e[j];
    }
  }
  for (size_t j = 0; j < cols; ++j) {
    out[j] = (j < valid && sum > 0.0) ? static_cast<Type>(e[j] / sum) : static_cast<Type>(0.0f);
  }
}

struct Outcome {
  size_t n = 0;
  // Elements whose encoding differs between the scale_lo and scale_hi runs.
  size_t diff_lo_hi = 0;
  // Elements whose encoding differs from the double-precision reference,
  // which is always evaluated at scale_ref.
  size_t mismatch_lo = 0;
  size_t mismatch_hi = 0;
};

template <typename Type>
Outcome runForward(SoftmaxOp op, float scale_lo, float scale_hi, float scale_ref) {
  using namespace test;
  const DType dtype = TypeInfo<Type>::dtype;
  const Geometry g = makeGeometry(op);
  const size_t n = g.rows * g.cols;

  Tensor input("input", g.shape, dtype);
  Tensor out_lo("out_lo", g.shape, dtype);
  Tensor out_hi("out_hi", g.shape, dtype);
  Tensor mask("mask", std::vector<size_t>{1, 1, g.rows_per_tile, g.cols}, DType::kByte);
  fillUniform(&input);

  uint8_t *mask_cpu = nullptr;
  if (op == SoftmaxOp::kMasked) {
    mask_cpu = mask.rowwise_cpu_dptr<uint8_t>();
    for (size_t i = 0; i < g.rows_per_tile * g.cols; ++i) mask_cpu[i] = (i % 7 == 0) ? 1 : 0;
    mask.from_cpu();
  }

  auto call = [&](Tensor *out, float scale) {
    switch (op) {
      case SoftmaxOp::kScaled:
        nvte_scaled_softmax_forward(input.data(), out->data(), scale, 0);
        break;
      case SoftmaxOp::kMasked:
        nvte_scaled_masked_softmax_forward(input.data(), mask.data(), out->data(), scale, 0);
        break;
      case SoftmaxOp::kUpperTriang:
        nvte_scaled_upper_triang_masked_softmax_forward(input.data(), out->data(), scale, 0);
        break;
      case SoftmaxOp::kAlignedCausal:
        nvte_scaled_aligned_causal_masked_softmax_forward(input.data(), out->data(), scale, 0);
        break;
    }
  };
  call(&out_lo, scale_lo);
  call(&out_hi, scale_hi);
  cudaDeviceSynchronize();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  out_lo.to_cpu();
  out_hi.to_cpu();

  const Type *in_cpu = input.rowwise_cpu_dptr<Type>();
  const Type *lo_cpu = out_lo.rowwise_cpu_dptr<Type>();
  const Type *hi_cpu = out_hi.rowwise_cpu_dptr<Type>();

  Outcome res;
  res.n = n;
  std::vector<Type> ref(g.cols);
  for (size_t r = 0; r < g.rows; ++r) {
    const size_t off = r * g.cols;
    // pad_batches == 1: one mask of rows_per_tile rows shared by every batch.
    const uint8_t *mask_row = (mask_cpu != nullptr) ? mask_cpu + (r % g.rows_per_tile) * g.cols
                                                    : nullptr;
    refRow<Type>(ref.data(), in_cpu + off, mask_row, validCount(op, g, r), g.cols,
                 static_cast<double>(scale_ref));
    for (size_t j = 0; j < g.cols; ++j) {
      if (bitsOf(lo_cpu[off + j]) != bitsOf(hi_cpu[off + j])) ++res.diff_lo_hi;
      if (bitsOf(lo_cpu[off + j]) != bitsOf(ref[j])) ++res.mismatch_lo;
      if (bitsOf(hi_cpu[off + j]) != bitsOf(ref[j])) ++res.mismatch_hi;
    }
  }
  return res;
}

// Round trip an fp32 scalar through the I/O dtype, i.e. exactly what a
// dispatcher that declares the scale as `input_t` does to it.
template <typename Type>
float quantizeScale(float scale) {
  return static_cast<float>(static_cast<Type>(scale));
}

// Widest pair of fp32 scales that `Type` cannot distinguish, found by walking
// out from 1.5f to the edges of its rounding bucket. Deriving it beats naming
// it: the bucket is asymmetric (below 1.0f the fp16 grid is twice as fine as
// above it), so a hand-written "1.0 +/- half an ulp" does not actually alias.
template <typename Type>
std::pair<float, float> aliasedScalePair() {
  const float mid = 1.5f;
  const uint16_t mid_bits = bitsOf(static_cast<Type>(mid));
  auto encodesToMid = [&](float v) { return bitsOf(static_cast<Type>(v)) == mid_bits; };
  float lo = mid;
  while (encodesToMid(std::nextafter(lo, 0.0f))) lo = std::nextafter(lo, 0.0f);
  float hi = mid;
  while (encodesToMid(std::nextafter(hi, std::numeric_limits<float>::infinity()))) {
    hi = std::nextafter(hi, std::numeric_limits<float>::infinity());
  }
  return {lo, hi};
}

template <typename Type>
void testAliasedScales(SoftmaxOp op, const char *dtype_name) {
  const auto [scale_lo, scale_hi] = aliasedScalePair<Type>();
  // Positive control: the pair really is indistinguishable in the I/O dtype, so a
  // dispatcher that narrows the scale cannot produce two different results.
  ASSERT_EQ(bitsOf(static_cast<Type>(scale_lo)), bitsOf(static_cast<Type>(scale_hi)))
      << "test bug: the scale pair is not aliased in this dtype";
  ASSERT_NE(scale_lo, scale_hi) << "test bug: the scale pair collapsed in fp32";

  const Outcome r = runForward<Type>(op, scale_lo, scale_hi, scale_hi);
  const double rate = static_cast<double>(r.diff_lo_hi) / static_cast<double>(r.n);
  std::printf("# %-9s %-14s aliased scales %.9g/%.9g differ in %zu/%zu (%.3f%%)\n", dtype_name,
              opName(op), scale_lo, scale_hi, r.diff_lo_hi, r.n, 100.0 * rate);
  EXPECT_GT(rate, 0.05)
      << opName(op) << ": the outputs for scales " << scale_lo << " and " << scale_hi
      << " differ in only " << r.diff_lo_hi << "/" << r.n << " elements (" << 100.0 * rate
      << "%). The forward dispatcher is quantizing the fp32 scale to the I/O dtype.";
}

// 2.1f is not representable in fp16 or bf16, and it is large enough that the
// exponents span about one output ulp, which is what makes quantizing the scale
// visible element by element rather than only in aggregate.
constexpr float kAccuracyScale = 2.1f;

template <typename Type>
void testAgainstDoubleReference(SoftmaxOp op, const char *dtype_name) {
  ASSERT_NE(quantizeScale<Type>(kAccuracyScale), kAccuracyScale)
      << "test bug: the scale is exactly representable in this dtype";

  const Outcome r = runForward<Type>(op, kAccuracyScale, kAccuracyScale, kAccuracyScale);
  const double rate = static_cast<double>(r.mismatch_hi) / static_cast<double>(r.n);
  std::printf("# %-9s %-14s scale=%-9.9g vs fp64 reference: %zu/%zu mismatches (%.5f%%)\n",
              dtype_name, opName(op), kAccuracyScale, r.mismatch_hi, r.n, 100.0 * rate);
  EXPECT_LT(rate, 0.005)
      << opName(op) << ": " << r.mismatch_hi << "/" << r.n << " elements (" << 100.0 * rate
      << "%) differ from the correctly rounded double-precision softmax. fp32 accumulation "
         "alone moves a few elements in a million; quantizing the scale to the I/O dtype "
         "moves tens of percent.";
}

// Negative control for the threshold above, kept in the suite permanently so the
// assertion cannot silently lose its discriminating power. It feeds the kernel
// the *pre-narrowed* scale while holding the reference at the requested one,
// which reproduces bit for bit what a dispatcher declaring `input_t scale` does.
// If that rate ever falls under the 0.5% bar, the bar proves nothing.
template <typename Type>
void testQuantizedScaleIsWorse(SoftmaxOp op, const char *dtype_name) {
  const float quantized = quantizeScale<Type>(kAccuracyScale);
  ASSERT_NE(quantized, kAccuracyScale) << "test bug: the scale survived the round trip";

  const Outcome good = runForward<Type>(op, kAccuracyScale, kAccuracyScale, kAccuracyScale);
  const Outcome bad = runForward<Type>(op, quantized, quantized, kAccuracyScale);
  const double good_rate = static_cast<double>(good.mismatch_hi) / static_cast<double>(good.n);
  const double bad_rate = static_cast<double>(bad.mismatch_hi) / static_cast<double>(bad.n);
  std::printf("# %-9s %-14s quantized scale %-9.9g -> %.9g: %.5f%% vs %.5f%% mismatches\n",
              dtype_name, opName(op), kAccuracyScale, quantized, 100.0 * good_rate,
              100.0 * bad_rate);
  EXPECT_GT(bad_rate, 0.05) << opName(op) << ": narrowing the scale is not measurable here, "
                               "so this case cannot validate the accuracy threshold";
  EXPECT_GT(bad_rate, 10.0 * good_rate)
      << opName(op) << ": the quantized scale is only " << bad_rate / good_rate
      << "x worse; the 0.5% threshold has too little margin";
}

}  // namespace

// Same dispatch trick as test_softmax.cu: use the test harness's fp16/bf16
// aliases so `Tensor` stays unambiguous.
#define SOFTMAX_PRECISION_DISPATCH_16BIT(dtype, fn, op)    \
  switch (dtype) {                                         \
    case DType::kFloat16:                                  \
      fn<test::fp16>(op, "float16");                       \
      break;                                               \
    case DType::kBFloat16:                                 \
      fn<test::bf16>(op, "bfloat16");                      \
      break;                                               \
    default:                                               \
      GTEST_FAIL() << "Unsupported 16-bit dtype for test"; \
  }

using SoftmaxPrecisionParams = std::tuple<DType, SoftmaxOp>;

class SoftmaxScalePrecisionTestSuite : public ::testing::TestWithParam<SoftmaxPrecisionParams> {};

TEST_P(SoftmaxScalePrecisionTestSuite, AliasedScalesAreDistinguished) {
  const auto [dtype, op] = GetParam();
  SOFTMAX_PRECISION_DISPATCH_16BIT(dtype, testAliasedScales, op);
}

TEST_P(SoftmaxScalePrecisionTestSuite, ForwardMatchesDoublePrecisionReference) {
  const auto [dtype, op] = GetParam();
  SOFTMAX_PRECISION_DISPATCH_16BIT(dtype, testAgainstDoubleReference, op);
}

TEST_P(SoftmaxScalePrecisionTestSuite, QuantizedScaleIsDetectablyWorse) {
  const auto [dtype, op] = GetParam();
  SOFTMAX_PRECISION_DISPATCH_16BIT(dtype, testQuantizedScaleIsWorse, op);
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, SoftmaxScalePrecisionTestSuite,
    ::testing::Combine(::testing::Values(DType::kFloat16, DType::kBFloat16),
                       ::testing::Values(SoftmaxOp::kScaled, SoftmaxOp::kMasked,
                                         SoftmaxOp::kUpperTriang, SoftmaxOp::kAlignedCausal)),
    [](const testing::TestParamInfo<SoftmaxPrecisionParams> &info) {
      const DType dtype = std::get<0>(info.param);
      return test::typeName(dtype) + "_" + opName(std::get<1>(info.param));
    });
