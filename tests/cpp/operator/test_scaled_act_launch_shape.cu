/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The scaled gated activation forward sizes its launch from the bound of the kernel's grid-stride
// loop, rows * num_segments * num_vectors: a tensor large enough to fill the device runs as
// 128-thread blocks carrying up to four vectors per thread, a smaller one keeps the shipped
// 512-thread one-vector-per-thread shape. Every loop iteration writes exactly one output vector
// and reads only the two input vectors and the one row scale that vector depends on, with no
// cross-thread or cross-iteration state, so output row r is a function of input row r alone and
// cannot depend on how many rows the launch covered or on how the iterations were distributed over
// blocks. These tests pin that invariant, which is what makes the launch-shape tuning safe. They
// mirror test_gated_launch_shape.cu, which covers the quantize gated kernels under the same rule.
//
// The scaled unary forward (nvte_scaled_srelu) is deliberately left at the shipped shape and is
// covered here only as a regression guard. Two sweeps over the same launch geometries, one of 78
// gated cells and one of 39 unary cells spanning working sets from 1.5 MiB to 3 GiB, put the gated
// forward at a 1.136x geomean with no cell below 0.9991x and the unary forward at a 0.991x geomean
// with 22 of 39 cells worse. The unary path has one input array instead of two, so at a given
// working set it moves half the bytes and is already close to the memory roofline; narrowing its
// block only costs it.
//
// The backward paths are not covered. Both reduce along a row, so their results legitimately
// depend on the thread count, and neither launch shape was changed.

#include <cstring>
#include <set>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>

// The positive control below asks the real launch-shape helper what it would pick, rather than
// restating its arithmetic, so this needs the internal header. That header pulls in common.h,
// which defines TRANSFORMER_ENGINE_TYPE_SWITCH_ALL differently from test_common.h's copy; hiding
// the internal one for the duration of the include keeps the redefinition warning out of the build.
// Nothing in this file uses the macro. It also brings transformer_engine::Tensor into scope, so the
// test fixture's Tensor is spelled test::Tensor throughout.
#pragma push_macro("TRANSFORMER_ENGINE_TYPE_SWITCH_ALL")
#undef TRANSFORMER_ENGINE_TYPE_SWITCH_ALL
#include "util/vectorized_pointwise.h"
#pragma pop_macro("TRANSFORMER_ENGINE_TYPE_SWITCH_ALL")

#include "../test_common.h"

using namespace transformer_engine;

namespace {

// Deterministic, varied, finite input whose value depends on the index, so that an off-by-one in
// the row, segment or vector indexing shows up. Uniform random data would hide it far more often.
template <typename T>
void fillDeterministic(T *data, const size_t size, const uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < size; ++i) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const double u = static_cast<double>(state >> 11) / 9007199254740992.0;
    // Spread over roughly [-2, 2]. The gated ops read a gate half that the clamped and SiTU
    // variants saturate, so this range exercises both the saturating and the linear branches.
    data[i] = static_cast<T>(4.0 * u - 2.0);
  }
}

// Stand-in for the launcher's kernel when asking the occupancy API how many 128-thread blocks fit
// per SM. Register-light on purpose: scaled_gated_forward_kernel is compiled under
// __launch_bounds__(512, 4), which caps it at 64 registers per thread, so at 128 threads both it
// and this probe are limited by maxThreadsPerMultiProcessor rather than by registers and report
// the same blocks per SM. Nothing below asserts an exact block count, so a device on which that
// reasoning did not hold would only move the bucket boundaries, not the conclusion.
__global__ void launchShapeProbe(float *out) {
  if (out != nullptr) out[threadIdx.x] = 0.f;
}

// Row counts chosen to land in every bucket of the launch rule. The bucket boundaries depend on
// the device, so these are derived from a 39 SM part with 2048 threads per SM and 16 blocks per SM
// at 128 threads, which gives a fill target of 312 blocks. With cols = 4096 the bf16 vector count
// per row is 256, so the iteration count is 256 * rows: the narrow block is selected at 39936
// iterations (156 rows), vectors-per-thread rises to 2 at 79872 (312 rows) and to 4 at 159744
// (624 rows). The fp32 cases carry 512 vectors per row and straddle the same boundaries at half
// these row counts. On a larger device the same sizes simply land in the shipped shape earlier in
// the ladder, which the assertions also allow.
std::vector<size_t> rowCounts() {
  return {1,    2,    8,    32,   60,   78,   79,   120,  155,  156,  157,
          200,  300,  311,  312,  313,  400,  600,  623,  624,  625,  1024,
          2048, 4096};
}

constexpr size_t kCols = 4096;
constexpr size_t kFullRows = 4096;

// Segment size for the interleaved cases. It has to be a multiple of the vector length that also
// divides kCols so the vectorized arm is taken; 1024 gives four segments per row.
constexpr int64_t kInterleave = 1024;

// A column count that is not a multiple of the vector length sends row_vector_alignment down its
// DIFFERENT arm, because input and input + cols then disagree on alignment. That arm cannot
// vectorize and keeps the shipped shape, so it is here as a regression guard rather than as
// coverage of the tuning.
constexpr size_t kOddCols = 4097;
constexpr size_t kOddFullRows = 512;

std::vector<size_t> oddRowCounts() { return {1, 7, 64, 120, 200, 512}; }

// Parameters of the saturating variants, matching what the PyTorch extension passes.
constexpr float kClampedLimit = 0.5f;
constexpr float kClampedAlpha = 1.702f;
constexpr float kClampedLinearOffset = 0.5f;
constexpr float kSiTUBeta1 = 4.0f;
constexpr float kSiTUBeta2 = 25.0f;

// Forward scaled activation: input [rows, input_cols], act_scales [rows] -> output [rows, cols].
// The gated ops take input_cols = 2 * cols, holding an activation half and a gate half; the unary
// one takes input_cols = cols, because it reads no gate.
template <typename InputType, typename ScaleType, typename OutputType, typename Op>
void performScaledFwdRowPrefixTest(const size_t cols, const size_t input_cols,
                                   const size_t full_rows, const std::vector<size_t> &prefix_rows,
                                   Op op, const char *tag) {
  std::vector<InputType> host(full_rows * input_cols);
  fillDeterministic(host.data(), host.size(), 0x9E3779B97F4A7C15ull);
  // A second, different sequence for the row scales. Reusing the input sequence would leave a
  // scale/activation mix-up undetected, because both operands would carry the same values.
  std::vector<ScaleType> host_scales(full_rows);
  fillDeterministic(host_scales.data(), host_scales.size(), 0xD1B54A32D192ED03ull);

  auto run = [&](const size_t rows) -> std::vector<OutputType> {
    test::Tensor input(tag, std::vector<size_t>{rows, input_cols},
                       test::TypeInfo<InputType>::dtype);
    test::Tensor scales(tag, std::vector<size_t>{rows}, test::TypeInfo<ScaleType>::dtype);
    test::Tensor output(tag, std::vector<size_t>{rows, cols}, test::TypeInfo<OutputType>::dtype);
    std::memcpy(input.rowwise_cpu_dptr<InputType>(), host.data(),
                rows * input_cols * sizeof(InputType));
    std::memcpy(scales.rowwise_cpu_dptr<ScaleType>(), host_scales.data(),
                rows * sizeof(ScaleType));
    input.from_cpu();
    scales.from_cpu();

    op(input.data(), scales.data(), output.data(), 0);
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    output.to_cpu();
    const OutputType *p = output.rowwise_cpu_dptr<OutputType>();
    return std::vector<OutputType>(p, p + rows * cols);
  };

  const std::vector<OutputType> full = run(full_rows);
  for (const size_t rows : prefix_rows) {
    ASSERT_LE(rows, full_rows);
    const std::vector<OutputType> got = run(rows);

    // Bit-exact, not tolerance-based: the launch shape cannot change any arithmetic, so anything
    // other than identical bytes is a defect.
    EXPECT_EQ(0, std::memcmp(got.data(), full.data(), rows * cols * sizeof(OutputType)))
        << tag << ": output depends on the row count at " << rows;

    // Negative control. The comparison above is byte-exact, so it would also catch a row, segment
    // or vector indexing error rather than passing vacuously. Shifting the expected data by one
    // row has to change it.
    if (rows < full_rows) {
      EXPECT_NE(0, std::memcmp(got.data(), full.data() + cols, rows * cols * sizeof(OutputType)))
          << tag << ": negative control failed at " << rows;
    }
  }
}

// The launch shape the rule picks for `rows` rows of `vectors_per_row` vectors each.
//
// This constructs PointwiseLaunchConfig directly instead of calling
// make_pointwise_narrow_launch_config, which is what the launcher calls. The two are equivalent
// here: that helper only adds an upper-bound prefilter, target = sm_count * (max_threads_per_sm /
// narrow_threads) / 2, which is at least the target the constructor computes from the occupancy
// query, so it can only return the shipped shape in cases the constructor returns it too. It is
// bypassed because it reaches cuda::device_props and cuda::kernel_blocks_per_sm, which are internal
// to libtransformer_engine and not exported, so a test cannot link them. The two device properties
// it reads are fetched here the same way it fetches them.
PointwiseLaunchConfig shapeAt(const size_t rows, const size_t vectors_per_row) {
  int device = 0;
  EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp prop;
  EXPECT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  int narrow_blocks_per_sm = 0;
  EXPECT_EQ(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &narrow_blocks_per_sm, launchShapeProbe, pointwise_narrow_kernel_threads, 0),
            cudaSuccess);
  return PointwiseLaunchConfig(rows * vectors_per_row, prop.multiProcessorCount,
                               narrow_blocks_per_sm);
}

}  // namespace

// Positive control. Every test below compares a prefix run against a full run, so they can only
// fail if the launcher picks a different shape for different row counts. On a device where the
// whole ladder landed in one bucket they would pass without exercising the rule at all. This asks
// the rule itself, with the iteration counts the launcher computes, and requires the ladder to span
// the shipped shape, the narrow block, and more than one vector per thread.
TEST(ScaledActLaunchShape, RowLadderSpansMultipleLaunchShapes) {
  const size_t bf16_vectors_per_row = kCols / (32 / sizeof(__nv_bfloat16));
  const size_t fp32_vectors_per_row = kCols / (32 / sizeof(float));

  // Interleaving splits a row into num_segments chunks of num_vectors vectors each, so its grid is
  // sized by their product. When the segment is a multiple of the vector length that product is
  // hidden / nvec, i.e. the interleaved layout sizes the same grid as the contiguous one and needs
  // no bucket of its own here. Pinning it keeps the interleaved prefix test honest: were the
  // segment not a multiple of nvec, the bound would grow and the ladder could miss a bucket.
  const size_t interleaved_vectors_per_row =
      (kCols / kInterleave) * (kInterleave / (32 / sizeof(__nv_bfloat16)));
  EXPECT_EQ(interleaved_vectors_per_row, bf16_vectors_per_row);

  for (const size_t vectors_per_row : {bf16_vectors_per_row, fp32_vectors_per_row}) {
    std::set<std::pair<size_t, size_t>> shapes;
    size_t top_rows = 0;
    PointwiseLaunchConfig top;
    for (const size_t rows : rowCounts()) {
      const PointwiseLaunchConfig cfg = shapeAt(rows, vectors_per_row);
      shapes.emplace(cfg.threads, cfg.vecs_per_thread);
      if (rows > top_rows) {
        top_rows = rows;
        top = cfg;
      }
    }
    EXPECT_GE(shapes.size(), 3u)
        << "vectors_per_row=" << vectors_per_row
        << ": the row ladder does not cross the launch rule's buckets on this device, so the "
           "prefix tests would pass vacuously";

    const PointwiseLaunchConfig smallest = shapeAt(1, vectors_per_row);
    EXPECT_EQ(smallest.threads, static_cast<size_t>(unary_kernel_threads))
        << "vectors_per_row=" << vectors_per_row << ": a single row left the shipped shape";
    EXPECT_EQ(smallest.vecs_per_thread, 1u);

    EXPECT_EQ(top.threads, static_cast<size_t>(pointwise_narrow_kernel_threads))
        << "vectors_per_row=" << vectors_per_row << ": the largest row count kept the wide block";
    EXPECT_GE(top.vecs_per_thread, 2u)
        << "vectors_per_row=" << vectors_per_row
        << ": the ladder never reached two vectors per thread";
  }
}

TEST(ScaledActLaunchShape, GatedFwdRowPrefixIsSizeIndependent) {
  auto swiglu = [](const NVTETensor in, const NVTETensor sc, NVTETensor out, cudaStream_t st) {
    nvte_scaled_swiglu(in, sc, out, 0, st);
  };
  performScaledFwdRowPrefixTest<__nv_bfloat16, float, __nv_bfloat16>(
      kCols, 2 * kCols, kFullRows, rowCounts(), swiglu, "scaled_swiglu_bf16");
  // A wider output than the input halves nvec and so doubles the vector count per row, which
  // reaches the narrow block at half the row count and exercises the rule's second boundary.
  performScaledFwdRowPrefixTest<__nv_bfloat16, float, float>(
      kCols, 2 * kCols, kFullRows, rowCounts(), swiglu, "scaled_swiglu_bf16_f32out");
  performScaledFwdRowPrefixTest<float, float, float>(
      kCols, 2 * kCols, kFullRows, rowCounts(), swiglu, "scaled_swiglu_fp32");
  performScaledFwdRowPrefixTest<__half, float, __half>(
      kCols, 2 * kCols, kFullRows, rowCounts(), swiglu, "scaled_swiglu_fp16");
}

TEST(ScaledActLaunchShape, GatedFwdInterleavedIsSizeIndependent) {
  // num_segments > 1 multiplies the grid-stride loop bound without changing what an iteration
  // reads, so the interleaved layout has to be equally shape-independent.
  auto swiglu_il = [](const NVTETensor in, const NVTETensor sc, NVTETensor out, cudaStream_t st) {
    nvte_scaled_swiglu(in, sc, out, kInterleave, st);
  };
  performScaledFwdRowPrefixTest<__nv_bfloat16, float, __nv_bfloat16>(
      kCols, 2 * kCols, kFullRows, rowCounts(), swiglu_il, "scaled_swiglu_bf16_interleaved");
  performScaledFwdRowPrefixTest<float, float, float>(
      kCols, 2 * kCols, kFullRows, rowCounts(), swiglu_il, "scaled_swiglu_fp32_interleaved");
}

TEST(ScaledActLaunchShape, SaturatingGatedFwdIsSizeIndependent) {
  auto clamped = [](const NVTETensor in, const NVTETensor sc, NVTETensor out, cudaStream_t st) {
    nvte_scaled_clamped_swiglu(in, sc, out, kClampedLimit, kClampedAlpha, kClampedLinearOffset, 0,
                               st);
  };
  auto situ = [](const NVTETensor in, const NVTETensor sc, NVTETensor out, cudaStream_t st) {
    nvte_scaled_situglu(in, sc, out, kSiTUBeta1, kSiTUBeta2, 0, st);
  };
  performScaledFwdRowPrefixTest<__nv_bfloat16, float, __nv_bfloat16>(
      kCols, 2 * kCols, kFullRows, rowCounts(), clamped, "scaled_clamped_swiglu_bf16");
  performScaledFwdRowPrefixTest<float, float, float>(
      kCols, 2 * kCols, kFullRows, rowCounts(), clamped, "scaled_clamped_swiglu_fp32");
  performScaledFwdRowPrefixTest<__nv_bfloat16, float, __nv_bfloat16>(
      kCols, 2 * kCols, kFullRows, rowCounts(), situ, "scaled_situglu_bf16");
  performScaledFwdRowPrefixTest<__half, float, __half>(
      kCols, 2 * kCols, kFullRows, rowCounts(), situ, "scaled_situglu_fp16");
}

// Regression guard for the two arms the tuning does not touch: the unary forward, which the sweep
// measured as a loss, and the non-vectorized fallback, which cannot amortize a narrow block.
TEST(ScaledActLaunchShape, UntunedArmsAreSizeIndependent) {
  auto srelu = [](const NVTETensor in, const NVTETensor sc, NVTETensor out, cudaStream_t st) {
    nvte_scaled_srelu(in, sc, out, st);
  };
  performScaledFwdRowPrefixTest<__nv_bfloat16, float, __nv_bfloat16>(
      kCols, kCols, kFullRows, rowCounts(), srelu, "scaled_srelu_bf16");
  performScaledFwdRowPrefixTest<float, float, float>(
      kCols, kCols, kFullRows, rowCounts(), srelu, "scaled_srelu_fp32");

  auto swiglu = [](const NVTETensor in, const NVTETensor sc, NVTETensor out, cudaStream_t st) {
    nvte_scaled_swiglu(in, sc, out, 0, st);
  };
  performScaledFwdRowPrefixTest<__nv_bfloat16, float, __nv_bfloat16>(
      kOddCols, 2 * kOddCols, kOddFullRows, oddRowCounts(), swiglu, "scaled_swiglu_bf16_oddcols");
}
