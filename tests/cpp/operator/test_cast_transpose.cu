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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/transpose.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename InputType, typename OutputType>
void compute_ref(const InputType *data, OutputType *output_c, OutputType *output_t,
                 const size_t N, const size_t H,
                 float *amax, float scale) {
  using compute_t = float;
  compute_t current_max = -1e100;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      current_max = fmaxf(current_max, fabsf(current));
      output_c[i * H + j] = OutputType(scale * current);
      output_t[j * N + i] = OutputType(scale * current);
    }
  }
  *amax = current_max;
}


// delayed tensor scaling test
template <typename InputType, typename OutputType>
void performTest(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  Tensor input("input", std::vector<size_t>{ N, H }, itype);
  Tensor output("output", std::vector<size_t>{ N, H }, otype, true, true);

  std::unique_ptr<OutputType[]> ref_output_c = std::make_unique<OutputType[]>(N * H);
  std::unique_ptr<OutputType[]> ref_output_t = std::make_unique<OutputType[]>(N * H);

  fillUniform(&input);
  setRandomScale(&output);
  const float ref_scale = isFp8Type(otype) ? output.scale() : 1.0f;

  nvte_quantize(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref<InputType, OutputType>(input.rowwise_cpu_dptr<InputType>(), ref_output_c.get(),
                                     ref_output_t.get(), N, H, &ref_amax, ref_scale);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  if (isFp8Type(otype)) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / ref_scale;
    compareResults("scale_inv", output.rowwise_scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output, ref_output_c.get(), true, atol, rtol);
  compareResults("output_t", output, ref_output_t.get(), false, atol, rtol);
}


template <typename InputType>
void performPackedBytewiseTest(const size_t N, const size_t H) {
  using namespace test;
  Tensor input("input", std::vector<size_t>{N, H}, TypeInfo<InputType>::dtype);
  Tensor output("output", std::vector<size_t>{N, H}, DType::kFloat8E4M3, true, true);
  // 相邻值不同，用于检测 packed 字节顺序；包含舍入边界、符号零和饱和输入。
  const std::array<float, 12> values = {
      1.0f, -2.0f, 1.0625f, -1.1875f, 0.0f, -0.0f,
      0.0009765625f, -0.001953125f, 448.0f, -512.0f, 1.0627f, -1.0627f};
  auto *host_input = input.rowwise_cpu_dptr<InputType>();
  for (size_t i = 0; i < N * H; ++i) {
    // 不同 tile 使用不同位模式，避免周期性输入掩盖预取错位或漏处理尾部。
    const size_t pattern = (i * size_t{2654435761}) ^ (i >> 7) ^ (i >> 17);
    host_input[i] = InputType(values[pattern % values.size()]);
  }
  input.from_cpu();
  std::vector<fp8e4m3> ref_c(N * H), ref_t(N * H);
  for (float scale : {1.0f, 1.25f, 1.0627f, -1.25f, 0.0f, 0.001f}) {
    SCOPED_TRACE(::testing::Message() << "N=" << N << ", H=" << H << ", scale=" << scale);
    output.set_scale(scale);
    output.set_amax(0.0f);
    float ref_amax;
    compute_ref<InputType, fp8e4m3>(host_input, ref_c.data(), ref_t.data(), N, H,
                                   &ref_amax, scale);
    nvte_cast_transpose(input.data(), output.data(), 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    output.to_cpu();
    EXPECT_EQ(std::memcmp(output.rowwise_cpu_dptr<fp8e4m3>(), ref_c.data(), N * H), 0);
    EXPECT_EQ(std::memcmp(output.columnwise_cpu_dptr<fp8e4m3>(), ref_t.data(), N * H), 0);
    EXPECT_FLOAT_EQ(output.amax(), ref_amax);
  }
}

std::vector<std::pair<size_t, size_t>> test_cases = {{2048, 12288},
                                                     {768, 1024},
                                                     {256, 65536},
                                                     {65536, 128},
                                                     {256, 256},
                                                     {120, 2080},
                                                     {8, 8},
                                                     {1, 3221},       // Prime 456
                                                     {2333, 1},       // Prime 345
                                                     {1481, 677}};    // Primes 234, 123
}  // namespace

TEST(CastTransposePackedTest, E4M3Bytewise) {
  // 覆盖普通 RTC、静态尾部、双阶段窄 pipeline 和单阶段宽 pipeline。
  for (const auto &shape : std::vector<std::pair<size_t, size_t>>{
           {512, 512}, {512, 32}, {32, 32}, {127, 255}, {1024, 4096}, {8192, 4096}}) {
    performPackedBytewiseTest<float>(shape.first, shape.second);
    performPackedBytewiseTest<half>(shape.first, shape.second);
    performPackedBytewiseTest<nv_bfloat16>(shape.first, shape.second);
  }
}

TEST(CastTransposePackedTest, WidePrefetchBytewise) {
  // 覆盖两种单 slot 预取、总量阈值、非整除 persistent grid 的尾部。
  for (const auto &shape : std::vector<std::pair<size_t, size_t>>{
           {2048, 4096}, {8192, 4096}, {4096, 8192}, {8256, 4096}, {8192, 8192}}) {
    performPackedBytewiseTest<float>(shape.first, shape.second);
    performPackedBytewiseTest<half>(shape.first, shape.second);
    performPackedBytewiseTest<nv_bfloat16>(shape.first, shape.second);
  }
}

class CTTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                               transformer_engine::DType,
                                                               std::pair<size_t, size_t>>> {};

TEST_P(CTTestSuite, TestCastTranspose) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      // delayed tensor scaling
      performTest<InputType, OutputType>(size.first, size.second);
    );
  );
}



INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  CTTestSuite,
  ::testing::Combine(
      ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
      ::testing::ValuesIn(test::all_fp_types),
      ::testing::ValuesIn(test_cases)),
  [](const testing::TestParamInfo<CTTestSuite::ParamType>& info) {
    std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                       test::typeName(std::get<1>(info.param)) + "X" +
                       std::to_string(std::get<2>(info.param).first) + "X" +
                       std::to_string(std::get<2>(info.param).second);
    return name;
  });
