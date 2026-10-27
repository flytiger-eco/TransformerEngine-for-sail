/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The fused_rope and fused_qkv_rope forward and backward kernels pick their blockDim.y from the
// grid size, the head count and the head dimension -- and, for qkv, the number of output tensors
// walked per head -- replacing fixed `h < 16 ? 4 : 8` and `h <= 8 ? h : 8` heuristics. Nothing
// about those choices can change the result: a block owns exactly one (s, b) pair -- or one packed
// token on the THD linear-grid path -- and inside the block the head loop strides by blockDim.y
// while the rotary and copy-the-rest loops stride by blockDim.x, so every warp count covers the
// same elements and each thread lane computes the same values in the same order. The only shared
// state is the cos/sin scratch, whose entry i is sincosf(freqs[s_id * d2 + i]) -- a function of
// (i, s_id) alone, written before a __syncthreads and read-only afterwards. There is no
// cross-thread reduction anywhere in either family, so the output at (s_id, b_id) depends only on
// the input at (s_id, b_id) and on freqs[s_id].
//
// fused_qkv_rope needs one extra clause, because its block helper is called three times per block,
// once each for q, k and v. In the forward those three calls write three separate output tensors;
// in the backward they write one packed tensor at the disjoint per-head row offsets 0, q_split and
// q_split + k_split. Either way every output element is written by exactly one thread with no
// accumulation, and the v call passes freqs == nullptr so it neither writes nor reads the scratch.
// The argument above therefore carries over unchanged -- but a qkv block retires three
// __syncthreads() rather than one, which is where its element weight comes from.
//
// These tests pin that invariant, which is what makes the launch-shape tuning safe, and they pin
// both rules' decision boundaries so that a change to either shows up.
//
// The rules themselves are not measured here. Three sweeps of 820, 600 and 480 cells over
// warps_per_block in {1, 2, 4, 8, 16} x grid mode in {shipped, legacy, linear} x {fp32, bf16} x
// {forward, backward} found all 1900 measurements byte-identical, and scored the adopted rule over
// 324 shape groups at a 1.2279x geomean against the old heuristic with a worst cell of 0.9845x,
// against a per-shape oracle of 1.2384x. Two alternatives were measured and rejected: a tail-aware
// refinement of the wave term, which added 0.0018 to one sweep's geomean but turned 3 cells into
// regressions and introduced 2 more on a disjoint sweep; and a fixed cost per wave rather than per
// warp, which cannot select the fastest measured warp count in 25 of those 324 groups at any value
// of its constant. Measurements and the fit: docs/common_op_optimization.md, Op 8.
//
// The qkv pair was swept separately: 1640 measurements over 41 shapes x 2 dtypes x 2 directions x
// blockDim.y in {1..8, 12, 16}, every one byte-identical, scored over the 148 cells whose own
// shipped-vs-shipped controls reproduced to within 1%. The adopted rule reaches a 1.1272x geomean
// against the old heuristic, worst cell 0.9502x, against a per-shape oracle of 1.1293x, with two
// provable regressions -- both the same h = 3 shape, whose measurement contradicts itself across
// dtype. Its element weight is derived rather than fitted, from the three barriers per block, and
// leave-one-group-out cross-validation has the derived value beating a re-fitted one out of sample
// in 4 of 6 folds. Three alternatives were measured and rejected: the smooth, non-ceil wave term,
// at 1.0418x with a worst cell of 0.6779x; a fixed cost per wave rather than per warp, for which
// 10 of the 148 groups have no constant at all that selects their fastest measured warp count; and
// powers-of-two-only candidates, which cost 0.5-2.6% at h in {5, 7} and leave ten provable
// regressions at every hysteresis margin in [0, 0.30]. Measurements and the fit:
// docs/common_op_optimization.md, Op 8b.
//
// One qkv pin is a regression test for a defect that was in the *model* rather than the kernel: the
// fit once assumed a single device-wide resident-warp capacity shared by every candidate
// blockDim.y, when blocks_per_sm(w) = floor(W1 / w) only divides evenly at powers of two. The
// shipped C++ never had that bug -- it queries the occupancy per candidate -- but a model that
// does will silently disagree with it, and QkvCapacityIsQuantisedNotConstant is the pin that
// catches the assumption.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/fused_rope.h>

// The positive controls below ask the real launch-shape rule what it would pick, rather than
// restating its arithmetic, so this needs the internal header. That header pulls in common.h, which
// defines TRANSFORMER_ENGINE_TYPE_SWITCH_ALL differently from test_common.h's copy; hiding the
// internal one for the duration of the include keeps the redefinition warning out of the build.
// Nothing in this file uses the macro. It also brings transformer_engine::Tensor into scope, so the
// test fixture's Tensor is spelled test::Tensor throughout.
#pragma push_macro("TRANSFORMER_ENGINE_TYPE_SWITCH_ALL")
#undef TRANSFORMER_ENGINE_TYPE_SWITCH_ALL
#include "fused_rope/fused_rope_launch.h"
#pragma pop_macro("TRANSFORMER_ENGINE_TYPE_SWITCH_ALL")

#include "../test_common.h"

using namespace transformer_engine;

namespace {

// Deterministic, varied, finite input whose value depends on the index, so that an off-by-one in
// the s, b, h or d indexing shows up. Uniform random data would hide it far more often.
template <typename T>
void fillDeterministic(T *data, const size_t size, const uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < size; ++i) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const double u = static_cast<double>(state >> 11) / 9007199254740992.0;
    // Spread over roughly [-2, 2], well inside what bf16 and fp16 represent exactly enough for a
    // byte-exact comparison of two runs of the same arithmetic to be meaningful.
    data[i] = static_cast<T>(4.0 * u - 2.0);
  }
}

// freqs are angles, not activations: the kernel feeds them straight to sincosf. A different seed
// and a different range from the input, so that mixing the two operands up cannot go unnoticed.
void fillFreqs(float *data, const size_t size, const uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < size; ++i) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const double u = static_cast<double>(state >> 11) / 9007199254740992.0;
    data[i] = static_cast<float>(12.0 * u - 6.0);
  }
}

// The device-wide number of warps that can be resident at once. No kernel can exceed this, so it is
// the upper end of the capacity range the positive control sweeps; on the 39 SM, 2048 thread per SM
// part the measurements were taken on it is 2496.
size_t maxResidentWarps() {
  int device = 0;
  EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp prop{};
  EXPECT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  return static_cast<size_t>(prop.multiProcessorCount) *
         static_cast<size_t>(prop.maxThreadsPerMultiProcessor) / THREADS_PER_WARP;
}

// The lower end of that range. The rule is given `sm_count * blocks_per_sm * warps_per_block`, and
// cudaOccupancyMaxActiveBlocksPerMultiprocessor returns at least 1 for a kernel that launches at
// all, so a capacity below the SM count cannot be reported by any device this code runs on.
size_t smCount() {
  int device = 0;
  EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp prop{};
  EXPECT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  return static_cast<size_t>(prop.multiProcessorCount);
}

// blockDim.y the rule would pick if every candidate block shape could keep `capacity` warps
// resident. That is exactly the structure the derivation in fused_rope_launch.h claims: these
// kernels are register-limited and their register count does not depend on warps_per_block, so
// blocks_per_sm scales as 1/warps_per_block and the product is constant.
int pickAtCapacity(const size_t capacity, const int h, const int d, const int d2,
                   const size_t grid_blocks) {
  std::array<size_t, kFusedRopeWarpCandidates> table;
  table.fill(capacity);
  return fused_rope_select_warps(h, d, d2, grid_blocks, table.data());
}

// pow2_ceil(h) clamped at kFusedRopeMaxWarpsPerBlock, restated here only so the assertions can name
// the clamp. This is the one piece of the rule the tests duplicate, and they duplicate it as a
// bound rather than as a picked value.
int warpCapOf(const int h) {
  int cap = 1;
  while (cap < h && cap < kFusedRopeMaxWarpsPerBlock) cap <<= 1;
  return cap;
}

// Shapes and grid sizes for the ladder. The grid sizes straddle the `grid_blocks * w == capacity`
// wave boundaries the cost model's ceil() turns into cliffs, and the head counts straddle the
// points where `ceil(h / w)` stops falling: at d = 128 lanes_per_head is 4, so h = 8 saturates at
// w = 8, h = 32 at w = 16 clamped from 32, and h = 128 exercises the clamp itself. h = 1 and h = 3
// pin the two ends of the warp_cap clamp.
const std::vector<std::array<int, 4>> &shapeLadder() {
  static const std::vector<std::array<int, 4>> ladder = {
      {8, 128, 128, 1},     {8, 128, 128, 64},    {8, 128, 128, 256},   {8, 128, 128, 8192},
      {32, 128, 128, 1},    {32, 128, 128, 64},   {32, 128, 128, 256},  {32, 128, 128, 8192},
      {64, 128, 128, 1},    {64, 128, 128, 64},   {64, 128, 128, 256},  {64, 128, 128, 8192},
      {16, 64, 32, 1},      {16, 64, 32, 1024},   {128, 128, 64, 312},  {128, 128, 64, 313},
      {1, 128, 128, 4096},  {3, 32, 32, 1},
  };
  return ladder;
}

// Runs one rope forward or backward over a contiguous [s, b, h, d] (SBHD) or [b, s, h, d] (BSHD)
// tensor and returns the output. cu_seqlens and start_positions are passed as zero-element tensors,
// whose device pointer is null; that is what selects the SBHD/BSHD branch in the kernel and what
// the PyTorch binding passes for those formats.
template <typename T>
std::vector<T> runDense(const NVTE_QKV_Format fmt, const bool forward, const bool interleaved,
                        const size_t s, const size_t b, const size_t h, const size_t d,
                        const size_t d2, const std::vector<T> &host_in,
                        const std::vector<float> &host_freqs, const char *tag) {
  const size_t row = h * d;
  const size_t numel = s * b * row;
  // The kernel works off the raw pointer and the explicit strides, but allocate the format's own
  // layout so the shape metadata agrees with what the strides describe.
  const std::vector<size_t> shape = (fmt == NVTE_SBHD) ? std::vector<size_t>{s, b, h, d}
                                                       : std::vector<size_t>{b, s, h, d};
  test::Tensor input(tag, shape, test::TypeInfo<T>::dtype);
  test::Tensor output(tag, shape, test::TypeInfo<T>::dtype);
  test::Tensor freqs(tag, std::vector<size_t>{s, d2}, test::TypeInfo<float>::dtype);
  test::Tensor null_ivp(tag, std::vector<size_t>{0}, test::TypeInfo<int>::dtype);

  std::memcpy(input.rowwise_cpu_dptr<T>(), host_in.data(), numel * sizeof(T));
  std::memcpy(freqs.rowwise_cpu_dptr<float>(), host_freqs.data(), s * d2 * sizeof(float));
  input.from_cpu();
  freqs.from_cpu();

  // SBHD is [s, b, h, d] in memory, so s strides over the whole batch; BSHD is [b, s, h, d], so b
  // does. The launcher derives the output strides from s, b, h and d itself and always writes the
  // contiguous layout of the same format.
  const int stride_s = (fmt == NVTE_SBHD) ? static_cast<int>(b * row) : static_cast<int>(row);
  const int stride_b = (fmt == NVTE_SBHD) ? static_cast<int>(row) : static_cast<int>(s * row);

  if (forward) {
    nvte_fused_rope_forward(input.data(), null_ivp.data(), freqs.data(), null_ivp.data(),
                            output.data(), fmt, interleaved, /*cp_size=*/1, /*cp_rank=*/0,
                            static_cast<int>(s), static_cast<int>(b), static_cast<int>(h),
                            static_cast<int>(d), static_cast<int>(d2), stride_s, stride_b,
                            static_cast<int>(d), 1, nullptr);
  } else {
    nvte_fused_rope_backward(input.data(), null_ivp.data(), freqs.data(), null_ivp.data(),
                             output.data(), fmt, interleaved, /*cp_size=*/1, /*cp_rank=*/0,
                             static_cast<int>(s), static_cast<int>(b), static_cast<int>(h),
                             static_cast<int>(d), static_cast<int>(d2), stride_s, stride_b,
                             static_cast<int>(d), 1, nullptr);
  }
  cudaDeviceSynchronize();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

  output.to_cpu();
  const T *p = output.rowwise_cpu_dptr<T>();
  return std::vector<T>(p, p + numel);
}

// The comparison every dense test below makes: run the full tensor, then run a leading slice of it
// as a tensor in its own right, and require the two to agree byte for byte. The slice is a
// contiguous prefix in both formats, and it changes the grid -- dim3(s, b) for SBHD, dim3(s, b) for
// BSHD -- which is what makes the launcher pick a different blockDim.y for different entries of the
// ladder.
template <typename T>
void performPrefixTest(const NVTE_QKV_Format fmt, const bool forward, const bool interleaved,
                       const bool vary_s, const size_t s_full, const size_t b_full, const size_t h,
                       const size_t d, const size_t d2, const std::vector<size_t> &prefixes,
                       const char *tag) {
  const size_t row = h * d;
  const size_t full_numel = s_full * b_full * row;
  std::vector<T> host_in(full_numel);
  fillDeterministic(host_in.data(), host_in.size(), 0x9E3779B97F4A7C15ull);
  std::vector<float> host_freqs(s_full * d2);
  fillFreqs(host_freqs.data(), host_freqs.size(), 0xD1B54A32D192ED03ull);

  const std::vector<T> full =
      runDense<T>(fmt, forward, interleaved, s_full, b_full, h, d, d2, host_in, host_freqs, tag);
  ASSERT_EQ(full.size(), full_numel);

  size_t compared = 0;
  for (const size_t prefix : prefixes) {
    const size_t ps = vary_s ? prefix : s_full;
    const size_t pb = vary_s ? b_full : prefix;
    // The ladders are shared between cases of different full sizes; entries that do not fit this
    // one are simply not prefixes of it.
    if (ps > s_full || pb > b_full) continue;
    const size_t numel = ps * pb * row;
    std::vector<T> sub_in(host_in.begin(), host_in.begin() + numel);
    std::vector<float> sub_freqs(host_freqs.begin(), host_freqs.begin() + ps * d2);

    const std::vector<T> got =
        runDense<T>(fmt, forward, interleaved, ps, pb, h, d, d2, sub_in, sub_freqs, tag);
    ASSERT_EQ(got.size(), numel);

    // Bit-exact, not tolerance-based: blockDim.y cannot change any arithmetic, so anything other
    // than identical bytes is a defect.
    EXPECT_EQ(0, std::memcmp(got.data(), full.data(), numel * sizeof(T)))
        << tag << ": output depends on the launch shape at s=" << ps << " b=" << pb;

    // Negative control. The comparison above is byte-exact, so it would also catch an s, b, h or d
    // indexing error rather than passing vacuously. Shifting the expected data by one (s, b) row
    // has to change it.
    if (numel < full_numel) {
      EXPECT_NE(0, std::memcmp(got.data(), full.data() + row, numel * sizeof(T)))
          << tag << ": negative control failed at s=" << ps << " b=" << pb;
    }
    ++compared;
  }
  // Guard against the ladder being filtered down to nothing by the size check above, which would
  // make the whole test vacuous.
  EXPECT_GE(compared, 8u) << tag << ": the prefix ladder does not fit this shape";
}

// s prefixes for a full [512, 8, 32, 128] SBHD tensor. The grid is s * 8 blocks, so these cross
// every `grid * w == capacity` boundary for capacities near 2496: s = 19 is the last one whose
// 16-warp launch still fits, s = 39 for 8 warps, s = 78 for 4, s = 156 for 2 and s = 312 for 1.
// Values just below and just above each of those are in the list.
std::vector<size_t> sPrefixes() {
  return {1,   2,   3,   7,   8,   9,   16,  19,  20,  31,  32,  39,  40,
          64,  78,  79,  100, 128, 156, 157, 200, 256, 312, 313, 384, 511, 512};
}

// b prefixes for a full [64, 128, 8, 128] BSHD tensor. The grid is 64 * b blocks, so this crosses
// the same boundaries at b = 19, 39, 78 for capacities near 2496, and h = 8 also pins the rule's
// warp_cap clamp to 8.
std::vector<size_t> bPrefixes() {
  return {1,  2,  3,  4,  8,   9,   16,  19,  20,  31,  32,  39,
          40, 64, 78, 79, 100, 120, 127, 128};
}

// ---------------------------------------------------------------------------------------------
// fused_qkv_rope
// ---------------------------------------------------------------------------------------------

// The part the Op 8b sweep was measured on: 39 SMs, and fused_qkv_rope kernels that keep 64
// one-warp blocks resident per SM. Both limits agree at 64 there -- 131072 registers per SM over
// 32 lanes times 64 registers per thread, and 2048 threads per SM over 32 lanes -- which is what
// makes blocks_per_sm(w) == floor(64 / w) for every candidate.
constexpr size_t kQkvMeasuredSmCount = 39;
constexpr size_t kQkvMeasuredWarpsPerSm = 64;

// Device-wide resident warps at a given blockDim.y. floor(W1 / w) only divides evenly at powers of
// two, so capacity(w) = sm_count * floor(W1 / w) * w is 2496 at w = 1, 2, 4, 8 and 16 on the
// measured part and strictly less at every other blockDim.y: 2457 at w = 3 and 7, 2340 at w = 5, 6
// and 12. The shipped rule obtains this per candidate from the occupancy query; the pins below
// restate the measured values so the arithmetic can be asserted on any device.
size_t quantisedCapacity(const size_t sm_count, const size_t warps_per_sm, const int warps) {
  const size_t w = static_cast<size_t>(warps);
  return sm_count * (warps_per_sm / w) * w;
}

size_t measuredCapacity(const int warps) {
  return quantisedCapacity(kQkvMeasuredSmCount, kQkvMeasuredWarpsPerSm, warps);
}

// The same quantisation evaluated on whatever part the test runs on. maxThreadsPerMultiProcessor /
// 32 is an upper bound on W1 -- the register file can bind lower -- so this over-estimates the
// capacity, which is the right bias for a positive control: the ladders are required to span the
// rule's buckets at the bound, and a part with less capacity only moves the boundaries down.
size_t localCapacityBound(const int warps) {
  int device = 0;
  EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp prop{};
  EXPECT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  return quantisedCapacity(static_cast<size_t>(prop.multiProcessorCount),
                           static_cast<size_t>(prop.maxThreadsPerMultiProcessor) / THREADS_PER_WARP,
                           warps);
}

// One capacity for every candidate: the assumption a model must not make. Kept only so that
// QkvCapacityIsQuantisedNotConstant can show it changes the pick.
size_t uniformCapacity(const int) { return kQkvMeasuredSmCount * kQkvMeasuredWarpsPerSm; }

using CapacityFn = size_t (*)(int);

// blockDim.y the shipped qkv rule picks, given how to obtain each candidate's capacity. Asks the
// real fused_rope_qkv_candidates and fused_rope_select_warps_qkv rather than restating them.
int pickQkvWith(const CapacityFn cap, const int h, const int d, const int d2, const int segments,
                const size_t grid_blocks) {
  int cand[kFusedRopeQkvWarpCandidates];
  const int n = fused_rope_qkv_candidates(h, cand);
  size_t capacity[kFusedRopeQkvWarpCandidates];
  for (int i = 0; i < n; ++i) capacity[i] = cap(cand[i]);
  return fused_rope_select_warps_qkv(h, d, d2, segments, grid_blocks, n, cand, capacity);
}

int pickQkv(const int h, const int d, const int d2, const int segments, const size_t grid_blocks) {
  return pickQkvWith(measuredCapacity, h, d, d2, segments, grid_blocks);
}

int pickQkvUniform(const int h, const int d, const int d2, const int segments,
                   const size_t grid_blocks) {
  return pickQkvWith(uniformCapacity, h, d, d2, segments, grid_blocks);
}

// The modelled cost of every candidate, in candidate order, from the same entry point the rule
// uses. The picks below carry their costs as trailing comments; this exists so the hysteresis test
// can assert the margin as a number rather than imply it from a pick.
std::vector<uint64_t> qkvCosts(const int h, const int d, const int d2, const int segments,
                               const size_t grid_blocks) {
  int cand[kFusedRopeQkvWarpCandidates];
  const int n = fused_rope_qkv_candidates(h, cand);
  int elems[kFusedRopeQkvWarpCandidates];
  size_t capacity[kFusedRopeQkvWarpCandidates];
  for (int i = 0; i < n; ++i) {
    elems[i] = fused_rope_qkv_elems_per_thread(h, d, d2, segments, cand[i]);
    capacity[i] = measuredCapacity(cand[i]);
  }
  std::vector<uint64_t> cost(n);
  fused_rope_select_from_candidates(n, cand, capacity, elems, kFusedRopeQkvWarpWeight,
                                    kFusedRopeQkvElemWeight, grid_blocks,
                                    fused_rope_qkv_fixed_warps(h), cost.data());
  return cost;
}

// True when the incumbent -- the candidate the hysteresis protects -- is inside the rule's margin
// of the argmin, i.e. when the rule keeps the shape that has been running rather than the model's
// pick.
bool qkvHysteresisFires(const std::vector<uint64_t> &cost, const size_t incumbent_index) {
  const uint64_t best = *std::min_element(cost.begin(), cost.end());
  return cost[incumbent_index] * kFusedRopeQkvHysteresisDen <= best * kFusedRopeQkvHysteresisNum;
}

// The largest blockDim.y the qkv candidate list can contain, i.e. pow2_ceil(h) under the same
// clamp. Restated as a bound so the assertions can name it rather than recompute it.
int qkvWarpCapOf(const int h) {
  int cap = 1;
  while (cap < h && cap < kFusedRopeMaxWarpsPerBlock) cap <<= 1;
  return cap;
}

// One qkv shape for the ladders. nq, nk and nv are the per-head row lengths of the three outputs in
// units of d, so segments == nq + nk + nv and the packed input row is (nq + nk + nv) * d long.
// tests/pytorch/test_fused_rope.py:469 passes qkv_split_arg_list = [hidden_size * 4, hidden_size,
// hidden_size], i.e. nq = 4 and nk = nv = 1: that is the seg = 6 row, and it is why the segments
// factor exists at all.
//
// This table is the single source of truth for both the positive control and the bit-exactness
// tests, so a ladder that stops crossing the rule's buckets cannot silently stop being exercised.
struct QkvLadder {
  const char *tag;
  NVTE_QKV_Format fmt;
  bool interleaved;
  DType dtype;
  int h, d, d2;
  int nq, nk, nv;
  bool vary_s;  // false varies b instead, which is the BSHD stride order
  size_t s_full, b_full;
  const std::vector<size_t> *prefixes;

  int segments() const { return nq + nk + nv; }
  size_t total_d() const { return static_cast<size_t>(segments()) * static_cast<size_t>(d); }
};

// Every ladder spans at least three distinct blockDim.y over its prefix list on the measured part;
// QkvPrefixLaddersSpanMultipleWarpCounts asserts that on the part actually running the test,
// because a bit-exactness comparison that never changes the launch shape proves nothing.
const std::vector<QkvLadder> &qkvLadders() {
  static const std::vector<size_t> s_prefixes = sPrefixes();
  static const std::vector<size_t> b_prefixes = bPrefixes();
  constexpr DType kBF16 = DType::kBFloat16, kFP32 = DType::kFloat32, kFP16 = DType::kFloat16;
  static const std::vector<QkvLadder> ladders = {
      // seg 6 is the shape the PyTorch caller actually produces, at the head count and dimension a
      // GQA qkv projection gives: h = 32 heads of d = 128, q four times as wide as k and v.
      {"qkv_sbhd_bf16_seg6", NVTE_SBHD, false, kBF16, 32, 128, 128, 4, 1, 1, true, 256, 4,
       &s_prefixes},
      // seg 3 drops the GQA widening: q, k and v all one d wide.
      {"qkv_sbhd_bf16_seg3", NVTE_SBHD, false, kBF16, 32, 128, 128, 1, 1, 1, true, 256, 8,
       &s_prefixes},
      // seg 10 at d = 64 pushes segments up while lanes_per_head falls to 2, so E(w) moves in the
      // other direction from the seg 3 row above.
      {"qkv_sbhd_bf16_seg10_d64", NVTE_SBHD, false, kBF16, 8, 64, 64, 8, 1, 1, true, 128, 32,
       &s_prefixes},
      // rotary_percent 0.5, which takes the `d > d2` copy-the-rest loop: a second strided loop over
      // blockDim.y that has to be equally shape-independent.
      {"qkv_sbhd_bf16_d2half", NVTE_SBHD, false, kBF16, 32, 128, 64, 4, 1, 1, true, 256, 4,
       &s_prefixes},
      // h = 3, 5 and 7 are the head counts where the candidate list is not powers of two, because
      // fused_rope_qkv_fixed_warps(h) == h there. These are the only h at which the resident-warp
      // capacity is quantised rather than constant, and they carry the rule's two provable
      // regressions.
      {"qkv_sbhd_bf16_h3_seg6", NVTE_SBHD, false, kBF16, 3, 128, 128, 4, 1, 1, true, 512, 8,
       &s_prefixes},
      {"qkv_sbhd_bf16_h5_seg6", NVTE_SBHD, false, kBF16, 5, 128, 128, 4, 1, 1, true, 512, 8,
       &s_prefixes},
      {"qkv_sbhd_bf16_h7_seg6", NVTE_SBHD, false, kBF16, 7, 128, 128, 4, 1, 1, true, 512, 8,
       &s_prefixes},
      // BSHD swaps which dimension strides over the batch, and interleaved swaps the rotate partner
      // from d_id +/- d2/2 to d_id +/- 1. h = 8 also pins the warp_cap clamp.
      {"qkv_bshd_bf16_interleaved_seg6", NVTE_BSHD, true, kBF16, 8, 128, 128, 4, 1, 1, false, 64,
       64, &b_prefixes},
      // The other two instantiations the type switch reaches.
      {"qkv_sbhd_fp32_seg3_d2half", NVTE_SBHD, false, kFP32, 16, 64, 32, 1, 1, 1, true, 128, 4,
       &s_prefixes},
      {"qkv_sbhd_fp16_seg6_d64", NVTE_SBHD, false, kFP16, 16, 64, 64, 4, 1, 1, true, 128, 4,
       &s_prefixes},
  };
  return ladders;
}

// The three forward outputs, kept separate so that each can carry its own negative control: q and k
// are roped with different angles and v is a straight copy, so an error that only touched one of
// the three would be invisible in a concatenation.
template <typename T>
struct QkvOut {
  std::vector<T> q, k, v;
};

template <typename T>
std::vector<T> detach(test::Tensor &t, const size_t n) {
  const T *p = t.rowwise_cpu_dptr<T>();
  return std::vector<T>(p, p + n);
}

// Contiguous [s, b, h, row] for SBHD and [b, s, h, row] for BSHD. The qkv kernels take no strides
// at all -- they derive every offset from s, b, h and the three split arguments -- so the buffers
// have to be in exactly this layout.
std::vector<size_t> qkvShape(const NVTE_QKV_Format fmt, const size_t s, const size_t b,
                             const size_t h, const size_t row) {
  return (fmt == NVTE_SBHD) ? std::vector<size_t>{s, b, h, row}
                            : std::vector<size_t>{b, s, h, row};
}

// One fused_qkv_rope forward over a packed input whose per-(s, b, h) row is q || k || v.
// start_positions is a zero-element tensor, whose device pointer is null; that is what makes the
// kernel take its begin_offset = 0 branch, and what the PyTorch binding passes for None.
template <typename T>
QkvOut<T> runQkvForwardDense(const NVTE_QKV_Format fmt, const bool interleaved, const size_t s,
                             const size_t b, const size_t h, const size_t d, const size_t d2,
                             const size_t nq, const size_t nk, const size_t nv,
                             const std::vector<T> &host_in,
                             const std::vector<float> &host_q_freqs,
                             const std::vector<float> &host_k_freqs, const char *tag) {
  const size_t q_split = nq * d, k_split = nk * d, v_split = nv * d;
  const size_t total_d = q_split + k_split + v_split;
  test::Tensor input(tag, qkvShape(fmt, s, b, h, total_d), test::TypeInfo<T>::dtype);
  test::Tensor q(tag, qkvShape(fmt, s, b, h, q_split), test::TypeInfo<T>::dtype);
  test::Tensor k(tag, qkvShape(fmt, s, b, h, k_split), test::TypeInfo<T>::dtype);
  test::Tensor v(tag, qkvShape(fmt, s, b, h, v_split), test::TypeInfo<T>::dtype);
  test::Tensor qf(tag, std::vector<size_t>{s, d2}, test::TypeInfo<float>::dtype);
  test::Tensor kf(tag, std::vector<size_t>{s, d2}, test::TypeInfo<float>::dtype);
  test::Tensor null_ivp(tag, std::vector<size_t>{0}, test::TypeInfo<int>::dtype);

  std::memcpy(input.rowwise_cpu_dptr<T>(), host_in.data(), s * b * h * total_d * sizeof(T));
  std::memcpy(qf.rowwise_cpu_dptr<float>(), host_q_freqs.data(), s * d2 * sizeof(float));
  std::memcpy(kf.rowwise_cpu_dptr<float>(), host_k_freqs.data(), s * d2 * sizeof(float));
  input.from_cpu();
  qf.from_cpu();
  kf.from_cpu();

  nvte_fused_qkv_rope_forward(input.data(), qf.data(), kf.data(), null_ivp.data(), q.data(),
                              k.data(), v.data(), fmt, interleaved, /*cp_size=*/1, /*cp_rank=*/0,
                              static_cast<int>(s), static_cast<int>(b), static_cast<int>(h),
                              static_cast<int>(d), static_cast<int>(d2),
                              static_cast<int>(q_split), static_cast<int>(k_split),
                              static_cast<int>(v_split), nullptr);
  cudaDeviceSynchronize();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

  q.to_cpu();
  k.to_cpu();
  v.to_cpu();
  QkvOut<T> out;
  out.q = detach<T>(q, s * b * h * q_split);
  out.k = detach<T>(k, s * b * h * k_split);
  out.v = detach<T>(v, s * b * h * v_split);
  return out;
}

// One fused_qkv_rope backward: three separate grad tensors in, one packed grad out, whose per-head
// row is q || k || v at the disjoint offsets 0, q_split and q_split + k_split.
template <typename T>
std::vector<T> runQkvBackwardDense(const NVTE_QKV_Format fmt, const bool interleaved,
                                   const size_t s, const size_t b, const size_t h, const size_t d,
                                   const size_t d2, const size_t nq, const size_t nk,
                                   const size_t nv, const std::vector<T> &host_q,
                                   const std::vector<T> &host_k, const std::vector<T> &host_v,
                                   const std::vector<float> &host_q_freqs,
                                   const std::vector<float> &host_k_freqs, const char *tag) {
  const size_t q_split = nq * d, k_split = nk * d, v_split = nv * d;
  const size_t total_d = q_split + k_split + v_split;
  test::Tensor q(tag, qkvShape(fmt, s, b, h, q_split), test::TypeInfo<T>::dtype);
  test::Tensor k(tag, qkvShape(fmt, s, b, h, k_split), test::TypeInfo<T>::dtype);
  test::Tensor v(tag, qkvShape(fmt, s, b, h, v_split), test::TypeInfo<T>::dtype);
  test::Tensor out(tag, qkvShape(fmt, s, b, h, total_d), test::TypeInfo<T>::dtype);
  test::Tensor qf(tag, std::vector<size_t>{s, d2}, test::TypeInfo<float>::dtype);
  test::Tensor kf(tag, std::vector<size_t>{s, d2}, test::TypeInfo<float>::dtype);

  std::memcpy(q.rowwise_cpu_dptr<T>(), host_q.data(), s * b * h * q_split * sizeof(T));
  std::memcpy(k.rowwise_cpu_dptr<T>(), host_k.data(), s * b * h * k_split * sizeof(T));
  std::memcpy(v.rowwise_cpu_dptr<T>(), host_v.data(), s * b * h * v_split * sizeof(T));
  std::memcpy(qf.rowwise_cpu_dptr<float>(), host_q_freqs.data(), s * d2 * sizeof(float));
  std::memcpy(kf.rowwise_cpu_dptr<float>(), host_k_freqs.data(), s * d2 * sizeof(float));
  q.from_cpu();
  k.from_cpu();
  v.from_cpu();
  qf.from_cpu();
  kf.from_cpu();

  nvte_fused_qkv_rope_backward(q.data(), k.data(), v.data(), qf.data(), kf.data(), out.data(), fmt,
                               interleaved, /*cp_size=*/1, /*cp_rank=*/0, static_cast<int>(s),
                               static_cast<int>(b), static_cast<int>(h), static_cast<int>(d),
                               static_cast<int>(d2), static_cast<int>(q_split),
                               static_cast<int>(k_split), static_cast<int>(v_split), nullptr);
  cudaDeviceSynchronize();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

  out.to_cpu();
  return detach<T>(out, s * b * h * total_d);
}

// The comparison every qkv bit-exactness test makes: run the full tensor, then run a leading slice
// of it as a tensor in its own right, and require the two to agree byte for byte. A leading slice
// is a contiguous prefix in both formats, and it changes the grid -- dim3(s, b) either way --
// which is what makes the launcher pick a different blockDim.y for different ladder entries.
template <typename T>
void performQkvPrefixTest(const bool forward, const QkvLadder &l, const char *tag) {
  const NVTE_QKV_Format fmt = l.fmt;
  const bool interleaved = l.interleaved;
  const size_t h = l.h, d = l.d, d2 = l.d2;
  const size_t q_split = l.nq * d, k_split = l.nk * d, v_split = l.nv * d;
  const size_t total_d = q_split + k_split + v_split;
  const size_t s_full = l.s_full, b_full = l.b_full;

  // Distinct seeds per operand. q and k are roped with different angles and v is copied, so sharing
  // a seed would let the block helper read the wrong frequency tensor, or the wrong output segment,
  // and still produce the expected bytes.
  std::vector<float> host_q_freqs(s_full * d2), host_k_freqs(s_full * d2);
  fillFreqs(host_q_freqs.data(), host_q_freqs.size(), 0xD1B54A32D192ED03ull);
  fillFreqs(host_k_freqs.data(), host_k_freqs.size(), 0xC2B2AE3D27D4EB4Full);

  // The forward takes one packed buffer, so host_a carries q || k || v and there is nothing else to
  // fill. The backward takes three separate grads, so host_a is the q grad and host_b, host_c are k
  // and v.
  std::vector<T> host_a(s_full * b_full * h * (forward ? total_d : q_split));
  fillDeterministic(host_a.data(), host_a.size(), 0x9E3779B97F4A7C15ull);
  std::vector<T> host_b, host_c;
  if (!forward) {
    host_b.resize(s_full * b_full * h * k_split);
    host_c.resize(s_full * b_full * h * v_split);
    fillDeterministic(host_b.data(), host_b.size(), 0xBF58476D1CE4E5B9ull);
    fillDeterministic(host_c.data(), host_c.size(), 0x94D049BB133111EBull);
  }

  const size_t row_q = h * q_split, row_k = h * k_split, row_v = h * v_split;
  const size_t row_in = h * total_d;

  QkvOut<T> full_fwd;
  std::vector<T> full_bwd;
  if (forward) {
    full_fwd = runQkvForwardDense<T>(fmt, interleaved, s_full, b_full, h, d, d2, l.nq, l.nk, l.nv,
                                     host_a, host_q_freqs, host_k_freqs, tag);
    ASSERT_EQ(full_fwd.q.size(), s_full * b_full * row_q);
    ASSERT_EQ(full_fwd.k.size(), s_full * b_full * row_k);
    ASSERT_EQ(full_fwd.v.size(), s_full * b_full * row_v);
  } else {
    full_bwd = runQkvBackwardDense<T>(fmt, interleaved, s_full, b_full, h, d, d2, l.nq, l.nk, l.nv,
                                      host_a, host_b, host_c, host_q_freqs, host_k_freqs, tag);
    ASSERT_EQ(full_bwd.size(), s_full * b_full * row_in);
  }

  size_t compared = 0;
  for (const size_t prefix : *l.prefixes) {
    const size_t ps = l.vary_s ? prefix : s_full;
    const size_t pb = l.vary_s ? b_full : prefix;
    // The prefix ladders are shared between ladders of different full sizes; entries that do not
    // fit this one are simply not prefixes of it.
    if (ps > s_full || pb > b_full) continue;
    const size_t n = ps * pb;

    if (forward) {
      const std::vector<T> sub_a(host_a.begin(), host_a.begin() + n * row_in);
      const std::vector<float> sub_qf(host_q_freqs.begin(), host_q_freqs.begin() + ps * d2);
      const std::vector<float> sub_kf(host_k_freqs.begin(), host_k_freqs.begin() + ps * d2);
      const QkvOut<T> got = runQkvForwardDense<T>(fmt, interleaved, ps, pb, h, d, d2, l.nq, l.nk,
                                                  l.nv, sub_a, sub_qf, sub_kf, tag);
      ASSERT_EQ(got.q.size(), n * row_q);
      ASSERT_EQ(got.k.size(), n * row_k);
      ASSERT_EQ(got.v.size(), n * row_v);

      // Bit-exact, not tolerance-based: blockDim.y cannot change any arithmetic, so anything other
      // than identical bytes is a defect.
      EXPECT_EQ(0, std::memcmp(got.q.data(), full_fwd.q.data(), n * row_q * sizeof(T)))
          << tag << ": q depends on the launch shape at s=" << ps << " b=" << pb;
      EXPECT_EQ(0, std::memcmp(got.k.data(), full_fwd.k.data(), n * row_k * sizeof(T)))
          << tag << ": k depends on the launch shape at s=" << ps << " b=" << pb;
      EXPECT_EQ(0, std::memcmp(got.v.data(), full_fwd.v.data(), n * row_v * sizeof(T)))
          << tag << ": v depends on the launch shape at s=" << ps << " b=" << pb;

      // Negative control, one per output. The comparisons above are byte-exact, so they would also
      // catch an s, b, h or d indexing error rather than passing vacuously; shifting the expected
      // data by one (s, b) row has to change each of them.
      if (n < s_full * b_full) {
        EXPECT_NE(0, std::memcmp(got.q.data(), full_fwd.q.data() + row_q, n * row_q * sizeof(T)))
            << tag << ": q negative control failed at s=" << ps << " b=" << pb;
        EXPECT_NE(0, std::memcmp(got.k.data(), full_fwd.k.data() + row_k, n * row_k * sizeof(T)))
            << tag << ": k negative control failed at s=" << ps << " b=" << pb;
        EXPECT_NE(0, std::memcmp(got.v.data(), full_fwd.v.data() + row_v, n * row_v * sizeof(T)))
            << tag << ": v negative control failed at s=" << ps << " b=" << pb;
      }
    } else {
      const std::vector<T> sub_a(host_a.begin(), host_a.begin() + n * row_q);
      const std::vector<T> sub_b(host_b.begin(), host_b.begin() + n * row_k);
      const std::vector<T> sub_c(host_c.begin(), host_c.begin() + n * row_v);
      const std::vector<float> sub_qf(host_q_freqs.begin(), host_q_freqs.begin() + ps * d2);
      const std::vector<float> sub_kf(host_k_freqs.begin(), host_k_freqs.begin() + ps * d2);
      const std::vector<T> got =
          runQkvBackwardDense<T>(fmt, interleaved, ps, pb, h, d, d2, l.nq, l.nk, l.nv, sub_a, sub_b,
                                 sub_c, sub_qf, sub_kf, tag);
      ASSERT_EQ(got.size(), n * row_in);
      EXPECT_EQ(0, std::memcmp(got.data(), full_bwd.data(), n * row_in * sizeof(T)))
          << tag << ": qkv grad depends on the launch shape at s=" << ps << " b=" << pb;
      if (n < s_full * b_full) {
        EXPECT_NE(0, std::memcmp(got.data(), full_bwd.data() + row_in, n * row_in * sizeof(T)))
            << tag << ": negative control failed at s=" << ps << " b=" << pb;
      }
    }
    ++compared;
  }
  // Guard against the prefix list being filtered down to nothing by the size check above, which
  // would make the whole test vacuous.
  EXPECT_GE(compared, 8u) << tag << ": the prefix ladder does not fit this shape";
}

// Both directions of one ladder at its own dtype. The dtype lives in the table rather than in the
// call site so that the positive control and the bit-exactness tests cannot drift apart.
void runQkvLadder(const QkvLadder &l) {
  const std::string fwd = std::string(l.tag) + "_fwd";
  const std::string bwd = std::string(l.tag) + "_bwd";
  switch (l.dtype) {
    case DType::kFloat32:
      performQkvPrefixTest<float>(/*forward=*/true, l, fwd.c_str());
      performQkvPrefixTest<float>(/*forward=*/false, l, bwd.c_str());
      break;
    case DType::kFloat16:
      performQkvPrefixTest<__half>(/*forward=*/true, l, fwd.c_str());
      performQkvPrefixTest<__half>(/*forward=*/false, l, bwd.c_str());
      break;
    case DType::kBFloat16:
      performQkvPrefixTest<__nv_bfloat16>(/*forward=*/true, l, fwd.c_str());
      performQkvPrefixTest<__nv_bfloat16>(/*forward=*/false, l, bwd.c_str());
      break;
    default:
      FAIL() << l.tag << ": the qkv ladder names a dtype the type switch does not reach";
  }
}

}  // namespace

// Positive control. Every comparison below runs a prefix against the full tensor, so it can only
// fail if the launcher picks a different blockDim.y for different grid sizes. On a device where the
// whole ladder landed on one warp count those comparisons would pass without exercising the rule at
// all. This asks the rule itself, over every resident-warp capacity the device could possibly
// report, and requires the ladder to span at least three distinct warp counts for each one.
TEST(FusedRopeLaunchShape, ShapeLadderSpansMultipleWarpCounts) {
  const size_t max_warps = maxResidentWarps();
  const size_t min_capacity = smCount();
  ASSERT_GT(max_warps, static_cast<size_t>(kFusedRopeMaxWarpsPerBlock));
  ASSERT_GE(max_warps, min_capacity);

  size_t worst_span = SIZE_MAX;
  size_t worst_capacity = 0;
  for (size_t capacity = min_capacity; capacity <= max_warps; ++capacity) {
    std::set<int> picks;
    for (const auto &cell : shapeLadder()) {
      const int w =
          pickAtCapacity(capacity, cell[0], cell[1], cell[2], static_cast<size_t>(cell[3]));
      // Whatever the capacity, the rule must stay inside its documented range and never exceed the
      // point where ceil(h / w) has saturated.
      EXPECT_GE(w, 1) << "capacity=" << capacity;
      EXPECT_LE(w, warpCapOf(cell[0])) << "capacity=" << capacity << " h=" << cell[0];
      EXPECT_LE(w, kFusedRopeMaxWarpsPerBlock) << "capacity=" << capacity;
      picks.insert(w);
    }
    if (picks.size() < worst_span) {
      worst_span = picks.size();
      worst_capacity = capacity;
    }
  }
  EXPECT_GE(worst_span, 3u)
      << "at a resident warp capacity of " << worst_capacity
      << " the ladder does not cross the rule's buckets on this device, so the prefix tests below "
         "would pass vacuously";
}

// Positive control for the ladders the bit-exactness tests actually use. Those fix h, d and d2 and
// vary only the grid, so whether they cross the rule's buckets depends on the device's resident
// warp capacity -- which a test cannot query, because cuda::kernel_blocks_per_sm is internal to
// libtransformer_engine and not in its version script. It is bounded above, though: no kernel can
// keep more than sm_count * maxThreadsPerSM / 32 warps resident, and on the 39 SM, 2048 thread per
// SM part the measurements were taken on that bound is 2496, which is also what the forward kernels
// actually reach (64 registers per thread, so 64 one-warp blocks per SM). Requiring the ladders to
// span buckets at the bound therefore pins the case that was measured; a device with less capacity
// only moves the boundaries down, and the ladder entries straddle a wide enough range of grids to
// follow them.
TEST(FusedRopeLaunchShape, PrefixLaddersSpanMultipleWarpCounts) {
  const size_t capacity = maxResidentWarps();
  ASSERT_GT(capacity, static_cast<size_t>(kFusedRopeMaxWarpsPerBlock));

  struct Ladder {
    const char *tag;
    int h, d, d2;
    size_t grid_scale;  // grid_blocks == grid_scale * prefix
    size_t full;
    const std::vector<size_t> *prefixes;
  };
  const std::vector<size_t> s_prefixes = sPrefixes();
  const std::vector<size_t> b_prefixes = bPrefixes();
  const std::vector<Ladder> ladders = {
      {"sbhd_bf16", 32, 128, 128, 8, 512, &s_prefixes},
      {"sbhd_bf16_d2half", 32, 128, 64, 8, 512, &s_prefixes},
      {"bshd_bf16_interleaved", 8, 128, 128, 64, 128, &b_prefixes},
      {"sbhd_fp32", 16, 64, 32, 4, 128, &s_prefixes},
      {"sbhd_fp16", 16, 64, 64, 4, 128, &s_prefixes},
  };

  for (const Ladder &l : ladders) {
    std::set<int> picks;
    for (const size_t prefix : *l.prefixes) {
      if (prefix > l.full) continue;
      picks.insert(pickAtCapacity(capacity, l.h, l.d, l.d2, l.grid_scale * prefix));
    }
    EXPECT_GE(picks.size(), 3u)
        << l.tag << ": the prefix ladder does not cross the rule's buckets at a resident warp "
                      "capacity of "
        << capacity << ", so its bit-exactness comparisons would pass vacuously";
  }
}

// The cost model, pinned regime by regime. cost(w) = waves(w) * (w + kFusedRopeElemWeight * E(w)),
// waves(w) = ceil(grid_blocks * w / capacity), E(w) = ceil(h / w) * lanes_per_head. The trailing
// comments are the modelled costs of the candidates in the order w = 1, 2, 4, 8, 16, so that
// changing the weight shows up here as a changed pick instead of as a silent re-fit.
TEST(FusedRopeLaunchShape, TermsAndClampsArePinned) {
  // E(w): lanes_per_head is DIVUP(d2, 32) plus DIVUP(d - d2, 32) when d > d2, times ceil(h / w).
  // d = 128 with d2 = 128 and d2 = 64 both give 4 lanes, which is why rotary_percent does not move
  // E at that head dimension.
  EXPECT_EQ(fused_rope_elems_per_thread(8, 128, 128, 1), 32);
  EXPECT_EQ(fused_rope_elems_per_thread(64, 128, 128, 4), 64);
  EXPECT_EQ(fused_rope_elems_per_thread(32, 128, 64, 2), 64);
  EXPECT_EQ(fused_rope_elems_per_thread(16, 64, 32, 1), 32);
  EXPECT_EQ(fused_rope_elems_per_thread(1, 128, 128, 16), 4);
  EXPECT_EQ(kFusedRopeElemWeight, 5);

  // Capacities of the device the sweep was measured on: 39 SMs, 64 registers per thread forward
  // (64 one-warp blocks per SM) and 80 backward (48). blocks_per_sm * warps_per_block is constant,
  // which is what makes the whole device capacity a single number per kernel.
  const size_t fwd_capacity = 2496;
  const size_t bwd_capacity = 1872;

  // One partial wave. waves(w) is 1 for every candidate, so only w + 5 * E(w) is left and E falls
  // faster than w rises: the pick is the warp_cap clamp. This is the s = 1, b = 1, h = 32 regime
  // the sweep measured at 0.21x for w = 1 and 1.38x for w = 16 against the old heuristic.
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 32, 128, 128, 1), 16);  // 641 322 164 88 56
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 8, 128, 128, 1), 8);    // 161 82 44 28
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 3, 128, 128, 1), 4);    // 61 42 24
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 16, 64, 32, 1), 16);    // 161 82 44 28 26

  // grid_blocks == capacity, so w = 1 is exactly one wave and every larger w opens another. The
  // barrier term then has nothing to trade against and w = 1 wins at every head count.
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 8, 128, 128, fwd_capacity), 1);   // 161 164 176 224
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 32, 128, 128, fwd_capacity), 1);  // 641 644 656 704 896
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 64, 128, 128, fwd_capacity), 1);

  // A ceil() cliff. h = 128 at d2 = 64 has E(8) = 64 and E(16) = 32; 312 blocks still fit one wave
  // at w = 8 but 313 does not, and the extra wave costs more than the halved E saves.
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 128, 128, 64, 312), 8);   // 2561 1282 644 328 352
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 128, 128, 64, 313), 16);  // 2561 1282 644 656 528

  // Interior optimum, and the cell that disproved the two-term rule this replaced. At 1024 blocks
  // on the backward pass w = 1 costs 1 wave and w = 8 costs 5, yet the sweep measures w = 1 at
  // 0.72x-0.82x of w = 8 and w = 16 at 0.83x-0.87x, so neither "stay inside one wave" nor "shrink
  // E to a fixed target" can select w = 8. The barrier term can: 440 against 641 and 504.
  EXPECT_EQ(pickAtCapacity(bwd_capacity, 32, 128, 128, 1024), 8);   // 641 644 492 440 504
  EXPECT_EQ(pickAtCapacity(bwd_capacity, 32, 128, 128, 2048), 8);   // 1282 966 820 792 1008
  EXPECT_EQ(pickAtCapacity(bwd_capacity, 64, 128, 128, 512), 16);   // 1281 642 648 504 480
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 64, 128, 128, 16384), 4);  // 8967 8988 8748 8904 10176
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 32, 128, 64, 4096), 4);    // 1282 1288 1148 1232 1512
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 16, 64, 32, 1024), 2);     // 161 82 88 112 182

  // Many waves. waves(w) becomes proportional to w, so the cost grows like w^2 plus a constant and
  // the pick collapses to w = 1.
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 64, 128, 128, 131072), 1);  // 67893 68052 68364 70728
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 32, 128, 128, 131072), 1);  // 33973 34132 34604 37048
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 8, 128, 128, 16384), 1);    // 1127 1148 1188 1484

  // warp_cap: ceil(h / w) saturates at 1 once w >= h, so warps past pow2_ceil(h) only add barrier
  // cost. h = 1 leaves a single candidate.
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 8, 128, 128, 256), 8);  // 161 82 44 28, w = 16 excluded
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 1, 128, 128, 4096), 1);
  EXPECT_EQ(pickAtCapacity(fwd_capacity, 3, 32, 32, 1), 4);  // 16 12 9

  // A capacity of 0 means cudaOccupancyMaxActiveBlocksPerMultiprocessor was unavailable, which
  // kernel_blocks_per_sm also reports when a kernel spills to local memory. There is then no wave
  // count and therefore no cost to compare, so the launcher must fall back to the fixed shape this
  // rule replaced rather than to 0 or to an arbitrary candidate.
  std::array<size_t, kFusedRopeWarpCandidates> no_capacity{};
  EXPECT_EQ(fused_rope_select_warps(64, 128, 128, 16384, no_capacity.data()), 8);
  EXPECT_EQ(fused_rope_select_warps(8, 128, 128, 1, no_capacity.data()), 4);
}

// SBHD, forward and backward. s is the outermost dimension, so a leading slice of the host buffer
// is a valid [s', b, h, d] tensor with the same strides and a grid of s' * b blocks.
TEST(FusedRopeLaunchShape, SbhdPrefixIsShapeIndependent) {
  constexpr size_t kS = 512, kB = 8, kH = 32, kD = 128;
  performPrefixTest<__nv_bfloat16>(NVTE_SBHD, /*forward=*/true, /*interleaved=*/false,
                                   /*vary_s=*/true, kS, kB, kH, kD, kD, sPrefixes(),
                                   "rope_sbhd_bf16_fwd");
  performPrefixTest<__nv_bfloat16>(NVTE_SBHD, /*forward=*/false, /*interleaved=*/false,
                                   /*vary_s=*/true, kS, kB, kH, kD, kD, sPrefixes(),
                                   "rope_sbhd_bf16_bwd");
  // rotary_percent 0.5 takes the `d > d2` copy-the-rest loop, which is a second strided loop over
  // blockDim.y and so has to be equally shape-independent.
  performPrefixTest<__nv_bfloat16>(NVTE_SBHD, /*forward=*/true, /*interleaved=*/false,
                                   /*vary_s=*/true, kS, kB, kH, kD, kD / 2, sPrefixes(),
                                   "rope_sbhd_bf16_fwd_d2half");
  performPrefixTest<__nv_bfloat16>(NVTE_SBHD, /*forward=*/false, /*interleaved=*/false,
                                   /*vary_s=*/true, kS, kB, kH, kD, kD / 2, sPrefixes(),
                                   "rope_sbhd_bf16_bwd_d2half");
}

// BSHD swaps which dimension strides over the batch, and interleaved swaps the rotate partner from
// d_id +/- d2/2 to d_id +/- 1. Both have to be equally shape-independent. fp32 and fp16 cover the
// other two instantiations the type switch reaches.
TEST(FusedRopeLaunchShape, BshdInterleavedAndOtherDtypesAreShapeIndependent) {
  performPrefixTest<__nv_bfloat16>(NVTE_BSHD, /*forward=*/true, /*interleaved=*/true,
                                   /*vary_s=*/false, 64, 128, 8, 128, 128, bPrefixes(),
                                   "rope_bshd_bf16_fwd_interleaved");
  performPrefixTest<__nv_bfloat16>(NVTE_BSHD, /*forward=*/false, /*interleaved=*/true,
                                   /*vary_s=*/false, 64, 128, 8, 128, 128, bPrefixes(),
                                   "rope_bshd_bf16_bwd_interleaved");
  performPrefixTest<float>(NVTE_SBHD, /*forward=*/true, /*interleaved=*/false, /*vary_s=*/true, 128,
                           4, 16, 64, 32, sPrefixes(), "rope_sbhd_fp32_fwd");
  performPrefixTest<float>(NVTE_SBHD, /*forward=*/false, /*interleaved=*/false, /*vary_s=*/true,
                           128, 4, 16, 64, 32, sPrefixes(), "rope_sbhd_fp32_bwd");
  performPrefixTest<__half>(NVTE_SBHD, /*forward=*/true, /*interleaved=*/false, /*vary_s=*/true,
                            128, 4, 16, 64, 64, sPrefixes(), "rope_sbhd_fp16_fwd");
}

// THD reaches the third launch branch. Both runs below describe the same packed tokens; the only
// difference is how many empty sequences pad the batch, which is what tips
// use_fused_rope_thd_linear_grid_launch over its 65536 wasted-block threshold. One therefore runs
// the linear-grid kernel over `total_tokens` blocks and the other runs the legacy kernel over
// `s * b` blocks, and they must agree byte for byte.
TEST(FusedRopeLaunchShape, ThdLinearGridMatchesLegacyGrid) {
  constexpr size_t kT = 1024, kH = 32, kD = 128;
  constexpr size_t kRow = kH * kD;

  std::vector<__nv_bfloat16> host_in(kT * kRow);
  fillDeterministic(host_in.data(), host_in.size(), 0x2545F4914F6CDD1Dull);
  std::vector<float> host_freqs(kT * kD);
  fillFreqs(host_freqs.data(), host_freqs.size(), 0x853C29A3F1D0E77Bull);

  // Two real sequences of 512 tokens, then enough zero-length sequences that s * b - total_tokens
  // clears the 65536 threshold. cu_seqlens has b + 1 entries and the kernel reads cu_seqlens[b] as
  // the total, so the trailing duplicates have to repeat the final offset.
  auto run = [&](const size_t b, const char *tag) -> std::vector<__nv_bfloat16> {
    std::vector<int32_t> host_cu(b + 1);
    host_cu[0] = 0;
    host_cu[1] = 512;
    for (size_t i = 2; i <= b; ++i) host_cu[i] = static_cast<int32_t>(kT);

    test::Tensor input(tag, std::vector<size_t>{kT, kH, kD},
                       test::TypeInfo<__nv_bfloat16>::dtype);
    test::Tensor output(tag, std::vector<size_t>{kT, kH, kD},
                        test::TypeInfo<__nv_bfloat16>::dtype);
    test::Tensor freqs(tag, std::vector<size_t>{kT, kD}, test::TypeInfo<float>::dtype);
    test::Tensor cu_seqlens(tag, std::vector<size_t>{b + 1}, test::TypeInfo<int32_t>::dtype);
    test::Tensor null_ivp(tag, std::vector<size_t>{0}, test::TypeInfo<int>::dtype);

    std::memcpy(input.rowwise_cpu_dptr<__nv_bfloat16>(), host_in.data(),
                kT * kRow * sizeof(__nv_bfloat16));
    std::memcpy(freqs.rowwise_cpu_dptr<float>(), host_freqs.data(), kT * kD * sizeof(float));
    std::memcpy(cu_seqlens.rowwise_cpu_dptr<int32_t>(), host_cu.data(),
                (b + 1) * sizeof(int32_t));
    input.from_cpu();
    freqs.from_cpu();
    cu_seqlens.from_cpu();

    nvte_fused_rope_forward(input.data(), cu_seqlens.data(), freqs.data(), null_ivp.data(),
                            output.data(), NVTE_THD, /*interleaved=*/false, /*cp_size=*/1,
                            /*cp_rank=*/0, /*s=*/static_cast<int>(kT), /*b=*/static_cast<int>(b),
                            /*h=*/static_cast<int>(kH), /*d=*/static_cast<int>(kD),
                            /*d2=*/static_cast<int>(kD), /*stride_s_or_t=*/static_cast<int>(kRow),
                            /*stride_b=*/0, /*stride_h=*/static_cast<int>(kD), /*stride_d=*/1,
                            nullptr);
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    output.to_cpu();
    const __nv_bfloat16 *p = output.rowwise_cpu_dptr<__nv_bfloat16>();
    return std::vector<__nv_bfloat16>(p, p + kT * kRow);
  };

  // b = 2 gives s * b = 2048 blocks against 1024 tokens: 1024 wasted, under the threshold, so the
  // legacy dim3(s, b) kernel runs. b = 256 gives 262144 against 1024: 261120 wasted and a 256x
  // overlaunch ratio, so the linear-grid kernel runs over 1024 blocks.
  const std::vector<__nv_bfloat16> legacy = run(2, "rope_thd_legacy");
  const std::vector<__nv_bfloat16> linear = run(256, "rope_thd_linear");
  ASSERT_EQ(legacy.size(), kT * kRow);
  ASSERT_EQ(linear.size(), kT * kRow);
  EXPECT_EQ(0, std::memcmp(legacy.data(), linear.data(), kT * kRow * sizeof(__nv_bfloat16)))
      << "the THD linear-grid launch and the legacy launch disagree";

  // Negative control: the two runs read the same input, so a one-row shift has to differ.
  EXPECT_NE(0, std::memcmp(linear.data(), legacy.data() + kRow,
                           (kT - 1) * kRow * sizeof(__nv_bfloat16)))
      << "negative control failed";
}

// ---------------------------------------------------------------------------------------------
// fused_qkv_rope
// ---------------------------------------------------------------------------------------------

// The qkv rule is given one capacity per candidate, not one for the whole device, and this pins the
// difference. blocks_per_sm(w) == floor(W1 / w) divides evenly only at powers of two, so on the
// part the Op 8b sweep was measured on the resident warp count falls off a plateau at every other
// blockDim.y. A cost model that assumes a single capacity disagrees with the shipped rule at h in
// {3, 5, 6, 7} -- and only there, which is why a sweep over power-of-two blockDim.y alone could
// not have caught the assumption.
TEST(FusedRopeLaunchShape, QkvCapacityIsQuantisedNotConstant) {
  // 39 SMs, and fused_qkv_rope kernels that keep 64 one-warp blocks resident per SM. Both limits
  // agree at 64 on that part: 131072 registers per SM over 32 lanes times 64 registers per thread,
  // and 2048 threads per SM over 32 lanes.
  EXPECT_EQ(measuredCapacity(1), 2496u);
  EXPECT_EQ(measuredCapacity(2), 2496u);
  EXPECT_EQ(measuredCapacity(3), 2457u);
  EXPECT_EQ(measuredCapacity(4), 2496u);
  EXPECT_EQ(measuredCapacity(5), 2340u);
  EXPECT_EQ(measuredCapacity(6), 2340u);
  EXPECT_EQ(measuredCapacity(7), 2457u);
  EXPECT_EQ(measuredCapacity(8), 2496u);
  EXPECT_EQ(measuredCapacity(12), 2340u);
  EXPECT_EQ(measuredCapacity(16), 2496u);

  // Why the table above is a function of w alone: the dynamic scratch never binds. It is 4 * d2
  // floats -- cos and sin for q and for k -- and fused_rope_warp_capacity clamps blocks_per_sm by
  // shared_mem_per_sm / shared_mem_size. At the largest d2 in the ladders that clamp still admits
  // 64 blocks, equal to the register- and thread-limited W1, so it is never the tighter of the two.
  constexpr size_t kQkvMeasuredSharedMemPerSm = 262144;
  EXPECT_GE(kQkvMeasuredSharedMemPerSm / (4 * 256 * sizeof(float)), kQkvMeasuredWarpsPerSm);

  // Exactly which head counts see a quantised capacity, i.e. have a candidate list whose entries do
  // not all sit on the top plateau.
  std::set<int> quantised_h;
  for (int h = 1; h <= 128; ++h) {
    int cand[kFusedRopeQkvWarpCandidates];
    const int n = fused_rope_qkv_candidates(h, cand);
    bool constant = true;
    for (int i = 1; i < n; ++i) {
      constant &= measuredCapacity(cand[i]) == measuredCapacity(cand[0]);
    }
    if (!constant) quantised_h.insert(h);
  }
  EXPECT_EQ(quantised_h.size(), 4u);
  for (const int h : {3, 5, 6, 7}) {
    EXPECT_EQ(quantised_h.count(h), 1u) << "h=" << h;
  }

  // And that the difference changes the pick, which is the whole point. h = 3 at 4096 blocks: with
  // the real capacity w = 3 needs 6 waves and loses to w = 1; under a uniform capacity it needs 5
  // and wins. Both are the shipped rule; only the capacity table differs.
  EXPECT_EQ(pickQkv(3, 128, 128, 6, 4096), 1);          // quantised: 726 984 774 924
  EXPECT_EQ(pickQkvUniform(3, 128, 128, 6, 4096), 3);   // uniform:   726 984 645 924
  EXPECT_NE(pickQkv(3, 128, 128, 6, 4096), pickQkvUniform(3, 128, 128, 6, 4096));
}

// The candidate list, pinned. fused_rope_qkv_candidates(h) is the powers of two up to pow2_ceil(h)
// under the kFusedRopeMaxWarpsPerBlock clamp, plus fused_rope_qkv_fixed_warps(h) -- h itself at
// h <= 8. That extra entry is what puts 3, 5, 6 and 7 into the list, and what bounds the damage at
// odd h: the rule can always fall back to the old shape by choosing it rather than by being forced
// to it. Measured, dropping it costs 0.5-2.6% at h in {5, 7} and leaves ten provable regressions at
// every hysteresis margin in [0, 0.30].
TEST(FusedRopeLaunchShape, QkvCandidateListsArePinned) {
  const std::vector<std::pair<int, std::vector<int>>> rows = {
      {1, {1}},                {2, {1, 2}},           {3, {1, 2, 3, 4}},
      {4, {1, 2, 4}},          {5, {1, 2, 4, 5, 8}},  {6, {1, 2, 4, 6, 8}},
      {7, {1, 2, 4, 7, 8}},    {8, {1, 2, 4, 8}},     {9, {1, 2, 4, 8, 16}},
      {12, {1, 2, 4, 8, 16}},  {16, {1, 2, 4, 8, 16}}, {17, {1, 2, 4, 8, 16}},
      {32, {1, 2, 4, 8, 16}},  {64, {1, 2, 4, 8, 16}}, {128, {1, 2, 4, 8, 16}},
  };
  for (const auto &row : rows) {
    const int h = row.first;
    int cand[kFusedRopeQkvWarpCandidates];
    const int n = fused_rope_qkv_candidates(h, cand);
    ASSERT_LE(static_cast<size_t>(n), row.second.size()) << "h=" << h;
    EXPECT_EQ(std::vector<int>(cand, cand + n), row.second) << "h=" << h;
    EXPECT_LE(n, kFusedRopeQkvWarpCandidates) << "h=" << h;
    EXPECT_EQ(fused_rope_qkv_fixed_warps(h), h <= 8 ? h : 8) << "h=" << h;
    EXPECT_NE(std::find(cand, cand + n, fused_rope_qkv_fixed_warps(h)), cand + n) << "h=" << h;
    for (int i = 0; i < n; ++i) {
      EXPECT_GE(cand[i], 1) << "h=" << h;
      EXPECT_LE(cand[i], qkvWarpCapOf(h)) << "h=" << h << " candidate " << i;
      EXPECT_LE(cand[i], kFusedRopeMaxWarpsPerBlock) << "h=" << h;
    }
    // Ascending, which fused_rope_select_from_candidates' strict `<` relies on to resolve a tie
    // towards fewer warps.
    for (int i = 1; i < n; ++i) EXPECT_LT(cand[i - 1], cand[i]) << "h=" << h;
  }
}

// The qkv cost model, pinned regime by regime.
// cost(w) = waves(w) * (kFusedRopeQkvWarpWeight * w + kFusedRopeQkvElemWeight * E(w)),
// waves(w) = max(1, ceil(grid_blocks * w / capacity(w))),
// E(w) = ceil(h / w) * segments * lanes_per_head(d, d2).
// The trailing comments are the modelled costs of the candidates in ascending order, so that a
// change to either weight shows up here as a changed pick instead of as a silent re-fit.
//
// The weights are 3 and 5, i.e. WEIGHT == elem_weight / warp_weight == 5/3. That ratio is derived,
// not fitted: WEIGHT == 1 / K1 where K1 == barriers_per_block * t_barrier / t_elem, and a qkv block
// retires three __syncthreads() -- one per fused_qkv_rope_block_* call -- against the rope block's
// one, so K1 is three times Op 8's and WEIGHT is a third of Op 8's 5.
TEST(FusedRopeLaunchShape, QkvTermsAndClampsArePinned) {
  EXPECT_EQ(kFusedRopeQkvWarpWeight, 3);
  EXPECT_EQ(kFusedRopeQkvElemWeight, 5);

  // E(w). segments and lanes_per_head enter only through their product, so d = 128 with d2 = 128
  // (4 lanes) at seg 3 and d = 64 with d2 = 32 (2 lanes) at seg 6 give the same E.
  EXPECT_EQ(fused_rope_qkv_elems_per_thread(8, 128, 128, 6, 1), 192);
  EXPECT_EQ(fused_rope_qkv_elems_per_thread(8, 128, 128, 6, 8), 24);
  EXPECT_EQ(fused_rope_qkv_elems_per_thread(64, 128, 128, 6, 4), 384);
  EXPECT_EQ(fused_rope_qkv_elems_per_thread(32, 128, 64, 3, 2), 192);
  EXPECT_EQ(fused_rope_qkv_elems_per_thread(16, 64, 32, 6, 1), 192);
  EXPECT_EQ(fused_rope_qkv_elems_per_thread(1, 128, 128, 6, 16), 24);
  EXPECT_EQ(fused_rope_qkv_elems_per_thread(8, 128, 128, 3, 1),
            fused_rope_qkv_elems_per_thread(8, 64, 32, 6, 1));

  // segments is the one factor the rope rule does not have, and it moves the pick on its own: at
  // h = 64 and 1024 blocks the same shape wants w = 2 when each head walks one d-wide output and
  // w = 16 when it walks six.
  EXPECT_EQ(pickQkv(64, 128, 128, 1, 1024), 2);    // 1283 646 664 736 896
  EXPECT_EQ(pickQkv(64, 128, 128, 6, 1024), 16);   // 7683 3846 3864 3936 3696
  EXPECT_EQ(pickQkv(64, 128, 128, 16, 1024), 16);  // 20483 10246 10264 10336 9296

  // One partial wave. waves(w) is 1 for every candidate, so only the bracket is left and E falls
  // faster than 3w rises: the pick is the warp_cap clamp.
  EXPECT_EQ(pickQkv(64, 128, 128, 6, 1), 16);   // 7683 3846 1932 984 528
  EXPECT_EQ(pickQkv(8, 128, 128, 6, 1), 8);     // 963 486 252 144
  EXPECT_EQ(pickQkv(32, 128, 128, 3, 1), 16);   // 1923 966 492 264 168

  // grid_blocks == capacity, so w = 1 is exactly one wave and every larger w opens another. The
  // barrier term then has nothing to trade against and w = 1 wins at every head count.
  EXPECT_EQ(pickQkv(8, 128, 128, 6, 2496), 1);   // 963 972 1008 1152
  EXPECT_EQ(pickQkv(64, 128, 128, 6, 2496), 1);  // 7683 7692 7728 7872 8448

  // A ceil() cliff. h = 128 at d2 = 64 has E(8) = 384 and E(16) = 192; 312 blocks still fit one
  // wave at w = 8 but 313 does not, and the extra wave costs more than the halved E saves.
  EXPECT_EQ(pickQkv(128, 128, 64, 6, 312), 8);   // 15363 7686 3852 1944 2016
  EXPECT_EQ(pickQkv(128, 128, 64, 6, 313), 16);  // 15363 7686 3852 3888 3024

  // Interior optima, i.e. cells where neither "stay inside one wave" nor "shrink E to a fixed
  // target" selects the fastest measured warp count but the two-term model does.
  EXPECT_EQ(pickQkv(16, 64, 32, 6, 1024), 2);  // 963 486 504 576 756
  EXPECT_EQ(pickQkv(64, 64, 64, 6, 4096), 4);  // 7686 7704 6804 7056 7776
  EXPECT_EQ(pickQkv(9, 128, 128, 6, 1024), 2);  // 1083 606 744 1056 1176

  // Many waves. waves(w) becomes proportional to w, so the cost grows like w^2 plus a constant and
  // the pick collapses to w = 1.
  EXPECT_EQ(pickQkv(32, 128, 128, 6, 131072), 1);  // 203679 204156 205092 212184 242208
  EXPECT_EQ(pickQkv(8, 128, 128, 6, 16384), 1);    // 6741 6804 6804 7632
  // That last one also has an exact tie at 6804 between w = 2 and w = 4, which the ascending
  // candidate list plus a strict `<` resolves towards fewer warps.
  {
    const std::vector<uint64_t> cost = qkvCosts(8, 128, 128, 6, 16384);
    ASSERT_EQ(cost.size(), 4u);
    EXPECT_EQ(cost[1], cost[2]);
  }

  // warp_cap: ceil(h / w) saturates at 1 once w >= h, so warps past pow2_ceil(h) only add barrier
  // cost. h = 1 leaves a single candidate, for which there is nothing to choose.
  EXPECT_EQ(pickQkv(8, 128, 128, 6, 256), 8);   // 963 486 252 144, w = 16 excluded
  EXPECT_EQ(pickQkv(1, 128, 128, 6, 4096), 1);  // 246
}

// A capacity of 0 means cudaOccupancyMaxActiveBlocksPerMultiprocessor was unavailable, which
// kernel_blocks_per_sm also reports when a kernel spills to local memory. There is then no wave
// count and hence no cost to compare, so the launcher must fall back to the fixed shape this rule
// replaced rather than to 0 or to an arbitrary candidate.
TEST(FusedRopeLaunchShape, QkvZeroCapacityFallsBack) {
  std::array<size_t, kFusedRopeQkvWarpCandidates> no_capacity{};
  for (const int h : {1, 2, 3, 5, 6, 7, 8, 9, 12, 16, 32, 64, 128}) {
    int cand[kFusedRopeQkvWarpCandidates];
    const int n = fused_rope_qkv_candidates(h, cand);
    for (const size_t grid : {1u, 2496u, 131072u}) {
      EXPECT_EQ(fused_rope_select_warps_qkv(h, 128, 128, 6, grid, n, cand, no_capacity.data()),
                fused_rope_qkv_fixed_warps(h))
          << "h=" << h << " grid=" << grid;
    }
    EXPECT_EQ(fused_rope_qkv_fixed_warps(h), h <= 8 ? h : 8) << "h=" << h;
  }

  // One zero among non-zero capacities is equally fatal. fused_rope_select_from_candidates returns
  // on the first zero rather than skipping that candidate, because a comparison against a cost that
  // could not be evaluated is meaningless in both directions.
  int cand[kFusedRopeQkvWarpCandidates];
  const int n = fused_rope_qkv_candidates(64, cand);
  std::array<size_t, kFusedRopeQkvWarpCandidates> partial;
  for (int i = 0; i < n; ++i) partial[i] = measuredCapacity(cand[i]);
  ASSERT_EQ(cand[2], 4);
  partial[2] = 0;
  EXPECT_EQ(fused_rope_select_warps_qkv(64, 128, 128, 6, 512, n, cand, partial.data()), 8);
  EXPECT_EQ(pickQkv(64, 128, 128, 6, 512), 8);  // and 8 is what it picks with every capacity known
}

// Hysteresis, pinned from both sides. The wave count is a step function of grid_blocks * w /
// capacity, so near a step the model has no resolution: it can prefer a candidate by a margin
// smaller than one wave out of many, which is less than the quantisation it is being asked to see
// through. Keeping the incumbent there is not a safety net for a broken model -- measured over the
// 148 usable cells of the Op 8b sweep it raises the geomean from 1.1268 to 1.1272, cuts provable
// regressions from 15 to 2 and cuts moves from 127 to 99.
TEST(FusedRopeLaunchShape, QkvHysteresisIsPinned) {
  EXPECT_EQ(kFusedRopeQkvHysteresisNum, 51u);
  EXPECT_EQ(kFusedRopeQkvHysteresisDen, 50u);

  // Fires. In each of these the incumbent -- min(fused_rope_qkv_fixed_warps(h), warp_cap) -- is
  // within the 1/50 margin of the argmin, so the shape that has been running is kept. The first two
  // are why the fixed-shape entry is in the candidate list at all: without hysteresis h = 5 and
  // h = 7 fall to w = 1 on a modelled margin of 0.7% and 0.4%.
  {
    const std::vector<uint64_t> c = qkvCosts(5, 128, 128, 6, 4096);
    ASSERT_EQ(c.size(), 5u);  // candidates 1, 2, 4, 5, 8
    EXPECT_EQ(c[0], 1206u);   // w = 1, the argmin
    EXPECT_EQ(c[3], 1215u);   // w = 5, the incumbent
    EXPECT_TRUE(qkvHysteresisFires(c, 3));
    EXPECT_EQ(pickQkv(5, 128, 128, 6, 4096), 5);
  }
  {
    const std::vector<uint64_t> c = qkvCosts(7, 128, 128, 6, 4096);
    ASSERT_EQ(c.size(), 5u);  // candidates 1, 2, 4, 7, 8
    EXPECT_EQ(c[0], 1686u);   // w = 1, the argmin
    EXPECT_EQ(c[3], 1692u);   // w = 7, the incumbent
    EXPECT_TRUE(qkvHysteresisFires(c, 3));
    EXPECT_EQ(pickQkv(7, 128, 128, 6, 4096), 7);
  }
  // At many waves every candidate is within a few tenths of a percent of every other, so the
  // incumbent wins the margin test even though w = 1 is the argmin.
  EXPECT_EQ(pickQkv(64, 128, 128, 6, 131072), 8);  // 407199 407676 407652 414264 444048
  // Two different (d, d2, segments) triples with the same E per candidate, hence the same costs and
  // the same pick: segments and lanes_per_head really do enter only as a product.
  EXPECT_EQ(pickQkv(64, 256, 256, 3, 512), 8);   // 7683 3846 1932 1968 2112
  EXPECT_EQ(pickQkv(64, 128, 128, 6, 512), 8);   // 7683 3846 1932 1968 2112
  {
    const std::vector<uint64_t> c = qkvCosts(64, 128, 128, 6, 512);
    ASSERT_EQ(c.size(), 5u);  // candidates 1, 2, 4, 8, 16
    EXPECT_EQ(c[2], 1932u);   // w = 4, the argmin
    EXPECT_EQ(c[3], 1968u);   // w = 8, the incumbent
    EXPECT_TRUE(qkvHysteresisFires(c, 3));
  }

  // Does not fire. The same h = 5 at a grid one wave smaller: the argmin moves to w = 1 by 12%
  // rather than 0.7%, which is outside the margin, and the rule follows the model.
  {
    const std::vector<uint64_t> c = qkvCosts(5, 128, 128, 6, 2048);
    ASSERT_EQ(c.size(), 5u);
    EXPECT_EQ(c[0], 603u);   // w = 1, the argmin
    EXPECT_EQ(c[3], 675u);   // w = 5, the incumbent
    EXPECT_FALSE(qkvHysteresisFires(c, 3));
    EXPECT_EQ(pickQkv(5, 128, 128, 6, 2048), 1);
  }
  {
    const std::vector<uint64_t> c = qkvCosts(3, 128, 128, 6, 1024);
    ASSERT_EQ(c.size(), 4u);  // candidates 1, 2, 3, 4
    EXPECT_EQ(c[1], 246u);    // w = 2, the argmin
    EXPECT_EQ(c[2], 258u);    // w = 3, the incumbent
    EXPECT_FALSE(qkvHysteresisFires(c, 2));
    EXPECT_EQ(pickQkv(3, 128, 128, 6, 1024), 2);
  }

  // The margin is tight, and it is what separates those two groups: 1968 / 1932 fires at 1/50 and
  // 258 / 246 does not. Widening it to 1/10 flips the second one, which is why the margin was
  // chosen against 1/10 by disagreement analysis rather than by score -- over the sweep the two
  // differ on 48 cells and 46 of them favour 1/50, at a 1.0328x geomean.
  EXPECT_LE(1968 * kFusedRopeQkvHysteresisDen, 1932 * kFusedRopeQkvHysteresisNum);
  EXPECT_GT(258 * kFusedRopeQkvHysteresisDen, 246 * kFusedRopeQkvHysteresisNum);
  EXPECT_LE(258 * 10, 246 * 11);
}

// Positive control for the qkv ladders. Those fix h, d, d2 and segments and vary only the grid, so
// whether they cross the rule's buckets depends on the device's resident warp capacity -- which a
// test cannot query, because cuda::kernel_blocks_per_sm is internal to libtransformer_engine and
// not in its version script. It is bounded above by sm_count * maxThreadsPerSM / 32, quantised per
// candidate exactly as the measured table is; on the 39 SM, 2048 thread per SM part the Op 8b
// sweep was measured on that bound is the measured capacity. Requiring the ladders to span buckets
// at the bound therefore pins the case that was measured, and a device with less capacity only
// moves the boundaries down, which the prefix lists straddle widely enough to follow.
TEST(FusedRopeLaunchShape, QkvPrefixLaddersSpanMultipleWarpCounts) {
  for (const QkvLadder &l : qkvLadders()) {
    std::set<int> picks;
    size_t cells = 0;
    for (const size_t prefix : *l.prefixes) {
      const size_t ps = l.vary_s ? prefix : l.s_full;
      const size_t pb = l.vary_s ? l.b_full : prefix;
      if (ps > l.s_full || pb > l.b_full) continue;
      ++cells;
      const int w = pickQkvWith(localCapacityBound, l.h, l.d, l.d2, l.segments(), ps * pb);
      EXPECT_GE(w, 1) << l.tag;
      EXPECT_LE(w, qkvWarpCapOf(l.h)) << l.tag << " prefix=" << prefix;
      EXPECT_LE(w, kFusedRopeMaxWarpsPerBlock) << l.tag;
      picks.insert(w);
    }
    EXPECT_GE(cells, 8u) << l.tag << ": the prefix ladder does not fit this shape";
    EXPECT_GE(picks.size(), 3u)
        << l.tag << ": the prefix ladder does not cross the qkv rule's buckets on this device, so "
                      "its bit-exactness comparisons would pass vacuously";
  }
}

// SBHD, forward and backward, over every SBHD ladder. A leading slice of the packed host buffer is
// a valid [s', b, h, total_d] tensor with the same layout and a grid of s' * b blocks, so the
// slice and the full tensor must agree byte for byte on all three outputs.
TEST(FusedRopeLaunchShape, QkvSbhdPrefixIsShapeIndependent) {
  size_t run = 0;
  for (const QkvLadder &l : qkvLadders()) {
    if (l.fmt != NVTE_SBHD) continue;
    runQkvLadder(l);
    ++run;
  }
  EXPECT_GE(run, 8u) << "the qkv ladder table lost its SBHD rows";
}

// BSHD swaps which dimension strides over the batch, and interleaved swaps the rotate partner from
// d_id +/- d2/2 to d_id +/- 1. Both have to be equally shape-independent, and so do the other two
// instantiations the type switch reaches. The backward writes one packed tensor at the disjoint
// per-head row offsets 0, q_split and q_split + k_split, which the forward does not exercise.
TEST(FusedRopeLaunchShape, QkvBshdInterleavedAndOtherDtypesAreShapeIndependent) {
  size_t bshd = 0, non_bf16 = 0;
  for (const QkvLadder &l : qkvLadders()) {
    if (l.fmt == NVTE_SBHD && l.dtype == DType::kBFloat16) continue;
    if (l.fmt == NVTE_BSHD) ++bshd;
    if (l.dtype != DType::kBFloat16) ++non_bf16;
    runQkvLadder(l);
  }
  EXPECT_GE(bshd, 1u) << "the qkv ladder table lost its BSHD row";
  EXPECT_GE(non_bf16, 2u) << "the qkv ladder table lost its fp32 and fp16 rows";
}
