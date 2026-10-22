/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The fused_rope forward and backward kernels pick their blockDim.y from the grid size, the head
// count and the head dimension, replacing the fixed `h < 16 ? 4 : 8` heuristic. Nothing
// about those choices can change the result: a block owns exactly one (s, b) pair -- or one packed
// token on the THD linear-grid path -- and inside the block the head loop strides by blockDim.y
// while the rotary and copy-the-rest loops stride by blockDim.x, so every warp count covers the
// same elements and each thread lane computes the same values in the same order. The only shared
// state is the cos/sin scratch, whose entry i is sincosf(freqs[s_id * d2 + i]) -- a function of
// (i, s_id) alone, written before a __syncthreads and read-only afterwards. There is no
// cross-thread reduction anywhere in either family, so the output at (s_id, b_id) depends only on
// the input at (s_id, b_id) and on freqs[s_id].
//
// These tests pin that invariant, which is what makes the launch-shape tuning safe, and they pin
// the rule's decision boundaries so that a change to it shows up.
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
