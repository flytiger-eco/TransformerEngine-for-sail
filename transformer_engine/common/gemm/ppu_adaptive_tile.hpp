/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <utility>

#include "../util/logging.h"
#include "../util/system.h"

namespace transformer_engine::grouped_gemm_rtc {

struct TileConfig {
  int tile_m, tile_n, tile_k, stages;  // Stage zero selects CUTLASS auto-carveout.
  int workers_per_cu = 0;              // Zero selects the occupancy-based grid.
  bool n_expand = false;

  constexpr bool matches(int m, int n, int k, int s) const {
    return tile_m == m && tile_n == n && tile_k == k && stages == s;
  }
};

constexpr TileConfig kDefaultTile{256, 256, 64, 3};

// Supported host instantiations, including configurations used by tag replay.
// clang-format off
#define NVTE_GG_GROUPED_TENSOR_TILES(F) \
  F(256, 256, 64, 3) F(256, 256, 64, 4) F(256, 256, 128, 2) \
  F(256, 128, 64, 3) F(256, 128, 64, 4) F(128, 256, 64, 2) \
  F(128, 256, 64, 3) F(128, 256, 64, 4) F(128, 128, 64, 2)

#define NVTE_GG_TILE_TABLE(F) \
  NVTE_GG_GROUPED_TENSOR_TILES(F) F(256, 256, 64, 0) F(256, 128, 64, 0) \
  F(128, 128, 64, 0) F(256, 64, 64, 0) F(128, 64, 64, 0) \
  F(256, 128, 64, 2) F(256, 256, 64, 2) F(256, 256, 32, 0) \
  F(256, 256, 32, 4) F(256, 256, 32, 6) F(256, 256, 32, 8) \
  F(256, 128, 128, 2) F(128, 256, 64, 0) F(128, 128, 64, 4) \
  F(128, 128, 32, 3) F(128, 256, 128, 2) F(128, 128, 128, 2) \
  F(256, 128, 32, 4)
// clang-format on

inline const std::optional<TileConfig>& forced_tile() {
  static const auto forced = []() -> std::optional<TileConfig> {
    const auto value = getenv<std::string>("NVTE_GG_FORCE_TILE");
    if (value.empty()) return std::nullopt;
    TileConfig tile{};
    if (std::sscanf(value.c_str(), "%d,%d,%d,%d", &tile.tile_m, &tile.tile_n, &tile.tile_k,
                    &tile.stages) < 3) {
      NVTE_WARN("Ignoring malformed NVTE_GG_FORCE_TILE=", value);
      return std::nullopt;
    }
#define NVTE_GG_MATCH(M, N, K, S) \
  if (tile.matches(M, N, K, S)) return tile;
    NVTE_GG_TILE_TABLE(NVTE_GG_MATCH)
#undef NVTE_GG_MATCH
    NVTE_WARN("Unsupported RTC tile ", value, ", using 256x256x64s3");
    return kDefaultTile;
  }();
  return forced;
}

struct TunedTileEntry {
  int m_lo, m_hi, n_lo, n_hi, k_lo, k_hi;  // Inclusive average expert dimensions.
  bool trans_a, trans_b;                   // Layout at the TE API boundary.
  int sm_arch;
  TileConfig tile;
};

// First match wins. The exact SM80 rows precede broader tile-only ranges.
// Their worker counts apply to 16 experts on a 64-CU device.
// clang-format off
constexpr TunedTileEntry kTunedTiles[] = {
    {0, 160, 448, INT_MAX, 1792, 2304, false, false, 89, {128, 256, 64, 4}},
    {256, INT_MAX, 896, INT_MAX, 3584, 4608, false, false, 89, {256, 256, 64, 3}},
    {256, INT_MAX, 3584, INT_MAX, 1344, 1728, false, false, 89, {256, 256, 64, 3}},
    {256, INT_MAX, 448, INT_MAX, 1792, 2304, false, false, 89, {256, 256, 64, 3}},
    {256, INT_MAX, 1792, INT_MAX, 448, 576, false, false, 89, {256, 256, 64, 3}},
    {256, INT_MAX, 896, INT_MAX, 2688, 3456, false, false, 89, {256, 256, 64, 3}},
    {256, INT_MAX, 2688, INT_MAX, 896, 1152, false, false, 89, {256, 256, 64, 3}},
    {256, INT_MAX, 1792, INT_MAX, 448, 576, true, false, 89, {256, 256, 64, 3}},
    {256, INT_MAX, 448, INT_MAX, 224, INT_MAX, false, true, 89, {256, 256, 64, 3}},
    {2048, 2048, 2048, 2048, 512, 512, true, false, 80, {128, 128, 64, 2, 24}},
    {2048, 2048, 4096, 4096, 1536, 1536, true, false, 80, {128, 128, 64, 2, 64}},
    {2048, 2048, 3072, 3072, 1024, 1024, true, false, 80, {128, 128, 64, 2, 64}},
    {2048, 2048, 4096, 4096, 1024, 1024, true, false, 80, {128, 128, 64, 2, 64}},
    {2048, 2048, 1536, 1536, 4096, 4096, false, false, 80, {128, 128, 64, 2, 5}},
    {2048, 2048, 512, 512, 2048, 2048, false, false, 80, {128, 128, 64, 2, 5}},
    {2048, 2048, 1024, 1024, 3072, 3072, false, false, 80, {128, 128, 64, 2, 5}},
    {2048, 2048, 1024, 1024, 4096, 4096, false, false, 80, {128, 128, 64, 2, 5}},
    {4096, 4096, 1536, 1536, 2048, 2048, false, true, 80, {128, 128, 64, 2, 5}},
    {2048, 2048, 512, 512, 2048, 2048, false, true, 80, {128, 128, 64, 2, 5}},
    {3072, 3072, 1024, 1024, 2048, 2048, false, true, 80, {128, 128, 64, 2, 5}},
    {4096, 4096, 1024, 1024, 2048, 2048, false, true, 80, {128, 128, 64, 2, 5}},
    {224, INT_MAX, 448, 8064, 448, 8064, true, false, 80, {128, 128, 64, 2}},
    {224, INT_MAX, 448, 8064, 448, 8064, false, false, 80, {128, 128, 64, 2}},
    {448, 8064, 448, 8064, 224, INT_MAX, false, true, 80, {128, 128, 64, 2}},
};
// clang-format on

inline TileConfig select_grouped_tile(int m, int n, int k, int groups, bool transa, bool transb,
                                      int sm_arch, int sms) {
  if (forced_tile()) return *forced_tile();
  for (const auto& entry : kTunedTiles) {
    if (entry.sm_arch == sm_arch && entry.trans_a == transa && entry.trans_b == transb &&
        m >= entry.m_lo && m <= entry.m_hi && n >= entry.n_lo && n <= entry.n_hi &&
        k >= entry.k_lo && k <= entry.k_hi)
      return entry.tile;
  }
  if (m > 0 && m <= 160) return {128, n <= 512 ? 256 : 128, 64, 2};

  TileConfig best = kDefaultTile;
  if (m > 0 && n > 0 && k > 0) {
    // Score persistent waves across all experts. For row-major B, narrowing BN
    // reduces contiguous load width. Equal costs prefer the wider tile.
    constexpr double kComputeToMemoryRatio = 96.0;
    constexpr std::pair<int, int> candidates[] = {
        {256, 256}, {256, 128}, {128, 128}, {256, 64}, {128, 64}};
    double best_cost = std::numeric_limits<double>::infinity();
    for (auto [bm, bn] : candidates) {
      const int64_t blocks =
          int64_t(groups) * ((int64_t(m) + bm - 1) / bm) * ((int64_t(n) + bn - 1) / bn);
      const int64_t waves = (blocks + sms - 1) / sms;
      const double live_m = std::min(bm, m);
      const double b_extent = transa ? bn : 256;
      const double cost = waves * std::max(live_m * bn / kComputeToMemoryRatio, live_m + b_extent);
      if (cost < best_cost) {
        best_cost = cost;
        best = {bm, bn, 64, bm == 256 && bn == 256 ? 3 : 0};
      }
    }
  }
  if (!transa && !transb && best.tile_m == 256 && best.tile_n == 256) {
    return {256, 256, 128, 2};
  }
  return best;
}

// Four N tiles per CTA need full tiles and enough work to cover the device.
__host__ __device__ inline bool ne4_shape(int m, int n, int k) {
  return m > 0 && m % 256 == 0 && n > 0 && n % 1024 == 0 && k > 0 && k <= 2048 && k % 256 == 0;
}

inline void select_n_expand(TileConfig& cfg, uint64_t tiles, int sms) {
  if (forced_tile() && !cfg.matches(256, 256, 64, 3)) return;
  auto waves = [sms](uint64_t work) { return (work + sms - 1) / sms; };
  if (tiles > 0 && 18 * waves(tiles) <= 5 * waves(4 * tiles)) {
    cfg = kDefaultTile;
    cfg.n_expand = true;
  }
}

// Specialize shared N/K when full-width output tiles can fill the device.
// Short aligned reductions profit from the four-column pipeline. For row-major
// B, deeper reductions amortize a separate loop per output tile.
inline bool select_common_shape(TileConfig& tile, bool trans_b, int64_t sum_m, int n, int k,
                                int sms) {
  constexpr int bm = kDefaultTile.tile_m, bn = kDefaultTile.tile_n, bk = kDefaultTile.tile_k;
  if ((tile.n_expand && (trans_b || k < 16 * bk)) || n <= 0 || n % bn != 0 ||
      k < kDefaultTile.stages * bk || k > 32 * bk || k % bk != 0 ||
      tile.tile_m != bm || tile.tile_n != bn || sum_m / bm < 1 + (sms - 1) / (n / bn) ||
      (forced_tile() && !tile.matches(bm, bn, bk, kDefaultTile.stages)))
    return false;
  tile = kDefaultTile;
  return true;
}

}  // namespace transformer_engine::grouped_gemm_rtc
