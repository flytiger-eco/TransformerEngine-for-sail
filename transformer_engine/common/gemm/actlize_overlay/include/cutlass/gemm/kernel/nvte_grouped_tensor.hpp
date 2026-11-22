// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE for license information.
#pragma once
#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"

namespace cutlass::gemm::kernel {

struct NativeExpert {
  uint64_t a, b, c, d, end;
  int m, n, k, columns;
  float alpha, beta;
};
struct NativeControl {
  uint64_t work, per_group;
};
struct NativeParams {
  const NativeExpert* experts;
  const NativeControl* control;
  NativeExpert uniform;
  cutlass::FastDivmod tiles_per_group, tiles_per_row;
  const float *alpha, *beta;
  int scalar_stride;
};

// Each work item belongs to exactly one expert. A four-column item only
// drains its A/B pipeline after the fourth output tile has been written.
template <class Mainloop, class Epilogue, bool EnableNExpand, int CommonN = 0, int CommonK = 0>
struct NvteNativeGrouped {
  using Params = NativeParams;
  using TileShape = typename Mainloop::TileShape;
  using TiledMma = typename Mainloop::TiledMma;
  using ElementA = typename Mainloop::ElementA;
  using ElementB = typename Mainloop::ElementB;
  using ElementD = typename Epilogue::ElementD;
  static constexpr int BM = cute::size<0>(TileShape{});
  static constexpr int BN = cute::size<1>(TileShape{});
  static constexpr int BK = cute::size<2>(TileShape{});
  static constexpr int SharedStorageSize = sizeof(typename Mainloop::SharedStorage);
  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});

  template <class Stride>
  CUTLASS_DEVICE static Stride packed_stride(int extent, int k) {
    Stride stride{};
    if constexpr (cute::is_static<decltype(cute::get<0>(stride))>::value)
      cute::get<1>(stride) = extent;
    else
      cute::get<0>(stride) = k;
    return stride;
  }

  CUTLASS_DEVICE static void run_nexpand(NativeExpert e, int mi, int ni, char* smem) {
    using namespace cute;
    static_assert(BM == 256 && BN == 256 && BK == 64 && Mainloop::DispatchPolicy::Stages == 3);
    if constexpr (CommonK != 0) {
      e.n = CommonN;
      e.k = CommonK;
    }
    int tid = int(threadIdx.x);
    auto* dst = reinterpret_cast<ElementD*>(e.d);
    TiledMma mma;
    auto acc = partition_fragment_C(mma, take<0, 2>(TileShape{}));
    clear(acc);
    auto da = packed_stride<typename Mainloop::InternalStrideA>(e.m, e.k);
    auto db = packed_stride<typename Mainloop::InternalStrideB>(e.n, e.k);
    Tensor a = make_tensor(make_gmem_ptr(reinterpret_cast<ElementA const*>(e.a)),
                           make_shape(e.m, e.k), da);
    Tensor b = make_tensor(make_gmem_ptr(reinterpret_cast<ElementB const*>(e.b)),
                           make_shape(e.n, e.k), db);
    Tensor ga = local_tile(make_mix_tensor_like(a), TileShape{}, make_coord(mi, _, _),
                           Step<_1, Underscore, _1>{});
    Tensor gb = local_tile(make_mix_tensor_like(b), take<1, 3>(TileShape{}),
                           make_coord(_, _));
    Mainloop mainloop(make_shape(e.m, e.n, e.k), da, db);
    // The four columns share one ring; only the final column drains it.
    mainloop.compute_nexpand(
        acc, ga, gb, size<2>(ga), tid, smem,
        [&](int column) {
          auto coordinate = make_coord(mi, ni + column, _, 0);
          Epilogue::store_full_tile(
              dst, e.m, e.n, TileShape{}, coordinate, acc, mma, tid);
          clear(acc);
        }, ni);
    cp_async_wait<0>();
    __syncthreads();
  }

  CUTLASS_DEVICE static void run_tile(const NativeExpert& expert, int mi, int ni, char* smem) {
    NativeExpert e = expert;
    if constexpr (CommonK != 0) {
      e.n = CommonN;
      e.k = CommonK;
      // Adjacent M tile rows traverse the expert's B tiles in opposite order.
      if constexpr (!EnableNExpand) {
        if (mi & 1) ni = CommonN / BN - 1 - ni;
      }
    }
    using namespace cute;
    if constexpr (EnableNExpand) {
      if (e.columns == 4) {
        run_nexpand(e, mi, ni, smem);
        return;
      }
    }
    const int tid = int(threadIdx.x);
    const auto coordinate = make_coord(mi, ni, _, 0);
    const int rows = e.m - BM * mi, cols = e.n - BN * ni;
    TiledMma mma;
    auto acc = partition_fragment_C(mma, take<0, 2>(TileShape{}));
    clear(acc);
    if (e.k > 0) {
      auto da = packed_stride<typename Mainloop::InternalStrideA>(e.m, e.k);
      auto db = packed_stride<typename Mainloop::InternalStrideB>(e.n, e.k);
      auto a = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementA*>(e.a)),
                           make_shape(e.m, e.k), da);
      auto b = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementB*>(e.b)),
                           make_shape(e.n, e.k), db);
      auto ga = local_tile(make_mix_tensor_like(a), TileShape{}, take<0,3>(coordinate),
                           Step<_1, Underscore, _1>{});
      auto gb = local_tile(make_mix_tensor_like(b), TileShape{}, take<0,3>(coordinate),
                           Step<Underscore, _1, _1>{});
      Mainloop mainloop(make_shape(e.m, e.n, e.k), da, db);
      int k_tiles = size<2>(ga);
      mainloop(acc, ga, gb, acc, make_coord_iterator(shape<2>(ga)), k_tiles,
               make_coord(rows, cols, e.k - int64_t(k_tiles) * BK), tid, smem);
    }
    cp_async_wait<0>();
    __syncthreads();
    auto* dst = reinterpret_cast<ElementD*>(e.d);
    bool aligned = e.d % 16 == 0 && (uint64_t(e.n) * sizeof(ElementD)) % 16 == 0;
    if (aligned && e.alpha == 1 && e.beta == 0 && cols >= BN) {
      if (rows >= BM)
        Epilogue::store_full_tile(dst, e.m, e.n, TileShape{}, coordinate, acc, mma, tid);
      else
        Epilogue::compute_native_rows(dst, e.m, e.n, TileShape{}, coordinate, acc, mma, tid);
    } else {
      Epilogue::compute_native(reinterpret_cast<const ElementD*>(e.c), dst, e.m, e.n,
                               TileShape{}, coordinate, acc, mma, tid, e.alpha, e.beta, aligned,
                               smem, Int<SharedStorageSize>{});
    }
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const& p, char* smem) {
    if (p.experts == nullptr) {
      // Uniform packed storage has fixed strides and needs no setup kernel.
      for (uint64_t id = blockIdx.x; id < p.uniform.end; id += gridDim.x) {
        int group, local, mi, ni;
        p.tiles_per_group(group, local, int(id));
        if constexpr (CommonK != 0 && !EnableNExpand) {
          constexpr int ntiles = CommonN / BN;
          mi = local / ntiles;
          ni = local % ntiles;
        } else {
          p.tiles_per_row(mi, ni, local);
        }
        const uint64_t g = group;
        auto e = p.uniform;
        if constexpr (CommonK != 0) {
          e.n = CommonN;
          e.k = CommonK;
        }
        e.a += g * e.m * e.k * sizeof(ElementA);
        e.b += g * e.n * e.k * sizeof(ElementB);
        const uint64_t offset = g * e.m * e.n * sizeof(ElementD);
        e.c += offset;
        e.d += offset;
        e.alpha = p.alpha[g * p.scalar_stride];
        e.beta = p.beta[g * p.scalar_stride];
        if constexpr (EnableNExpand) {
          ni *= e.columns;
          if (e.alpha == 1 && e.beta == 0) {
            run_nexpand(e, mi, ni, smem);
          } else {
            // Device scalars may change between launches. Each column needs
            // the general epilogue when the caller requests accumulation.
            const int columns = e.columns;
            e.columns = 1;
            for (int col = 0; col < columns; ++col) run_tile(e, mi, ni + col, smem);
          }
        } else {
          run_tile(e, mi, ni, smem);
        }
      }
      return;
    }
    const uint64_t work = p.control->work;
    if constexpr (CommonK != 0 && !EnableNExpand) {
      // Process each expert's tile range before loading the next descriptor.
      // Empty experts have identical consecutive end offsets.
      static_assert(CommonN % BN == 0);
      constexpr int ntiles = CommonN / BN;
      uint64_t id = blockIdx.x, start = 0;
      for (unsigned group = 0; id < work; ++group) {
        const auto e = p.experts[group];
        for (; id < e.end; id += gridDim.x) {
          const uint64_t local = id - start;
          run_tile(e, int(local / ntiles), int(local % ntiles), smem);
        }
        start = e.end;
      }
      return;
    }
    const uint64_t per_group = p.control->per_group;
    uint64_t group = 0, start = 0;
    for (uint64_t id = blockIdx.x; id < work; id += gridDim.x) {
      if (per_group) {
        group = id / per_group;
        start = group * per_group;
      } else {
        while (p.experts[group].end <= id) {
          start = p.experts[group].end;
          ++group;
        }
      }
      const auto& e = p.experts[group];
      const int columns = EnableNExpand ? e.columns : 1;
      const int ntiles = ((e.n + BN - 1) / BN) / columns;
      const uint64_t local = id - start;
      // Keep large experts addressable, and use a 32-bit divide for ordinary tile counts.
      const int mi = local <= UINT32_MAX ? int(uint32_t(local) / uint32_t(ntiles))
                                         : int(local / ntiles);
      const int ni = int(local - uint64_t(mi) * ntiles);
      run_tile(e, mi, ni * columns, smem);
    }
  }
};

// The list API already stages per-expert shapes and pointers. Use its existing
// GemmUniversal ABI and workspace cache even when the allocations are disjoint.
template <class Ordinary, class Mainloop, class Epilogue, bool Expand, int CommonN, int CommonK>
struct NvteListGrouped {
  using Params = typename Ordinary::Params;
  CUTLASS_DEVICE void operator()(Params const& p, char* smem) {
    using namespace cute;
    uint64_t id = blockIdx.x + uint64_t(gridDim.x) * blockIdx.y;
    uint64_t stride = uint64_t(gridDim.x) * gridDim.y;
    for (int g = 0; g < p.problem_shape.num_groups; ++g) {
      auto shape = p.problem_shape.get_problem_shape(g);
      const int m = get<0>(shape);
      const int n = CommonK ? CommonN : get<1>(shape);
      const int k = CommonK ? CommonK : get<2>(shape);
      const int ntiles = n / (Expand ? 1024 : 256);
      const uint64_t mtiles = Expand ? uint64_t(m / 256) : (uint64_t(m) + 255) / 256;
      const uint64_t tiles = mtiles * ntiles;
      if (id < tiles) {
        NativeExpert e{};
        e.m = m; e.n = n; e.k = k;
        e.a = reinterpret_cast<uint64_t>(p.mainloop.ptr_A[g]);
        e.b = reinterpret_cast<uint64_t>(p.mainloop.ptr_B[g]);
        e.d = reinterpret_cast<uint64_t>(p.epilogue.ptr_D[g]);
        e.alpha = 1;
        for (; id < tiles; id += stride) {
          const int mi = int(id / ntiles), ni = int(id % ntiles);
          if constexpr (Expand)
            NvteNativeGrouped<Mainloop, Epilogue, true, CommonN, CommonK>::run_nexpand(
                e, mi, ni * 4, smem);
          else
            NvteNativeGrouped<Mainloop, Epilogue, false, CommonN, CommonK>::run_tile(
                e, mi, ni, smem);
        }
      }
      id -= tiles;
    }
  }
};

}  // namespace cutlass::gemm::kernel
