// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE for license information.
#pragma once
#include "cutlass/gemm/kernel/nvte_contiguous_grouped_params.hpp"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {
template <class Mainloop, class Epilogue, bool WithC, int NExpand = 1, int KMax = 512>
struct NvteContiguousGrouped {
  static_assert(NExpand == 1 || NExpand == 4);
  static_assert(KMax == 512 || KMax == 1024 || KMax == 2048);
  static_assert(NExpand == 1 || !WithC, "NExpand requires noC.");
  using Params = nvte_contiguous::Params;
  using TileShape = typename Mainloop::TileShape;
  using TiledMma = typename Mainloop::TiledMma;
  using ElementA = typename Mainloop::ElementA;
  using ElementB = typename Mainloop::ElementB;
  using ElementD = typename Epilogue::ElementD;
  struct SharedStorage {
    typename Mainloop::SharedStorage mainloop;
    // 独立于 A/B，两个 fp32 C chunk 共 64 KiB；无 fusion SMEM。
    alignas(128) char c_pool[WithC ? 65536 : 128];
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static_assert(SharedStorageSize <= 256 * 1024);
  static constexpr int BN = cute::size<1>(TileShape{});
  static_assert(BN == 128 || (BN == 256 && !WithC), "宽 BN 仅启用 noC。");
  static_assert(NExpand == 1 || BN == 256, "Full contiguous NE4 requires BN256.");
  static_assert(cute::size<0>(TileShape{}) == 256 &&
                cute::size<2>(TileShape{}) == 64 && Mainloop::DispatchPolicy::Stages == 3);

  struct Work {
    int expert, m, k, mi, ni;
    uint64_t a_offset, b_offset, d_offset;
  };
  CUTLASS_DEVICE static Work decode(Params const& p, uint64_t id) {
    if (p.ragged_k) {
      uint64_t per_expert = uint64_t(p.m_tiles) * p.n_tiles;
      int expert = int(id / per_expert);
      uint64_t local = id % per_expert;
      auto e = p.experts[expert];
      return {expert, p.m, e.k, int(local / p.n_tiles), int(local % p.n_tiles) * NExpand,
              e.a_offset, e.b_offset, e.d_offset};
    }
    auto block = p.m_blocks[id / p.n_tiles];
    return {block.expert, block.rows, p.k, block.row / 256, int(id % p.n_tiles) * NExpand,
            block.a_offset, block.b_offset, block.d_offset};
  }

  template <class Stride>
  CUTLASS_DEVICE static Stride packed_stride(int extent, int k) {
    Stride stride{};
    if constexpr (cute::is_static<decltype(cute::get<0>(stride))>::value)
      cute::get<1>(stride) = extent;
    else
      cute::get<0>(stride) = k;
    return stride;
  }

  CUTLASS_DEVICE void operator()(Params const& p, char* smem) {
    using namespace cute;
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
    int tid = int(threadIdx.x);
    for (uint64_t id = blockIdx.x; id < p.work_count; id += gridDim.x) {
      Work w = decode(p, id);
      if constexpr (WithC) { if (w.k == 0) continue; }
      TiledMma mma;
      Tensor acc = partition_fragment_C(mma, take<0, 2>(TileShape{}));
      clear(acc);
      auto store_column = [&](int column) {
        auto coordinate = make_coord(w.mi, w.ni + column, _, 0);
        Epilogue::template compute_contiguous<WithC>(
            static_cast<ElementD*>(p.d) + w.d_offset, w.m, p.n, TileShape{}, coordinate,
            acc, mma, tid, storage.c_pool);
      };
      if constexpr (NExpand > 1) {
        // M/N整除由host验证并编码在work中；逐expert共享同一编译期KMax。
        if (w.k > 0 && w.k <= KMax) {
          auto da = packed_stride<typename Mainloop::InternalStrideA>(w.m, w.k);
          auto db = packed_stride<typename Mainloop::InternalStrideB>(p.n, w.k);
          Tensor a = make_tensor(make_gmem_ptr(static_cast<ElementA const*>(p.a) + w.a_offset),
                                 make_shape(w.m, w.k), da);
          Tensor b = make_tensor(make_gmem_ptr(static_cast<ElementB const*>(p.b) + w.b_offset),
                                 make_shape(p.n, w.k), db);
          Tensor ga = local_tile(make_mix_tensor_like(a), TileShape{}, make_coord(w.mi, _, _),
                                 Step<_1, Underscore, _1>{});
          // (BN,BK,N tiles,K tiles)，加载端逐列 K 回绕，不跨 expert。
          Tensor gb = local_tile(make_mix_tensor_like(b), take<1, 3>(TileShape{}),
                                 make_coord(_, _));
          Mainloop mainloop(make_shape(w.m, p.n, w.k), da, db);
          mainloop.compute_nexpand(
              acc, ga, gb, size<2>(ga), tid, reinterpret_cast<char*>(&storage.mainloop),
              [&](int column) {
                store_column(column);
                clear(acc);
              }, w.ni, cute::true_type{});
          // 四列只在末尾排空；列间不重新填充 prologue 或复位 stage。
          cp_async_wait<0>();
          __syncthreads();
          continue;
        }
      }
      // 混合 ragged K 的不合格 expert 仍计算独立列；K=0 写零/保持 C。
      CUTLASS_PRAGMA_UNROLL
      for (int column = 0; column < NExpand; ++column) {
        clear(acc);
        auto coordinate = make_coord(w.mi, w.ni + column, _, 0);
        if (w.k > 0) {
          auto da = packed_stride<typename Mainloop::InternalStrideA>(w.m, w.k);
          auto db = packed_stride<typename Mainloop::InternalStrideB>(p.n, w.k);
          Tensor a = make_tensor(make_gmem_ptr(static_cast<ElementA const*>(p.a) + w.a_offset),
                                 make_shape(w.m, w.k), da);
          Tensor b = make_tensor(make_gmem_ptr(static_cast<ElementB const*>(p.b) + w.b_offset),
                                 make_shape(p.n, w.k), db);
          Tensor ga = local_tile(make_mix_tensor_like(a), TileShape{}, take<0, 3>(coordinate),
                                 Step<_1, Underscore, _1>{});
          Tensor gb = local_tile(make_mix_tensor_like(b), TileShape{}, take<0, 3>(coordinate),
                                 Step<Underscore, _1, _1>{});
          Mainloop mainloop(make_shape(w.m, p.n, w.k), da, db);
          mainloop.compute_contiguous(
              acc, ga, gb, acc, make_coord_iterator(shape<2>(ga)), size<2>(ga),
              tid, reinterpret_cast<char*>(&storage.mainloop));
          cp_async_wait<0>();
          __syncthreads();
        }
        store_column(column);
      }
    }
  }
};
}  // namespace cutlass::gemm::kernel
