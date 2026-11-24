/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "mega_moe_device.cuh"

namespace {

using namespace transformer_engine::rtc::mega_moe;

template <bool Gated>
union alignas(32) SharedTiles {
  struct {
    unsigned short input[kRows * kDepth];
    unsigned short weights[Gated ? 2 : 1][kColumns * kDepth];
  } operands;
  float accum[Gated ? 2 : 1][kRows * kColumns];
};

template <bool Peer>
__device__ const Element* source_row(const Element* matrix, long long first, int tile_row, int end,
                                     int depth, Element* const* peer_rows) {
  if constexpr (Peer) return peer_rows[tile_row];
  const long long row = first + tile_row;
  return row < end ? matrix + row * depth : nullptr;
}

template <int TileRows, bool Peer = false>
__device__ void load_rows(Element* tile, const Element* matrix, long long first_row, int end_row,
                          int depth, int first_k, Element* const* peer_rows = nullptr) {
  constexpr int kVector = sizeof(uint4) / sizeof(Element);
  // The CUDA symmetric allocator aligns every peer base to a page boundary.
  if (depth % kVector == 0 &&
      (Peer || reinterpret_cast<unsigned long long>(matrix) % sizeof(uint4) == 0)) {
    auto* dest = reinterpret_cast<uint4*>(tile);
    constexpr int vectors_per_row = kDepth / kVector;
    for (int i = threadIdx.x; i < TileRows * vectors_per_row; i += kThreads) {
      const auto* row =
          source_row<Peer>(matrix, first_row, i / vectors_per_row, end_row, depth, peer_rows);
      const int k = first_k / kVector + i % vectors_per_row;
      dest[i] = row != nullptr && k < depth / kVector ? reinterpret_cast<const uint4*>(row)[k]
                                                      : make_uint4(0, 0, 0, 0);
    }
    return;
  }
  for (int i = threadIdx.x; i < TileRows * kDepth; i += kThreads) {
    const auto* row = source_row<Peer>(matrix, first_row, i / kDepth, end_row, depth, peer_rows);
    const long long k = static_cast<long long>(first_k) + i % kDepth;
    tile[i] = row != nullptr && k < depth ? row[k] : round_element(0.f);
  }
}

struct PeerRoutes {
  const long long* bases;
  const long long* routes;  // [route, (source rank, token, top-k slot)]
  int world, max_tokens, max_slots;

  template <bool Gated>
  __device__ Element* address(long long route, int hidden) const {
    const long long rank = routes[3 * route];
    const long long token = routes[3 * route + 1];
    const long long slot = routes[3 * route + 2];
    if (rank < 0 || rank >= world || token < 0 || token >= max_tokens || slot < 0 ||
        slot >= max_slots)
      return nullptr;
    const long long base = bases[rank];
    if (base == 0) return nullptr;
    return reinterpret_cast<Element*>(base) + (Gated ? token : max_tokens + slot) * hidden;
  }
};

template <bool Gated, bool Peer = false>
__device__ void grouped_linear(const Element* input, const float* probs, const int* offsets,
                               const long long* weight_ptrs, Element* projection, Element* output,
                               int rows, int hidden, int intermediate, int experts,
                               SharedTiles<Gated>& shared, PeerRoutes peers = {},
                               Element** peer_rows = nullptr) {
  const int width = Gated ? intermediate : hidden;
  const int depth = Gated ? hidden : intermediate;
  const int column_tiles = (width - 1) / kColumns + 1;
  const long long row_tiles = (static_cast<long long>(rows) + kRows - 1) / kRows + experts;
  const int warp_col = (threadIdx.x / kWarpThreads) * kWarpTile;
  for (long long tile = blockIdx.x; tile < row_tiles * column_tiles; tile += gridDim.x) {
    const long long m_tile = tile / column_tiles;
    const int e = expert_for_tile(m_tile, offsets, rows, experts);
    const int first = route_offset(offsets, e, rows), end = route_offset(offsets, e + 1, rows);
    const long long m = first_route_in_tile(m_tile, e, first);
    if (m < first || m >= end) continue;
    if constexpr (Peer) {
      // A tile can mix source ranks. Resolve once and reuse the resulting row
      // addresses across the full K reduction or the output scatter.
      if (threadIdx.x < kRows) {
        const long long row = m + threadIdx.x;
        peer_rows[threadIdx.x] = row < end ? peers.address<Gated>(row, hidden) : nullptr;
      }
      __syncthreads();
    }
    const int n = static_cast<int>(tile % column_tiles) * kColumns;
    const auto* weights = reinterpret_cast<const Element*>(weight_ptrs[(Gated ? 0 : experts) + e]);
    Accumulator acc[Gated ? 2 : 1];
#pragma unroll
    for (int part = 0; part < (Gated ? 2 : 1); ++part) wmma::fill_fragment(acc[part], 0.f);
    auto* a = reinterpret_cast<Element*>(shared.operands.input);
    for (long long k = 0; k < depth; k += kDepth) {
      load_rows<kRows, Peer && Gated>(a, input, m, end, depth, k, peer_rows);
#pragma unroll
      for (int part = 0; part < (Gated ? 2 : 1); ++part) {
        auto* b = reinterpret_cast<Element*>(shared.operands.weights[part]);
        load_rows<kColumns>(b, weights + static_cast<long long>(part) * width * depth, n, width,
                            depth, k);
      }
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < kDepth; kk += kWarpTile) {
        MatrixA af;
        wmma::load_matrix_sync(af, a + kk, kDepth);
#pragma unroll
        for (int part = 0; part < (Gated ? 2 : 1); ++part) {
          MatrixB<wmma::col_major> bf;
          auto* b = reinterpret_cast<Element*>(shared.operands.weights[part]);
          wmma::load_matrix_sync(bf, b + warp_col * kDepth + kk, kDepth);
          wmma::mma_sync(acc[part], af, bf, acc[part]);
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int part = 0; part < (Gated ? 2 : 1); ++part)
      wmma::store_matrix_sync(shared.accum[part] + warp_col, acc[part], kColumns,
                              wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kRows * kColumns; i += kThreads) {
      const long long row = m + i / kColumns;
      const long long col = static_cast<long long>(n) + i % kColumns;
      if (row >= end || col >= width) continue;
      Element value = round_element(shared.accum[0][i]);
      if constexpr (Gated) {
        const Element up = round_element(shared.accum[1][i]);
        if (projection != nullptr) {
          const long long pos = row * (2LL * width) + col;
          projection[pos] = value;
          projection[pos + width] = up;
        }
        const float gate = promote(value);
        value = round_element((gate / (1.f + expf(-gate))) * promote(up) * probs[row]);
      }
      if constexpr (Peer && !Gated) {
        auto* dest = peer_rows[i / kColumns];
        if (dest != nullptr) dest[col] = value;
      } else {
        output[row * width + col] = value;
      }
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" __global__ __launch_bounds__(kThreads) void nvte_mega_moe_fc1(
    const void* x, const float* probs, const int* offsets, const long long* weights,
    void* projection, void* activation, int rows, int hidden, int intermediate, int experts) {
  __shared__ SharedTiles<true> tiles;
  grouped_linear<true>(static_cast<const Element*>(x), probs, offsets, weights,
                       static_cast<Element*>(projection), static_cast<Element*>(activation), rows,
                       hidden, intermediate, experts, tiles);
}

extern "C" __global__ __launch_bounds__(kThreads) void nvte_mega_moe_fc2(
    const void* activation, const int* offsets, const long long* weights, void* output, int rows,
    int hidden, int intermediate, int experts) {
  __shared__ SharedTiles<false> tiles;
  grouped_linear<false>(static_cast<const Element*>(activation), nullptr, offsets, weights, nullptr,
                        static_cast<Element*>(output), rows, hidden, intermediate, experts, tiles);
}

extern "C" __global__ __launch_bounds__(kThreads) void nvte_mega_moe_peer_fc1(
    const float* probs, const int* offsets, const long long* weights, const long long* peers,
    const long long* routes, void* activation, int rows, int hidden, int intermediate, int experts,
    int world, int max_tokens, int max_slots) {
  __shared__ SharedTiles<true> tiles;
  __shared__ Element* peer_rows[kRows];
  grouped_linear<true, true>(nullptr, probs, offsets, weights, nullptr,
                             static_cast<Element*>(activation), rows, hidden, intermediate, experts,
                             tiles, {peers, routes, world, max_tokens, max_slots}, peer_rows);
}

extern "C" __global__ __launch_bounds__(kThreads) void nvte_mega_moe_peer_fc2(
    const void* activation, const int* offsets, const long long* weights, const long long* peers,
    const long long* routes, int rows, int hidden, int intermediate, int experts, int world,
    int max_tokens, int max_slots) {
  __shared__ SharedTiles<false> tiles;
  __shared__ Element* peer_rows[kRows];
  grouped_linear<false, true>(static_cast<const Element*>(activation), nullptr, offsets, weights,
                              nullptr, nullptr, rows, hidden, intermediate, experts, tiles,
                              {peers, routes, world, max_tokens, max_slots}, peer_rows);
}

namespace {

template <int Width>
struct alignas(Width * sizeof(Element)) SlotVector {
  Element values[Width];
};

template <int Width>
__device__ void combine_slots(const void* slots, void* output, long long tokens, int topk,
                              int hidden) {
  const auto* input = static_cast<const SlotVector<Width>*>(slots);
  auto* result = static_cast<SlotVector<Width>*>(output);
  const int cols = hidden / Width;
  const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < tokens * cols; i += step) {
    const long long token = i / cols;
    const long long first = token * topk * cols + i % cols;
    float sum[Width] = {};
    for (int k = 0; k < topk; ++k) {
      const auto chunk = input[first + static_cast<long long>(k) * cols];
#pragma unroll
      for (int j = 0; j < Width; ++j) sum[j] += promote(chunk.values[j]);
    }
    SlotVector<Width> combined;
#pragma unroll
    for (int j = 0; j < Width; ++j) combined.values[j] = round_element(sum[j]);
    result[i] = combined;
  }
}

}  // namespace

extern "C" __global__ __launch_bounds__(128) void nvte_mega_moe_combine(const void* slots,
                                                                        void* output,
                                                                        long long tokens, int topk,
                                                                        int hidden) {
  const auto addresses =
      reinterpret_cast<unsigned long long>(slots) | reinterpret_cast<unsigned long long>(output);
  if (hidden % 4 == 0 && addresses % alignof(SlotVector<4>) == 0)
    combine_slots<4>(slots, output, tokens, topk, hidden);
  else
    combine_slots<1>(slots, output, tokens, topk, hidden);
}

namespace {

struct RouteRow {
  Element* data;
  bool remote;
};

__device__ RouteRow transport_row(int id, int position, Element* peer0, Element* peer1,
                                  long long capacity, int hidden, int experts, int rank) {
  if (id < 0 || id >= 2 * experts || position < 0 || position >= capacity) return {nullptr, false};
  const int owner = id >= experts;
  return {(owner ? peer1 : peer0) + static_cast<long long>(position) * hidden, owner != rank};
}

#if defined(__HGGCCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890 && !NVTE_MOE_FP16
// Every lane passes the same 512-byte segment address. Each receives/stores
// 128 bits. Remote fabric accesses have dedicated load and store instructions.
__device__ __forceinline__ uint4 load_route(const RouteRow& row, long long offset) {
  if (!row.data) return make_uint4(0, 0, 0, 0);
  const auto* src = row.data + offset;
  return row.remote ? __ppu_remote_load_bulk_b32x4(src) : __ppu_global_ldcg_bulk_b32x4(src);
}

__device__ __forceinline__ void store_route(const RouteRow& row, long long offset, uint4 value) {
  if (!row.data) return;
  auto* dst = row.data + offset;
  if (row.remote)
    __ppu_remote_store_bulk_volatile_b32x4(dst, value);
  else
    __ppu_global_store_bulk_volatile_b32x4(dst, value);
}

__device__ __forceinline__ uint4 sum_bf16_vectors(uint4 a, uint4 b) {
  uint4 result;
  const auto* av = reinterpret_cast<const __nv_bfloat162*>(&a);
  const auto* bv = reinterpret_cast<const __nv_bfloat162*>(&b);
  auto* out = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __bfloat1622float2(av[i]), y = __bfloat1622float2(bv[i]);
    out[i] = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
  }
  return result;
}

template <bool Pull>
__device__ void transport_two(Element* local, RouteRow a, RouteRow b, int hidden) {
  constexpr int segment = 512 / sizeof(Element);
  const int warp = threadIdx.x / kWarpThreads;
  for (long long offset = warp * segment; offset < hidden; offset += 4 * segment) {
    if constexpr (Pull) {
      const uint4 value = sum_bf16_vectors(load_route(a, offset), load_route(b, offset));
      __ppu_global_store_bulk_volatile_b32x4(local + offset, value);
    } else {
      const uint4 value = __ppu_global_ldcg_bulk_b32x4(local + offset);
      store_route(a, offset, value);
      store_route(b, offset, value);
    }
  }
}
#endif

template <bool Pull>
__device__ void transport_tokens(void* data, const int* ids, const int* positions, void* buffer0,
                                 void* buffer1, long long tokens, long long capacity, int hidden,
                                 int topk, int experts, int rank, long long row_stride,
                                 long long col_stride, long long id_row_stride,
                                 long long id_slot_stride) {
  auto* local = static_cast<Element*>(data);
  auto* p0 = static_cast<Element*>(buffer0);
  auto* p1 = static_cast<Element*>(buffer1);
  for (long long t = blockIdx.x; t < tokens; t += gridDim.x) {
    auto* row = local + t * row_stride;
    const auto* id = ids + t * id_row_stride;
    const auto* pos = positions + t * topk;
#if defined(__HGGCCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890 && !NVTE_MOE_FP16
    const auto alignment = reinterpret_cast<unsigned long long>(row) |
                           reinterpret_cast<unsigned long long>(p0) |
                           reinterpret_cast<unsigned long long>(p1);
    if (topk == 2 && hidden % 256 == 0 && col_stride == 1 && alignment % 512 == 0) {
      auto a = transport_row(id[0], pos[0], p0, p1, capacity, hidden, experts, rank);
      auto b = transport_row(id[id_slot_stride], pos[1], p0, p1, capacity, hidden, experts, rank);
      transport_two<Pull>(row, a, b, hidden);
      continue;
    }
#endif
    // Arbitrary top-k, source strides, and row tails use the same route order.
    for (long long col = threadIdx.x; col < hidden; col += blockDim.x) {
      float sum = 0.f;
      Element value;
      if constexpr (!Pull) value = row[col * col_stride];
      for (int k = 0; k < topk; ++k) {
        auto route =
            transport_row(id[k * id_slot_stride], pos[k], p0, p1, capacity, hidden, experts, rank);
        if (!route.data) continue;
        if constexpr (Pull)
          sum += promote(route.data[col]);
        else
          route.data[col] = value;
      }
      if constexpr (Pull) row[col * col_stride] = round_element(sum);
    }
  }
}
}  // namespace

extern "C" __global__ __launch_bounds__(128) void nvte_mega_moe_bulk_push(
    void* local, const int* ids, const int* positions, void* peer0, void* peer1, long long tokens,
    long long capacity, int hidden, int topk, int experts, int rank, long long row_stride,
    long long col_stride, long long id_row_stride, long long id_slot_stride) {
  transport_tokens<false>(local, ids, positions, peer0, peer1, tokens, capacity, hidden, topk,
                          experts, rank, row_stride, col_stride, id_row_stride, id_slot_stride);
}

extern "C" __global__ __launch_bounds__(128) void nvte_mega_moe_bulk_pull(
    void* local, const int* ids, const int* positions, void* peer0, void* peer1, long long tokens,
    long long capacity, int hidden, int topk, int experts, int rank, long long row_stride,
    long long col_stride, long long id_row_stride, long long id_slot_stride) {
  transport_tokens<true>(local, ids, positions, peer0, peer1, tokens, capacity, hidden, topk,
                         experts, rank, row_stride, col_stride, id_row_stride, id_slot_stride);
}
