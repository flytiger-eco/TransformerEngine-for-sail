// ===========================================================================
// NVTE actlize overlay copy -- maintained in TransformerEngine, NOT in
// 3rdparty/actlize (which stays at pristine upstream v1.0.0, commit 664f5c19).
//
// This file carries the CUDA-converted RTC grouped-GEMM epilogue maintained
// by TransformerEngine. It wins include resolution over the submodule copy
// because the overlay
// include dir precedes 3rdparty/actlize/include in every -I list (host nvcc
// build and runtime NVRTC alike).
//
// Maintenance: when actlize upstream changes these files, port the delta
// here by hand. Do NOT edit the submodule copy -- it will be overwritten.
// ===========================================================================
/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/fusion/ppu_callbacks.hpp"

// RTC in-place accumulation folds C into the direct output fragment; the host
// guarantees alpha==beta==1.
#ifndef NVTE_GG_PRELOAD_C
#define NVTE_GG_PRELOAD_C 0
#endif
#ifndef NVTE_GG_PROLOGUE_C
#define NVTE_GG_PROLOGUE_C 0
#endif
#ifndef NVTE_GG_ATOMIC_C
#define NVTE_GG_ATOMIC_C 0
#endif
#ifndef NVTE_GG_ATOMIC_GLOBAL
#define NVTE_GG_ATOMIC_GLOBAL 0
#endif
#ifndef NVTE_GG_ATOMIC_INTERLEAVE
#define NVTE_GG_ATOMIC_INTERLEAVE 0
#endif
// A3-native: bulk-prefetch C into smem with an AIU tensor copy instead of a
// per-thread global vector load. Measured over eight shapes, the ACU tensor
// core active rate gains a geometric mean of +23.78% and the net cost of C
// drops to 0.339x, with D bit-identical to the baseline — hence on by default.
// A tile whose shape does not qualify falls back to the fragment path silently
// via kCanStageC.
#ifndef NVTE_GG_C_AIU_SMEM
#define NVTE_GG_C_AIU_SMEM 1
#endif
// Upper bound on the C prefetch pipeline depth. Depth 2 measured a further
// +6.3% over depth 1 (TFU 59.88 vs 56.33, non-overlapping distributions), so 2
// is the default. The second stage must be borrowed from the kernel union's
// slack and must never live in the epilogue's own SharedStorage: that would be
// subtracted from the 256KB by StageCountAutoCarveout, and 4 of 7 tiles with
// stages=0 measurably lost a mainloop stage as a result (256x256x64 dropped
// straight to 1 and also tripped the mainloop's Stages>=2 assertion).
// Borrowing the slack leaves sizeof(SharedStorage) completely unchanged.
#ifndef NVTE_GG_C_AIU_STAGES
#define NVTE_GG_C_AIU_STAGES 2
#endif

#if NVTE_GG_C_AIU_SMEM
#include "cute/arch/copy_ppu.hpp"
#include "cute/arch/copy_ppu0015_aiu.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

// Direct store only supports a LinearCombination with no coordinate dependency
// and no reduction node.
template <class Operation>
struct IsDirectStoreLinearCombination {
  static constexpr bool value = false;
};

template <
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle
>
struct IsDirectStoreLinearCombination<
    fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>> {
  static constexpr bool value = true;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes it out to destination storage.
///
/// Ways to generalize this:
/// - CTA tile shape
/// - vectorization requirements (GMEM)
/// - vectoriz(able) transform()
///
template <
  class StrideC_,
  class StrideD_,
  class FusionCallbacks_,
  class SmemLayout_,
  class CopyAtomR2S_,
  class TiledCopyS2R_,
  class CopyAtomR2G_,
  class EpilogueScheduleType_,
  class CopyAtomG2R_ = CopyAtomR2G_
>
class EpilogueEvt {
public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator

  using SmemLayout   = SmemLayout_;
  using CopyAtomR2S  = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2G  = CopyAtomR2G_;
  using CopyAtomG2R  = CopyAtomG2R_;

  using ElementAccumulator =  typename CopyAtomR2S::ValType;
  using ElementCompute = ElementAccumulator;
  using ElementScalar = ElementCompute;
  using ElementOutput = typename CopyAtomR2G::ValType;
  using ElementC = typename CopyAtomG2R::ValType;
  using StrideC = StrideC_;
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = ElementOutput;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;
  using FusionCallbacks = FusionCallbacks_;
  using FusionStorage = typename FusionCallbacks::SharedStorage;

  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;

  // only for build if GmemTiledCopyC or kAlignmentC is used
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static constexpr bool IsUsePtrArray = cute::is_same_v<EpilogueScheduleType_, EpiloguePtrArraySimtVectorized>;
  static constexpr bool IsDirectStoreSupported =
      IsUsePtrArray && IsDirectStoreLinearCombination<ThreadEpilogueOp>::value;
  static constexpr bool IsGroupGemm = IsUsePtrArray and !cute::is_same_v<InternalStrideC, StrideC>;
  using PtrCType = cute::conditional_t<!IsUsePtrArray, ElementC const*, ElementC const**>;
  using PtrDType = cute::conditional_t<!IsUsePtrArray, ElementD*, ElementD**>;
  static_assert(rank(InternalStrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(InternalStrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayout>> smem_epilogue;
    FusionStorage thread;
  };

  // Host side epilogue arguments
  struct Arguments {
#ifndef __CUDACC_RTC__
    typename FusionCallbacks::Arguments thread{};
#endif
    PtrCType ptr_C = nullptr;
    StrideC dC{};
    PtrDType ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  // different to Arguments, construct thread params in kernel
  // gemm_configs will always use EpilogueEvt
  // both aiu/cutlass gemm will force use rtc since evt params is hard to construct in static
  struct Params {
    typename FusionCallbacks::Params thread{};
    PtrCType ptr_C = nullptr;
    StrideC dC{};
    PtrDType ptr_D = nullptr;
    StrideD dD{};
  };

  //
  // Methods
  //

  template <class IndexType>
  static constexpr ElementC const*
  get_c_ptr(const Params& params, IndexType l, bool isSrcNeeded) {
    if (!isSrcNeeded || params.ptr_C == nullptr) {
      return nullptr;
    }
    if constexpr (IsUsePtrArray) {
      return params.ptr_C[l];
    } else {
      return params.ptr_C;
    }
  }

  template <class IndexType>
  static constexpr ElementD*
  get_d_ptr(const Params& params, IndexType l) {
    if constexpr (IsUsePtrArray) {
      return params.ptr_D[l];
    } else {
      return params.ptr_D;
    }
  }

  template <class IndexType>
  static constexpr InternalStrideC
  get_stride_c(const Params& params, IndexType l) {
    if constexpr (IsGroupGemm) {
      return params.dC[l];
    } else {
      return params.dC;
    }
  }

  template <class IndexType>
  static constexpr InternalStrideD
  get_stride_d(const Params& params, IndexType l) {
    if constexpr (IsGroupGemm) {
      return params.dD[l];
    } else {
      return params.dD;
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    Params params;
    params.ptr_C = args.ptr_C;
    params.dC = args.dC;
    params.ptr_D = args.ptr_D;
    params.dD = args.dD;

#ifndef __CUDACC_RTC__
    params.thread = FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace);
#endif
    return params;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int cu_count=0) {
#ifndef __CUDACC_RTC__
    return FusionCallbacks::get_workspace_size(problem_shape, args.thread);
#else
    return 0;
#endif
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    HostAdapter* host_adapter = nullptr) {
    return FusionCallbacks::initialize_workspace(problem_shape, args.thread, workspace, stream, host_adapter);
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  EpilogueEvt(Params const& params_, SharedStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {}

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return fusion_callbacks.is_C_load_needed();
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK,
    // Bytes actually usable from smem_buf onwards (the size of the kernel-side
    // union). The C prefetch pipeline borrows the union's slack for its second
    // stage, and the epilogue cannot see how much of it the mainloop occupies,
    // so this bound can only come from the kernel. 0 = use only the epilogue's
    // own block.
    class CSmemBytes = cute::Int<0>
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf,
      CSmemBytes c_smem_bytes = {})
  {
#if (defined(__CUDACC_RTC__) || defined(NVTE_GG_PROFILE_RTC)) && \
    NVTE_GG_ATOMIC_C && NVTE_GG_PRELOAD_C && \
    !NVTE_GG_PROLOGUE_C && NVTE_GG_TRANS_A && !NVTE_GG_TRANS_B && NVTE_GG_OUT_FP32
    // RTC fp32 NT wgrad only: the host pins alpha=1 and requests beta=1, while
    // the EVT sees beta=0. Compare the actual per-group C/D address and stride;
    // anything not aliased stays on the original fragment path.
    static_assert(IsDirectStoreSupported && cute::is_same_v<ElementC, float> &&
                  cute::is_same_v<ElementD, float> &&
                  cute::is_same_v<ElementAccumulator, float>,
                  "Atomic C requires fp32 pointer-array LinearCombination");
    auto l = cute::get<3>(blk_coord_mnkl);
    auto source = get_c_ptr(params, l, true);
    if (__builtin_expect(source != nullptr, 0) && source == get_d_ptr(params, l) &&
        get_stride_c(params, l) == get_stride_d(params, l)) {
      compute_epilogue_atomic(problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl,
                              accumulators, tiled_mma, residue_mnk, thread_idx);
      return;
    }
#endif
#if NVTE_GG_PRELOAD_C && !NVTE_GG_PROLOGUE_C
    // Must come before the beta==0 direct short circuit; ptr_C is the host's
    // signal that it wants C folded in.
    if (__builtin_expect(get_c_ptr(params, cute::get<3>(blk_coord_mnkl), true) != nullptr, 0)) {
      static_assert(IsDirectStoreSupported,
                    "Preload C requires pointer-array LinearCombination");
      compute_epilogue_direct<ProblemShapeMNKL, BlockShapeMNK, BlockCoordMNKL,
          FrgEngine, FrgLayout, TiledMma, ResidueMNK, true>(
          problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl,
          accumulators, tiled_mma, residue_mnk, thread_idx, smem_buf, c_smem_bytes);
      return;
    }
#endif
#if !defined(NVTE_GG_DISABLE_DIRECT_EPILOGUE)
    if constexpr (IsDirectStoreSupported) {
      if (!__builtin_expect(is_source_needed(), 0)) {
        compute_epilogue_direct(
          problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl,
          accumulators, tiled_mma, residue_mnk, thread_idx);
        return;
      }
    }
#endif
    if (__builtin_expect(is_source_needed(), 0)) {
      compute_epilogue<ProblemShapeMNKL, BlockShapeMNK, BlockCoordMNKL, FrgEngine, FrgLayout, TiledMma, ResidueMNK, true>(
        problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk, thread_idx, smem_buf);
    } else {
      compute_epilogue<ProblemShapeMNKL, BlockShapeMNK, BlockCoordMNKL, FrgEngine, FrgLayout, TiledMma, ResidueMNK, false>(
        problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk, thread_idx, smem_buf);
    }
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK,
    bool isSrcNeeded
  >
  CUTLASS_DEVICE void
  compute_epilogue(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // synchronizing function for smem reads/writes
#if PPU_BARRIER_ENABLED
    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(typename TiledCopyS2R::TiledNumThr{}, 0); };
#else
    auto synchronize = [] () { __syncthreads(); };
#endif
    // Kernel level shared memory storage
    SharedStorage& shared_tensors = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    // Batches are managed by using appropriate pointers to C and D matrices
    const int32_t mock_L = IsUsePtrArray ? 1 : L;

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    const int32_t mock_l_coord = IsUsePtrArray ? 0 : l_coord;
    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(get_c_ptr(params, l_coord, isSrcNeeded)), make_shape(M,N,mock_L), get_stride_c(params, l_coord));      //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(get_d_ptr(params, l_coord)), make_shape(M,N,mock_L), get_stride_d(params, l_coord));      //             (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,mock_l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,mock_l_coord);                                                   // (BLK_M,BLK_N)

    // Construct a tensor in SMEM that we can partition for rearranging data
    Tensor sC = make_tensor(make_smem_ptr(shared_tensors.smem_epilogue.data()), SmemLayout{});              // (SMEM_M,SMEM_N)

    // Partition sC to match the accumulator partitioning
    auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto thread_r2s = tiled_r2s.get_slice(thread_idx);
    auto tC     = tiled_r2s.get_thread_slice(thread_idx);
    Tensor tCaC = tC.retile_S(accumulators);                                          // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tCsC = tC.partition_D(sC);                                                 // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Tile gD and gC by the shape of SmemLayout first
    auto tile  = make_shape(size<0>(sC), size<1>(sC));
    Tensor gCt = flat_divide(gC, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gDt = flat_divide(gD, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)

    // Partition sC, gC, and gD for the output
    auto tiled_s2r = TiledCopyS2R{};
    auto tD     = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tDsC = tD.partition_S(sC);                                   //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDgC = tD.partition_D(gCt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tDgD = tD.partition_D(gDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Allocate intermediate registers on the dst tensors
    Tensor tDrAcc = make_tensor<ElementAccumulator>(take<0,3>(shape(tDgC)));            // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDrD = make_tensor<ElementOutput>(shape(tDrAcc));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD)));          // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt  = flat_divide(cD, tile);                                //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tDcD = tD.partition_D(cDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    CUTE_STATIC_ASSERT(size<1>(tCaC) % size<3>(tDgC) == 0);  // TILE_M divides MMA_M
    CUTE_STATIC_ASSERT(size<2>(tCaC) % size<4>(tDgC) == 0);  // TILE_N divides MMA_N
    // TiledCopyS2R::TiledNumThr is block size, size<0>(typename TiledMma::AtomLayoutC_TV{}) is warp size, why should be equal?????
    // CUTE_STATIC_ASSERT(typename TiledCopyS2R::TiledNumThr{} == size<0>(typename TiledMma::AtomLayoutC_TV{}));

    constexpr int EvtFragSize = size(tDrAcc(_,0,0));
    // frag for one epilogue outer loop
    Tensor evt_c_tensor = make_tensor<ElementC>(take<0,3>(shape(tDgC)));

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(blk_shape_MNK), make_coord(m_coord, n_coord));          // (CTA_M,CTA_N)
    using EpilogueTile = decltype(shape(coalesce(make_layout(shape(SmemLayout{})), Step<_1, _1>{})));
    Tensor tRS_cD_mn = thread_r2s.partition_S(flat_divide(cD_mn, EpilogueTile{}));     // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)

    // Relative coordinate tensors (static)
    Tensor tRS_cD = make_counting_tensor(tRS_cD_mn.layout());                          // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});                                                           // (m,n)
    auto residue_tRS_cD = make_coord(M,N) - tRS_cD_mn(_0{});

#if SAIL_EPILOGUE_OPT >= 2
    using MmaAccType = typename FrgEngine::value_type;
    static const bool do_epilogue_opt = !cute::is_same_v<MmaAccType, ElementAccumulator>;
    using ArrayType = cutlass::Array<MmaAccType, decltype(cute::size<0>(tCsC))::value>;
    cutlass::NumericArrayConverter<ElementD, MmaAccType, decltype(size<0>(tCsC))::value, FloatRoundStyle::round_to_nearest> converter;
#endif

    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      blk_shape_MNK,
                      blk_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      tiled_s2r,
                      cD,
                      residue_cD,
                      tRS_cD,
                      residue_tRS_cD,
                      evt_c_tensor,
                      thread_idx
                    };
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<true>(cst_args);

    // Pre-loop fusion callback entry point
    cst_callbacks.begin();

    // Check https://github.com/ColfaxResearch/cfx-article-src/blob/master/evt/node_types.md for various nodes(very helpful)
    // For each tiling needed for SmemLayout to cover shape(gD)
    CUTLASS_PRAGMA_UNROLL
    for (int step_m = 0; step_m < size<2>(cDt); ++step_m)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int step_n = 0; step_n < size<3>(cDt); ++step_n)
      {
        bool is_last_iteration = step_m == size<2>(cDt)-1 && step_n == size<3>(cDt)-1;
        // Step 1. Copy to SMEM
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
            int mma_m = step_m * size<1>(tCsC) + pipe_m;
            int mma_n = step_n * size<2>(tCsC) + pipe_n;
#if SAIL_EPILOGUE_OPT >= 2
            if (do_epilogue_opt) {
              const auto result = converter(*reinterpret_cast<const ArrayType *>(tCaC(_,mma_m,mma_n).data()));
              Tensor tCaC_convert = make_tensor(result.data(), tCaC(_,mma_m,mma_n).layout());
              copy(tiled_r2s, tCaC_convert, tCsC(_,pipe_m,pipe_n));
            } else {
              copy(tiled_r2s, tCaC(_,mma_m,mma_n), tCsC(_,pipe_m,pipe_n));
            }
#else
            copy(tiled_r2s, tCaC(_,mma_m,mma_n), tCsC(_,pipe_m,pipe_n));
#endif
          }
        }

        // Step 2. Wait for SMEM writes to complete
        synchronize();

        // Step 3. Copy from SMEM into a fragment
        copy(tiled_s2r, tDsC, tDrAcc);

        // Step 4. Wait for SMEM reads to complete
        synchronize();

        Tensor tDgDmn = tDgD(_,_,_,step_m,step_n);
        Tensor tDcDmn = tDcD(_,_,_,step_m,step_n);

        if (isSrcNeeded) {
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tDgDmn); ++m)
          {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tDgDmn); ++n)
            {
              if (elem_less(tDcDmn(0,m,n), take<0,2>(residue_mnk))) {
                // CopyAtomR2G is Vector Copy which can also used for G2R
                // copy(CopyAtomR2G{}, tDgC(_,m,n,step_m,step_n), evt_c_tensor(_,m,n));
                copy_g2r_ldcs(CopyAtomG2R{}, tDgC(_,m,n,step_m,step_n), evt_c_tensor(_,m,n));
              }
            }
          }

          // follow original epilogue logic, transfer part of global tensor to epilogue_op
          // read to vreg in each sub loop, avoid alloc vreg for the whole subloop
          // evt_c_tensor = tDgC(_,_,_,step_m,step_n);
        }

        cst_callbacks.begin_loop(step_m, step_n);

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(tDgDmn); ++m)
        {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < size<2>(tDgDmn); ++n)
          {
            int sub_loop_idx = m * size<2>(tDgDmn) + n;
            // Predication
            if (get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk) &&
                get<1>(tDcDmn(0,m,n)) < get<1>(residue_mnk))
            {
              // run evt
              Tensor tDrAcc_array = recast<Array<ElementAccumulator, EvtFragSize>>(tDrAcc(_,m,n));
              Tensor tDrD_array = recast<Array<ElementOutput, EvtFragSize>>(tDrD(_,m,n));
              // run output op for all regs in cur epi loop
              // corresponds to multiple vmem.st
              tDrD_array[0] = cst_callbacks.visit(tDrAcc_array[0], sub_loop_idx, step_m, step_n);
              copy(CopyAtomR2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
            }
          }
        }

        // Smem reduction callback entry point using shared memory for workspace
        // cst_callbacks.reduce(tDsC, synchronize, step_m, step_n, is_last_iteration, tDrD);
        cst_callbacks.reduce(sC, synchronize, step_m, step_n, is_last_iteration, tDrD);


        cst_callbacks.end_loop(step_m, step_n);
      }
    }

    cst_callbacks.end();
  }

#if (defined(__CUDACC_RTC__) || defined(NVTE_GG_PROFILE_RTC)) && \
    NVTE_GG_ATOMIC_C && NVTE_GG_PRELOAD_C && \
    !NVTE_GG_PROLOGUE_C && NVTE_GG_TRANS_A && !NVTE_GG_TRANS_B && NVTE_GG_OUT_FP32
  template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
            class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK>
  CUTLASS_DEVICE void
  compute_epilogue_atomic(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx)
  {
    using namespace cute;
    using X = Underscore;
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor mD = make_tensor(make_gmem_ptr(get_d_ptr(params, l_coord)),
                           make_shape(M, N, Int<1>{}), get_stride_d(params, l_coord));
    Tensor gD_mnl = local_tile(mD, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1,X>{});
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,0);
    auto tiled_copy = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto thread_copy = tiled_copy.get_thread_slice(thread_idx);
    Tensor tCrAcc = thread_copy.retile_S(accumulators);
    Tensor tCgD = thread_copy.partition_D(gD);
    Tensor cD = make_identity_tensor(take<0,2>(blk_shape_MNK));
    Tensor tCcD = thread_copy.partition_D(cD);
    CUTE_STATIC_ASSERT(size(tCrAcc) == size(tCgD));
    constexpr int FragSize = decltype(size(tCrAcc(_,0,0)))::value;
    auto issue = [&](auto rotate) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<1>(tCrAcc); ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<2>(tCrAcc); ++j) {
          // The two 256-thread warp groups interleave their accesses to the N
          // fragments, still one writer per element. The fragment index is
          // expanded statically in each branch, so the accumulator is never
          // indexed dynamically (which would spill).
          constexpr int NumN = decltype(size<2>(tCrAcc))::value;
          int n = (j + decltype(rotate)::value * (NumN / 2)) % NumN;
          Tensor tCrAcc_frag = tCrAcc(_,m,n);
          Tensor tCgD_frag = tCgD(_,m,n);
          Tensor tCcD_frag = tCcD(_,m,n);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < FragSize; ++i) {
            auto coord = tCcD_frag(i);
            if (get<0>(coord) < get<0>(residue_mnk) &&
                get<1>(coord) < get<1>(residue_mnk)) {
#if NVTE_GG_ATOMIC_GLOBAL && NVTE_GG_ARCH == 89
              // D is proven to be gmem, so address it explicitly and discard the
              // result, which skips the per-element generic/TSM test.
              // The memory clobber stays to stop the compiler from reordering
              // related memory accesses across this RMW.
              asm volatile("ppu.atom.global.add.f32 _, [%0], %1;\n"
                           :: "l"(&tCgD_frag(i)), "f"(tCrAcc_frag(i)) : "memory");
#else
              // The returned old value is unused; the underlying op is still an
              // RMW.
              atomicAdd(&tCgD_frag(i), tCrAcc_frag(i));
#endif
            }
          }
        }
      }
    };
#if NVTE_GG_ATOMIC_INTERLEAVE
    if constexpr (decltype(size(tiled_mma))::value == 512) {
      if (thread_idx >= 256) {
        issue(Int<1>{});
      } else {
        issue(Int<0>{});
      }
    } else {
      issue(Int<0>{});
    }
#else
    issue(Int<0>{});
#endif
  }
#endif

#if !defined(NVTE_GG_DISABLE_DIRECT_EPILOGUE) || NVTE_GG_PRELOAD_C
  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK,
    bool preloadC = false,
    class CSmemBytes = cute::Int<0>
  >
  CUTLASS_DEVICE void
  compute_epilogue_direct(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf = nullptr,
      CSmemBytes c_smem_bytes = {})
  {
    using namespace cute;
    using X = Underscore;

    static_assert(IsDirectStoreSupported,
                  "Direct store requires pointer-array LinearCombination");

    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    const int32_t mock_L = IsUsePtrArray ? 1 : L;

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    const int32_t mock_l_coord = IsUsePtrArray ? 0 : l_coord;

    Tensor mD_mnl = make_tensor(
        make_gmem_ptr(get_d_ptr(params, l_coord)), make_shape(M,N,mock_L),
        get_stride_d(params, l_coord));
    Tensor gD_mnl = local_tile(
        mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,mock_l_coord);

    // Map D by the MMA-C fragment just like DeepGemm does, which avoids the
    // detour through SMEM.
    auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto thread_r2s = tiled_r2s.get_slice(thread_idx);
    auto tC = tiled_r2s.get_thread_slice(thread_idx);
    Tensor tCaC = tC.retile_S(accumulators);
    Tensor tCaD = tC.partition_D(gD);
    CUTE_STATIC_ASSERT(size(tCaD) == size(tCaC));
    CUTE_STATIC_ASSERT(size<1>(tCaD) == size<1>(tCaC));
    CUTE_STATIC_ASSERT(size<2>(tCaD) == size<2>(tCaC));

    Tensor cD = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD)));
    Tensor tCcD = tC.partition_D(cD);

    // With beta==0 the LinearCombination only consumes alpha and the
    // accumulator. The original callback is kept here so that a scalar alpha
    // still works, with a placeholder C tensor to satisfy the EVT argument
    // contract.
    Tensor mC_mnl = make_tensor(
        make_gmem_ptr(get_c_ptr(params, l_coord, preloadC)), make_shape(M,N,mock_L),
        get_stride_c(params, l_coord));
    Tensor gC_mnl = local_tile(
        mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,mock_l_coord);
    Tensor tCaG = tC.partition_D(gC);
    auto tile = make_shape(size<0>(SmemLayout{}), size<1>(SmemLayout{}));
    Tensor gCt = flat_divide(gC, tile);
    auto tiled_s2r = TiledCopyS2R{};
    auto tD = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tDgC = tD.partition_D(gCt);

    constexpr int EvtFragSize = decltype(size(tCaC(_,0,0)))::value;
    Tensor evt_c_tensor = make_tensor<ElementC>(take<0,3>(shape(tDgC)));

    Tensor mD_crd = make_identity_tensor(make_shape(M,N));
    Tensor cD_mn = local_tile(
        mD_crd, take<0,2>(blk_shape_MNK), make_coord(m_coord, n_coord));
    using EpilogueTile = decltype(
        shape(coalesce(make_layout(shape(SmemLayout{})), Step<_1, _1>{})));
    Tensor tRS_cD_mn = thread_r2s.partition_S(flat_divide(cD_mn, EpilogueTile{}));
    Tensor tRS_cD = make_counting_tensor(tRS_cD_mn.layout());
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});
    auto residue_tRS_cD = make_coord(M,N) - tRS_cD_mn(_0{});

    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
        problem_shape_mnkl,
        blk_shape_MNK,
        blk_coord_mnkl,
        tiled_mma,
        EpilogueTile{},
        tiled_s2r,
        cD,
        residue_cD,
        tRS_cD,
        residue_tRS_cD,
        evt_c_tensor,
        thread_idx
    };
    auto cst_callbacks =
        fusion_callbacks.template get_consumer_store_callbacks<true>(cst_args);
    cst_callbacks.begin();

#if NVTE_GG_C_AIU_SMEM
    // A3-native: instead of every thread issuing a global vector load for C,
    // an AIU bulk tensor copy moves the whole C block into smem and the MMA-C
    // fragment is then read from smem. A single AIU instruction moves one cube,
    // is issued at warp granularity, costs no vreg and is asynchronous, so the
    // transfer overlaps the accumulate/store work.
    //
    // This path imposes hard requirements on C's type, layout and tile shape.
    // They have to degrade silently rather than static_assert: once this is on
    // by default every tile reaches here, and a tile_m=128 chunk (kMmaM=2) is
    // exactly twice the smem tile, so a hard assertion would make small tiles
    // that are live in production fail at RTC compile time.
    constexpr int kAiuMmaM   = decltype(size<1>(tCaC))::value;
    constexpr int kAiuBlockM = decltype(size<0>(cD))::value;
    constexpr int kAiuBlockN = decltype(size<1>(cD))::value;
    // The AIU's contiguous extent tops out at 128B, i.e. 32 fp32 columns.
    constexpr int kAiuCubeW  = 128 / int(sizeof(ElementC));
    constexpr bool kCanStageC =
        sizeof(ElementC) == 4 &&
        cute::is_same_v<ElementC, ElementAccumulator> &&
        // the AIU's 2D descriptor can only express a C that is contiguous in N
        cute::is_constant<1, decltype(cute::get<1>(InternalStrideC{}))>::value &&
        // a chunk has to divide evenly into cubes
        kAiuBlockN % kAiuCubeW == 0 &&
        // a chunk must tile the epilogue's smem block exactly, or it would run
        // past the end and corrupt something else
        (kAiuBlockM / kAiuMmaM) * kAiuBlockN == int(cute::cosize_v<SmemLayout>);

    if constexpr (preloadC && kCanStageC) {
      // One chunk = one mma_m row band x all of mma_n, which is exactly the
      // shape of SmemLayout.
      constexpr int kMmaN   = decltype(size<2>(tCaC))::value;
      constexpr int kChunkM = kAiuBlockM / kAiuMmaM;
      constexpr int kChunkN = kAiuBlockN;
      constexpr int kChunks = kAiuMmaM;
      constexpr int kCubeH  = kChunkM;
      constexpr int kCubeW  = kAiuCubeW;
      constexpr int kCubes  = kChunkN / kCubeW;
      constexpr int kCubeElems  = kCubeH * kCubeW;
      constexpr int kChunkElems = kChunkM * kChunkN;
      // One stage fills smem_epilogue exactly, so the second one lands on the
      // FusionStorage that follows it. Double buffering is only allowed when
      // that really is an empty shell AND the union as a whole has room for it.
      // If either condition fails, fall back to single buffering silently: that
      // gives up roughly a third of the gain but never runs out of bounds.
      constexpr int kChunkBytes  = kChunkElems * int(sizeof(ElementC));
      constexpr int kWantStages  = NVTE_GG_C_AIU_STAGES;
      constexpr bool kSlackFitsC = sizeof(FusionStorage) <= 1 &&
                                   int(CSmemBytes::value) >= kWantStages * kChunkBytes;
      constexpr int kStages =
          (kWantStages > 1 && kSlackFitsC && kChunks >= kWantStages) ? kWantStages : 1;

      // Swzl=false selects padz.linear: in smem each cube is stored row-major
      // and the cubes are laid out one after another.
      using CAiuLoad = cute::PPU0015_AIU_LOAD<
          cute::C<kCubeElems * cute::sizeof_bits<ElementC>::value>,
          ElementC, /*Trans=*/false, kCubeH, kCubeW, /*Swzl=*/false>;
      using CChunkLayout = Layout<Shape <Int<kCubeH>, Shape <Int<kCubeW>, Int<kCubes>>>,
                                  Stride<Int<kCubeW>, Stride<        _1, Int<kCubeElems>>>>;

      ElementC* smem_c = reinterpret_cast<ElementC*>(smem_buf);
      ElementC const* ptr_c = get_c_ptr(params, l_coord, preloadC);

      cute::AiuDesc desc;
      desc.init(reinterpret_cast<const uint8_t*>(ptr_c), int(M), int(N),
                int(cute::get<0>(get_stride_c(params, l_coord))));

      const int m_base = int(m_coord) * kAiuBlockM;
      const int n_base = int(n_coord) * kAiuBlockN;
      const int warp_idx = thread_idx / 32;

      // The AIU is issued per warp, so a chunk's kCubes cubes are spread over
      // the first kCubes warps.
      auto issue_chunk = [&](int chunk, int stage) {
        if (warp_idx < kCubes) {
          CAiuLoad::copy(smem_c + stage * kChunkElems + warp_idx * kCubeElems,
                         ptr_c, desc,
                         /*coord_w=*/n_base + warp_idx * kCubeW,
                         /*coord_h=*/m_base + chunk * kChunkM);
        }
      };

      // The mainloop leaves its trailing AIU groups undrained on exit, and the
      // smem_a/smem_b they write aliases the C buffer used here through the
      // union, so this wait_all is required for correctness — it is not merely
      // group-count bookkeeping.
      cp_async_wait<0>();
      __syncthreads();

      CUTLASS_PRAGMA_UNROLL
      for (int st = 0; st < kStages; ++st) {
        issue_chunk(st, st);
        cp_async_fence();
      }

      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kChunks; ++s) {
        // Groups are committed in order, so every chunk but the last may leave
        // kStages-1 groups still in flight.
        if (s + 1 < kChunks) {
          cp_async_wait<kStages - 1>();
        } else {
          cp_async_wait<0>();
        }
        __syncthreads();

        Tensor sC_chunk = make_tensor(
            make_smem_ptr(smem_c + (s % kStages) * kChunkElems), CChunkLayout{});
        Tensor tCsC = tC.partition_D(sC_chunk);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < kMmaN; ++mma_n) {
          Tensor tCaC_frag = tCaC(_,s,mma_n);
          Tensor tCaD_frag = tCaD(_,s,mma_n);
          Tensor tCcD_frag = tCcD(_,s,mma_n);

          bool any_valid = false;
          bool all_valid = true;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < EvtFragSize; ++i) {
            auto coord = tCcD_frag(i);
            bool valid = get<0>(coord) < get<0>(residue_mnk) &&
                         get<1>(coord) < get<1>(residue_mnk);
            any_valid = any_valid || valid;
            all_valid = all_valid && valid;
          }

          if (any_valid) {
            using MmaAccType = typename FrgEngine::value_type;
            using AccArray = cutlass::Array<MmaAccType, EvtFragSize>;
            auto const& accumulator = *reinterpret_cast<const AccArray*>(tCaC_frag.data());
            // padz already zeroed the out-of-bounds elements in smem, so the
            // tail chunk does not need a per-element predicate on the C read.
            Tensor tCrC_frag = make_tensor<ElementC>(shape(tCaC_frag));
            copy(CopyAtomR2S{}, tCsC(_,0,mma_n), tCrC_frag);
            AccArray combined;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < EvtFragSize; ++i) {
              combined[i] = accumulator[i] + static_cast<MmaAccType>(tCrC_frag(i));
            }
            auto output = cst_callbacks.visit(combined, s * kMmaN + mma_n, s, mma_n);
            Tensor tDrD_frag = make_tensor(output.data(), tCaC_frag.layout());

            if (all_valid) {
              copy(tiled_r2s, tDrD_frag, tCaD_frag);
            } else {
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < EvtFragSize; ++i) {
                auto coord = tCcD_frag(i);
                if (get<0>(coord) < get<0>(residue_mnk) &&
                    get<1>(coord) < get<1>(residue_mnk)) {
                  tCaD_frag(i) = tDrD_frag(i);
                }
              }
            }
          }
        }

        if (s + kStages < kChunks) {
          // Before a buffer is reused, every thread must have finished reading
          // the current chunk out of it.
          __syncthreads();
          issue_chunk(s + kStages, s % kStages);
          cp_async_fence();
        }
      }

      // The next work tile's mainloop immediately issues AIU copies into the
      // aliased smem, so a barrier here is mandatory on the way out.
      __syncthreads();
      cst_callbacks.end();
      return;
    }
#endif
    // The fallback path (macro off, non-fp32 C, or a tile shape that fails
    // kCanStageC) never touches smem.
    (void) smem_buf;
    (void) c_smem_bytes;

    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < size<1>(tCaC); ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_n = 0; mma_n < size<2>(tCaC); ++mma_n) {
        Tensor tCaC_frag = tCaC(_,mma_m,mma_n);
        Tensor tCaD_frag = tCaD(_,mma_m,mma_n);
        Tensor tCcD_frag = tCcD(_,mma_m,mma_n);

        bool any_valid = false;
        bool all_valid = true;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < EvtFragSize; ++i) {
          auto coord = tCcD_frag(i);
          bool valid = get<0>(coord) < get<0>(residue_mnk) &&
                       get<1>(coord) < get<1>(residue_mnk);
          any_valid = any_valid || valid;
          all_valid = all_valid && valid;
        }

        if (any_valid) {
          using MmaAccType = typename FrgEngine::value_type;
          using AccArray = cutlass::Array<MmaAccType, EvtFragSize>;
          auto const& accumulator = *reinterpret_cast<const AccArray*>(tCaC_frag.data());
          auto output = [&]() {
            if constexpr (preloadC) {
              // C and D share the same MMA-C fragment; a full fragment is read
              // as a vector, a tail one element at a time.
              // Only the current fragment is held, which avoids a writable view
              // over the whole accumulator and a second pass over it.
              Tensor tCrC_frag = make_tensor<ElementC>(shape(tCaC_frag));
              Tensor tCgC_frag = tCaG(_,mma_m,mma_n);
              if (all_valid) {
                copy(CopyAtomG2R{}, tCgC_frag, tCrC_frag);
              } else {
                clear(tCrC_frag);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < EvtFragSize; ++i) {
                  auto coord = tCcD_frag(i);
                  if (get<0>(coord) < get<0>(residue_mnk) &&
                      get<1>(coord) < get<1>(residue_mnk)) {
                    tCrC_frag(i) = tCgC_frag(i);
                  }
                }
              }
              AccArray combined;
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < EvtFragSize; ++i) {
                combined[i] = accumulator[i] + static_cast<MmaAccType>(tCrC_frag(i));
              }
              // bf16 has to be accumulated in fp32 before the conversion; the
              // EVT's beta==0 keeps it from consuming C a second time.
              return cst_callbacks.visit(combined,
                  mma_m * size<2>(tCaC) + mma_n, mma_m, mma_n);
            } else {
              return cst_callbacks.visit(accumulator,
                  mma_m * size<2>(tCaC) + mma_n, mma_m, mma_n);
            }
          }();
          Tensor tDrD_frag = make_tensor(output.data(), tCaC_frag.layout());

          if (all_valid) {
            copy(tiled_r2s, tDrD_frag, tCaD_frag);
          } else {
            // Only a tail fragment stores element by element; a vector store
            // would overwrite the next group's D whenever just the leading
            // elements of the fragment are valid.
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < EvtFragSize; ++i) {
              auto coord = tCcD_frag(i);
              if (get<0>(coord) < get<0>(residue_mnk) &&
                  get<1>(coord) < get<1>(residue_mnk)) {
                tCaD_frag(i) = tDrD_frag(i);
              }
            }
          }
        }
      }
    }

    cst_callbacks.end();
  }
#endif

private:
  Params params;
  FusionCallbacks fusion_callbacks;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
