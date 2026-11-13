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

#include "cute/arch/copy_ppu.hpp"
#include "cute/arch/copy_ppu0015_aiu.hpp"

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
#if NVTE_GG_PRELOAD_C
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
    if constexpr (IsDirectStoreSupported) {
      if (!__builtin_expect(is_source_needed(), 0)) {
        compute_epilogue_direct(
          problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl,
          accumulators, tiled_mma, residue_mnk, thread_idx);
        return;
      }
    }
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
      // Borrow union slack only when FusionStorage is empty, preserving the SMEM ABI.
      constexpr int kChunkBytes  = kChunkElems * int(sizeof(ElementC));
      constexpr bool kSlackFitsC = sizeof(FusionStorage) <= 1 &&
                                   int(CSmemBytes::value) >= 2 * kChunkBytes;
      constexpr int kStages = (kSlackFitsC && kChunks >= 2) ? 2 : 1;

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
    // The fragment fallback for non-fp32 C or ineligible tiles never touches smem.
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

  // Requires host-proved full M/N tiles and 16B D origins/row strides.
  template <bool WithC, class TileShapeType, class Coord, class Acc, class Mma>
  CUTLASS_DEVICE static void compute_contiguous(
      ElementD* dst, int M, int N, TileShapeType tile_shape, Coord coord,
      Acc const& acc, Mma mma, int thread_idx, char* c_pool) {
    using namespace cute;
    auto stride = make_stride(int64_t(N), _1{}, _0{});
    Tensor matrix = make_tensor(make_gmem_ptr(dst), make_shape(M, N, _1{}), stride);
    static_assert(sizeof(decltype(matrix.layout()(make_coord(0, 0, 0)))) >= sizeof(int64_t));
    Tensor output = local_tile(matrix, tile_shape, make_coord(_, _, _), Step<_1, _1, Underscore>{});
    Tensor gD = output(_, _, get<0>(coord), get<1>(coord), 0);
    auto copy_c = make_tiled_copy_C(CopyAtomR2S{}, mma);
    auto thread = copy_c.get_thread_slice(thread_idx);
    Tensor tAcc = thread.retile_S(acc);
    Tensor tDst = thread.partition_D(gD);
    constexpr int Frag = decltype(size(tAcc(_, 0, 0)))::value;
    constexpr int MmaM = decltype(size<1>(tAcc))::value;
    constexpr int MmaN = decltype(size<2>(tAcc))::value;
    constexpr int BM = decltype(size<0>(gD))::value;
    constexpr int BN = decltype(size<1>(gD))::value;
    constexpr int ChunkM = BM / MmaM;
    constexpr int CubeW = 128 / sizeof(ElementD);
    constexpr int CubeElems = ChunkM * CubeW;
    constexpr int ChunkElems = ChunkM * BN;
    using ChunkLayout = Layout<Shape<Int<ChunkM>, Shape<Int<CubeW>, Int<BN / CubeW>>>,
                               Stride<Int<CubeW>, Stride<_1, Int<CubeElems>>>>;
    // noC 不使用 C pool；宽 noC tile 不受有 C chunk 的 32 KiB 预算约束。
    static_assert(!WithC || ChunkElems * sizeof(ElementD) <= 32768);
    // Use the same automatic copy grain for D stores and full BF16 C loads.
    // Prove actual fragment/vector origins, not that a scattered fragment is 16B-wide.
    auto vector_bytes = [&](auto const& registers, auto const& global) {
      using CopyOp = typename CPY_Op<typename CopyAtomR2S::Traits>::type;
      constexpr int CopyBits = sizeof(typename CopyOp::SRegisters) * 8;
      constexpr int CommonElems = CUTE_STATIC_V(max_common_vector(registers, global));
      constexpr int AlignBits = CUTE_STATIC_V(
          gcd(max_alignment(registers), max_alignment(global), Int<CopyBits>{}));
      constexpr int VectorBits = gcd(CommonElems * sizeof_bits<ElementD>::value, AlignBits);
      constexpr int VectorBytes = VectorBits / 8;
      static_assert(VectorBytes >= sizeof(ElementD));
      static_assert(VectorBytes <= 16 && 16 % VectorBytes == 0);
      static_assert(Frag % (VectorBytes / sizeof(ElementD)) == 0);
      // Host proves 16B expert origins and row strides. This static layout
      // proof completes that guarantee for both WithC and noC fragments.
      constexpr bool fragment_origins_aligned = []() {
        using Copy = decltype(copy_c);
        constexpr auto tv = Copy::get_layoutD_TV();
        constexpr int CopyM = size<0>(typename Copy::Tiler_MN{});
        constexpr int CopyN = size<1>(typename Copy::Tiler_MN{});
        constexpr int Threads = size<0>(tv);
        constexpr int Vectors = size<1>(tv) / (VectorBytes / sizeof(ElementD));
        if (CopyN * sizeof(ElementD) % 16 != 0) return false;
        // Affine TV modes are additive; cache coordinates to bound constexpr evaluation.
        int value_offsets[Vectors]{};
        for (int v = 0; v < Vectors; ++v)
          value_offsets[v] = tv(0, v * (VectorBytes / sizeof(ElementD)));
        for (int t = 0; t < Threads; ++t) {
          int thread_offset = tv(t, 0);
          for (int v = 0; v < Vectors; ++v)
            if (((thread_offset + value_offsets[v]) / CopyM * sizeof(ElementD)) %
                VectorBytes != 0) return false;
        }
        return true;
      }();
      static_assert(fragment_origins_aligned, "host 对齐证明不覆盖此 MMA-C copy 布局。");
      return Int<VectorBytes>{};
    };
    auto store = [&](auto mma_m, auto mma_n, auto const& source) {
      Tensor a = tAcc(_, mma_m, mma_n);
      Tensor d = tDst(_, mma_m, mma_n);
      using Array = cutlass::Array<ElementAccumulator, Frag>;
      Array combined;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Frag; ++i) {
        combined[i] = a(i);
        if constexpr (WithC) combined[i] += float(source(i));
      }
      cutlass::NumericArrayConverter<ElementD, ElementAccumulator, Frag> convert;
      auto converted = convert(combined);
      Tensor result = make_tensor(converted.data(), a.layout());
      (void) vector_bytes(result, d);
      copy(copy_c, result, d);
    };
    if constexpr (WithC && sizeof(ElementD) == 4) {
      static_assert(BN % CubeW == 0 && MmaM >= 2);
      using Load = PPU0015_AIU_LOAD<C<CubeElems * sizeof_bits<ElementD>::value>,
                                    ElementD, false, ChunkM, CubeW, false>;
      auto* smem = reinterpret_cast<ElementD*>(c_pool);
      AiuDesc desc;
      desc.init(reinterpret_cast<const uint8_t*>(dst), M, N, N);
      auto issue = [&](int chunk, int slot) {
        int warp = thread_idx / 32;
        if (warp < BN / CubeW)
          Load::copy(smem + slot * ChunkElems + warp * CubeElems, dst, desc,
                     int(get<1>(coord)) * BN + warp * CubeW,
                     int(get<0>(coord)) * BM + chunk * ChunkM);
      };
      issue(0, 0); cp_async_fence();
      issue(1, 1); cp_async_fence();
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < MmaM; ++s) {
        if (s + 1 < MmaM) {
          cp_async_wait<1>();
        } else {
          cp_async_wait<0>();
        }
        __syncthreads();
        Tensor sm = make_tensor(make_smem_ptr(smem + (s % 2) * ChunkElems), ChunkLayout{});
        Tensor tSm = thread.partition_D(sm);
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaN; ++n) {
          Tensor source = make_tensor<ElementD>(shape(tAcc(_, s, n)));
          copy(CopyAtomR2S{}, tSm(_, 0, n), source);
          store(s, n, source);
        }
        __syncthreads();
        if (s + 2 < MmaM) { issue(s + 2, s % 2); cp_async_fence(); }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaM; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaN; ++n) {
          Tensor source = make_tensor<ElementD>(shape(tAcc(_, m, n)));
          if constexpr (WithC) {
            Tensor d = tDst(_, m, n);
            (void) vector_bytes(source, d);
            copy(copy_c, d, source);
          }
          store(m, n, source);
        }
      }
      __syncthreads();
    }
  }

private:
  Params params;
  FusionCallbacks fusion_callbacks;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
