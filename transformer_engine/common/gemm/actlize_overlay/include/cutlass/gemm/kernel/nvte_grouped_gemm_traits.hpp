/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

// PPU specializes the CUTLASS collective builders after their primary templates.
// clang-format off
#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"
// clang-format on

// The host constructs Params and launch geometry for the exact specialization
// compiled by NVRTC. Keep both sides of that ABI on this definition.
namespace transformer_engine::grouped_gemm_rtc {

template <bool trans_a, bool trans_b, int SmArch, typename Element_ = cutlass::bfloat16_t,
          int TileM = 256, int TileN = 128, int TileK = 64, int Stages = 0, bool OutFp32 = false>
struct GemmTraits {
  static_assert(SmArch == 80 || SmArch == 89,
                "RTC grouped GEMM supports PPU 1.0 (sm80) and PPU 1.5 (sm89)");
  using ArchTag = cute::conditional_t<SmArch == 80, cutlass::arch::PPU0010, cutlass::arch::PPU0015>;

  using ElementA = Element_;
  using ElementB = Element_;
  using ElementOut = cute::conditional_t<OutFp32, float, Element_>;
  using ElementC = ElementOut;
  using ElementD = ElementOut;
  using ElementAccumulator = float;

  using LayoutA =
      cute::conditional_t<trans_a, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using LayoutB =
      cute::conditional_t<trans_b, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using LayoutC = cutlass::layout::RowMajor;

  static constexpr int kAlignmentA = 8;
  static constexpr int kAlignmentB = 8;
  // Keep each vectorized C/D access at 16 bytes for both output types.
  static constexpr int kAlignmentC = OutFp32 ? 4 : 8;
  static constexpr int kAlignmentD = kAlignmentC;

  using TileShape = cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
  using ClusterShape = cute::Shape<cute::_64, cute::_64, cute::_64>;

  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::EpiloguePtrArraySimtVectorized;

  using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ProblemShapeType = typename ProblemShape::UnderlyingProblemShape;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, kAlignmentC, ElementD, LayoutC*, kAlignmentD, EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>>::CollectiveOp;

  // Stage zero lets CUTLASS choose the pipeline depth from the shared memory budget.
  using StageCountPolicy =
      cute::conditional_t<Stages == 0,
                          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                              sizeof(typename CollectiveEpilogue::SharedStorage))>,
                          cutlass::gemm::collective::StageCount<Stages>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp, ElementA, LayoutA*, kAlignmentA, ElementB, LayoutB*,
      kAlignmentB, ElementAccumulator, TileShape, ClusterShape, StageCountPolicy,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
};

}  // namespace transformer_engine::grouped_gemm_rtc
