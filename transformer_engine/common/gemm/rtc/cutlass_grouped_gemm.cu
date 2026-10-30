/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVRTC kernel template for PPU CUTLASS 3.x grouped GEMM.
//
// This source is compiled at runtime by NVRTC. The host TU
// (cutlass_grouped_gemm_rtc.cu) instantiates the exact same Traits and
// builds `GemmKernel::Params` on the host via
// `GemmKernel::to_underlying_arguments(args, workspace_ptr)` before launching
// this kernel through cuLaunchKernel. ABI compatibility between the host-built
// `Params` and the device-side `Params` is required.
//
// Kernel selected: the grouped (GroupProblemShape) specialization of PPU
// `cutlass::gemm::kernel::GemmUniversal` at
//   3rdparty/cutlass3/include/ppu/cutlass/gemm/kernel/acompute_aiu_gemm_array_group.hpp
// matched by `KernelAiuMultistageBatchArray` schedule + `isGroupProblemShape_v`.
// kGrouped mode, rank-3 GroupProblemShape<(M,N,K)>, per-group M/N/K.
//
// Parameterized via -D macros (must match the host TU instantiation exactly,
// because the host builds GemmKernel::Params from the same Traits):
//   NVTE_GG_TRANS_A : 0 or 1
//   NVTE_GG_TRANS_B : 0 or 1
//   NVTE_GG_ARCH    : 80 or 89  (selects the PPU0010 / PPU0015 ArchTag)
//   NVTE_GG_DTYPE   : 0 = bf16 (default), 1 = fp16  (A/B and, unless
//                     NVTE_GG_OUT_FP32, also C/D)
//   NVTE_GG_OUT_FP32: 0 = C/D carry the A/B dtype (default), 1 = C/D are fp32
//   NVTE_GG_WITH_C  : 0 or 1    (0 = ElementC=void; skip source-C load)
//   NVTE_GG_TILE_M  : tile rows (default 256)
//   NVTE_GG_TILE_N  : tile cols (default 128)
//   NVTE_GG_TILE_K  : tile K    (default 64)
//   NVTE_GG_STAGES  : pipeline stages (default 0 = StageCountAutoCarveout)
// The tile and stage macros MUST be honored here: the host derives the launch
// grid, block size and kernel symbol from them, so a cubin that ignored them
// would run with the wrong thread count and read/write out of bounds.

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

#ifndef NVTE_GG_TRANS_A
#error "NVTE_GG_TRANS_A must be defined"
#endif
#ifndef NVTE_GG_TRANS_B
#error "NVTE_GG_TRANS_B must be defined"
#endif
#ifndef NVTE_GG_ARCH
#error "NVTE_GG_ARCH must be defined"
#endif
#ifndef NVTE_GG_WITH_C
#define NVTE_GG_WITH_C 0
#endif
#ifndef NVTE_GG_TILE_M
#define NVTE_GG_TILE_M 256
#endif
#ifndef NVTE_GG_TILE_N
#define NVTE_GG_TILE_N 128
#endif
#ifndef NVTE_GG_TILE_K
#define NVTE_GG_TILE_K 64
#endif
#ifndef NVTE_GG_DTYPE
#define NVTE_GG_DTYPE 0  // 0 = bf16, 1 = fp16
#endif
#ifndef NVTE_GG_OUT_FP32
#define NVTE_GG_OUT_FP32 0
#endif
#ifndef NVTE_GG_STAGES
#define NVTE_GG_STAGES 0  // 0 = StageCountAutoCarveout
#endif

namespace {

#if NVTE_GG_DTYPE == 1
using ElementAB = cutlass::half_t;
#else
using ElementAB = cutlass::bfloat16_t;
#endif
using ElementA = ElementAB;
using ElementB = ElementAB;

// Gradient accumulation (wgrad with fuse_wgrad_accumulation) writes straight
// into the fp32 main_grad, so D and the aliased C carry 32-bit values while the
// operands stay 16-bit. Everything else keeps C/D at the operand dtype, which
// is the latency-optimal form for a 16-bit destination.
#if NVTE_GG_OUT_FP32
using ElementD = float;
#else
using ElementD = ElementAB;
#endif

using ElementAccumulator = float;

// When accumulate=false (no source C load), ElementC=void tells the epilogue
// builder to elide the gmem load of C entirely. PPU's ppu_builder.inl handles
// `cute::is_void_v<ElementC_>` to remap downstream types so the rest of the
// epilogue still type-checks; see ppu_builder.inl:141-145.
#if NVTE_GG_WITH_C
using ElementC = ElementD;
#else
using ElementC = void;
#endif

#if NVTE_GG_TRANS_A
using LayoutA = cutlass::layout::ColumnMajor;
#else
using LayoutA = cutlass::layout::RowMajor;
#endif

#if NVTE_GG_TRANS_B
using LayoutB = cutlass::layout::ColumnMajor;
#else
using LayoutB = cutlass::layout::RowMajor;
#endif

using LayoutC = cutlass::layout::RowMajor;

// actlize dispatches the AIU copy atoms and MMA instructions off the ArchTag
// (see cutlass/gemm/config/gemm_operands.hpp: `is_same_v<Arch, arch::PPU0015>`
// selects PPU0015_AIU_LOAD / PPU0015_* MMA, anything else falls back to the
// PPU0010 variants). A standard CUTLASS tag such as Sm80 therefore silently
// selects PPU 1.0 instructions, which trap at codegen time on a PPU 1.5 part
// ("Support for AIU_LOAD has not been enabled"). Mirror the static PPU path
// (cutlass_grouped_gemm_ppu.cuh:377-386) and pick the tag per compute
// capability instead.
#if NVTE_GG_ARCH == 80
using ArchTag = cutlass::arch::PPU0010;
#elif NVTE_GG_ARCH == 89
using ArchTag = cutlass::arch::PPU0015;
#else
#error "Unsupported NVTE_GG_ARCH (expected 80 or 89)"
#endif

constexpr int kAlignmentA = 8;
constexpr int kAlignmentB = 8;
// ppu_builder.inl types the gmem store atom as
// AutoVectorizingCopyWithAssumedAlignment<sizeof(ElementD) * AlignmentD * 8>,
// and that trait only accepts MaxVecBits <= 128. A 4-byte ElementD therefore
// caps the alignment at 4 elements (= 128 bit); 8 would not compile.
constexpr int kAlignmentC = NVTE_GG_OUT_FP32 ? 4 : 8;
constexpr int kAlignmentD = kAlignmentC;

// Tile comes from the host-provided macros; the host uses the same values to
// size the launch grid and block, so they must agree. ClusterShape mirrors the
// non-RTC PPU path (cutlass_grouped_gemm.cuh, __HGGCCC__ branch) and PPU
// reuses it as the warp tile shape.
using TileShape = cute::Shape<cute::Int<NVTE_GG_TILE_M>,
                              cute::Int<NVTE_GG_TILE_N>,
                              cute::Int<NVTE_GG_TILE_K>>;
using ClusterShape = cute::Shape<cute::_64, cute::_64, cute::_64>;

using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::EpiloguePtrArraySimtVectorized;

// Rank-3 (M,N,K) GroupProblemShape — per-group (M, N, K) for grouped GEMM
// with heterogeneous shapes.
using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, kAlignmentC,
    ElementD, LayoutC*, kAlignmentD,
    EpilogueSchedule,
    cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>>::CollectiveOp;

// Stages selection: 0 means "let CUTLASS auto-pick from the smem budget after
// carving out the epilogue's SharedStorage"; non-zero means "use exactly this
// many stages". Must match the host reference Traits' StageCountPolicy so the
// resulting Params layout is ABI-identical.
#if NVTE_GG_STAGES == 0
using StageCountTraits = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
#else
using StageCountTraits = cutlass::gemm::collective::StageCount<NVTE_GG_STAGES>;
#endif

// Pass LayoutA*/LayoutB* (pointer-to-layout) so StrideA/StrideB become
// pointer types. Combined with GroupProblemShape, this selects the
// acompute_aiu_gemm_array_group.hpp kernel specialization.
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA*, kAlignmentA,
    ElementB, LayoutB*, kAlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    StageCountTraits,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

}  // namespace

// The kernel function name is rewritten per-instantiation by the host TU,
// which prepends an `#undef`/`#define NVTE_GG_KERNEL_NAME <ident>` pair to
// the source string before calling NVRTC. This is done at the source level
// (not via -D) because PPU's hgrtc silently drops macro values passed via
// -D when they contain long identifiers (verified against cubin dumps).
#ifndef NVTE_GG_KERNEL_NAME
#define NVTE_GG_KERNEL_NAME nvte_cutlass_grouped_gemm_rtc_kernel
#endif

extern "C" __global__ void NVTE_GG_KERNEL_NAME(
    typename GemmKernel::Params params) {
  extern __shared__ char smem_buf[];
  GemmKernel op;
  op(params, smem_buf);
}
