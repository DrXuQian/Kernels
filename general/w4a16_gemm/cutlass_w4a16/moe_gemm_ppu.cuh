// MoE W4A16 grouped GEMM on actlize v1.0.0 -- the cutlass MIXED-PRECISION grouped GEMM.
//
// KEY FACT that makes this small: actlize's mixed-input kernel (ppu_aiu_gemm_mixed_input.hpp) is already
// rank-4 / batch-aware -- get_grid_shape sets grid.z = size<3>(problem_shape) and the collective slices
// A/B/scale/zero by l_coord = blockIdx.z (ppu_mma_aiu_multistage_mixed_input.hpp:220,367,371,380). For MoE
// the per-expert weight/scale/zero planes are uniform-strided (B=[L][N][K], S=[L][N][scale_k]), so that
// L-slicing IS the grouping. With A grouped as [L][M][K] and one uniform M per expert, this is exactly the
// mixed-precision grouped GEMM -- the same combination NVIDIA builds as MoeFCGemm<mixed-input Mma>, except the
// actlize collective already carries the L axis so no new kernel is needed for the UNIFORM-m_per_expert case.
//
//   this file  = UNIFORM m_per_expert (batched). Matches the machete/prefill MoE bench (--m_per_expert=N).
//   TODO       = RAGGED grouped (variable tokens/expert): port array_group's GroupScheduler + total_tokens
//                onto this kernel (A addressing becomes ragged; B/S/Z stay L-strided).
//
// Structurally this is fpA_intB_ppu.cuh's launcher with a 4th (L=experts) problem dimension and NO
// SplitKSerialScheduler (the batched mixed-input kernel path). Everything else -- FinegrainedGs schedules,
// ScaleTileShape, AiuInterleaved, block_k>=gs -- is identical.
#pragma once

#include <cstdio>
#include <type_traits>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"

namespace moe_gemm_ppu {
using namespace cute;

// format axis (actlize has no cutlass::WeightOnlyQuantOp; carry it locally, as in fpA_intB_ppu.cuh)
enum class QuantMode { PerColScaleOnly, FinegrainedScaleOnly, FinegrainedScaleZero };
constexpr bool is_finegrained(QuantMode q) { return q != QuantMode::PerColScaleOnly; }
constexpr bool has_zero(QuantMode q) { return q == QuantMode::FinegrainedScaleZero; }

// One batched (uniform-m) mixed-input instantiation. A=[L][M][K] fp16, B=[L][N][K] int4, S=[L][N][scale_k].
template <QuantMode QuantOp, class KernelSchedule,
          class TileShape, class ScaleTileShape, class WarpShape, int Stages, bool AiuInterleaved>
void generic_launcher(const cutlass::half_t* A, const cutlass::int4b_t* B,
                      const cutlass::half_t* scales, const cutlass::half_t* zeros, cutlass::half_t* D,
                      int m, int n, int k, int L, int group_size,
                      char* workspace, size_t workspace_bytes, hggcStream_t stream) {
  using ElementA = cutlass::half_t;
  using LayoutA  = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::int4b_t;
  using LayoutB  = std::conditional_t<AiuInterleaved,
                     cutlass::layout::ColumnMajorInterleaved<256>, cutlass::layout::ColumnMajor>;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementScale = cutlass::half_t;
  using ElementZero  = cutlass::half_t;
  using PackedScale     = cute::tuple<ElementB, ElementScale>;
  using PackedScaleZero = cute::tuple<ElementB, ElementScale, ElementZero>;
  using ElementBInfo = std::conditional_t<has_zero(QuantOp), PackedScaleZero, PackedScale>;

  using ElementC = cutlass::half_t;
  using LayoutC  = cutlass::layout::RowMajor;
  using ElementD = cutlass::half_t;
  using LayoutD  = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape  = WarpShape;    // ppu1.0 has no cluster; builder takes WarpShape here
  using EpilogueSchedule = cutlass::epilogue::EpilogueSimtVectorizedWithoutEvt;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
      ElementAccumulator, ElementAccumulator, ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD, EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, OperatorClass, ElementA, LayoutA, AlignmentA,
      ElementBInfo, LayoutB, AlignmentB, ElementAccumulator,
      cute::tuple<TileShape, ScaleTileShape>, ClusterShape, cute::Int<Stages>, KernelSchedule>::CollectiveOp;

  // rank-4 problem shape [M,N,K,L]; NO SplitKSerialScheduler -> the batched mixed-input kernel path.
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;

  const int scale_k = (k + group_size - 1) / group_size;
  // strides carry the L (expert) dimension -> the collective slices each plane by l_coord = blockIdx.z.
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, L));
  StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, L));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, L));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, L));
  StrideS sS = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, L));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, L},
    { A, sA, B, sB, scales, sS, group_size, zeros },
    { {ElementAccumulator(1.f), ElementAccumulator(0.f)}, (ElementC*)nullptr, sC, D, sD }
  };

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return;
  if (gemm.get_workspace_size(args) > workspace_bytes) return;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return;
  gemm.run(stream);
}

// group_size -> schedule + ScaleTileShape (mirrors fpA_intB_ppu.cuh dispatch_gs; adds the L arg).
template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages, bool AiuInterleaved>
void dispatch_gs(const cutlass::half_t* A, const cutlass::int4b_t* B, const cutlass::half_t* scales,
                 const cutlass::half_t* zeros, cutlass::half_t* D, int m, int n, int k, int L, int group_size,
                 char* ws, size_t ws_bytes, hggcStream_t stream) {
  using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using WarpShape = cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>;
  if (k % 64 || n % 64) { std::printf("[moe_gemm] n,k must be multiples of 64\n"); return; }

  if constexpr (is_finegrained(QuantOp)) {
    switch (group_size) {
      case 128: { constexpr int SK = (TK + 127) / 128;
        generic_launcher<QuantOp, cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs128,
            TileShape, cute::Shape<cute::Int<TN>, cute::Int<SK>>, WarpShape, Stages, AiuInterleaved>(
            A, B, scales, zeros, D, m, n, k, L, group_size, ws, ws_bytes, stream); break; }
      case 64: { constexpr int SK = (TK + 63) / 64;
        generic_launcher<QuantOp, cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs64,
            TileShape, cute::Shape<cute::Int<TN>, cute::Int<SK>>, WarpShape, Stages, AiuInterleaved>(
            A, B, scales, zeros, D, m, n, k, L, group_size, ws, ws_bytes, stream); break; }
      default: std::printf("[moe_gemm] gs %d unsupported (finegrained: 64/128)\n", group_size);
    }
  } else {
    generic_launcher<QuantOp, cutlass::gemm::KernelAiuMultistageMixedInputPerCol,
        TileShape, cute::Shape<cute::Int<TN>, cute::_1>, WarpShape, Stages, AiuInterleaved>(
        A, B, scales, zeros, D, m, n, k, L, k, ws, ws_bytes, stream);
  }
}

// AiuInterleaved from n,k divisibility (mirrors fpA_intB_ppu.cuh filter_and_run).
template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages>
void filter_and_run(const cutlass::half_t* A, const cutlass::int4b_t* B, const cutlass::half_t* scales,
                    const cutlass::half_t* zeros, cutlass::half_t* D, int m, int n, int k, int L, int group_size,
                    char* ws, size_t ws_bytes, hggcStream_t stream) {
  if (n % 256 == 0 && k % 256 == 0)
    dispatch_gs<QuantOp, TM, TN, TK, WM, WN, Stages, true >(A,B,scales,zeros,D,m,n,k,L,group_size,ws,ws_bytes,stream);
  else
    dispatch_gs<QuantOp, TM, TN, TK, WM, WN, Stages, false>(A,B,scales,zeros,D,m,n,k,L,group_size,ws,ws_bytes,stream);
}

} // namespace moe_gemm_ppu
