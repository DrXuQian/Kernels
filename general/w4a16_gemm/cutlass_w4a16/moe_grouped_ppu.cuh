// STEP 2a launcher: grouped mixed-input GEMM via the GroupScheduler (ppu_aiu_gemm_mixed_input_group.hpp).
// DEGENERATE / uniform-M for now (mainloop args = step-1 batched: single L-strided A/B/S base; A sliced by
// l_coord). Purpose: prove the GroupProblemShape + GroupScheduler + mixed-input collective type stack compiles
// and runs. Ragged A (step 2b) adds group_row_offsets to the collective; nothing here changes structurally.
#pragma once

#include <cstdio>
#include <type_traits>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"
#include "ppu_aiu_gemm_mixed_input_group.hpp"   // the new grouped mixed-input GemmUniversal specialization

namespace moe_grouped_ppu {
using namespace cute;

enum class QuantMode { PerColScaleOnly, FinegrainedScaleOnly, FinegrainedScaleZero };
constexpr bool is_finegrained(QuantMode q) { return q != QuantMode::PerColScaleOnly; }
constexpr bool has_zero(QuantMode q) { return q == QuantMode::FinegrainedScaleZero; }

using GroupShape = cute::Shape<int,int,int>;                            // per-expert [M,N,K]
using GroupProblemShape = cutlass::gemm::GroupProblemShape<GroupShape>;

// group_shapes_dev/host: L entries of [M_e,N,K]. A/B/scales single L-strided bases (2a uniform). L=num_experts.
template <QuantMode QuantOp, class KernelSchedule,
          class TileShape, class ScaleTileShape, class WarpShape, int Stages, bool AiuInterleaved>
void launch(const cutlass::half_t* A, const cutlass::int4b_t* B, const cutlass::half_t* scales,
            const cutlass::half_t* zeros, cutlass::half_t* D,
            int m, int n, int k, int L, int group_size,
            GroupShape* group_shapes_dev, GroupShape const* group_shapes_host,
            int const* group_row_offsets,   // ragged: per-expert cumulative A row start; null=uniform
            char* workspace, size_t workspace_bytes, hggcStream_t stream) {
  using ElementA = cutlass::half_t;  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  using ElementB = cutlass::int4b_t;
  using LayoutB  = std::conditional_t<AiuInterleaved, cutlass::layout::ColumnMajorInterleaved<256>, cutlass::layout::ColumnMajor>;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  using ElementScale = cutlass::half_t;  using ElementZero = cutlass::half_t;
  using ElementBInfo = std::conditional_t<has_zero(QuantOp),
      cute::tuple<ElementB, ElementScale, ElementZero>, cute::tuple<ElementB, ElementScale>>;
  using ElementC = cutlass::half_t;  using LayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::half_t;  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape = WarpShape;
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

  // GroupProblemShape -> hits ppu_aiu_gemm_mixed_input_group.hpp's specialization.
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<GroupProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename CollectiveEpilogue::StrideC;  using StrideD = typename CollectiveEpilogue::StrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;

  const int scale_k = (k + group_size - 1) / group_size;
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, L));
  StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, L));
  // C/D strides are REVERSED (n,m,L) because mixed-input swaps+transposes when B is the narrow (int4) operand
  // -> the kernel writes D transposed as [N][M]. (example 16: "Reverse stride here due to swap and transpose".)
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, L));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, L));
  StrideS sS = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, L));

#ifdef MOEG_DEBUG
  // GET THE LAYOUT, don't guess: print the actual strides the kernel uses for A/B/C/D/scale + the swap flag.
  { static bool once = true; if (once) { once = false;
    std::printf("[moe_grouped layout] m=%d n=%d k=%d L=%d gs=%d\n", m,n,k,L,group_size);
    std::printf("  StrideA="); cute::print(sA); std::printf("\n  StrideB="); cute::print(sB);
    std::printf("\n  StrideC="); cute::print(sC); std::printf("\n  StrideD="); cute::print(sD);
    std::printf("\n  StrideScale="); cute::print(sS);
    // THE decisive flag: mixed-input swaps M/N in to_underlying when B is narrow. actlize's single-GEMM kernel
    // does that swap; this grouped kernel does NOT (yet). If this prints 1, that's the L=1 bug.
    std::printf("\n  Has_SwapAB=%d  (strides above are the ARGUMENT/host stage, pre-to_underlying/pre-swap)\n",
                int(cutlass::gemm::kernel::detail::Has_SwapAB_v<CollectiveMainloop>)); } }
#endif
  GroupProblemShape ps; ps.num_groups = L; ps.problem_shapes = group_shapes_dev; ps.host_problem_shapes = group_shapes_host;
  cutlass::KernelHardwareInfo hw{};   // cu_count auto-queried in to_underlying_arguments

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    ps,
    { A, sA, B, sB, scales, sS, group_size, zeros, group_row_offsets },
    { {ElementAccumulator(1.f), ElementAccumulator(0.f)}, (ElementC*)nullptr, sC, D, sD },
    hw
  };

  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) { std::printf("[moe_grouped] can_implement: %s\n", cutlassGetStatusString(st)); return; }
  size_t need = gemm.get_workspace_size(args);
  if (need > workspace_bytes) { std::printf("[moe_grouped] workspace %zu > %zu\n", need, workspace_bytes); return; }
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) { std::printf("[moe_grouped] init failed\n"); return; }
  gemm.run(stream);
}

template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages>
void filter_and_run(const cutlass::half_t* A, const cutlass::int4b_t* B, const cutlass::half_t* scales,
                    const cutlass::half_t* zeros, cutlass::half_t* D, int m, int n, int k, int L, int group_size,
                    GroupShape* gsd, GroupShape const* gsh, int const* group_row_offsets,
                    char* ws, size_t ws_bytes, hggcStream_t stream) {
  using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using WarpShape = cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>;
  const bool il = (n % 256 == 0 && k % 256 == 0);
  #define MOEG_CALL(SCH, STK, IL) launch<QuantOp, SCH, TileShape, cute::Shape<cute::Int<TN>, STK>, WarpShape, Stages, IL>( \
      A,B,scales,zeros,D,m,n,k,L,group_size,gsd,gsh,group_row_offsets,ws,ws_bytes,stream)
  if constexpr (is_finegrained(QuantOp)) {
    if (group_size == 128) { constexpr int SK=(TK+127)/128;
      if (il) MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs128, cute::Int<SK>, true);
      else    MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs128, cute::Int<SK>, false); }
    else if (group_size == 64) { constexpr int SK=(TK+63)/64;
      if (il) MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs64, cute::Int<SK>, true);
      else    MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs64, cute::Int<SK>, false); }
    else std::printf("[moe_grouped] gs %d unsupported\n", group_size);
  } else {
    if (il) MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputPerCol, cute::_1, true);
    else    MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputPerCol, cute::_1, false);
  }
  #undef MOEG_CALL
}

} // namespace moe_grouped_ppu
