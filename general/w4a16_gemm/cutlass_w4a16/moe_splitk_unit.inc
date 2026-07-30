// GENERATED-INCLUDED per-unit body for the grouped split-K bench. One unit = one (TM,TN,TK,WM,WN,Stages); the
// S ladder is a runtime loop inside it, because split-K does not change the instantiation.
//
// The body must be a TEMPLATE: `if constexpr (false)` does not suppress instantiation in a plain function, and
// an illegal (TileM, WarpM) pair would otherwise trip gemm_operands.hpp's division by zero in a unit that
// never runs. The MoE sweep's unit body records the same trap.
#include "moe_splitk_bench_common.hpp"

template <int Dummy>
static void unit_body(Band const& bd, SkCtx const& cx, SkBest& b1, SkBest& bS) {
  sk_config<UNIT_TM, UNIT_TN, UNIT_TK, UNIT_WM, UNIT_WN, UNIT_ST>(bd, cx, b1, bS);
}

void UNIT_FN(Band const& bd, SkCtx const& cx, SkBest& b1, SkBest& bS) { unit_body<0>(bd, cx, b1, bS); }
