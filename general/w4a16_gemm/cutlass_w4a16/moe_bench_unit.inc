// ONE SWEEP UNIT = ONE SHAPE, all of its stage counts. Included by the generated moe_units/*.cu stubs, each of which
// #defines the UNIT_* macros below and nothing else.
//
// WHY THE UNIT IS A SHAPE AND NOT A FORMAT. Splitting by translation unit is the only way to make the sweep compile in
// parallel (cutlass_build_dev_kernels emits one add_custom_command per .cu), and the granularity is chosen by MEASUREMENT
// rather than taste. Locally, the header alone costs ~5 s to parse and each two-plane kernel adds ~2.5 s, so wall clock
// for a unit is roughly 5 + 2.5*kernels: 42 kernels is ~109 s, 3 kernels is ~12 s.
//
// The axis that must NOT be split is the stage count, and that is a RUNTIME cost, not a compile one: the offline pack is
// per SHAPE, and stages do not change the shape, so all three stage counts share one pack. Measured on this machine, one
// shape costs 31 ms to pack for a two-plane format (19.2 ms int2 + 11.8 ms int1) and 24 ms single-plane; at ~112 shapes
// that is ~3 s, already the same order as the GPU time, so splitting stages into their own units would triple it for
// nothing. TileM, TileN, WarpM and WarpN are all part of the shape, so splitting on those is free.
//
// The (TileM, WarpM) L-shape is NOT enumerated anywhere: moe_ok's WM <= TM rejects the illegal corner, so a unit for
// (TM=32, WM=64) simply compiles to an empty function. One less relation written down by hand.
#include "lowbit_moe_bench.hpp"

#define UNIT_STR2_(x) #x
#define UNIT_STR_(x)  UNIT_STR2_(x)

// THE BODY MUST BE A TEMPLATE, and this is not stylistic. `if constexpr (false)` suppresses instantiation of the
// discarded branch ONLY inside a templated entity; in a plain function the branch is still instantiated, so moe_ok
// returning false does NOT make an illegal shape compile away. Measured: of four probe units, the three legal ones built
// clean and (TileM=32, WarpM=64) produced 99 errors headed by `gemm_operands.hpp: division by zero` -- warpOnM = TM/WM =
// 0, so the collective builder degrades to `int`, which is the failure already recorded in the skill for TM=16/WM=32.
//
// It is the same rule that made a static_assert fire from inside a discarded branch earlier in this work, so the evidence
// for it was already in hand; only the consequence was missed.
template <int Dummy>
static void unit_body(const Band& bd, Best& best) {
#if UNIT_2PLANE
  MOE2(bd, best, UNIT_STR_(UNIT_NAME), UNIT_LOELEM, UNIT_HIELEM, UNIT_LOB, UNIT_HIB,
       UNIT_TM, UNIT_TN, MOE_TK, UNIT_WM, UNIT_WN);
#else
  MOE1(bd, best, UNIT_STR_(UNIT_NAME), UNIT_LOELEM, UNIT_LOB,
       UNIT_TM, UNIT_TN, MOE_TK, UNIT_WM, UNIT_WN);
#endif
}
void UNIT_FN(const Band& bd, Best& best) { unit_body<0>(bd, best); }

// Static-init vote so main can report whether all units agree on PPU_B_CHUNK -- see moe_chunk_tally() in the header.
namespace { const struct MoeUnitVote { MoeUnitVote() { moe_chunk_vote(PPU_CHUNK_ON); } } _moe_unit_vote; }
