// Q5 = int4 + int1. Same shape space as Q3 -- the int1 plane pins WarpN to 64 -- and that is deliberate: Q3 and Q5 must
// be swept on the IDENTICAL grid, because the 20.7% gap between them is the open question and a gap measured across two
// different grids is not a measurement of the formats.
#include "lowbit_moe_bench.hpp"
#define ROW(TMv, WMv)                                                    \
  MOE2(bd, best, "q5", int4_t, uint1_t, 4, 1, TMv,  64, MOE_TK, WMv, 64); \
  MOE2(bd, best, "q5", int4_t, uint1_t, 4, 1, TMv, 128, MOE_TK, WMv, 64);
void sweep_q5(const Band& bd, Best& best) {
  std::printf("  --- Q5 = int4 + int1   (WN=64 forced; the SAME grid as Q3 -- that IS the Q3-vs-Q5 comparison) --- [%s]\n", PPU_CHUNK_STR);
  MOE_GRID(ROW)
}
