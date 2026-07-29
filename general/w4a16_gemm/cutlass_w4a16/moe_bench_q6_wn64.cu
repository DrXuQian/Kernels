// Q6 = int4 + int2, WarpN=64. This is the row that compares against Q3 and Q5, which are pinned to WN=64 by their int1
// plane: Q6 at WN=64 is the only Q6 shape on the same grid as the other two two-plane formats. Its own translation unit
// so it compiles in parallel with the WN=32 half.
#include "lowbit_moe_bench.hpp"
#define ROW(TMv, WMv)                                                     \
  MOE2(bd, best, "q6", int4_t, uint2_t, 4, 2, TMv,  64, MOE_TK, WMv, 64); \
  MOE2(bd, best, "q6", int4_t, uint2_t, 4, 2, TMv, 128, MOE_TK, WMv, 64);
void sweep_q6_wn64(const Band& bd, Best& best) {
  std::printf("  --- Q6 = int4 + int2, WN=64   (the same WarpN Q3/Q5 are pinned to -- the cross-format row) --- [%s]\n", PPU_CHUNK_STR);
  MOE_GRID(ROW)
}
