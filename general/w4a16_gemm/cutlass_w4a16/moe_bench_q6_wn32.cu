// Q6 = int4 + int2, WarpN=32. Q6's sparsest plane is int2, so 32*TK*2 >= 4096 holds at TK=64 and w*x32 is legal here
// where Q3/Q5 cannot reach it. Its own translation unit so this half compiles in parallel with the WN=64 half.
#include "lowbit_moe_bench.hpp"
#define ROW(TMv, WMv)                                                   \
  MOE2(bd, best, "q6", int4_t, uint2_t, 4, 2, TMv,  64, MOE_TK, WMv, 32); \
  MOE2(bd, best, "q6", int4_t, uint2_t, 4, 2, TMv, 128, MOE_TK, WMv, 32);
void sweep_q6_wn32(const Band& bd, Best& best) {
  std::printf("  --- Q6 = int4 + int2, WN=32   (legal here unlike Q3/Q5: the sparsest plane is int2, not int1) --- [%s]\n", PPU_CHUNK_STR);
  MOE_GRID(ROW)
}
