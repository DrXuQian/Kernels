// Single-plane i2 (int2), WarpN=32. Single plane means the delivery bound is just WN*TK*2 >= 4096, so
// both WarpN values are legal at TK=64 and each gets its own translation unit -- these two files are what makes the
// int2 half of the sweep compile in parallel rather than one after the other.
//
// The old hand-written list gave int2 and int4 DIFFERENT stage counts at the same shape (i2 s2, i4 s3), so the
// single-plane verdict was comparing stages, not formats. Stages is an axis here.
#include "lowbit_moe_bench.hpp"
#define ROW(TMv, WMv)                                              \
  MOE1(bd, best, "i2", uint2_t, 2, TMv,  64, MOE_TK, WMv, 32); \
  MOE1(bd, best, "i2", uint2_t, 2, TMv, 128, MOE_TK, WMv, 32);
void sweep_i2_wn32(const Band& bd, Best& best) {
  std::printf("  --- i2 = single-plane int2, WN=32 --- [%s]\n", PPU_CHUNK_STR);
  MOE_GRID(ROW)
}
