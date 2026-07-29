// Q3 = int2 + int1. WarpN is pinned to 64 at TK=64 by the int1 plane (WN*TK*1 >= 4096), so this format has ONE legal
// WarpN and needs no split: the whole (TileM, WarpM) x TileN x stages product fits in one translation unit.
#include "lowbit_moe_bench.hpp"
#define ROW(TMv, WMv)                                                     \
  MOE2(bd, best, "q3", uint2_t, uint1_t, 2, 1, TMv,  64, MOE_TK, WMv, 64); \
  MOE2(bd, best, "q3", uint2_t, uint1_t, 2, 1, TMv, 128, MOE_TK, WMv, 64);
void sweep_q3(const Band& bd, Best& best) {
  std::printf("  --- Q3 = int2 + int1   (int1 plane forces WN >= 4096/TK; TN=64 rows self-filter when WN > TN) --- [%s]\n", PPU_CHUNK_STR);
  MOE_GRID(ROW)
}
