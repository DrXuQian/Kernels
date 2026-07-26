// Will the new build-time guard reject anything that works today?
//
// The guard (fold::Check) depends on NOTHING but (Bits, TileShape.K). So enumerating the distinct (Bits, TK)
// pairs that appear anywhere in the tree is an EXHAUSTIVE answer -- tile M, tile N, warp shape and stage count
// cannot change the verdict. Section 1 instantiates every such pair; if this file compiles, no existing kernel
// instantiation can be rejected.
//
// Section 2 is the wider, informational pass: every concrete (Bits, TM, TN, TK, Stages, WM, WN) tuple found in
// the tree, run through the full FoldTraits including the invariants the guard deliberately leaves out. This is
// where a config with too small a TN, or an occupancy cliff, would show up.
//
//   g++ -O2 -std=c++17 sweep_shapes.cpp -o sweep && ./sweep
#include "../fold_traits.hpp"
#include <cstdio>

// ---------------------------------------------------------------- section 1: the guard, exhaustively
// Every distinct (Bits, TK) reachable from the tree. Sources:
//   test_q3_bconcat_bench.cu   BC/I1 @ TK=256 (uint2+uint1), I2 @ TK=128, I4 @ TK=64 and 128
//   test_q3_bconcat_real.cu    (uint2b_t, uint1b_t) @ TK=256
//   test_fold_int2.cu          int1 @ TK=128, int2 @ TK=64 and 128, int4 @ TK=64 and 128
//   moe_grouped_ppu.cuh users  int4 (default ElementB) @ TK=64 and 128
//   misc W4A16/W2A16 tests     @ TK=256
template <int Bits, int TK> struct GuardOK { static constexpr bool v = fold::Check<Bits, TK>::ok; };
static_assert(GuardOK<1, 128>::v && GuardOK<1, 256>::v, "int1");
static_assert(GuardOK<2,  64>::v && GuardOK<2, 128>::v && GuardOK<2, 256>::v, "int2");
static_assert(GuardOK<4,  64>::v && GuardOK<4, 128>::v && GuardOK<4, 256>::v, "int4");
static_assert(fold::Check<16, 64>::ok && fold::Check<8, 64>::ok && fold::Check<0, 64>::ok,
              "non-sub-byte / absent plane must be skipped, not rejected");

// ---------------------------------------------------------------- section 2: full traits on real tuples
struct Row { const char* src; int bits; int tm, tn, tk, st, wm, wn; int F, Ng, deliv, slots, smem, warps, blocks; };
static Row rows[64]; static int nrows = 0;

template <int B, int TM, int TN, int TK, int ST, int WM, int WN>
void add(const char* src) {
  using T = fold::FoldTraits<B, TM, TN, TK, ST, WM, WN>;
  rows[nrows++] = {src, B, TM, TN, TK, ST, WM, WN,
                   T::F, T::Ng, T::delivery, T::slots, T::smem, T::warps, T::blocks};
}

int main() {
  // --- test_q3_bconcat_bench.cu : BC() is two-plane, so BOTH widths are checked at the same tile
  #define BC(TM,TN,TK,WM,WN,S) add<2,TM,TN,TK,S,WM,WN>("BC lo int2"); add<1,TM,TN,TK,S,WM,WN>("BC hi int1")
  BC(16, 64,256,16,32,3); BC(16,128,256,16,32,3); BC(16,256,256,16,32,3);
  BC(32, 32,256,32,32,3); BC(32, 64,256,32,32,3); BC(32, 64,256,32,32,2);
  BC(32,128,256,32,32,3); BC(64, 64,256,32,32,3); BC(64, 64,256,32,32,2);
  #undef BC
  // --- single-plane sweeps in the same file
  add<1,16, 64,256,3,16,32>("I1"); add<1,16,128,256,3,16,32>("I1"); add<1,16,256,256,3,16,32>("I1");
  add<1,32, 32,256,3,32,32>("I1"); add<1,32, 64,256,3,32,32>("I1"); add<1,32,128,256,3,32,32>("I1");
  add<1,64, 64,256,3,32,32>("I1");
  add<2,32, 64,128,3,32,32>("I2"); add<2,32,128,128,3,32,32>("I2"); add<2,64, 64,128,3,32,32>("I2");
  add<2,64, 64,128,4,32,32>("I2"); add<2,64,128,128,3,32,64>("I2");
  add<4,32, 64, 64,3,32,32>("I4"); add<4,64, 64, 64,3,32,32>("I4"); add<4,64, 64, 64,4,32,32>("I4");
  add<4,64,128, 64,3,32,64>("I4"); add<4,64, 64,128,3,32,32>("I4");
  // --- test_fold_int2.cu (both the correctness ladder and the fixed perf lambda)
  add<1,32,128,128,3,32,32>("fold int1 winner"); add<1,64, 64,128,3,32,32>("fold int1 alt");
  add<2,64, 64, 64,3,32,32>("fold int2");        add<2,64,128, 64,3,32,32>("fold int2 wideN");
  add<2,64, 64,128,3,32,32>("fold int2 TK128");  add<4,64, 64, 64,3,32,32>("fold int4 ref");
  add<4,64, 64,128,3,32,32>("fold int4 TK128 (the SK experiment)");
  // --- misc W4A16 / W2A16 / real-weight tests
  add<4,64, 64,256,3,32,32>("misc"); add<2,64, 64,256,3,32,32>("misc"); add<1,64, 64,256,3,32,32>("misc");

  std::printf("SECTION 1 -- guard exhaustive over distinct (Bits,TK): compiled, so nothing in the tree is rejected.\n");
  std::printf("             pairs covered: int1{128,256} int2{64,128,256} int4{64,128,256} + skipped{0,8,16}\n\n");
  std::printf("SECTION 2 -- full FoldTraits on %d concrete tuples from the tree\n\n", nrows);
  std::printf("  %-32s %5s %-16s %2s %3s %5s %5s %7s %5s %6s\n",
              "source", "width", "tile(M,N,K)/warp", "F", "Ng", "deliv", "slots", "smem", "warps", "blocks");
  int tight = 0;
  for (int i = 0; i < nrows; ++i) {
    Row& r = rows[i];
    char tile[32]; std::snprintf(tile, sizeof tile, "(%d,%d,%d)/%dx%d", r.tm, r.tn, r.tk, r.wm, r.wn);
    char w[8];     std::snprintf(w, sizeof w, "int%d", r.bits);
    bool eq = (r.deliv == r.slots); tight += eq;
    std::printf("  %-32s %5s %-16s %2d %3d %5d %5d %7d %5d %6d%s\n",
                r.src, w, tile, r.F, r.Ng, r.deliv, r.slots, r.smem, r.warps, r.blocks,
                eq ? "   (I1 exactly tight)" : "");
  }
  std::printf("\n  all %d instantiated -- every FoldTraits invariant holds, including the ones the kernel guard\n"
              "  deliberately omits (I1, alignment, run size). %d sit exactly at the I1 boundary: correct today,\n"
              "  but they have no headroom, so lowering their TN would silently drop weights.\n", nrows, tight);
  return 0;
}
