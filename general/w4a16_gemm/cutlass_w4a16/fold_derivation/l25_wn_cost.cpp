// L25 -- what does the WN=64 escape actually cost? Derived, at the SAME (TM,TN,TK).
//
// int1 cannot use TK=64 at WN=32 (delivery 128 codes vs slots = WN*TK/32 = 64), so the only way in is WN=64. That
// looked free -- the placement is WN-invariant, verified byte-for-byte in l20. It is not free in REGISTERS, and this
// is derivable without a box.
//
// WN is the N extent ONE WARP covers. Doubling it halves the warps per block, so every per-thread fragment doubles:
//     warps           = (TM/WM) * (TN/WN)
//     accum/thread    = TM*TN / (32*warps)                floats  -> 1 reg each
//     B frag/thread   = WN*TK/32                          half    -> /2 regs   (= slots)
//     A frag/thread   = TM*TK / (32*(TM/WM))              half    -> /2 regs   (A is duplicated across the N warps)
//     scale/thread    = slots/4                           half    -> /8 regs   (cosize, measured in l21: size/4)
// The accumulator is the big term because it is fp32 and scales with TM*TN/threads.
#include <cstdio>
#include <initializer_list>
#include <utility>
struct R { int warps, accum, bfrag, afrag, scale, total, blocks, smem; };
static R regs(int Bits, int TM, int TN, int TK, int WM, int WN, int ST, int gs, bool zero) {
  R r; r.warps = (TM/WM) * (TN/WN);
  const int thr = 32 * r.warps;
  const int slots = WN*TK/32;
  r.accum = TM*TN / thr;                       // fp32
  r.bfrag = slots / 2;                         // fp16 -> 2 per reg
  r.afrag = (TM*TK / (32*(TM/WM))) / 2;
  r.scale = (slots/4) / 2 * (zero ? 2 : 1);
  r.total = r.accum + r.bfrag + r.afrag + r.scale;
  const int sk = TK/gs;
  r.smem = (TM*TK*2 + TN*TK*Bits/8 + TN*sk*2*(zero?2:1)) * ST;
  const int bs = 262144/r.smem, bw = 64/r.warps;
  r.blocks = bs < bw ? bs : bw;
  return r;
}
static void row(const char* t, int Bits, int TM, int TN, int TK, int WM, int WN, int ST, int gs, bool z) {
  R r = regs(Bits,TM,TN,TK,WM,WN,ST,gs,z);
  printf("  %-26s int%d (%d,%d,%d) w%dx%-3d | warps=%d thr=%-4d | accum=%-3d B=%-3d A=%-2d S=%-2d = %-3d regs/thr"
         " | smem=%6dB blocks=%d\n", t, Bits, TM,TN,TK, WM,WN, r.warps, 32*r.warps,
         r.accum, r.bfrag, r.afrag, r.scale, r.total, r.smem, r.blocks);
}
int main() {
  printf("L25 -- the register cost of raising WN, at identical tiles\n\n");
  printf("== CLEAN A/B: same (64,64,64), only Bits/F/Ng/WN differ. int1 must use WN=64 (delivery bound).\n");
  row("int2 F=2 WN=32", 2, 64,64,64, 32,32, 3, 32, false);
  row("int1 F=4 WN=64 (forced)", 1, 64,64,64, 32,64, 3, 32, false);
  printf("\n== the two configs actually MEASURED (not a clean A/B -- TN differs too)\n");
  row("int2 53.2%%", 2, 64, 64,64, 32,32, 3, 32, false);
  row("int1 46.4%%", 1, 64,128,64, 32,64, 3, 32, false);
  printf("\n== and the TK=128 pair, where int1 WINS (42.0 vs 37.1) and does NOT need a raised WN\n");
  row("int2 37.1%%", 2, 32,128,128, 32,32, 3, 32, false);
  row("int1 42.0%%", 1, 32,128,128, 32,32, 3, 32, false);
  printf("\n== with zero, gs=16: the case Q3/Q5 actually run\n");
  row("int1 TK=128 WN=32", 1, 32,128,128, 32,32, 3, 16, true);
  row("int1 TK=64  WN=64", 1, 64,128, 64, 32,64, 3, 16, true);
  printf("\n== THE PRESCRIPTION. accum/thread = TM*TN/(32*(TM/WM)*(TN/WN)) = WM*WN/32 -- it depends ONLY on the WARP\n");
  printf("   shape, not on TM or TN. The delivery bound constrains WN alone, so WM is free: keep WN=64 to clear the\n");
  printf("   bound and drop WM to 16, and the accumulator halves.\n");
  row("int1 TK=64 w32x64 (now)", 1, 64,128,64, 32,64, 3, 32, false);
  row("int1 TK=64 w16x64 (fix)", 1, 32,128,64, 16,64, 3, 32, false);
  row("int1 TK=64 w16x64 s2",   1, 32,128,64, 16,64, 2, 32, false);
  row("int1 TK=64 w16x64 TN256", 1, 32,256,64, 16,64, 3, 32, false);
  printf("   and the case Q3/Q5 run, gs=16 with zero:\n");
  row("int1 TK=64 w32x64 gs16 Z", 1, 64,128,64, 32,64, 3, 16, true);
  row("int1 TK=64 w16x64 gs16 Z", 1, 32,128,64, 16,64, 3, 16, true);

  printf("\n  READ: at the clean A/B the smem and blocks are nearly equal, but int1 pays 2x the registers per\n");
  printf("  thread -- because WN=64 halves the warps, so accum, B and scale all double per thread. The accumulator\n");
  printf("  dominates (fp32, TM*TN/threads). Against 256 regs/thread and 131072 regs/CU that is the plausible\n");
  printf("  mechanism for int1 losing at TK=64 while winning at TK=128, where no WN raise is needed.\n");
  printf("\n  regs/CU check: blocks * threads * regs must fit 131072.\n");
  for (auto c : {std::pair<const char*,int>{"int2 (64,64,64) w32x32",2}, {"int1 (64,64,64) w32x64",1}}) {
    R r = c.second==2 ? regs(2,64,64,64,32,32,3,32,false) : regs(1,64,64,64,32,64,3,32,false);
    const long need = (long)r.blocks * 32*r.warps * r.total;
    printf("    %-24s blocks=%d x thr=%d x regs=%d = %6ld  %s\n", c.first, r.blocks, 32*r.warps, r.total, need,
           need > 131072 ? "<-- EXCEEDS 131072, occupancy is register-limited" : "fits");
  }
  return 0;
}
