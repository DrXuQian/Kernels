// L24 -- at the SAME (TM, TN, TK), what actually differs between int1 and int2?
// Every quantity below is from the builder/collective, not estimated:
//   F           = 32/(TK*Bits/8)                        fold factor, so the 32B AIU run is reached
//   Ng          = TN/F                                  physical rows of the smem cube
//   B-smem      = Ng * (F*TK*Bits/8) = TN*TK*Bits/8     F cancels
//   A-smem      = TM*TK*2                               Bits-INDEPENDENT
//   slots       = WN*TK/32                              fp16 fragment slots -- Bits-INDEPENDENT
//   delivery    = 16*8/Bits                             codes one swzl hands a thread
//   AiuContElem = min(TK*Bits/8,128)*8/Bits             elements per AIU instruction
//   bits/AIU    = Ng * AiuContElem * Bits               payload of one AIU instruction
//   converter   = 2 asm ops per 8 output fp16 per vreg   (16 _E per 128 outputs for int1, 8 per 64 for int2)
#include <cstdio>
#include <initializer_list>
struct Q { int F, Ng, bsmem, asmem, slots, deliv, aiuelem, aiubits, ncopy, nconv, mma; };
static Q q(int Bits, int TM, int TN, int TK, int WN) {
  Q r; const int contig = TK*Bits/8;
  r.F = contig >= 32 ? 1 : 32/contig;
  r.Ng = TN/r.F;
  r.bsmem = TN*TK*Bits/8;  r.asmem = TM*TK*2;
  r.slots = WN*TK/32;      r.deliv = 16*8/Bits;
  const int ab = contig > 128 ? 128 : contig;
  r.aiuelem = ab*8/Bits;   r.aiubits = r.Ng*r.aiuelem*Bits;
  r.ncopy = r.slots / r.deliv;                 // swzl instructions to fill the fragment
  r.nconv = r.slots / (4*32/Bits);             // converter calls (CPY_VEC = 4*32/Bits)
  r.mma   = TK/16;                             // mma k-atoms per iteration
  return r;
}
int main() {
  printf("int1 vs int2 at the SAME (TM,TN,TK). WN=32 unless the delivery bound forbids it.\n\n");
  for (int TK : {128, 64}) {
    const int TM = 32, TN = 128;
    printf("== TM=%d TN=%d TK=%d\n", TM, TN, TK);
    for (int Bits : {2, 1}) {
      int WN = 32;
      while ((long)WN*TK*Bits < 4096) WN *= 2;          // smallest legal WN
      Q r = q(Bits, TM, TN, TK, WN);
      printf("   int%d WN=%-3d | F=%d Ng=%-3d | B-smem=%5dB A-smem=%5dB | slots=%3d deliv=%3d -> %d swzl, %d conv"
             " | AIU %5d bits/instr | mma atoms=%d\n",
             Bits, WN, r.F, r.Ng, r.bsmem, r.asmem, r.slots, r.deliv, r.ncopy, r.nconv, r.aiubits, r.mma);
    }
    printf("\n");
  }
  printf("WHAT IS IDENTICAL at fixed (TM,TN,TK,gs):\n");
  printf("   mma instructions        -- same tile, same atoms\n");
  printf("   A operand entirely      -- A-smem = TM*TK*2, fp16, Bits-independent\n");
  printf("   fragment size (slots)   -- WN*TK/32 counts fp16, and both widths produce the same fp16 count\n");
  printf("   converter ALU per output-- 0.25 asm ops per output fp16 for BOTH (16 _E/128 vs 8 _E/64)\n");
  printf("   scale and zero          -- per (n, k/gs) in fp16, so identical in ABSOLUTE terms\n");
  printf("   AIU instruction count   -- same number of instructions, each carrying half the payload for int1\n\n");
  printf("WHAT ACTUALLY DIFFERS:\n");
  printf("   1. B-smem halves -- but A dominates. At TM=32/TN=128/TK=128: A=8KB, B goes 4KB -> 2KB, i.e. ~17%% of\n");
  printf("      the tile. That is the whole occupancy story, and it is small.\n");
  printf("   2. F doubles, so Ng halves: the cube has half the rows and each row still spans 32B.\n");
  printf("   3. The delivery bound is 2x stricter. delivery = 128/Bits, slots is Bits-independent, so int1 hits\n");
  printf("      over-delivery at TWICE the TK int2 does -- int1 needs TK>=128 at WN=32, int2 only TK>=64.\n");
  printf("   4. scale/zero is unchanged in absolute cost, so RELATIVE to weight bytes it is 2x more expensive.\n");
  printf("   5. Half the swzl instructions fill the same fragment, and each AIU instruction carries half the bytes.\n\n");
  printf("CONSEQUENCE: at equal (TN,TK) int1 does the same mma, the same scale work, the same AIU instruction count\n");
  printf("and the same converter ALU as int2. It only saves smem. So int1 should be MARGINALLY faster than int2, not\n");
  printf("faster in proportion to being half the size -- and that is what the box says: int2@TK128 37.1%%,\n");
  printf("int1@TK128 42.0%%, a 4.9 point gap consistent with ~17%% less smem.\n");
  return 0;
}
