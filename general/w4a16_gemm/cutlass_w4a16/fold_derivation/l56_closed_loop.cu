// L56 -- the CLOSED LOOP I should have built five rounds ago.
//
// Every earlier harness modelled a SEGMENT of the chain and compared it against something, so each could be
// self-consistent and wrong. The missing question is non-circular and gated by the MMA itself:
//
//     the fp16 the converter actually writes at fragment memory offset o -- which logical (n,k) is it,
//     and is that the (n,k) the mma expects at o, i.e. part(pi(o)) ?
//
// It is non-circular because plane_map was BUILT assuming the destination is
//     flat = chunk*VEC + MixGemmEmit<Bits>::index(j, v)
// while the converter actually writes
//     dest = stride1*ii + 2*at_plain(t_line, v) + half
// with at_plain from MixGemm2Plane_uint2_uint1's own _E2 lines. Those two expressions have never been compared.
//
// Checks, per thread / copy-step / N-instance / (_E2 line, half):
//   A. plane 1's delivered code carries the (n,k) the mma wants at dest        <- emission vs placement
//   B. plane 2's delivered bit carries the SAME (n,k)                          <- cross-plane pairing
//   C. the kernel's scale group for dest's k-atom equals k / gs                <- FINE / APG schedule
// Gate: Block_K=128 w32x32 measures bad=0 on the box, so it must come out clean here or the model is wrong.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l56_closed_loop.cu -o l56 && ./l56
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include "xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;

// the 2-plane converter's own (line, vreg) -> half2 slot map, l41's cute Layout, transcribed as arithmetic
static inline int at_plain(int t, int v) { return (t & 1) + 4 * (t >> 1) + 16 * (v & 1) + 2 * (v >> 1); }

template <int TM,int TN,int TK,int WM,int WN,int F1,int F2,int GS>
static bool loop(const char* tag) {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  constexpr int warpM = (WM > 16) ? WM : 16, warpN = (WN > 16) ? WN : 16;
  constexpr int WOM = TM / warpM, WON = TN / warpN, RPI = WON * 16, NTHR = 32 * WOM * WON;
  constexpr int CPW1 = 16, CPW2 = 32, Ng1 = TN / F1, Ng2 = TN / F2;
  constexpr int DL1 = (F1 * TK * 2 / 8) / 32, DL2 = (F2 * TK / 8) / 32;
  constexpr int NI1 = Ng1 / RPI, NI2 = Ng2 / RPI;
  constexpr int P2_DIV = (DL1 / DL2) ? (DL1 / DL2) : 1;
  constexpr int MmaPermK1 = (F1 > 1) ? TK : (32 * 8 / 2);
  using Mma = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<MmaPermK1>>>;
  auto sB   = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                          make_layout(Shape<Int<TN>, Int<TK>>{}, Stride<Int<TK>, _1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  auto pi   = right_inverse(frag.layout());
  const int MMA_K = int(size<2>(frag.layout()));
  const int stride1 = 4 * 32 / 2;                 // cvt_in's mode-1 stride in fp16: one 4-vreg chunk of int2 = 64
  const int K_ATOM = MMA_K / 1;                   // CPY_K == 1 at every config here
  const int Scale_TileK = TK / GS, APG = (Scale_TileK > 1) ? (MMA_K / Scale_TileK) : 1;

  const auto m1 = xplane::plane_map<2, TM, TN, TK, WM, WN, F1>();
  const auto m2 = xplane::tile_map_int1<TM, TN, TK, WM, WN, F2, F1>();

  long nA = 0, nB = 0, nC = 0, tot = 0;
  int showA = 0;
  // COVERAGE. The per-slot checks only look at slots that ARE fed. If some logical (n,k) of the tile is never consumed
  // by any mma atom the result is wrong and nothing above notices -- exactly the hole a per-element check leaves. Every
  // element must be fed once per M-warp (B is split across N warps only, so all WOM M-warps read the same B).
  std::vector<int> cover((size_t)TN * TK, 0);
  for (int t = 0; t < NTHR; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int kb = 0; kb < P2_DIV; ++kb)
      for (int ii = 0; ii < NI1; ++ii)
        for (int v = 0; v < 4; ++v)
          for (int lt = 0; lt < 8; ++lt)
            for (int half = 0; half < 2; ++half, ++tot) {
              // ---- what the mma wants at the destination the converter writes
              const int dest = stride1 * ii + 2 * at_plain(lt, v) + half;
              auto c = part(pi(dest));
              const int want_n = int(get<0>(c)), want_k = int(get<1>(c));
              // ---- plane 1's delivered code at (t, ii, v, j)
              const int j1 = (lt % 4) + 4 * (lt / 4) + 8 * half;
              const int row1 = ii * RPI + 16 * warp_n + (v / 2) * 8 + lane / 4, wd1 = (v % 2) * 4 + lane % 4;
              const int lo = m1[(((size_t)row1 * DL1 + kb % DL1) * 8 + wd1) * CPW1 + j1];
              // ---- plane 2's delivered bit
              const int hv = (kb % P2_DIV) + P2_DIV * (ii / (NI2 ? NI2 : 1)) + 2 * (v >> 1);
              const int j2 = 8 * (v & 1) + lt + 16 * half;
              const int inst2 = (NI2 > 1) ? (ii % NI2) : 0;
              const int row2 = inst2 * RPI + 16 * warp_n + (hv / 2) * 8 + lane / 4, wd2 = (hv % 2) * 4 + lane % 4;
              const int hi = (hv < 4 && row2 < Ng2)
                           ? m2[(((size_t)row2 * DL2 + (kb / P2_DIV) % DL2) * 8 + wd2) * CPW2 + j2] : -1;
              const int wantE = want_n * TK + want_k;
              if (lo != wantE) { ++nA; if (showA++ < 6)
                printf("      A t=%d ii=%d v=%d lt=%d h=%d dest=%d | mma wants (n=%d,k=%d) got (n=%d,k=%d)\n",
                       t, ii, v, lt, half, dest, want_n, want_k, lo / TK, lo % TK); }
              if (hi != wantE) ++nB;
              if (want_n >= 0 && want_n < TN && want_k >= 0 && want_k < TK) ++cover[(size_t)want_n*TK + want_k];
              // ---- scale group the kernel will apply at dest's k-atom
              // pi(dest) is a FLAT domain index, not a coordinate tuple -- get<2> on it is out of range. The k-atom
              // comes from the fragment's documented structure instead: MMA_K stride 8, MMA_N stride 8*MMA_K.
              const int katom = (dest / 8) % MMA_K;
              if ((katom / APG) != (want_k / GS)) ++nC;
            }
  }
  long uncov = 0, wrongmult = 0;
  for (int e : cover) { if (!e) ++uncov; else if (e != WOM) ++wrongmult; }
  printf("  %-32s pairs=%ld | A=%ld B=%ld C=%ld | coverage: never-fed=%ld mult!=WOM(%d)=%ld  %s\n",
         tag, tot, nA, nB, nC, uncov, WOM, wrongmult,
         (!nA && !nB && !nC && !uncov && !wrongmult) ? "<-- CLEAN" : "<-- see above");
  return !nA && !nB && !nC && !uncov && !wrongmult;
}

int main() {
  printf("L56 -- closed loop: does the converter's destination agree with what the mma expects there?\n\n");
  printf("  GATE: Block_K=128 w32x32 measures bad=0 on the box\n");
  bool g = loop<64, 64,128,32,32, 1,2, 16>("(64, 64,128) w32x32 GATE");
  printf("\n  the ladder's rungs\n");
  loop<64,128,128,32,32, 1,2, 16>("(64,128,128) w32x32");
  loop<64,128,128,32,64, 1,2, 16>("(64,128,128) w32x64");
  loop<64,128,128,64,64, 1,2, 16>("(64,128,128) w64x64");
  loop<64,128, 64,64,64, 2,4, 16>("(64,128, 64) w64x64 F1=2 F2=4");
  printf("\n  %s\n", g ? "gate clean -- the failures above are real"
                       : "GATE DIRTY -- the model is wrong, not the kernel; fix the model first");
  return 0;
}
