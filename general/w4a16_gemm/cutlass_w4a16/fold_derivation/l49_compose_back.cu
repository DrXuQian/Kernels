// L49 -- compose the whole chain FORWARD and require it to land on the identity. The check I kept almost doing.
//
// Reverse derivation is the right tool; my three failures were all wrong STARTING POINTS, not a wrong method:
//   l44  identified a plane's PHYSICAL row with a logical N column -- true only if the offline is the identity
//   l46  built T2 from partition_B over the physical shape, inheriting the same conflation
//   the kernel index rewrite then followed l46 and made the box worse (15010 -> 29666 bad)
// None of them closed the loop. This one does, and it is still pure composition -- no simulator, no box:
//
//     delivered position (thread, vreg, code)  --[ tile_map = the offline placement ]-->  logical (n, k)
//
// Do that for BOTH planes, pair them the way MixGemm2Plane_uint2_uint1 actually pairs them, and require the two
// logical elements to be equal. F2=1 MUST pass -- that configuration measures bad=0 on the box -- so the check has a
// built-in gate, which is exactly what l44/l46 lacked.
//
// The pairing, from the _E2 lines (l37):
//     line (t, v):  LOW  crumb of lo[v] at code (t%4) + 4*(t/4) [+8]
//                   HIGH bit of hi[hi_vreg0 + 2*(v>>1)] at 8*(v&1) + t [+16]
//     hi_vreg0 = k_block % P2_DIV        <-- the SHIPPED expression, restored after the revert
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l49_compose_back.cu -o l49 && ./l49
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using namespace cute;

// l20's tile_map, verbatim in structure: destination from the swzl TV formula, source from partition_B o pi o
// MixGemmEmit. Validated byte-identical to the shipped offline for int1/int2/int4 (l20) and, for a folded plane, in
// l47 at F2=1.
// DL = deliveries per physical row = (F*TK*Bits/8)/32. l20's frame silently assumed DL==1 (8*CPW == TK), which holds
// for int1@256 / int2@128 / int4@64 and for any folded plane, but NOT for int2@TK=256 -- exactly the configuration
// needed as the gate. With DL==1 this is byte-for-byte l20. `Order` picks which of (delivery, N-instance) is the outer
// chunk in the fragment's linear order; it is a single bit, resolved against the gate rather than guessed.
template <int Bits, int TM, int TN, int TK, int WM, int WN, int F, int Order = 0>
static std::vector<int> tmap() {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  constexpr int WOM = TM / warpM, WON = TN / warpN;
  constexpr int CPW = 32 / Bits, Ng = TN / F, RPI = WON * 16, VEC = 4 * 32 / Bits;
  constexpr int DL = (F * TK * Bits / 8) / 32;                   // deliveries per physical row
  static_assert(DL >= 1 && DL * 8 * CPW == F * TK, "row must be a whole number of 32B deliveries");
  constexpr int MmaPermK = 32 * 8 / Bits;                        // non-fold builder rule
  using Mma = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<MmaPermK>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>, Int<TK>>{}, Stride<Int<TK>, _1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));
  auto pi = right_inverse(frag.layout());

  std::vector<int> m((size_t)Ng * DL * 8 * CPW, -1);
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int dl = 0; dl < DL; ++dl)
      for (int inst = 0; inst < Ng / RPI; ++inst)
        for (int v = 0; v < 4; ++v) {
          const int row = inst * RPI + 16 * warp_n + (v / 2) * 8 + lane / 4;
          const int wd  = (v % 2) * 4 + lane % 4;
          const int chunk = Order ? (inst * DL + dl) : (dl * (Ng / RPI) + inst);
          for (int j = 0; j < CPW; ++j) {
            const int e = cutlass::MixGemmEmit<Bits>::index(j, v);
            const int flat = chunk * VEC + e;
            if (flat < 0 || flat >= NS) continue;
            auto c = part(pi(flat));
            m[(((size_t)row * DL + dl) * 8 + wd) * CPW + j] = int(get<0>(c)) * TK + int(get<1>(c));
          }
        }
  }
  return m;
}

// HV picks the candidate high-vreg-offset formula. hi_vreg0 decides which of plane 2's four vregs a given
// (k_block, ii) reads, and it is the ONE thing the fold changes. Enumerated against the gate instead of guessed --
// the previous guess (ii moved into the offset) made the box worse.
template <int TM, int TN, int TK, int WM, int WN, int F2, int P2_DIV, int Order, int HV>
static bool compose(const char* tag, bool quiet = false, std::vector<int>* want = nullptr) {
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  constexpr int WOM = TM / warpM, WON = TN / warpN, RPI = WON * 16;
  constexpr int CPW1 = 16, CPW2 = 32;                       // int2, int1
  constexpr int Ng1 = TN, Ng2 = TN / F2;

  constexpr int DL1 = (TK * 2 / 8) / 32, DL2 = (F2 * TK / 8) / 32;
  const auto m1 = tmap<2, TM, TN, TK, WM, WN, 1,  Order>();
  const auto m2 = tmap<1, TM, TN, TK, WM, WN, F2, Order>();

  constexpr int NI1 = Ng1 / RPI, NI2 = Ng2 / RPI;      // N-instances per plane == MMA_N
  long paired = 0, bad = 0, skipped = 0;
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
   for (int ii = 0; ii < NI1; ++ii)
    for (int kb = 0; kb < P2_DIV; ++kb)
      for (int v = 0; v < 4; ++v)
        for (int lt = 0; lt < 8; ++lt)
          for (int half = 0; half < 2; ++half) {
            // LOW: vreg v, code (lt%4) + 4*(lt/4) [+8]  -- plane 1's delivered position
            const int j1 = (lt % 4) + 4 * (lt / 4) + 8 * half;
            const int row1 = ii * RPI + 16 * warp_n + (v / 2) * 8 + lane / 4, wd1 = (v % 2) * 4 + lane % 4;
            if (row1 >= Ng1) { ++skipped; continue; }
            const int e1 = m1[(((size_t)row1 * DL1 + kb % DL1) * 8 + wd1) * CPW1 + j1];
            // HIGH: vreg hi_vreg0 + 2*(v>>1), bit 8*(v&1) + lt [+16] -- plane 2's delivered position
            const int inst2 = (NI2 > 1) ? (ii % NI2) : 0;
            int base;
            switch (HV) {
              case 0: base = (kb % P2_DIV); break;                                  // shipped
              case 1: base = (kb % P2_DIV) + P2_DIV * (ii / NI2); break;            // the rewrite that made it worse
              case 2: base = (kb % P2_DIV) + (ii / NI2); break;
              case 3: base = (ii / NI2) * 2 + (kb % P2_DIV); break;
              default: base = 0;
            }
            const int hv = base + 2 * (v >> 1), hb = 8 * (v & 1) + lt + 16 * half;
            const int v2 = hv, j2 = hb;
            if (v2 >= 4) { ++skipped; continue; }
            const int row2 = inst2 * RPI + 16 * warp_n + (v2 / 2) * 8 + lane / 4, wd2 = (v2 % 2) * 4 + lane % 4;
            if (row2 >= Ng2) { ++skipped; continue; }
            const int e2 = m2[(((size_t)row2 * DL2 + (kb / P2_DIV) % DL2) * 8 + wd2) * CPW2 + j2];
            if (e1 < 0) { ++skipped; continue; }
            // RECORD what plane 2 must hold here. This is T2, but now derived through a composition whose gate
            // PASSES (F2=1 -> 0 mispaired, matching the box's bad=0) -- unlike l45/l46, whose gate was self-consistency.
            if (want) {
              const size_t at = (((size_t)row2 * DL2 + (kb / P2_DIV) % DL2) * 8 + wd2) * CPW2 + j2;
              if ((*want)[at] >= 0 && (*want)[at] != e1) ++bad;      // two threads demanding different elements = fatal
              (*want)[at] = e1;
            }
            if (e2 < 0) { ++skipped; continue; }
            ++paired; if (!want && e1 != e2) ++bad;
          }
  }
  if (!quiet) printf("  %-30s HV=%d : paired=%ld  MISPAIRED=%ld  skipped=%ld  %s\n",
         tag, HV, paired, bad, skipped,
         bad ? "<-- the two planes do NOT agree" : "<-- composition lands on the identity");
  return bad == 0;
}

int main() {
  printf("L49 -- compose offline o delivery o converter-pairing and require identity\n\n");
  printf("  GATE: F2=1 measures bad=0 on the box, so it MUST pass here or this check is worthless\n");
  bool g = compose<64, 64,256,32,32, 1, 2, 1, 0>("Q3 (64,64,256) shipped HV");
  printf("\n  the folded config the box reports wrong\n");
  compose<64, 64,128,32,32, 2, 1, 1, 0>("Q3 (64,64,128) HV=0 shipped");
  compose<64, 64,128,32,32, 2, 1, 1, 1>("Q3 (64,64,128) HV=1 the bad rewrite");
  compose<64, 64,128,32,32, 2, 1, 1, 2>("Q3 (64,64,128) HV=2");
  compose<64, 64,128,32,32, 2, 1, 1, 3>("Q3 (64,64,128) HV=3");
  // ---- read the required placement straight out of the composition, and gate THAT on F2=1
  {
    constexpr int CPW2 = 32;
    auto derive = [&](auto tagv, auto&& runner, int Ng2, int DL2) {
      std::vector<int> want((size_t)Ng2 * DL2 * 8 * CPW2, -1);
      runner(&want);
      long unset = 0; for (int e : want) if (e < 0) ++unset;
      return std::make_pair(want, unset);
    };
    printf("\n  the placement the composition DEMANDS of plane 2\n");
    { std::vector<int> want((size_t)64 * 1 * 8 * 32, -1);
      compose<64, 64,256,32,32, 1, 2, 1, 0>("(derive F2=1)", true, &want);
      const auto ship = tmap<1, 64, 64,256,32,32, 1, 1>();
      long d = 0, unset = 0; for (size_t i = 0; i < want.size(); ++i) { if (want[i] < 0) ++unset; else if (want[i] != ship[i]) ++d; }
      printf("    F2=1 demanded vs shipped map : %ld differ, %ld unset  %s\n", d, unset,
             (!d && !unset) ? "<-- GATE PASSED: the derivation reproduces what works" : "<-- derivation is wrong");
    }
    // 4096 unset under the shipped HV means vregs 1 and 3 are NEVER READ -- half the tile's high bits cannot arrive
    // at all, whatever the placement. So hi_vreg0 must vary with ii. That is HV=1, the rewrite that made the box worse
    // ON ITS OWN: it changed WHICH vregs are read without moving the bits. Kernel and offline are one change.
    for (int hv : {0, 1}) {
      std::vector<int> want((size_t)32 * 1 * 8 * 32, -1);
      if (hv == 0) compose<64, 64,128,32,32, 2, 1, 1, 0>("(derive)", true, &want);
      else         compose<64, 64,128,32,32, 2, 1, 1, 1>("(derive)", true, &want);
      const auto cur = tmap<1, 64, 64,128,32,32, 2, 1>();
      long d = 0, unset = 0; for (size_t i = 0; i < want.size(); ++i) { if (want[i] < 0) ++unset; else if (want[i] != cur[i]) ++d; }
      printf("    F2=2 HV=%d : demanded vs in-use %ld / %zu differ, %ld unset  %s\n", hv, d, want.size(), unset,
             unset ? "<-- half the high bits are never fetched; no placement can fix this"
                   : (d ? "<-- COMPLETE demand: this is the offline the folded path needs" : "<-- already in use"));
    }
  }

  printf("\n  %s\n", g ? "gate passed -- the mispaired count above is a real measurement of the folded defect"
                       : "GATE FAILED -- this check models something other than the working kernel; fix it first");
  return 0;
}
