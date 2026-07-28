// L41 -- the 2-plane chunk gate, DERIVED. Replaces l38's model (which used the fold path's predicate) now that l40
// has the real layouts.
//
// l40: the 2-plane convert's flat write IS an exact partition of tCrB_mma, because cvt_in's mode-1 stride (128) equals
// tCrB_mma's MMA_N stride (128). So per `ii` the converter fills 64 CONTIGUOUS fp16 = 8 k-atoms of one n, laid out
// val + 8*k with k INNER. That makes the chunk axis a plain division of the h2 index -- no right_inverse composition
// needed, unlike the fold path.
//
// The claim, in three lines:
//   at_plain(t,v) = 16*(v&1) + 2*(v>>1) + 4*(t>>1) + (t&1)      == a cute Layout, colex over ((2,4),(2,2))
//   k-atom        = at_plain / 4                                 (4 half2 == 8 fp16 == one mma B atom)
//   at_chunk      = at_plain % 4                                 (index inside a ONE-ATOM buffer)
// at_plain must reproduce the eight hardcoded `h2[base + {0,1,4,5,8,9,12,13}]` indices of the shipping converter with
// base = 16*(v&1) + 2*(v>=2), or the gate is gating something else. Checked against them literally.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l41_2plane_chunk_gate.cu -o l41 && ./l41
#include "cute/tensor.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using namespace cute;

// the shipping converter's own indices, transcribed from MixGemm2Plane_uint2_uint1::convert
static int shipped_at(int t, int v) {
  const int base = 16 * (v & 1) + 2 * (v >= 2);
  static const int off[8] = {0, 1, 4, 5, 8, 9, 12, 13};
  return base + off[t];
}

using H2 = Layout<Shape <Shape<_2,_4>, Shape< _2,_2>>,
                  Stride<Stride<_1,_4>, Stride<_16,_2>>>;

int main() {
  constexpr int NLINE = 8, NVREG = 4, NH2 = 32, PER = 4;   // 32 half2 = 64 fp16 = 8 atoms x 4 half2
  printf("H2 layout: "); print(H2{}); printf("   size=%d cosize=%d\n", int(size(H2{})), int(cosize(H2{})));

  // 1. the layout must reproduce the shipping converter's hardcoded destinations
  long bad = 0;
  for (int v = 0; v < NVREG; ++v)
    for (int t = 0; t < NLINE; ++t) {
      const int got = int(H2{}(t + NLINE * v)), want = shipped_at(t, v);
      if (got != want) { if (++bad <= 8) printf("  t=%d v=%d : layout %2d  shipped %2d  <-- DIFFER\n", t, v, got, want); }
    }
  printf("1. at_plain == shipping converter's indices : %ld / %d differ  %s\n",
         bad, NLINE * NVREG, bad ? "<-- the layout is NOT the converter's map" : "OK");

  // 2. it must be a bijection onto the 32 half2 slots
  std::vector<int> hit(NH2, 0);
  for (int v = 0; v < NVREG; ++v) for (int t = 0; t < NLINE; ++t) ++hit[int(H2{}(t + NLINE * v))];
  int once = 0; for (int h : hit) if (h == 1) ++once;
  printf("2. bijection onto %d half2 slots            : once=%d  %s\n", NH2, once, once == NH2 ? "OK" : "<-- NOT");

  // 3. per chunk: exactly PER half2, all four at_chunk values distinct, and the union over chunks a partition
  constexpr int NCHUNK = NH2 / PER;
  std::vector<int> cover(NH2, 0);
  bool ok3 = true;
  for (int c = 0; c < NCHUNK; ++c) {
    int lines = 0; std::vector<int> slot(PER, 0);
    for (int v = 0; v < NVREG; ++v)
      for (int t = 0; t < NLINE; ++t) {
        const int e = int(H2{}(t + NLINE * v));
        if (e / PER != c) continue;                       // keep()
        ++lines; ++slot[e % PER]; ++cover[e];             // at() inside a one-atom buffer
      }
    int d = 0; for (int s : slot) if (s == 1) ++d;
    if (lines != PER || d != PER) { ok3 = false;
      printf("  chunk %d: lines=%d (want %d) distinct-slots=%d (want %d)  <-- WRONG\n", c, lines, PER, d, PER); }
  }
  int cov = 0; for (int h : cover) if (h == 1) ++cov;
  printf("3. %d chunks x %d half2, exact partition     : %s (covered once=%d/%d)\n",
         NCHUNK, PER, (ok3 && cov == NH2) ? "OK" : "<-- WRONG", cov, NH2);

  // 4. which (vreg, line) each chunk needs -- this is what the emitter's `if constexpr` will select, so print it:
  //    a chunk must touch only TWO of the four vregs, or a chunked pass would still have to read all four.
  printf("4. per-chunk source vregs (a chunk should need only 2 of 4, so the OTHER two vregs' lop3 are skipped):\n");
  for (int c = 0; c < NCHUNK; ++c) {
    printf("     chunk %d : vregs", c);
    int nv = 0;
    for (int v = 0; v < NVREG; ++v) {
      bool used = false;
      for (int t = 0; t < NLINE; ++t) if (int(H2{}(t + NLINE * v)) / PER == c) used = true;
      if (used) { printf(" %d", v); ++nv; }
    }
    printf("   (%d vregs)%s\n", nv, nv == 2 ? "" : "  <-- not 2");
  }
  // 5. THE HARNESS MUST TEST THE SHIPPED CODE, NOT A TRANSCRIPTION OF IT (l29/l35 pattern). Bind to the real
  //    converter's own at/keep/vreg_used, so an edit to the header that diverges from this derivation fails HERE.
  bool ok5 = true;
  for (int v = 0; v < NVREG; ++v)
    for (int t = 0; t < NLINE; ++t) {
      if (cutlass::MixGemm2Plane_uint2_uint1<>::at(t, v) != shipped_at(t, v)) ok5 = false;      // Chunk<0 path
      if (!cutlass::MixGemm2Plane_uint2_uint1<>::keep(t, v)) ok5 = false;                       // Chunk<0 keeps all
    }
  for (int c = 0; c < NCHUNK; ++c) {
    int lines = 0, vregs = 0;
    for (int v = 0; v < NVREG; ++v) {
      bool uv = false;
      for (int t = 0; t < NLINE; ++t) {
        const bool k = (int(H2{}(t + NLINE * v)) / PER) == c;
        if (k) { ++lines; uv = true; }
      }
      if (uv) ++vregs;
    }
    if (lines != PER || vregs != 2) ok5 = false;
  }
  // spot-check every (chunk, t, v) against the real template, instantiated for each chunk
  auto chk = [&](auto cc) {
    constexpr int C = decltype(cc)::value;
    using M = cutlass::MixGemm2Plane_uint2_uint1<C>;
    for (int v = 0; v < NVREG; ++v)
      for (int t = 0; t < NLINE; ++t) {
        const bool want_keep = (int(H2{}(t + NLINE * v)) / PER) == C;
        if (M::keep(t, v) != want_keep) ok5 = false;
        if (want_keep && M::at(t, v) != int(H2{}(t + NLINE * v)) % PER) ok5 = false;
        if (M::vreg_used(v) != [&]{ for (int tt=0;tt<NLINE;++tt) if ((int(H2{}(tt+NLINE*v))/PER)==C) return true; return false; }()) ok5 = false;
      }
  };
  for_each(make_int_sequence<8>{}, chk);
  printf("5. the SHIPPED converter's at/keep/vreg_used  : %s\n", ok5 ? "agrees on all 8 chunks x 32 lines" : "<-- DIVERGES");

  printf("\nVERDICT: %s\n", (!bad && once == NH2 && ok3 && cov == NH2 && ok5)
         ? "gate is at_plain/4 == Chunk, at_plain%4 -- one cute Layout, no right_inverse needed"
         : "DO NOT implement -- see the WRONG lines above");
  return 0;
}
