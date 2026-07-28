// L67 -- the generalised cross-plane map: Q3 byte-identical, Q6 and Q5 complete bijections.
//
// tile_map_int1 was int1-specific in five places (CPW2, DL2, the two pairing layouts, the hi-vreg stride, and the loop
// bound over half2 pairs). tile_map_hi<LowBits, HiBits, ...> drives all five from the widths. Two things must hold:
//
//  (1) Q3 IS UNCHANGED. The old int1-only body is reproduced here verbatim as the reference and diffed entry by entry.
//      This is the only configuration where a working answer exists, so it is the only real check on the generalisation.
//  (2) Q6 AND Q5 ARE COMPLETE. Every slot of the high tile is assigned, every logical (n,k) of the tile appears exactly
//      once, and nothing is left unset. A map that merely "looks derived" can still miss half the tile -- that is
//      literally the hi_vreg0 defect, which cost a box round.
//
// Note what the widths imply: int4's contiguous run is >= 32 B for every Block_K >= 64, so Q6 and Q5 never fold their
// LOW plane (F1 = 1) and plane 1 takes the interleave-256 walk. Only the high plane folds. That makes them structurally
// simpler than Q3, where both planes fold at Block_K=64.
//
//   nvcc -std=c++17 -Istub_inc -I.. -I../../../../third_party/actlize/include l67_q65_maps.cu -o l67 && ./l67
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "../xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;

// ---- the ORIGINAL int1-only body, verbatim, as the reference for (1).
template <int TM, int TN, int TK, int WM, int WN, int F2, int F1 = 1>
static std::vector<int> old_tile_map_int1() {
  constexpr int InstM = 16, warpM = (WM > InstM) ? WM : InstM, warpN = (WN > 16) ? WN : 16;
  constexpr int WOM = TM / warpM, WON = TN / warpN, RPI = WON * 16;
  constexpr int CPW1 = 16, CPW2 = 32, Ng1 = TN / F1, Ng2 = TN / F2;
  constexpr int DL1 = (F1 * TK * 2 / 8) / 32, DL2 = (F2 * TK / 8) / 32;
  constexpr int NI1 = Ng1 / RPI, NI2 = Ng2 / RPI;
  constexpr int P2_DIV = (DL1 / DL2) ? (DL1 / DL2) : 1;
  using LoCodeL = Layout<Shape<_8, _2>,     Stride<_1, _8>>;
  using HiCodeL = Layout<Shape<_8, _2, _2>, Stride<_1, _8, _16>>;
  using CTV1 = xplane::CubeTV<2, TM, TN, TK, WM, WN, F1>;
  using CTV2 = xplane::CubeTV<1, TM, TN, TK, WM, WN, F2>;
  const auto m1 = xplane::plane_map<2, TM, TN, TK, WM, WN, F1>();
  std::vector<int> m((size_t)Ng2 * DL2 * 8 * CPW2, -1);
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32;
    for (int ii = 0; ii < NI1; ++ii)
      for (int kb = 0; kb < P2_DIV; ++kb)
        for (int v = 0; v < 4; ++v)
          for (int lt = 0; lt < 8; ++lt)
            for (int half = 0; half < 2; ++half) {
              const int j1 = int(LoCodeL{}(lt, half));
              const int row1 = CTV1::base_row(w, ii) + CTV1::cube_row(lane, v), wd1 = CTV1::cube_wd(lane, v, kb % DL1);
              if (row1 >= Ng1) continue;
              const int e1 = m1[(((size_t)row1 * DL1 + kb % DL1) * 8 + wd1) * CPW1 + j1];
              if (e1 < 0) continue;
              const int base = (kb % P2_DIV) + P2_DIV * (ii / (NI2 ? NI2 : 1));
              const int v2 = base + 2 * (v >> 1), j2 = int(HiCodeL{}(lt, v & 1, half));
              if (v2 >= 4) continue;
              const int inst2 = (NI2 > 1) ? (ii % NI2) : 0;
              const int row2 = CTV2::base_row(w, inst2) + CTV2::cube_row(lane, v2),
                        wd2 = CTV2::cube_wd(lane, v2, (kb / P2_DIV) % DL2);
              if (row2 >= Ng2) continue;
              m[(((size_t)row2 * DL2 + (kb / P2_DIV) % DL2) * 8 + wd2) * CPW2 + j2] = e1;
            }
  }
  return m;
}

template <int TM, int TN, int TK, int WM, int WN, int F2, int F1 = 1>
static bool same_as_old(const char* tag) {
  const auto a = old_tile_map_int1<TM, TN, TK, WM, WN, F2, F1>();
  const auto b = xplane::tile_map_hi<2, 1, TM, TN, TK, WM, WN, F2, F1>();
  size_t d = (a.size() != b.size()) ? a.size() : 0;
  if (!d) for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++d;
  printf("  %-40s %zu / %zu entries differ  %s\n", tag, d, a.size(), d ? "<-- Q3 CHANGED" : "IDENTICAL");
  return d == 0;
}

template <int LowBits, int HiBits, int TM, int TN, int TK, int WM, int WN, int F2, int F1 = 1>
static bool complete(const char* tag) {
  const auto m = xplane::tile_map_hi<LowBits, HiBits, TM, TN, TK, WM, WN, F2, F1>();
  std::vector<int> seen((size_t)TN * TK, 0);
  size_t unset = 0, oob = 0;
  for (size_t i = 0; i < m.size(); ++i) {
    if (m[i] < 0) { ++unset; continue; }
    if (m[i] >= TN * TK) { ++oob; continue; }
    ++seen[(size_t)m[i]];
  }
  size_t miss = 0, dbl = 0;
  for (size_t i = 0; i < seen.size(); ++i) { if (!seen[i]) ++miss; else if (seen[i] > 1) ++dbl; }
  const bool ok = !unset && !oob && !miss && !dbl && m.size() == (size_t)TN * TK;
  printf("  %-40s slots %5zu (want %5d) unset %zu oob %zu | logical: unreached %zu doubled %zu  %s\n",
         tag, m.size(), TN * TK, unset, oob, miss, dbl, ok ? "BIJECTION" : "<-- INCOMPLETE");
  return ok;
}

int main() {
  printf("L67 -- tile_map_hi: Q3 unchanged, Q6 and Q5 complete\n\n");
  int bad = 0;
  printf("  (1) Q3 against the original int1-only body\n");
  if (!same_as_old<64, 128,  64, 64, 64, 4, 2>("Q3 (64,128, 64) w64x64 F1=2 F2=4")) ++bad;
  if (!same_as_old<64, 128, 128, 32, 64, 2, 1>("Q3 (64,128,128) w32x64 F1=1 F2=2")) ++bad;
  if (!same_as_old<64,  64, 128, 32, 32, 2, 1>("Q3 (64, 64,128) w32x32 F1=1 F2=2")) ++bad;
  if (!same_as_old<64,  64, 256, 32, 32, 1, 1>("Q3 (64, 64,256) w32x32 F1=1 F2=1")) ++bad;

  printf("\n  (2) Q6 = int4 + int2. int4's run is >= 32 B for TK >= 64, so F1 = 1 always; only int2 folds.\n");
  if (!complete<4, 2, 64, 128, 128, 32, 32, 1, 1>("Q6 (64,128,128) w32x32 F1=1 F2=1")) ++bad;
  if (!complete<4, 2, 64, 128,  64, 64, 32, 2, 1>("Q6 (64,128, 64) w64x32 F1=1 F2=2")) ++bad;
  if (!complete<4, 2, 64, 128,  64, 64, 64, 2, 1>("Q6 (64,128, 64) w64x64 F1=1 F2=2")) ++bad;
  if (!complete<4, 2, 64,  64, 128, 32, 32, 1, 1>("Q6 (64, 64,128) w32x32 F1=1 F2=1")) ++bad;

  printf("\n  (3) Q5 = int4 + int1. int1 needs WN >= 4096/TK, so TK=64 forces w*x64.\n");
  if (!complete<4, 1, 64, 128, 128, 32, 32, 2, 1>("Q5 (64,128,128) w32x32 F1=1 F2=2")) ++bad;
  if (!complete<4, 1, 64, 128,  64, 64, 64, 4, 1>("Q5 (64,128, 64) w64x64 F1=1 F2=4")) ++bad;
  if (!complete<4, 1, 64, 128, 256, 32, 32, 1, 1>("Q5 (64,128,256) w32x32 F1=1 F2=1")) ++bad;

  printf("\n  %s\n", bad ? "NOT ready" : "Q3 byte-identical, Q6 and Q5 are complete bijections over their tiles");
  return bad ? 1 : 0;
}
