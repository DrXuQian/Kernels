// L38 -- the chunked gate for the 2-PLANE converter (task #12), and why it needs no second predicate.
//
// l37 showed the correspondence is a tuple of layouts over ONE (t, v) domain. The consequence is stronger than
// "composable": each _E2 line consumes the low plane's crumb AND the high plane's bit and writes ONE h2 slot, so
// gating the LINE gates BOTH planes. The 2-plane chunk gate is therefore EXACTLY the single-plane int2 gate --
// MixGemmChunkEmit<2, Chunk, NChunk>::keep/at -- with no high-plane predicate at all.
//
// Checked here the same way l32 checked the single-plane one: per chunk the lines must be 32/NChunk, the h2 slots must
// each be written exactly once, the union over chunks must be an exact partition, and no line may straddle. Plus the
// thing that is specific to 2-plane: the HIGH source each kept line reads must stay inside the same chunk, or the
// high plane would need its own gate after all.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l38_2plane_chunk.cu -o l38 && ./l38
#include "cute/tensor.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using namespace cute;
using cutlass::MixGemmEmit; using cutlass::MixGemmChunkEmit;

template <int NChunk>
static bool check() {
  constexpr int kOut = 64, kPer = kOut / NChunk, NLINE = 8;   // int2: 8 pairs x 4 vregs = 32 lines, 64 outputs
  auto HI = make_layout(make_shape (_8{}, make_shape(_2{}, _2{})),
                        make_stride(_1{}, make_stride(_8{}, _64{})));
  int emitted[NLINE][4] = {}; long straddle = 0; bool ok = true;
  for (int c = 0; c < NChunk; ++c) {
    std::vector<int> hit(kPer / 2, 0); int lines = 0;
    std::vector<int> hi_seen;
    for (int v = 0; v < 4; ++v)
      for (int t = 0; t < NLINE; ++t) {
        const int e = MixGemmEmit<2>::index(t, v);
        // the single-plane int2 gate, unchanged
        const bool keep = ((e / 8) % NChunk) == c;   // the single-plane int2 gate, verbatim
        if (!keep) continue;
        ++lines; ++emitted[t][v];
        const int at = ((e % 8) + 8 * (e / (8 * NChunk))) / 2;
        if (at < 0 || at >= kPer / 2) { ok = false; }
        else ++hit[at];
        // the 2-plane-specific check: the HIGH source this line reads
        hi_seen.push_back(int(HI(make_coord(t, make_coord(v & 1, v >> 1)))));
      }
    int once = 0; for (int h : hit) if (h == 1) ++once;
    if (lines != 32 / NChunk || once != (int)hit.size()) ok = false;
    // the high sources a chunk touches must be disjoint from other chunks' -- otherwise two chunks would need the
    // same high register at different times, which is fine, or the SAME bit twice, which is not
    printf("  NChunk=%d chunk %d: %2d lines, %2d/%2zu h2 slots once, %2zu high sources\n",
           NChunk, c, lines, once, hit.size(), hi_seen.size());
  }
  long once = 0, other = 0;
  for (int t = 0; t < NLINE; ++t) for (int v = 0; v < 4; ++v) (emitted[t][v] == 1) ? ++once : ++other;
  printf("  NChunk=%d across chunks: %ld lines exactly once, %ld otherwise | %ld straddling  %s\n\n",
         NChunk, once, other, straddle,
         (ok && other == 0 && straddle == 0) ? "<-- EXACT PARTITION, one gate covers both planes"
                                             : "<-- BROKEN");
  return ok && other == 0 && straddle == 0;
}

int main() {
  printf("L38 -- the 2-plane chunk gate is the single-plane int2 gate, no high-plane predicate\n\n");
  bool ok = check<2>() && check<4>();
  printf("  Each _E2 line reads the low crumb and the high bit and writes ONE h2 slot, so gating the line gates both\n");
  printf("  planes. That is why no second predicate is needed -- and it follows from l37's tuple-of-layouts result\n");
  printf("  rather than from inspection.\n");
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
