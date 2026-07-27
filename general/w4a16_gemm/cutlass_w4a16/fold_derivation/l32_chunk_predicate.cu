// L32 -- which (code, vreg) pairs belong to k-atom chunk `a`? This is the fact the whole N/K-chunked conversion rests
// on, and it comes straight out of MixGemmEmit -- which is precisely why making MixGemmEmit the emission source (l29)
// was the prerequisite rather than a nicety.
//
// WHY A PREDICATE IS NEEDED AT ALL. convert_tensor converts a WHOLE delivery: CPY_VEC = 4*32/Bits, i.e. 128 fp16 for
// int1, because MixGemmNumericArrayConverter<half_t,uint1b_t,128> is built that way. And one source vreg's 32 codes
// do NOT form a contiguous 32-element output block -- MixGemmEmit scatters them across all 128. So converting "one
// chunk" means emitting only the (code, vreg) pairs whose output index lands in that chunk.
//
// WHICH CHUNK. tCrB_mma is compact (8, MMA_N, MMA_K), so flat index e = val + 8*n + 8*MMA_N*k, and k-atom a owns the
// CONTIGUOUS range [32a, 32a+32) when MMA_N = 4. From MixGemmEmit<1>:
//     e = bit4 + 2*b0 + 8*b1 + 16*b2 + 32*bit3 + 64*(v&1) + 4*(v>>1)
// every term except 32*bit3 and 64*(v&1) is < 32, so
//     e / 32  ==  bit3(code) + 2*(vreg & 1)
// i.e. the chunk is a STATIC function of (code, vreg) -- no runtime selection, and each chunk gets exactly a quarter
// of the pairs. Verified below for all three widths rather than trusted.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l32_chunk_predicate.cu -o l32 && ./l32
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using cutlass::MixGemmEmit;

template <int Bits, int MMA_N>
static bool check(const char* tag) {
  constexpr int CPW = 32 / Bits, NOUT = 4 * CPW, CHUNK = 8 * MMA_N;
  if (NOUT % CHUNK != 0) { printf("  %-22s int%d MMA_N=%d: %d outputs does not split into %d -- skipped\n",
                                  tag, Bits, MMA_N, NOUT, CHUNK); return true; }
  constexpr int NCH = NOUT / CHUNK;
  // the claimed predicate: chunk = (top code bit below the half selector) + 2*(vreg & 1), generalised as
  // e/CHUNK read off the weights -- derive it per width instead of hardcoding int1's
  long bad = 0; std::vector<int> per(NCH, 0);
  for (int v = 0; v < 4; ++v)
    for (int c = 0; c < CPW; ++c) {
      const int e = MixGemmEmit<Bits>::index(c, v);
      const int actual = e / CHUNK;
      // predicted: the sum of exactly those weight terms that are >= CHUNK, divided by CHUNK
      int pred = 0;
      const int nb = MixGemmEmit<Bits>::kNumCodeBits;
      for (int i = 0; i < nb - 1; ++i) { const int w = (i == 0 ? 2 : (4 << i)); if (w >= CHUNK) pred += ((c >> i) & 1) * w; }
      if (2 * CPW >= CHUNK) pred += (v & 1) * 2 * CPW;
      if (4 >= CHUNK)       pred += (v >> 1) * 4;
      pred /= CHUNK;
      if (pred != actual) ++bad;
      if (actual >= 0 && actual < NCH) ++per[actual];
    }
  printf("  %-22s int%d MMA_N=%d | %3d outputs / %2d per chunk = %d chunks | %ld mismatch | pairs per chunk:",
         tag, Bits, MMA_N, NOUT, CHUNK, NCH, bad);
  for (int i = 0; i < NCH; ++i) printf(" %d", per[i]);
  printf("  %s\n", bad ? "<-- NOT a static predicate" : "<-- static, and evenly split");
  return bad == 0;
}

// ============================================================================================================
// THE QUESTION THAT ACTUALLY DECIDES WHETHER CHUNKING IS FREE: is any conversion DUPLICATED?
//
// Each _EC line writes one uint32 = TWO halves, i.e. codes t (low lane) and t+16 (high lane). If those two codes fell
// in DIFFERENT chunks, the line would have to be emitted twice and the converter work would grow. And across the
// NChunk emit() calls the kept lines must partition the unchunked set exactly -- no line emitted twice, none dropped.
// Counting it, not arguing it.
static bool no_duplicate_work() {
  // AGAINST THE MEASURED LAYOUT. tCrB_mma is ((2,2,2),MMA_N,MMA_K):((1,2,4),32,8), so e = val + 32*n + 8*k and
  // e/32 == n_atom, NOT k_atom. The first version of this function used e/32 as "the chunk" and called it a k-atom;
  // the arithmetic was right and the INTERPRETATION was wrong, which is worse than a wrong number because it reads as
  // verified. cute::gemm wants (val, MMA_N) at fixed k, so the chunk is a k-atom:
  //     keep = ((e/8) % MMA_K) == Chunk        at = ((e%8) + 8*(e/(8*MMA_K)))/2
  constexpr int MMA_K = 4, NPAIR = 16;
  int emitted[NPAIR][4] = {}; long straddle = 0; bool shape_ok = true;
  for (int c = 0; c < MMA_K; ++c) {
    int hit[16] = {}, lines = 0;
    for (int v = 0; v < 4; ++v) for (int t = 0; t < NPAIR; ++t) {
      const int e = MixGemmEmit<1>::index(t, v), e2 = MixGemmEmit<1>::index(t + 16, v);
      if (((e / 8) % MMA_K) != c) continue;
      ++lines; ++emitted[t][v];
      if (e2 != e + 1 || ((e2 / 8) % MMA_K) != c) ++straddle;
      const int at = ((e % 8) + 8 * (e / (8 * MMA_K))) / 2;
      if (at < 0 || at >= 16) shape_ok = false; else ++hit[at];
    }
    int once = 0; for (int i = 0; i < 16; ++i) if (hit[i] == 1) ++once;
    if (lines != 16 || once != 16) shape_ok = false;
    printf("  chunk %d: %2d lines, %2d/16 h2 slots written exactly once\n", c, lines, once);
  }
  long once = 0, other = 0;
  for (int t = 0; t < NPAIR; ++t) for (int v = 0; v < 4; ++v) (emitted[t][v] == 1) ? ++once : ++other;
  printf("  across chunks: %ld lines exactly once, %ld otherwise | %ld straddling  %s\n", once, other, straddle,
         (shape_ok && other == 0 && straddle == 0) ? "<-- EXACT PARTITION, zero duplicated work" : "<-- BROKEN");
  return shape_ok && other == 0 && straddle == 0;
}

int main() {
  printf("L32 -- the chunk predicate: chunk = (weight terms >= CHUNK) / CHUNK, a STATIC function of (code, vreg)\n\n");
  bool ok = true;
  ok &= check<1, 4>("int1 the #9 target");
  ok &= check<1, 2>("int1 MMA_N=2");
  ok &= check<2, 4>("int2");
  ok &= check<2, 2>("int2 MMA_N=2");
  ok &= check<4, 4>("int4");
  ok &= check<4, 2>("int4 MMA_N=2");
  printf("\n  == is any conversion DUPLICATED by chunking?\n");
  ok &= no_duplicate_work();
  printf("\n  Because the chunk is static in (code, vreg), a chunk-aware converter is a COMPILE-TIME gate on the\n");
  printf("  existing emission loop -- both t and v are unrolled constants -- not a runtime selection. That is the\n");
  printf("  whole reason MixGemmEmit had to become the emission source first.\n");
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

