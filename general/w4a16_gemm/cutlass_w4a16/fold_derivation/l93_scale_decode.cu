// L93 -- PHASE 2 GATE (the decode unit, before any collective change). gguf_scale_decode.hpp turns native GGUF scale
// bytes into the (scale, zero) an mma atom needs. Three things can be wrong in it and all three are checked here
// against something that is not itself:
//
//   (1) the int -> fp16 magic conversion, over its WHOLE claimed range [-128, 895], against the integer
//   (2) the per-format code centre (Traits::kScaleBias), against the formulas in dump_real_weights.py -- Q3_K's -32 is
//       the one non-zero constant here and I would have guessed 0
//   (3) the (scale, zero) formation over REAL Q4_K superblocks, against a double-precision reference, including that
//       the fp16 product is bit-identical to the host's fp32-then-round -- the claim dump_packed_scale.py gate (b)
//       makes in python, re-made here in the C++ that will actually run
//
// Reads real_weight/scale_blocks_q4k.bin (GSB0), the same fixture l91 uses: raw 12-byte blocks out of a real gguf plus
// the reference sc/mn from dump_real_weights.get_scale_min_k4.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "gguf_scale_decode.hpp"

using namespace gguf_scale;
using cutlass::half_t;
static int g_fail = 0;

// ---------------------------------------------------------------------------------------------------------------
static void check_conversion() {
  int bad = 0, lo = 1 << 30, hi = -(1 << 30);
  for (int v = -128; v < 896; ++v) {
    half_t h = int_to_half_small(v);
    if (float(h) != float(v)) { if (bad < 4) std::printf("    [FAIL] int_to_half_small(%d) = %g\n", v, float(h)); ++bad; }
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  // And that it FAILS outside the claimed range, so the bound is real and not decoration.
  bool out_of_range_differs = (float(int_to_half_small(896)) != 896.f);
  std::printf("  (1) int_to_half_small exact over [%d, %d] -> %d bad; 896 is outside and %s\n",
              lo, hi, bad, out_of_range_differs ? "does differ (bound is real)" : "still matches (bound is loose)");
  g_fail += bad;
}

// ---------------------------------------------------------------------------------------------------------------
// (2) the code centres, against dump_real_weights.py. Written out as the formulas that file uses, so the check is
// against the reference implementation rather than against my memory of it.
static void check_centres() {
  struct { char const* name; int got; int want; char const* src; } c[] = {
    {"Q4_K", Traits<KType::Q4_K>::kScaleBias, 0,  "unpack_q4k_expert: sv = d * sc[sub]"},
    {"Q3_K", Traits<KType::Q3_K>::kScaleBias, 32, "unpack_q3k_expert: dl = d * (sc6[is_full] - 32)"},
    {"Q2_K", Traits<KType::Q2_K>::kScaleBias, 0,  "unpack_q2k_expert: dl = d * (sc & 0xF)"},
    {"Q6_K", Traits<KType::Q6_K>::kScaleBias, 0,  "scales are signed int8; scale_of sign-extends"},
  };
  for (auto& e : c) {
    bool ok = e.got == e.want;
    std::printf("  (2) %s kScaleBias = %-2d (want %-2d)  %-52s -> %s\n", e.name, e.got, e.want, e.src, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
  }
  // Q6_K's signed path must survive the conversion: scale_of returns [-128,127] and the conversion must cover it.
  int bad = 0;
  for (int v = -128; v < 128; ++v) if (float(int_to_half_small(v - Traits<KType::Q6_K>::kScaleBias)) != float(v)) ++bad;
  std::printf("  (2) Q6_K signed int8 scale codes through the same conversion -> %d bad\n", bad);
  g_fail += bad;
}

// ---------------------------------------------------------------------------------------------------------------
static void check_real_blocks(char const* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) { std::printf("  (3) SKIPPED: %s not found (run real_weight/dump_scale_blocks.py)\n", path); return; }
  char magic[4]; uint32_t hdr[4];
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GSB0", 4) != 0 || std::fread(hdr, 4, 4, f) != 4) {
    std::printf("  (3) FAIL: %s is not a GSB0 file\n", path); ++g_fail; std::fclose(f); return;
  }
  uint32_t const nb = hdr[1], bb = hdr[2], ng = hdr[3];
  int bad_sc = 0, bad_prod = 0, seen = 0;
  double max_rel = 0;
  // d / dmin are not in the fixture (it is a scale-BLOCK dump), so drive them over a spread of fp16 values. The first
  // three are NON-dyadic on purpose: with powers of two only, d*sc is exactly representable and max_rel comes out 0.000
  // for the trivial reason that nothing ever rounds -- which reads like a strong result and tests nothing. The claim
  // being gated is the BIT-IDENTITY with the host's fp32-then-round, so the rounding path has to be entered.
  half_t const ds[]  = {half_t(0.0123f), half_t(0.00789f), half_t(0.0316f), half_t(0.0078125f), half_t(1.f)};
  for (uint32_t i = 0; i < nb; ++i) {
    uint8_t raw[16]; int32_t rsc[16], rmn[16];
    if (std::fread(raw, 1, bb, f) != bb) break;
    if (std::fread(rsc, 4, ng, f) != ng) break;
    if (std::fread(rmn, 4, ng, f) != ng) break;
    half_t const d = ds[i % 5], dmin = ds[(i + 2) % 5];
    Superblock<KType::Q4_K> sb;
    sb.decode(raw, d, dmin);
    for (uint32_t g = 0; g < ng; ++g) {
      // the field extraction, against the reference ints
      if (scale_of<KType::Q4_K>(raw, int(g)) != rsc[g] || min_of<KType::Q4_K>(raw, int(g)) != rmn[g]) ++bad_sc;
      // the formation, against fp64 -- and against the host's fp32-then-round, which must be BIT-identical
      double const want_s = double(float(d))    * double(rsc[g]);
      double const want_z = -double(float(dmin)) * double(rmn[g]);
      float const host_s = float(half_t(float(want_s)));            // one rounding of the exact product
      float const host_z = float(half_t(float(want_z)));
      if (float(sb.g[g].scale) != host_s || float(sb.g[g].zero) != host_z) {
        if (bad_prod < 4)
          std::printf("    [FAIL] blk=%u g=%u scale %.9g vs %.9g   zero %.9g vs %.9g\n",
                      i, g, float(sb.g[g].scale), host_s, float(sb.g[g].zero), host_z);
        ++bad_prod;
      }
      double const rel = std::abs(double(float(sb.g[g].scale)) - want_s) / (std::abs(want_s) + 1e-12);
      if (rel > max_rel) max_rel = rel;
      ++seen;
    }
  }
  std::fclose(f);
  std::printf("  (3) real Q4_K: %d groups | field extraction %d bad | (scale,zero) vs host fp32-then-round %d bad"
              " | max_rel vs fp64 %.3e\n", seen, bad_sc, bad_prod, max_rel);
  g_fail += bad_sc + bad_prod;
}

int main() {
  std::printf("== plan #20 phase 2: the scale DECODE unit ==\n");
  check_conversion();
  check_centres();
  check_real_blocks("../real_weight/scale_blocks_q4k.bin");
  // The traffic claim, as numbers off the object rather than prose.
  std::printf("  bytes/group/col: Q4_K native %.3f vs fp16 scale+zero %.1f  (%.2fx) | Q3_K %.3f vs %.1f (%.2fx)\n",
              bytes_per_group_per_col<KType::Q4_K>(), bytes_per_group_per_col_fp16<KType::Q4_K>(),
              bytes_per_group_per_col_fp16<KType::Q4_K>() / bytes_per_group_per_col<KType::Q4_K>(),
              bytes_per_group_per_col<KType::Q3_K>(), bytes_per_group_per_col_fp16<KType::Q3_K>(),
              bytes_per_group_per_col_fp16<KType::Q3_K>() / bytes_per_group_per_col<KType::Q3_K>());
  std::printf("== %s: %d mismatches ==\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
