// L96 -- ARE THE TWO MICRO-OPTIMISATIONS IDENTITIES? Both claims are checked against the code they replace, on real
// Q4_K bytes, before either one reaches the collective.
//
//   (A) group_pair_of_words == group_of_words, BIT FOR BIT. Not "within a tolerance" -- the packed form exists to be
//       free, and a form that changes the last bit would invalidate test_q4k_packed_gemm's exact gate rather than
//       merely nudge it. Four identities are at stake and each one is checked where it can fail:
//         0x6400 | ((v+128) & 0x3FF) == 0x6480 + v      over the WHOLE unsigned code range, and shown to FAIL
//                                                       outside it, so the gate on unsigned codes is real
//         half(1152) == 0x6480                          the constant that biases up is the one that subtracts down
//         half2(d, -dmin) == u[0] ^ 0x80000000          including that the lane order is (d, dmin) and not (dmin, d)
//         fma(x, m, -0) == x * m                        for signed zeros, where + (+0) would differ
//
//   (B) two adjacent columns are ADJACENT IN SHARED MEMORY, which is what makes a 4-byte store legal at all. The
//       store rewrite is worth ~0.6% and would be worth nothing if it were also wrong, so the layout is asked rather
//       than assumed -- l95 already established that this layout IS the collective's SmemLayoutScale.
//
// Host-only and needs no PPU: CUTLASS_GGUF_PACKED_F16X2_ASM is 0 under nvcc, so the scalar fallback is what runs, and
// the fallback is the thing the asm has to match. That direction is deliberate -- the asm is two attested mnemonics
// over the same operands, while the arithmetic is where a mistake would be silent.
//
//   nvcc -std=c++17 -x cu -arch=sm_80 -w -I stub_inc -I <actlize>/include -I .. -I . -o /tmp/l96 l96_packed_pair.cu
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "cute/tensor.hpp"
#include "cutlass/gguf_packed_scale.h"

using cutlass::half_t;
using namespace cutlass::gguf_packed;
static int g_fail = 0;

// -------------------------------------------------------------------------------------------------------------------
// (A0) the four identities, in isolation, before any real data.
static void check_identities() {
  // 0x6400 | ((v+128) & 0x3FF) == 0x6480 + v, over the range the gate claims and just outside it.
  int bad = 0;
  for (int v = 0; v < 64; ++v) {
    uint32_t const a = 0x6400u | uint32_t((v + 128) & 0x3FF);
    uint32_t const b = 0x6480u + uint32_t(v);
    if (a != b) ++bad;
  }
  // I FIRST GATED THIS ON UNSIGNED CODES AND THE GATE WAS WRONG -- too narrow, and this check is what said so. The
  // identity holds over the WHOLE of int_to_half_small's range: for v in [-128, 895], (v+128) is in [0, 1023] so the
  // mask is a no-op and the OR is an add, giving 0x6400 + v + 128 = 0x6480 + v. It is the RANGE that bounds it, not
  // the sign, so a signed code is fine and Q3_K's -32 just folds into the constant. So probe the real boundary.
  int bad_full = 0;
  for (int v = -128; v < 896; ++v) {
    uint32_t const a = 0x6400u | uint32_t((v + 128) & 0x3FF);
    uint32_t const b = uint32_t(int(0x6480) + v);
    if (a != b) ++bad_full;
  }
  // And that it DOES break outside, so the bound is real rather than decoration: v = 896 overflows the mantissa
  // field, v = -129 makes (v+128) negative and the mask wraps.
  auto agree = [](int v) {
    return (0x6400u | uint32_t((v + 128) & 0x3FF)) == uint32_t(int(0x6480) + v);
  };
  bool const bound_real = !agree(896) && !agree(-129);
  // The packed form needs one more thing the scalar form does not: no carry between the two 16-bit lanes.
  // 0x6480 + 895 = 0x67FF and 0x6480 - 128 = 0x6400, so the lane stays inside 16 bits across the whole range.
  bool const no_carry = (uint32_t(0x6480 + 895) < 0x10000u) && (0x6480 - 128 >= 0x6400);
  std::printf("  (A0) 0x6400|((v+128)&0x3FF) == 0x6480+v over v in [0,63] -> %d bad; over ALL of [-128,895] -> %d bad;"
              " outside %s; inter-lane carry %s\n",
              bad, bad_full, bound_real ? "it DOES break (bound is real)" : "it still holds (bound is loose!)",
              no_carry ? "impossible" : "POSSIBLE -- packing is unsound");
  g_fail += bad + bad_full + (bound_real ? 0 : 1) + (no_carry ? 0 : 1);

  // half(1152) is the magic word, in both lanes.
  bool const magic_ok = (half_t(1152.f).raw() == 0x6480) && (kMagic1152x2 == 0x64806480u);
  std::printf("  (A0) half(1152) = 0x%04X and kMagic1152x2 = 0x%08X -> %s\n",
              half_t(1152.f).raw(), kMagic1152x2, magic_ok ? "ok" : "FAIL");
  if (!magic_ok) ++g_fail;

  // half2(d, -dmin) == u[0] ^ 0x80000000, INCLUDING the lane order. A swapped pair would still look like a plausible
  // half2 and would produce plausible-looking scales, so the check pins which lane is which.
  {
    half_t const d(0.0123f), dmin(0.0456f);
    uint32_t u[4] = {pack_h2(d, dmin), 0, 0, 0};
    uint32_t const m2 = mul2_of_words(u);
    bool const ok = (lo_h2(m2) == d) && (hi_h2(m2) == -dmin);
    std::printf("  (A0) mul2_of_words -> lo %g (want %g), hi %g (want %g) -> %s\n",
                float(lo_h2(m2)), float(d), float(hi_h2(m2)), float(-dmin), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
  }

  // fma(x, m, -0) == x*m, and the signed-zero cases are the whole reason the addend is -0 and not +0.
  {
    int nz = 0;
    half_t const cases[] = {half_t(0.f), half_t(-0.f), half_t(1.5f), half_t(-2.25f), half_t(63.f)};
    for (half_t x : cases) for (half_t m : cases) {
      uint32_t const got  = fma_f16x2(pack_h2(x, x), pack_h2(m, m), kNegZeroX2);
      half_t   const want = x * m;
      if (lo_h2(got).raw() != want.raw() || hi_h2(got).raw() != want.raw()) ++nz;
    }
    // and that +0 as the addend would NOT do, on the case that separates them
    uint32_t const with_pos = fma_f16x2(pack_h2(half_t(0.f), half_t(0.f)), pack_h2(half_t(-1.f), half_t(-1.f)), 0u);
    bool const pos_differs = (lo_h2(with_pos).raw() != (half_t(0.f) * half_t(-1.f)).raw());
    std::printf("  (A0) fma(x,m,-0) == x*m over signed zeros -> %d bad; addend +0 %s\n", nz,
                pos_differs ? "DOES differ (so -0 is required)" : "agrees (either would do)");
    g_fail += nz;
  }
}

// -------------------------------------------------------------------------------------------------------------------
// (A) the real thing: every group of every real Q4_K superblock, packed form against scalar form, bit for bit.
// Reads the GSB0 fixture l91 and l93 use -- raw 12-byte scale blocks out of a real gguf, with d/dmin prepended here
// the way the offline lays the 16-byte unit out.
static void check_real(char const* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) { std::printf("  (A) SKIPPED: cannot open %s\n", path); return; }
  // The GSB0 record layout is READ OFF l93 rather than remembered: magic, four u32 of header, then per block `bb` raw
  // bytes and ng int32 each of sc and mn. My first version of this loop invented "12 raw bytes, then d and dmin as
  // floats, then ng uint8 each" and smashed the stack -- the same defect as everywhere else in this task, a relation
  // written from memory when the file that defines it was one directory away.
  char magic[4]; uint32_t hdr[4];
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GSB0", 4) != 0 || std::fread(hdr, 4, 4, f) != 4) {
    std::printf("  (A) SKIPPED: %s is not GSB0\n", path); std::fclose(f); return;
  }
  uint32_t const nb = hdr[1], bb = hdr[2], ng = hdr[3];
  if (bb > 16 || ng > 16) { std::printf("  (A) FAIL: bb=%u ng=%u beyond this reader\n", bb, ng); ++g_fail; std::fclose(f); return; }
  int bad = 0, seen = 0, nonzero_zero = 0;
  // d and dmin are not in a scale-BLOCK dump, so drive them the way l93 does -- non-dyadic first, so the products
  // actually round and the bit-identity claim is entered rather than sidestepped.
  half_t const ds[] = {half_t(0.0123f), half_t(0.00789f), half_t(0.0316f), half_t(0.0078125f), half_t(1.f)};
  for (uint32_t i = 0; i < nb; ++i) {
    uint8_t raw[16]; int32_t rsc[16], rmn[16];
    if (std::fread(raw, 1, bb, f) != bb) break;
    if (std::fread(rsc, 4, ng, f) != ng) break;
    if (std::fread(rmn, 4, ng, f) != ng) break;
    (void)rsc; (void)rmn;
    half_t const d = ds[i % 5], dmin = ds[(i + 2) % 5];
    // The 16-byte unit exactly as dump_packed_scale.py writes it: d, dmin, then the 12 scale bytes.
    uint32_t u[4];
    uint8_t bytes[16] = {};
    bytes[0] = uint8_t(d.raw() & 0xFF);    bytes[1] = uint8_t(d.raw() >> 8);
    bytes[2] = uint8_t(dmin.raw() & 0xFF); bytes[3] = uint8_t(dmin.raw() >> 8);
    std::memcpy(bytes + 4, raw, bb);
    std::memcpy(u, bytes, 16);

    auto const h  = head_of_words(u);
    uint32_t const m2 = mul2_of_words(u);
    // ZMul = 8, the value the collective actually uses, so the check covers the term too.
    cute::for_each(cute::make_int_sequence<8>{}, [&] (auto g_) {
      constexpr int G = decltype(g_)::value;
      auto const a = group_of_words<G, 0, true, 8>(u, h);
      auto const b = group_pair_of_words<G, 8>(u, m2);
      if (a.scale.raw() != b.scale.raw() || a.zero.raw() != b.zero.raw()) {
        if (bad < 4)
          std::printf("    [FAIL] blk %u g %d: scale %04X vs %04X, zero %04X vs %04X\n",
                      i, G, a.scale.raw(), b.scale.raw(), a.zero.raw(), b.zero.raw());
        ++bad;
      }
      if (a.zero.raw() != 0) ++nonzero_zero;
      ++seen;
    });
  }
  std::fclose(f);
  // A gate that passes because everything is zero passes for the wrong reason, so say how much was non-trivial.
  std::printf("  (A) real Q4_K: %d groups compared bit-for-bit -> %d bad  (%d had a non-zero zero, so the min term "
              "is exercised)\n", seen, bad, nonzero_zero);
  g_fail += bad;
}

// -------------------------------------------------------------------------------------------------------------------
// (B) are two adjacent columns adjacent in shared memory? l95 pinned this layout as the collective's own
// SmemLayoutScale, so asking it here is asking the kernel.
static void check_layout() {
  using namespace cute;
  auto sS = tile_to_shape(Layout<Shape<_8, _1>>{}, make_shape(_128{}, _8{}, _2{}));
  std::printf("  (B) SmemLayoutScale = "); print(sS); std::printf("\n");
  int bad = 0, worst_n = -1;
  for (int s = 0; s < 2; ++s)
    for (int g = 0; g < 8; ++g)
      for (int n = 0; n < 128; n += 2)
        if (sS(n + 1, g, s) != sS(n, g, s) + 1) { if (bad == 0) worst_n = n; ++bad; }
  std::printf("  (B) (n+1,g,s) == (n,g,s)+1 for every even n -> %d bad%s\n", bad,
              bad ? " -- a 4-byte store of two columns would be WRONG" : " (a 4-byte store of two columns is legal)");
  if (bad) std::printf("      first failure at n = %d\n", worst_n);
  g_fail += bad;

  // And the bank story the store rewrite is meant to fix: with one column per lane the warp touches 32 halfs = 64 B,
  // which is 16 of 32 banks; with two columns per lane it touches 128 B, which is all 32. Counted, not asserted.
  int const banks_1col = 0, banks_2col = 0; (void)banks_1col; (void)banks_2col;
  bool seen1[32] = {}, seen2[32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    int const a1 = int(sS(lane, 0, 0));               // one column per lane, 2 B each
    seen1[(a1 * 2 / 4) % 32] = true;
    int const a2 = int(sS(2 * lane, 0, 0));           // two columns per lane, 4 B each
    seen2[(a2 * 2 / 4) % 32] = true;
  }
  int n1 = 0, n2 = 0;
  for (int b = 0; b < 32; ++b) { n1 += seen1[b]; n2 += seen2[b]; }
  std::printf("  (B) banks touched by one warp: %d/32 at one column per lane, %d/32 at two -> %s\n",
              n1, n2, (n2 == 32 && n1 < 32) ? "the rewrite is what removes the conflict" : "no conflict to remove");
}

int main(int argc, char** argv) {
  std::printf("== l96: are the two micro-optimisations identities? ==\n");
  check_identities();
  check_real(argc > 1 ? argv[1] : "real_weight/scale_blocks_q4k.bin");
  check_layout();
  std::printf("== l96 %s (%d failures) ==\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
