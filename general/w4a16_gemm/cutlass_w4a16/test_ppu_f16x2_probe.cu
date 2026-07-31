// DO THE TWO f16x2 ASM INSTRUCTIONS DO WHAT THE SCALAR FALLBACK DOES? Nothing has ever asked, and that is why this
// file exists.
//
// gguf_packed_scale.h's group_pair_of_words decodes both 6-bit fields of a Q4_K group in one 32-bit lane pair, using
//     ppu.sub.f16x2      %0, %1, %2
//     ppu.fma.rtte.f16x2 %0, %1, %2, %3
// under the guard CUTLASS_GGUF_PACKED_F16X2_ASM == (__HGGCCC__ && !__NVCC__). The host gate l96_packed_pair.cu
// compares that function against the scalar group_of_words over 32768 REAL Q4_K groups and reports 0 bad -- but it
// compiles with nvcc, where the guard picks the SCALAR FALLBACK. So l96 proves the arithmetic and proves NOTHING about
// the two instructions the device actually executes. That gap is not theoretical: with the packed decode on, rowC of
// test_q4k_packed_gemm mismatches, and a bisect (PPU_PACKED_PAIR=0) puts the fault inside group_pair_of_words while
// everything around it -- the native transport, the shared stores, the loop -- stays correct.
//
// This harness is deliberately NOT a GEMM. It runs one tiny kernel, no shared memory, no cp.async, no mma, so a
// failure here cannot be confused with tile or fragment plumbing. Three sections, narrowing:
//
//   (1) the two instructions in isolation, against the scalar operations they claim to equal, over operands chosen to
//       hit the cases an ISA is most likely to differ on: signed zeros (fma(x, m, -0) == x*m is load-bearing), the
//       exact magic constants this decode uses, denormals, and both lanes carrying DIFFERENT values -- a packed op
//       that ignores one lane, or swaps them, passes any test that puts the same number in both.
//   (2) the whole per-group decode, on device: group_pair_of_words (asm) against group_of_words (scalar), over real
//       16-byte units built from a fixture, so a disagreement is attributed to a specific (group, field).
//   (3) which guard the device build actually took, printed. If CUTLASS_GGUF_PACKED_F16X2_ASM came out 0 on the real
//       compiler, the packed path never used the asm and every conclusion drawn from its timings is about something
//       else -- that is worth one printf.
//
//   build: TARGET=test_ppu_f16x2_probe ./build.sh
//   run:   $BIN/test_ppu_f16x2_probe [real_weight/q4k_packed.bin]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "cutlass/half.h"
#include "cutlass/util/device_memory.h"
#include "ppu_include.hpp"
#include "helper.h"
#include "cutlass/gguf_packed_scale.h"
#include "rwmoep_loader.hpp"

using cutlass::half_t;
namespace gp = cutlass::gguf_packed;

// ---------------------------------------------------------------------------------------------------------------
// (1) the instructions, alone. `want` is computed on the DEVICE with scalar half ops, so this compares two device
// code paths rather than device against host -- host fp16 emulation would be a third thing to doubt.
struct OpCase { uint32_t a, b, c; };

__global__ void probe_ops(OpCase const* in, int n, uint32_t* sub_got, uint32_t* sub_want,
                          uint32_t* fma_got, uint32_t* fma_want) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t const a = in[i].a, b = in[i].b, c = in[i].c;

  sub_got[i]  = gp::sub_f16x2(a, b);
  sub_want[i] = gp::pack_h2(gp::lo_h2(a) - gp::lo_h2(b), gp::hi_h2(a) - gp::hi_h2(b));

  fma_got[i]  = gp::fma_f16x2(a, b, c);
  fma_want[i] = gp::pack_h2(gp::lo_h2(a) * gp::lo_h2(b) + gp::lo_h2(c),
                            gp::hi_h2(a) * gp::hi_h2(b) + gp::hi_h2(c));
}

// ---------------------------------------------------------------------------------------------------------------
// (2) the whole per-group decode, both ways, on device. ZMul = 8 and ScaleBias = 0 are the values the collective
// uses, so this is the mainloop's call, minus the mainloop.
__global__ void probe_groups(uint32_t const* units, int nunits, half_t* s_pair, half_t* z_pair,
                             half_t* s_scal, half_t* z_scal) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nunits) return;
  uint32_t u[4];
  CUTLASS_PRAGMA_UNROLL
  for (int w = 0; w < 4; ++w) u[w] = units[i * 4 + w];
  uint32_t const m2 = gp::mul2_of_words(u);
  auto const h = gp::head_of_words(u);
  CUTLASS_PRAGMA_UNROLL
  for (int g = 0; g < 8; ++g) {
    // A runtime g would not compile (every bit position is a template argument), so unroll by hand through a switch
    // on a compile-time constant. Written as a fold over an index sequence would be neater and needs cute here.
#define ONE(G)                                                                                   \
    if (g == G) {                                                                                \
      auto const p = gp::group_pair_of_words<G, 8, 0>(u, m2);                                    \
      auto const s = gp::group_of_words<G, 0, true, 8>(u, h);                                    \
      s_pair[i * 8 + G] = p.scale; z_pair[i * 8 + G] = p.zero;                                   \
      s_scal[i * 8 + G] = s.scale; z_scal[i * 8 + G] = s.zero;                                   \
    }
    ONE(0) ONE(1) ONE(2) ONE(3) ONE(4) ONE(5) ONE(6) ONE(7)
#undef ONE
  }
}

// ---------------------------------------------------------------------------------------------------------------
// (4) IS THE DESTINATION ALLOWED TO ALIAS AN INPUT? This is the leading hypothesis for rowC's PARTIAL failure, and it
// is the one thing "=r" versus "=&r" changes. In packed_decode_stage the multiplier m2 is live across all eight
// unrolled groups and dies at the last one, so an allocator may pick dest == m2 for the FINAL fma only -- which
// corrupts some groups and not others, and can differ between builds. That is the observed shape (bad=128 in one
// build, 724 in another) and nothing else on the list predicts it.
//
// No hardcoded expected values: each op is run once with a separate destination and once with the destination tied to
// each input in turn, and the results are compared to each other. If any aliased form differs, "=r" was never safe
// here, whatever the reference uses in fast_numeric_conversion_for_mix_gemm.h happen to get away with.
__global__ void probe_alias(uint32_t const* in3, uint32_t* out) {
  uint32_t const x = in3[0], m = in3[1], z = in3[2];
  uint32_t r;
  asm volatile("ppu.fma.rtte.f16x2 %0, %1, %2, %3;\n" : "=&r"(r) : "r"(x), "r"(m), "r"(z));
  out[0] = r;                                             // the reference: no overlap possible
  uint32_t a1 = x; asm volatile("ppu.fma.rtte.f16x2 %0, %0, %1, %2;\n" : "+r"(a1) : "r"(m), "r"(z));
  out[1] = a1;                                            // dest == source 1
  uint32_t a2 = m; asm volatile("ppu.fma.rtte.f16x2 %0, %1, %0, %2;\n" : "+r"(a2) : "r"(x), "r"(z));
  out[2] = a2;                                            // dest == source 2  <-- the m2 case
  uint32_t a3 = z; asm volatile("ppu.fma.rtte.f16x2 %0, %1, %2, %0;\n" : "+r"(a3) : "r"(x), "r"(m));
  out[3] = a3;                                            // dest == source 3
  uint32_t s;
  asm volatile("ppu.sub.f16x2 %0, %1, %2;\n" : "=&r"(s) : "r"(x), "r"(m));
  out[4] = s;
  uint32_t b1 = x; asm volatile("ppu.sub.f16x2 %0, %0, %1;\n" : "+r"(b1) : "r"(m));
  out[5] = b1;                                            // dest == source 1 (what the reference uses)
  uint32_t b2 = m; asm volatile("ppu.sub.f16x2 %0, %1, %0;\n" : "+r"(b2) : "r"(x));
  out[6] = b2;                                            // dest == source 2
}

static int g_fail = 0;

static void report_ops(char const* name, std::vector<OpCase> const& cs,
                       std::vector<uint32_t> const& got, std::vector<uint32_t> const& want) {
  int bad = 0, shown = 0;
  for (size_t i = 0; i < cs.size(); ++i) {
    if (got[i] == want[i]) continue;
    ++bad;
    if (shown < 8) {
      std::printf("    [%s] a=%08X b=%08X c=%08X   got %08X  want %08X   "
                  "(lo %g vs %g | hi %g vs %g)\n",
                  name, cs[i].a, cs[i].b, cs[i].c, got[i], want[i],
                  float(half_t::bitcast(uint16_t(got[i] & 0xFFFF))),
                  float(half_t::bitcast(uint16_t(want[i] & 0xFFFF))),
                  float(half_t::bitcast(uint16_t(got[i] >> 16))),
                  float(half_t::bitcast(uint16_t(want[i] >> 16))));
      ++shown;
    }
  }
  std::printf("  (1) %-16s %zu cases -> %d disagree with the scalar form\n", name, cs.size(), bad);
  g_fail += bad;
}

int main(int argc, char** argv) {
  std::printf("== ppu f16x2 probe: do the two asm ops equal the scalar ops they replace? ==\n");
  // (3) first, because everything else is conditional on it.
  std::printf("  (3) f16x2 form compiled: sub=\"%s\" fma=\"%s\"\n",
              CUTLASS_PPU_F16X2_SUB, CUTLASS_PPU_F16X2_FMA);
  std::printf("  (3) CUTLASS_GGUF_PACKED_F16X2_ASM = %d  -> the device build uses %s\n",
              CUTLASS_GGUF_PACKED_F16X2_ASM,
              CUTLASS_GGUF_PACKED_F16X2_ASM ? "the ppu.*.f16x2 ASM" : "the SCALAR FALLBACK (asm never runs!)");

  // ---- (1) operands. Lanes are given DIFFERENT values throughout: an instruction that broadcasts one lane, or
  // swaps them, is invisible to any case where lo == hi.
  auto H = [](float v) { return uint32_t(half_t(v).raw()); };
  auto P = [&](float lo, float hi) { return H(lo) | (H(hi) << 16); };
  std::vector<OpCase> cs;
  // the decode's own constants and a plausible (d, -dmin)
  cs.push_back({0x64806480u + 0x0001u + (0x0002u << 16), gp::kMagic1152x2, gp::kNegZeroX2});
  cs.push_back({0x64806480u + 63u   + (31u << 16),      gp::kMagic1152x2, gp::kNegZeroX2});
  cs.push_back({P(3.f, -5.f),  P(0.0123f, -0.0456f),    gp::kNegZeroX2});
  // signed zeros in every position -- fma(x, m, -0) == x*m is the identity the packed multiply rests on
  cs.push_back({P(0.f, -0.f),  P(-1.f, 1.f),            gp::kNegZeroX2});
  cs.push_back({P(-0.f, 0.f),  P(1.f, -1.f),            gp::kNegZeroX2});
  cs.push_back({P(0.f, 0.f),   P(-2.f, -3.f),           0u});               // addend +0, the case that differs
  // asymmetric lanes, and a lane whose value is large while the other is small
  cs.push_back({P(1024.f, 0.0001f), P(0.5f, 4096.f),    P(-1.f, 2.f)});
  cs.push_back({P(-1152.f, 1152.f), P(1.f, 1.f),        gp::kNegZeroX2});
  // SUBNORMALS, IN THE EXACT SHAPE Q4_K PRODUCES. On the real fixture d spans 1.585e-05 to 9.484e-04 against fp16's
  // smallest normal 6.104e-05, so 3914 of 4864 superblock d values are SUBNORMAL while dmin (1.1e-04 to 1.1e-02) is
  // not -- and the PRODUCTS d*sc are not either, which is why the fp16-plane baseline never meets this. If the f16x2
  // ops flush a subnormal INPUT, the scale lane dies and the zero lane lives. That asymmetry is the signature.
  cs.push_back({P(50.f, 40.f),   P(1.585e-05f, 1.096e-04f), gp::kNegZeroX2});   // smallest d, normal dmin
  cs.push_back({P(63.f, 63.f),   P(3.0e-05f,  9.484e-04f),  gp::kNegZeroX2});   // mid subnormal d, largest dmin
  cs.push_back({P(1.f, 1.f),     P(6.0e-05f,  6.2e-05f),    gp::kNegZeroX2});   // straddling the normal threshold
  cs.push_back({P(6e-8f, 1.f),  P(1.f, 6e-8f),          gp::kNegZeroX2});
  cs.push_back({P(6e-5f, 6e-5f), P(6e-5f, 1e-3f),       gp::kNegZeroX2});
  // and the SUB, which also sees a subnormal if one ever reaches it
  cs.push_back({P(3.0e-05f, 1.f), P(0.f, 0.f),          gp::kNegZeroX2});
  // the two whole ranges the identity claims, sampled at the ends
  for (int v = -128; v <= 895; v += 73)
    cs.push_back({uint32_t(0x6480 + v) | (uint32_t(0x6480 + (v % 64)) << 16), gp::kMagic1152x2, gp::kNegZeroX2});

  int const n = int(cs.size());
  cutlass::DeviceAllocation<OpCase> dIn(n);
  cutlass::DeviceAllocation<uint32_t> dSg(n), dSw(n), dFg(n), dFw(n);
  dIn.copy_from_host(cs.data());
  probe_ops<<<(n + 63) / 64, 64>>>(dIn.get(), n, dSg.get(), dSw.get(), dFg.get(), dFw.get());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<uint32_t> sg(n), sw(n), fg(n), fw(n);
  dSg.copy_to_host(sg.data()); dSw.copy_to_host(sw.data());
  dFg.copy_to_host(fg.data()); dFw.copy_to_host(fw.data());
  report_ops("sub.f16x2", cs, sg, sw);
  report_ops("fma.f16x2", cs, fg, fw);

  // ---- (2) the whole decode, on real units if a fixture is there, else synthetic ones covering every code.
  char const* path = argc > 1 ? argv[1] : "real_weight/q4k_packed.bin";
  std::vector<uint32_t> units;
  rwmoep::File f;
  if (f.load(path) && f.mode == 1) {
    int const sk = f.scale_k(), nb = f.nsb(), N = f.N, L = f.L;
    (void)sk;
    // rebuild the 16 B unit the way the offline lays it out: d, dmin, then the 6-bit codes
    for (int l = 0; l < L; ++l)
      for (int b = 0; b < nb; ++b)
        for (int n2 = 0; n2 < N; ++n2) {
          uint8_t bytes[16] = {};
          size_t const hidx = ((size_t)l * nb + b) * N + n2;
          uint16_t const dr = f.d[hidx].raw(), dmr = f.dmin[hidx].raw();
          bytes[0] = uint8_t(dr & 0xFF);  bytes[1] = uint8_t(dr >> 8);
          bytes[2] = uint8_t(dmr & 0xFF); bytes[3] = uint8_t(dmr >> 8);
          for (int g = 0; g < 8; ++g) {
            size_t const gi = ((size_t)l * f.scale_k() + b * 8 + g) * N + n2;
            gp::put_code(bytes, g, 0, int(f.sc[gi]) & 0x3F);
            gp::put_code(bytes, g, 1, int(f.mn[gi]) & 0x3F);
          }
          uint32_t w[4]; std::memcpy(w, bytes, 16);
          for (int k = 0; k < 4; ++k) units.push_back(w[k]);
        }
    std::printf("  (2) fixture %s: %zu real units\n", path, units.size() / 4);
  } else {
    // EVERY code value in both fields, so no group escapes: 64 units, code = (g*8 + u) % 64.
    for (int u = 0; u < 64; ++u) {
      uint8_t bytes[16] = {};
      uint16_t const dr = half_t(0.0123f).raw(), dmr = half_t(0.0456f).raw();
      bytes[0] = uint8_t(dr & 0xFF);  bytes[1] = uint8_t(dr >> 8);
      bytes[2] = uint8_t(dmr & 0xFF); bytes[3] = uint8_t(dmr >> 8);
      for (int g = 0; g < 8; ++g) {
        gp::put_code(bytes, g, 0, (g * 8 + u) % 64);
        gp::put_code(bytes, g, 1, (g * 5 + u * 3) % 64);
      }
      uint32_t w[4]; std::memcpy(w, bytes, 16);
      for (int k = 0; k < 4; ++k) units.push_back(w[k]);
    }
    std::printf("  (2) no fixture at %s -- 64 synthetic units covering every code\n", path);
  }

  int const nu = int(units.size() / 4);
  cutlass::DeviceAllocation<uint32_t> dU(units.size());
  cutlass::DeviceAllocation<half_t> dSp(nu * 8), dZp(nu * 8), dSs(nu * 8), dZs(nu * 8);
  dU.copy_from_host(units.data());
  probe_groups<<<(nu + 127) / 128, 128>>>(dU.get(), nu, dSp.get(), dZp.get(), dSs.get(), dZs.get());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> sp(nu * 8), zp(nu * 8), ss(nu * 8), zs(nu * 8);
  dSp.copy_to_host(sp.data()); dZp.copy_to_host(zp.data());
  dSs.copy_to_host(ss.data()); dZs.copy_to_host(zs.data());

  // PER GROUP, because the two straddling fields are group-specific: Q4_K's bit 62 (group 1's min) and bit 92
  // (group 6's scale) are the only two that cross a 32-bit word, so a straddle bug shows as those groups alone.
  int bad_g[8] = {}, shown = 0, bad = 0;
  for (int i = 0; i < nu; ++i)
    for (int g = 0; g < 8; ++g) {
      size_t const k = (size_t)i * 8 + g;
      if (sp[k].raw() == ss[k].raw() && zp[k].raw() == zs[k].raw()) continue;
      ++bad; ++bad_g[g];
      if (shown < 8) {
        std::printf("    unit %d group %d: scale %04X vs %04X (%g vs %g)   zero %04X vs %04X (%g vs %g)\n",
                    i, g, sp[k].raw(), ss[k].raw(), float(sp[k]), float(ss[k]),
                    zp[k].raw(), zs[k].raw(), float(zp[k]), float(zs[k]));
        ++shown;
      }
    }
  std::printf("  (2) packed vs scalar decode on device: %d of %d (unit, group) disagree\n", bad, nu * 8);
  std::printf("      per group:");
  for (int g = 0; g < 8; ++g) std::printf(" g%d=%d", g, bad_g[g]);
  std::printf("      (only g1's min and g6's scale straddle a word)\n");
  g_fail += bad;

  // ---- (4) the aliasing question, on the decode's own operand shapes
  {
    uint32_t h3[3] = {0x00360032u,                      // an x2 as the sub would produce it
                      0x89640172u,                      // an m2 = u[0] ^ 0x80000000 from the real fixture
                      gp::kNegZeroX2};
    cutlass::DeviceAllocation<uint32_t> d3(3), dOut(7);
    d3.copy_from_host(h3);
    probe_alias<<<1, 1>>>(d3.get(), dOut.get());
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    uint32_t o[7]; dOut.copy_to_host(o);
    char const* nm[7] = {"fma dest separate", "fma dest==src1", "fma dest==src2", "fma dest==src3",
                         "sub dest separate", "sub dest==src1", "sub dest==src2"};
    int alias_bad = 0;
    for (int i = 0; i < 7; ++i) {
      uint32_t const ref = (i < 4) ? o[0] : o[4];
      bool const ok = (o[i] == ref);
      if (!ok) ++alias_bad;
      std::printf("  (4) %-18s = %08X %s\n", nm[i], o[i], ok ? "" : "  <-- DIFFERS: \"=r\" was never safe here");
    }
    std::printf("  (4) destination/input aliasing changes the answer in %d of 5 aliased forms\n", alias_bad);
    g_fail += alias_bad;
  }

  std::printf("== probe %s (%d disagreements) ==\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
