// L91 -- PHASE 0 GATE for plan #20's native scale channel. Host-only, runnable, no kernel changes.
//
// Subject:  gguf_scale_layout.hpp's cute-Layout-driven decode.
// Oracle:   a LITERAL transcription of real_weight/dump_real_weights.py, quoted beside each line. That file is
//           itself regressed against marlin_gguf_ppu.cuh:gguf_q4k_scales.
//
// The comparison is EXHAUSTIVE, not sampled. Each decode is a selection of bits, hence linear over GF(2) in the
// input bits, so agreement on the all-zero block and on every single-bit block proves agreement on all 2^96 (or
// 2^128) inputs. Random blocks are run too, but they add nothing -- they are there only to make that argument
// falsifiable if it is wrong.
//
// Plus a KNOWN-ANSWER row per format, hand-computed in the comments, so the gate cannot be vacuously green.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "gguf_scale_layout.hpp"
using namespace gguf_scale;

// ---- ORACLE: dump_real_weights.py:get_scale_min_k4, transcribed literally --------------------------------------
//   for g in range(4):     sc[g] = q[g] & 63                                  ; mn[g] = q[g+4] & 63
//   for g in range(4,8):   sc[g] = (q[g+4]&0xF) | ((q[g-4]>>6)<<4)            ; mn[g] = (q[g+4]>>4) | ((q[g]>>6)<<4)
static void oracle_q4k(uint8_t const* q, int* sc, int* mn) {
  for (int g = 0; g < 4; ++g) { sc[g] = q[g] & 63;            mn[g] = q[g + 4] & 63; }
  for (int g = 4; g < 8; ++g) { sc[g] = (q[g + 4] & 0xF) | ((q[g - 4] >> 6) << 4);
                                mn[g] = (q[g + 4] >> 4)  | ((q[g]     >> 6) << 4); }
}
// ---- ORACLE: dump_real_weights.py:unpack_q3k_expert's 12-byte -> 16 six-bit loop -------------------------------
static void oracle_q3k(uint8_t const* s, int* sc) {
  for (int t = 0; t < 4; ++t) {
    sc[ 0 + t] = ( s[0 + t]       & 0xF) | (((s[8 + t] >> 0) & 0x3) << 4);
    sc[ 4 + t] = ( s[4 + t]       & 0xF) | (((s[8 + t] >> 2) & 0x3) << 4);
    sc[ 8 + t] = ((s[0 + t] >> 4) & 0xF) | (((s[8 + t] >> 4) & 0x3) << 4);
    sc[12 + t] = ((s[4 + t] >> 4) & 0xF) | (((s[8 + t] >> 6) & 0x3) << 4);
  }
}
// ---- ORACLE: Q2_K / Q6_K, byte = g -----------------------------------------------------------------------------
static void oracle_q2k(uint8_t const* q, int* sc, int* mn) {
  for (int g = 0; g < 16; ++g) { sc[g] = q[g] & 0xF; mn[g] = q[g] >> 4; }
}
static void oracle_q6k(uint8_t const* q, int* sc) { for (int g = 0; g < 16; ++g) sc[g] = int(int8_t(q[g])); }

static int g_fail = 0;
template <KType T>
static void check_block(uint8_t const* p, char const* what) {
  using Tr = Traits<T>;
  int osc[16] = {}, omn[16] = {};
  if constexpr (T == KType::Q4_K) oracle_q4k(p, osc, omn);
  if constexpr (T == KType::Q3_K) oracle_q3k(p, osc);
  if constexpr (T == KType::Q2_K) oracle_q2k(p, osc, omn);
  if constexpr (T == KType::Q6_K) oracle_q6k(p, osc);
  for (int g = 0; g < Tr::kGroups; ++g) {
    int s = scale_of<T>(p, g);
    if (s != osc[g]) { if (g_fail < 8) std::printf("  [FAIL] %s sc g=%d layout=%d oracle=%d\n", what, g, s, osc[g]); ++g_fail; }
    if constexpr (Tr::kHasMin) {
      int m = min_of<T>(p, g);
      if (m != omn[g]) { if (g_fail < 8) std::printf("  [FAIL] %s mn g=%d layout=%d oracle=%d\n", what, g, m, omn[g]); ++g_fail; }
    }
  }
}

template <KType T>
static void sweep(char const* name) {
  using Tr = Traits<T>;
  constexpr int NB = Tr::kBlockBytes;
  std::vector<uint8_t> b(NB, 0);
  check_block<T>(b.data(), name);                                    // the zero vector
  for (int bit = 0; bit < NB * 8; ++bit) {                           // every unit vector
    std::memset(b.data(), 0, NB);
    b[bit / 8] = uint8_t(1u << (bit % 8));
    check_block<T>(b.data(), name);
  }
  srand(1234);
  for (int it = 0; it < 20000; ++it) {                               // falsifiers for the linearity argument
    for (int i = 0; i < NB; ++i) b[i] = uint8_t(rand() & 0xFF);
    check_block<T>(b.data(), name);
  }
  std::printf("  %-5s bits tile exactly: %s | exhaustive over %d unit vectors + zero + 20000 random\n",
              name, bits_tile_exactly<T>() ? "yes" : "NO", NB * 8);
}

int main() {
  std::printf("== plan #20 phase 0: GGUF scale block as cute Layouts ==\n");
  sweep<KType::Q4_K>("Q4_K");
  sweep<KType::Q3_K>("Q3_K");
  sweep<KType::Q2_K>("Q2_K");
  sweep<KType::Q6_K>("Q6_K");

  // ---- KNOWN ANSWERS, hand-computed, so a broken oracle cannot make this vacuous -------------------------------
  // Q4_K q = {0x41,0,0,0, 0x82,0,0,0, 0x35,0,0,0}
  //   sc[0] = q[0]&63 = 0x41&0x3F = 1                 mn[0] = q[4]&63 = 0x82&0x3F = 2
  //   sc[4] = (q[8]&0xF) | ((q[0]>>6)<<4) = 5|16 = 21 mn[4] = (q[8]>>4) | ((q[4]>>6)<<4) = 3|32 = 35
  {
    uint8_t q[12] = {0x41,0,0,0, 0x82,0,0,0, 0x35,0,0,0};
    struct { int g; int sc; int mn; } ka[] = {{0,1,2},{4,21,35}};
    for (auto& e : ka) {
      int s = scale_of<KType::Q4_K>(q, e.g), m = min_of<KType::Q4_K>(q, e.g);
      bool ok = (s == e.sc && m == e.mn);
      std::printf("  Q4_K known-answer g=%d: sc=%d (want %d), mn=%d (want %d) -> %s\n", e.g, s, e.sc, m, e.mn, ok?"ok":"FAIL");
      if (!ok) ++g_fail;
    }
  }
  // Q3_K s = {0x51,0,0,0, 0x62,0,0,0, 0x9B,0,0,0}
  //   sc[0]  = (s[0]&0xF) | ((s[8]&3)<<4)        = 1 | (3<<4) = 49
  //   sc[4]  = (s[4]&0xF) | (((s[8]>>2)&3)<<4)   = 2 | (2<<4) = 34
  //   sc[8]  = (s[0]>>4)  | (((s[8]>>4)&3)<<4)   = 5 | (1<<4) = 21
  //   sc[12] = (s[4]>>4)  | (((s[8]>>6)&3)<<4)   = 6 | (2<<4) = 38
  {
    uint8_t s[12] = {0x51,0,0,0, 0x62,0,0,0, 0x9B,0,0,0};
    struct { int g; int sc; } ka[] = {{0,49},{4,34},{8,21},{12,38}};
    for (auto& e : ka) {
      int v = scale_of<KType::Q3_K>(s, e.g);
      bool ok = (v == e.sc);
      std::printf("  Q3_K known-answer g=%2d: sc=%d (want %d) -> %s\n", e.g, v, e.sc, ok?"ok":"FAIL");
      if (!ok) ++g_fail;
    }
  }

  // ---- REAL GGUF DATA: raw scale blocks from a real Q4_K tensor, against the PYTHON's own decode ---------------
  // real_weight/dump_scale_blocks.py imports dump_real_weights and writes, per superblock, the raw 12 bytes plus its
  // get_scale_min_k4 output. So this compares the cute Layouts against the reference decoder ON REAL BYTES, exact
  // integer, which is what plan #20's phase 0 asks for.
  //
  // NOTE the earlier version of this block read real_q3k_concat.bin at offset 96 as if it were a raw Q3_K record.
  // It is not -- those .bin files are the already-decoded harness inputs for test_moe_grouped_real.cu, and
  // 286736 % 110 != 0 gives it away. That check compared two decoders on twelve arbitrary bytes and called it a
  // real-file test. The exhaustive sweep above proves the BIT MAP; only this block proves the RECORD OFFSETS.
  if (FILE* f = std::fopen("../real_weight/scale_blocks_q4k.bin", "rb")) {
    char magic[4] = {}; uint32_t hdr[4] = {};
    if (std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "GSB0", 4) == 0 && std::fread(hdr, 4, 4, f) == 4) {
      uint32_t const nb = hdr[1], bb = hdr[2], ng = hdr[3];
      std::vector<uint8_t> raw(bb);
      std::vector<int32_t> rsc(ng), rmn(ng);
      int bad = 0;
      for (uint32_t i = 0; i < nb; ++i) {
        if (std::fread(raw.data(), 1, bb, f) != bb) break;
        if (std::fread(rsc.data(), 4, ng, f) != ng) break;
        if (std::fread(rmn.data(), 4, ng, f) != ng) break;
        for (uint32_t g = 0; g < ng; ++g) {
          if (scale_of<KType::Q4_K>(raw.data(), int(g)) != rsc[g]) ++bad;
          if (min_of  <KType::Q4_K>(raw.data(), int(g)) != rmn[g]) ++bad;
        }
      }
      std::printf("  Q4_K real GGUF: %u blocks x %u groups against dump_real_weights.get_scale_min_k4 -> %d bad\n",
                  nb, ng, bad);
      g_fail += bad;
    }
    std::fclose(f);
  } else std::printf("  (scale_blocks_q4k.bin missing -- run real_weight/dump_scale_blocks.py <file.gguf>)\n");

  std::printf("== %s: %d mismatches ==\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
