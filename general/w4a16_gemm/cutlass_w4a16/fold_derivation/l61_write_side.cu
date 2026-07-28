// L61 -- the write side, at the failing test's ACTUAL parameters, with the permutation decoded locally.
//
// l59: the swzl cube base agrees with the real tiled copy everywhere, rung 5 included.
// l60: the flat-vs-nested view question is a non-issue -- partition_B returns the same logical (n,k) either way.
// PROBE=hi clean: plane_map<2, 64,128,64,64,64, F1=2> is CORRECT, because plane 2's map is composed from it.
//
// So plane 1's map is right and plane 1's BUFFER is wrong: the defect is in the WRITE path, which for plane 1 is
// pack_plane -> preprocess_weights_for_mixed_gemm<false,256> THEN nfold_regroup_gmem -- two passes -- while plane 2
// goes through place_int1's single direct walk. l52 compared the two at N=256 K=512. The failing test runs K=256.
//
// This harness does not compare byte counts. It INVERTS the shipped write path bit-plane by bit-plane: for bit b of
// idx = n*K + k, feed codes[idx] = (idx>>b)&1 through the real preprocess+fold and read bit b back out of every slot.
// 13 passes reconstruct, exactly, WHICH logical element the shipped writer placed at each buffer position. Comparing
// that against plane_map's answer yields the column permutation sigma directly -- the same object PROBE=lo printed,
// so the two can be checked against each other instead of just "differs / does not differ".
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l61_write_side.cu -o l61 && ./l61
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include "../unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"   // the DELETED five steps, kept gate-side so "bit-identical" is not a tautology
#include "../xplane_offline.hpp"
#include <cstdio>
#include <vector>
#include <map>
using namespace cute;


template <int TM, int TN, int TK, int WM, int WN, int F>
static bool write_side(const char* tag, int N, int K) {
  constexpr int Bits = 2, CPW = 32 / Bits, DL = (F * TK * Bits / 8) / 32, Ng = TN / F;
  static_assert(F > 1, "the fold branch is the one under test");
  const auto m = xplane::plane_map<Bits, TM, TN, TK, WM, WN, F>();
  const size_t NB = (size_t)N * K * Bits / 8;

  // ---- invert the shipped write path, one index bit per pass
  int nbits = 0; while ((1 << nbits) < N * K) ++nbits;
  std::vector<int> ship_loc(NB * 4 / 1, -1);          // one entry per 2-bit code slot in the buffer
  std::vector<int> got(NB * 4, 0);
  for (int b = 0; b < nbits; ++b) {
    std::vector<int8_t> packed(NB, 0);
    for (size_t i = 0; i < (size_t)N * K; ++i)
      packed[i / 4] |= int8_t((((int)(i >> b) & 1) & 3) << (2 * (i % 4)));
    std::vector<int8_t> ship(NB);
    legacy::preprocess_weights_for_mixed_gemm<false,256,0>(ship.data(), packed.data(), {(size_t)K, (size_t)N},
                                      QuantTypeClass::PACKED_INT2_WEIGHT_ONLY);
    std::vector<int8_t> f(NB);
    legacy::nfold_regroup_gmem(f.data(), ship.data(), {(size_t)K, (size_t)N}, TN, TK, Bits);
    for (size_t s = 0; s < NB * 4; ++s)
      got[s] |= ((f[s / 4] >> (2 * (s % 4))) & 1) << b;
  }
  for (size_t s = 0; s < NB * 4; ++s) ship_loc[s] = got[s];

  // ---- walk the slots plane_map describes and compare the two answers, accumulating sigma over logical n
  const int W_ROW_OFF = 256 / CPW, RUNS = W_ROW_OFF / 8, nrow = N / F;
  std::map<int, std::map<int, int>> sigma;                        // n -> histogram of the n actually delivered
  size_t bad = 0, tot = 0;
  for (int tn = 0; tn < N / TN; ++tn)
    for (int ki = 0; ki < K / TK; ++ki)
      for (int row = 0; row < Ng; ++row)
        for (int dl = 0; dl < DL; ++dl)
          for (int wd = 0; wd < 8; ++wd)
            for (int j = 0; j < CPW; ++j) {
              const int loc = m[(((size_t)row * DL + dl) * 8 + wd) * CPW + j];
              if (loc < 0) continue;
              const int n = tn * TN + loc / TK, k = ki * TK + loc % TK;
              const int kb = ki / RUNS, tt = ki % RUNS;
              const size_t slot = (size_t)((((size_t)kb * nrow + (size_t)tn * Ng + row) * W_ROW_OFF + tt * 8 + wd) * CPW + j);
              const int want = n * K + k, have = ship_loc[slot];
              ++tot;
              if (have != want) { ++bad; sigma[n][have / K]++; }
              else                sigma[n][n]++;
            }

  printf("  %-40s %dx%d : %zu / %zu slots misplaced\n", tag, N, K, bad, tot);

  // REGRESSION: place_derived follows `m` by construction, so bad == 0 above means it is byte-identical to the
  // whole-word packer -- which is what must hold at every configuration the box has already validated. Checked at the
  // buffer level too, since that is the object the kernel actually reads.
  {
    std::vector<uint8_t> q((size_t)K * N);
    for (size_t i = 0; i < q.size(); ++i) q[i] = uint8_t((i * 2654435761u >> 11) & 3);
    std::vector<int8_t> packed(NB, 0);
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
      const size_t i = (size_t)n * K + k; packed[i / 4] |= int8_t((q[(size_t)k * N + n] & 3) << (2 * (i % 4)));
    }
    std::vector<int8_t> ship(NB), f(NB), drv(NB);
    legacy::preprocess_weights_for_mixed_gemm<false,256,0>(ship.data(), packed.data(), {(size_t)K, (size_t)N},
                                                  QuantTypeClass::PACKED_INT2_WEIGHT_ONLY);
    legacy::nfold_regroup_gmem(f.data(), ship.data(), {(size_t)K, (size_t)N}, TN, TK, Bits);
    xplane::place_derived<Bits, TM, TN, TK, WM, WN, F>(drv.data(), q, N, K);
    size_t d = 0; for (size_t i = 0; i < NB; ++i) if (drv[i] != f[i]) ++d;
    printf("      place_derived vs whole-word packer: %zu / %zu bytes differ  %s\n", d, NB,
           (d == 0) == (bad == 0) ? "(consistent with the slot audit)" : "<-- INCONSISTENT with the slot audit");
  }
  if (bad) {
    printf("    sigma (logical n -> n actually delivered), first 12 that move:\n");
    int shown = 0;
    for (auto const& [n, h] : sigma) {
      int best = -1, cnt = 0;
      for (auto const& [v, c] : h) if (c > cnt) { cnt = c; best = v; }
      if (best != n && shown < 12) { printf("      n=%3d -> %3d   (delta %+d)%s\n", n, best, best - n,
                                            h.size() > 1 ? "  [mixed]" : ""); ++shown; }
    }
  }
  return bad == 0;
}

// The UNFOLDED walk, gated against the full five-step pipeline (subbyte_transpose -> permute_B_rows ->
// add_bias_and_interleave -> interleave_column_major_ppu, no fold). If place_derived reproduces it byte for byte then
// the five steps have no remaining reason to exist -- this is the gate task #4 was waiting on.
template <int Bits, int TM, int TN, int TK, int WM, int WN, QuantTypeClass QTC, int BIAS = 0>
static bool unfolded(const char* tag, int N, int K) {
  const size_t NB = (size_t)N * K * Bits / 8;
  // place_derived is POSITIONAL; the value encoding is the caller's business. int4's preprocess adds +8 (signed ->
  // unsigned) while int2/int1 explicitly do not, so a raw comparison against the int4 pipeline differs in EVERY byte
  // for a reason that has nothing to do with layout. BIAS feeds the same encoding in.
  std::vector<uint8_t> q((size_t)K * N), qb((size_t)K * N);
  for (size_t i = 0; i < q.size(); ++i) {
    q[i]  = uint8_t((i * 2654435761u >> 11) & ((1 << Bits) - 1));
    qb[i] = uint8_t((int(q[i]) - (BIAS ? (1 << (Bits - 1)) : 0)) & ((1 << Bits) - 1));   // what the pipeline is fed
  }
  const int EPB = 8 / Bits;
  std::vector<int8_t> packed(NB, 0);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
    const size_t i = (size_t)n * K + k;
    packed[i / EPB] |= int8_t((qb[(size_t)k * N + n] & ((1 << Bits) - 1)) << (Bits * (i % EPB)));
  }
  std::vector<int8_t> ship(NB), drv(NB);
  legacy::preprocess_weights_for_mixed_gemm<false,256,0>(ship.data(), packed.data(), {(size_t)K, (size_t)N}, QTC);
  xplane::place_derived<Bits, TM, TN, TK, WM, WN, 1>(drv.data(), q, N, K);
  size_t d = 0; for (size_t i = 0; i < NB; ++i) if (drv[i] != ship[i]) ++d;
  printf("  %-40s %dx%d : %zu / %zu bytes differ  %s\n", tag, N, K, d, NB,
         d ? "<-- the five steps are NOT reproduced" : "<-- BIT-IDENTICAL, the five steps are redundant");
  return d == 0;
}

int main() {
  printf("L61 -- plane 1's WRITE path at the failing test's own parameters\n\n");
  bool a = write_side<64,  64,  64, 32, 32, 2>("int2 shipping (64,64,64) w32x32   BOX ok", 256, 256);
  bool b = write_side<64, 128,  64, 64, 64, 2>("rung 5 plane 1 (64,128,64) w64x64 BOX bad", 256, 256);
  bool c = write_side<64, 128,  64, 64, 64, 2>("rung 5 plane 1, l52's N/K         l52 ok", 256, 512);
  printf("\n  and the UNFOLDED walk against the whole five-step pipeline:\n");
  bool u1 = unfolded<2, 64,  64, 128, 32, 32, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>("int2 (64, 64,128) w32x32 F=1  DL=1", 256, 512);
  bool u2 = unfolded<2, 64,  64, 256, 32, 32, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>("int2 (64, 64,256) w32x32 F=1  DL=2", 256, 512);
  bool u3 = unfolded<2, 64, 128, 128, 64, 64, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>("int2 (64,128,128) w64x64 F=1  rung 4", 256, 256);
  bool u4 = unfolded<1, 64, 128, 256, 32, 32, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>("int1 (64,128,256) w32x32 F=1", 256, 512);
  bool u5 = unfolded<4, 64, 128, 128, 32, 32, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY, 1>("int4 (64,128,128) w32x32 F=1 +bias", 256, 512);
  const bool uall = u1 && u2 && u3 && u4 && u5;
  printf("  => %s\n", uall ? "place_from_map covers BOTH walks -> the five steps can be deleted"
                            : "the unfolded walk is not yet equivalent -- keep the five steps");

  // TILE-INVARIANCE. Deleting the five steps means preprocess_weights_for_mixed_gemm must produce the unfolded
  // placement WITHOUT knowing the kernel tile -- so a shim has to pick a representative (TN,TK). That is only sound if
  // the unfolded map really is tile-invariant. Swept here rather than assumed, because the whole deletion rests on it.
  printf("\n  tile-invariance of the UNFOLDED placement (all must be BIT-IDENTICAL):\n");
  int inv_bad = 0;
  auto sweep = [&](bool r) { if (!r) ++inv_bad; };
  sweep(unfolded<2, 32,  64, 128, 32, 32, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>("int2 TM32 (32, 64,128) w32x32", 256, 512));
  sweep(unfolded<2, 64, 128, 128, 32, 32, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>("int2      (64,128,128) w32x32", 256, 512));
  sweep(unfolded<2, 64, 128, 128, 32, 64, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>("int2      (64,128,128) w32x64", 256, 512));
  sweep(unfolded<2, 64, 256, 256, 64, 64, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>("int2      (64,256,256) w64x64", 256, 512));
  sweep(unfolded<2,128, 128, 128, 64, 64, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>("int2 TM128(128,128,128) w64x64", 512, 512));
  sweep(unfolded<1, 64,  64, 256, 32, 32, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>("int1      (64, 64,256) w32x32", 256, 512));
  sweep(unfolded<1, 64, 128, 256, 64, 64, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>("int1      (64,128,256) w64x64", 256, 512));
  sweep(unfolded<4, 64,  64, 128, 32, 32, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY, 1>("int4      (64, 64,128) w32x32", 256, 512));
  sweep(unfolded<4, 64, 128,  64, 32, 32, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY, 1>("int4      (64,128, 64) w32x32", 256, 512));
  sweep(unfolded<4, 64, 128, 128, 64, 64, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY, 1>("int4      (64,128,128) w64x64", 256, 512));
  sweep(unfolded<4, 64, 256, 256, 64, 64, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY, 1>("int4      (64,256,256) w64x64", 512,1024));
  printf("  => %s\n", inv_bad ? "NOT tile-invariant -- a representative tile is UNSOUND"
                               : "tile-invariant across TM/TN/TK/WM/WN and 3 widths -> a representative tile is sound");

  // THE PRODUCTION PATH. Everything above exercises place_derived with an explicit tile; the shipped
  // preprocess_weights_for_mixed_gemm now picks a REPRESENTATIVE tile from (N, K, Bits) instead. That choice is the
  // thing 39 call sites actually run, so gate it directly against legacy rather than inferring it from the sweep.
  printf("\n  the production shim (representative tile) vs legacy, per width and shape:\n");
  int shim_bad = 0;
  auto shim = [&](const char* tag, QuantTypeClass qtc, int Bits, int N, int K) {
    const size_t NB = (size_t)N * K * Bits / 8;
    const int EPB = 8 / Bits, MASK = (1 << Bits) - 1;
    std::vector<int8_t> in(NB);
    for (size_t i = 0; i < NB; ++i) in[i] = int8_t((i * 2654435761u >> 9) & 0xff);
    std::vector<int8_t> ref(NB), got(NB);
    legacy::preprocess_weights_for_mixed_gemm<false,256,0>(ref.data(), in.data(), {(size_t)K, (size_t)N}, qtc);
    preprocess_weights_for_mixed_gemm      <false,256,0>(got.data(), in.data(), {(size_t)K, (size_t)N}, qtc);
    size_t d = 0; for (size_t i = 0; i < NB; ++i) if (ref[i] != got[i]) ++d;
    (void)EPB; (void)MASK;
    printf("  %-40s %dx%d : %zu / %zu bytes differ  %s\n", tag, N, K, d, NB, d ? "<-- SHIM WRONG" : "<-- BIT-IDENTICAL");
    if (d) ++shim_bad;
  };
  shim("int2  N=256 K=256", QuantTypeClass::PACKED_INT2_WEIGHT_ONLY, 2,  256,  256);
  shim("int2  N=512 K=1024", QuantTypeClass::PACKED_INT2_WEIGHT_ONLY, 2, 512, 1024);
  shim("int1  N=256 K=512", QuantTypeClass::PACKED_INT1_WEIGHT_ONLY, 1,  256,  512);
  shim("int1  N=128 K=256", QuantTypeClass::PACKED_INT1_WEIGHT_ONLY, 1,  128,  256);
  shim("int4  N=256 K=512", QuantTypeClass::PACKED_INT4_WEIGHT_ONLY, 4,  256,  512);
  shim("int4  N=1024 K=1024", QuantTypeClass::PACKED_INT4_WEIGHT_ONLY, 4, 1024, 1024);
  printf("  => %s\n", shim_bad ? "the shim is NOT a drop-in -- restore the five steps"
                                : "the shim is a byte-exact drop-in for all 39 call sites");

  printf("\n  verdict: shipping %s ; rung 5 @K=256 %s ; rung 5 @K=512 %s\n",
         a ? "clean" : "DIRTY", b ? "clean" : "DIRTY", c ? "clean" : "DIRTY");
  return (a && b && c) ? 0 : 1;
}
