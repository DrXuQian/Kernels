// L64 -- the gate for retiring nfold_regroup_gmem and legacy::nfold_place_bits_int1_tk64(task #13).
//
// WHY THEY MUST GO. nfold_regroup_gmem moves whole uint32 words, so each word carries ONE logical column, which is what
// the mma wants only while cols_per_word == 1 -- i.e. warp N extent 32. Every remaining caller happens to be w32x32, so
// they are all correct today. But this session measured w64x32 as worth +7 to +9 points on int2 and int4, so WN=64 is
// exactly where the tuning is going, and the first caller that follows it would silently miscompute. A landmine that is
// only safe because nobody has walked there yet.
//
// WHAT THE MIGRATION IS. xplane::place_derived covers BOTH destination walks -- the fold one and the interleave-256 one
// -- so one call replaces `preprocess_weights_for_mixed_gemm` + `nfold_regroup_gmem` AND the separate bit-granular
// nfold_place_bits_int1_tk64. Already proven byte-identical at the two configurations the box validated: the int2 fold
// config (64,64,64) w32x32 F=2 in l61, and int1 at F=4 against nfold_place_bits_int1_tk64 in l58.
//
// WHAT THIS ADDS. Those two points are not the set the harnesses actually use. This diffs place_derived against the
// legacy path for EVERY (Bits, TN, TK) tuple test_int1_sweep, test_width_acu and test_fold_int2 pass, at two TM values,
// so the migration is verified on the configurations being migrated rather than on two neighbours of them.
//
// The harnesses reuse ONE buffer across several TM while holding WM=WN=32, so they already depend on the map being
// TM-invariant. plane_map's destination contains RPI = WON*16 and no TM at all -- TM enters only through the thread
// count, whose slices union to the same map -- and l61 swept TM 32/64/128 identically. Both TM values are checked here
// anyway, since "already depends on" is not the same as "verified".
//
//   nvcc -std=c++17 -Istub_inc -I.. -I../../../../third_party/actlize/include l64_migrate_gate.cu -o l64 && ./l64
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include "../unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"   // (f) the N-fold packers live here now -- gate-side only
#include "../xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;

// The legacy path, exactly as the harnesses call it: bit-granular for int1 at TK=64, otherwise preprocess then fold.
template <int Bits, int TN, int TK>
static std::vector<int8_t> legacy_pack(const std::vector<uint8_t>& q_kn, int N, int K) {
  constexpr int EPB = 8 / Bits, MASK = (1 << Bits) - 1;
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  const size_t bytes = (size_t)N * K / EPB;
  std::vector<int8_t> nk(bytes, 0), out(bytes, 0);
  for (int n = 0; n < N; ++n)                                     // [N][K] packed, what both legacy entry points take
    for (int k = 0; k < K; ++k) {
      const size_t i = (size_t)n * K + k;
      nk[i / EPB] |= int8_t((q_kn[(size_t)k * N + n] & MASK) << (Bits * (i % EPB)));
    }
  if (Bits == 1 && TK == 64) {
    legacy::nfold_place_bits_int1_tk64(out.data(), nk.data(), N, K, TN, TK);
  } else {
    const auto qtc = Bits == 1 ? QuantTypeClass::PACKED_INT1_WEIGHT_ONLY
                   : Bits == 2 ? QuantTypeClass::PACKED_INT2_WEIGHT_ONLY
                               : QuantTypeClass::PACKED_INT4_WEIGHT_ONLY;
    preprocess_weights_for_mixed_gemm<false, 256, 0>(out.data(), nk.data(), {(size_t)K, (size_t)N}, qtc);
    if (F > 1) { std::vector<int8_t> f(bytes);
                 legacy::nfold_regroup_gmem(f.data(), out.data(), {(size_t)K, (size_t)N}, TN, TK, Bits); out.swap(f); }
  }
  return out;
}

template <int Bits, int TM, int TN, int TK, int WM = 32, int WN = 32>
static bool one(const char* who, int N, int K) {
  constexpr int EPB = 8 / Bits, MASK = (1 << Bits) - 1;
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  std::vector<uint8_t> q((size_t)K * N);
  for (size_t i = 0; i < q.size(); ++i) q[i] = uint8_t((i * 2654435761u >> 7) & MASK);
  // int4's preprocess adds +8 (signed -> unsigned); place_derived is positional, so feed the same encoding.
  std::vector<uint8_t> qb = q;
  if (Bits == 4) for (size_t i = 0; i < qb.size(); ++i) qb[i] = uint8_t((int(q[i]) + 8) & MASK);

  auto ref = legacy_pack<Bits, TN, TK>(q, N, K);
  std::vector<int8_t> got((size_t)N * K / EPB);
  xplane::place_derived<Bits, TM, TN, TK, WM, WN, F>(got.data(), qb, N, K);
  size_t d = 0; for (size_t i = 0; i < ref.size(); ++i) if (ref[i] != got[i]) ++d;
  printf("  %-30s int%d TM=%-3d TN=%-3d TK=%-3d w%dx%-2d F=%d : %zu / %zu bytes differ  %s\n",
         who, Bits, TM, TN, TK, WM, WN, F, d, ref.size(), d ? "<-- DO NOT MIGRATE" : "BIT-IDENTICAL");
  return d == 0;
}

int main() {
  printf("L64 -- place_derived vs the legacy packers, on the configurations actually being migrated\n\n");
  int bad = 0;
  auto T = [&](bool r) { if (!r) ++bad; };

  printf("  test_int1_sweep: upload_weights<TN=128, TK> for TK 64 / 128 / 256, TM 32 and 64\n");
  // int1 at TK=64 CANNOT run at WN=32: delivery 128 > slots WN*TK/32 = 64, which is precisely why the bit-granular
  // packer exists. The harness runs w32x64. Writing 32,32 here made the gate call a real config broken -- the sixth
  // time this session that a relation written into a harness lost to one read off the thing it describes.
  T(one<1,  64, 128,  64, 32, 64>("int1_sweep w32x64", 512, 512));
  T(one<1,  32, 128,  64, 32, 64>("int1_sweep w32x64", 512, 512));
  T(one<1,  32, 128, 128>("int1_sweep", 512, 512));
  T(one<1,  64, 128, 128>("int1_sweep", 512, 512));
  T(one<1,  32, 128, 256>("int1_sweep", 512, 512));
  T(one<1,  64, 128, 256>("int1_sweep", 512, 512));

  printf("\n  test_width_acu: pack<Bits, TNp, TK> across the three widths\n");
  T(one<1,  64, 128,  64, 64, 64>("width_acu w64x64", 512, 512));
  T(one<1,  32, 128,  64, 32, 64>("width_acu w32x64", 512, 512));
  T(one<2,  32, 128,  64>("width_acu", 512, 512));
  T(one<2,  32, 128, 128>("width_acu", 512, 512));
  T(one<4,  32, 128,  64>("width_acu", 512, 512));
  T(one<4,  32, 128, 128>("width_acu", 512, 512));
  T(one<4,  64,  64, 128>("width_acu", 512, 512));

  printf("\n  test_fold_int2: int2 folded, and the int1 companion\n");
  T(one<2,  64,  64,  64>("fold_int2", 512, 512));
  T(one<2,  32,  64,  64>("fold_int2", 512, 512));
  T(one<2,  64, 128,  64>("fold_int2", 512, 512));
  T(one<1,  64, 128,  64, 32, 64>("fold_int2 w32x64", 512, 512));

  // EVERY ROW OF test_fold_int2's FOLD_CONFIGS, which is what it now dispatches after the #13 migration. The rows
  // above were written for the pre-migration harness and miss two of them -- and "the neighbours match" is exactly the
  // reasoning that let rung 5 ship broken (l52 passed because its probe value was invariant under the displacement).
  printf("\n  test_fold_int2 FOLD_CONFIGS: all 10 rows, so the gate covers what the harness actually runs\n");
  T(one<2,  32, 128, 128>("FOLD_CONFIGS", 512, 512));  T(one<2,  32, 128,  64>("FOLD_CONFIGS", 512, 512));
  T(one<2,  64, 128, 128>("FOLD_CONFIGS", 512, 512));  T(one<2,  64, 128,  64>("FOLD_CONFIGS", 512, 512));
  T(one<2,  64,  64, 128>("FOLD_CONFIGS", 512, 512));  T(one<2,  64,  64,  64>("FOLD_CONFIGS", 512, 512));
  T(one<1,  32, 128, 128>("FOLD_CONFIGS", 512, 512));  T(one<1,  64, 128, 128>("FOLD_CONFIGS", 512, 512));
  T(one<1,  64,  64, 128>("FOLD_CONFIGS", 512, 512));
  T(one<1,  64, 128,  64, 32, 64>("FOLD_CONFIGS", 512, 512));

  printf("\n  %s\n", bad ? "SOME CONFIGURATIONS DIFFER -- migrate only the ones that match"
                         : "every migrated configuration is byte-identical -- safe to delete both legacy packers");
  return bad ? 1 : 0;
}
