// L60 -- what PROBE=hi being CLEAN actually proves, and the two remaining candidates.
//
// The probe result is more informative than "plane 1 is wrong". Plane 2's map is COMPOSED from plane 1's:
// tile_map_int1 calls plane_map<2,...,F1> and, for every slot the converter reads, records the logical element whose
// high bit plane 1 will pair with it. So plane 2's buffer is correct ONLY IF plane_map<2,...,F1=2> is correct.
//
//     PROBE=hi clean  =>  plane_map<2, 64,128,64,64,64, F1=2> is CORRECT.
//     PROBE=lo +32    =>  plane 1's BUFFER is wrong anyway.
//
// A correct map and a wrong buffer means the defect is in the WRITE side, not the derivation. Two candidates:
//
//   (1) plane_map partitions the FLAT (TN, TK) identity, while the kernel's tCrB_mma partitions the NESTED
//       ((P1Fold, TN/P1Fold), TK) SmemLayoutB_MmaView. partition_fragment_B only reads the shape, so the FRAGMENT is
//       identical -- that much was already established -- but partition_B on an IDENTITY tensor returns a coordinate,
//       and a nested shape returns a nested coord whose flattening order is a real question, not an assumption.
//
//   (2) plane 1's buffer is produced by pack_plane -> preprocess_weights_for_mixed_gemm<false,256,0> followed by
//       nfold_regroup_gmem, i.e. interleave-256 THEN fold, two passes. plane 2's is produced by place_int1's single
//       direct walk. l52 compared them at N=256 K=512; the failing test runs N=256 K=256.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l60_flat_vs_nested.cu -o l60 && ./l60
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <vector>
using namespace cute;

// (1) FLAT vs NESTED partition_B. Same fragment either way; the question is whether the logical column each slot
// resolves to is the same. Reported as a flat n in both cases so they are directly comparable.
template <int TM, int TN, int TK, int WM, int WN, int F>
static bool flat_vs_nested(const char* tag) {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  constexpr int warpM = (WM > 16) ? WM : 16, warpN = (WN > 16) ? WN : 16;
  constexpr int WOM = TM / warpM, WON = TN / warpN;
  constexpr int MmaPermK = (F > 1) ? TK : (32 * 8 / 2);
  using Mma = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<MmaPermK>>>;

  auto id_flat   = make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{}));
  // exactly SmemLayoutB_MmaView's shape: ((P1Fold, TN/P1Fold), TK), f fastest
  auto id_nested = make_identity_tensor(make_shape(make_shape(Int<F>{}, Int<TN/F>{}), Int<TK>{}));

  int bad = 0, shown = 0;
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    auto pf = Mma{}.get_thread_slice(t).partition_B(id_flat);
    auto pn = Mma{}.get_thread_slice(t).partition_B(id_nested);
    if (size(pf) != size(pn)) { printf("  %-44s SIZE differs %d vs %d\n", tag, int(size(pf)), int(size(pn))); return false; }
    for (int i = 0; i < int(size(pf)); ++i) {
      auto cf = pf(i);
      auto cn = pn(i);
      const int nf = int(get<0>(cf)), kf = int(get<1>(cf));
      // flatten the nested coord back to a logical column: n = f + F*g
      const int nn = int(get<0>(get<0>(cn))) + F * int(get<1>(get<0>(cn))), kn = int(get<1>(cn));
      if (nf != nn || kf != kn) {
        ++bad;
        if (shown < 6) { printf("    t=%3d i=%3d : flat (n=%3d,k=%3d)  nested (n=%3d,k=%3d)\n", t, i, nf, kf, nn, kn); ++shown; }
      }
    }
  }
  printf("  %-44s %s\n", tag, bad ? "DIFFER <-- plane_map partitions the wrong view" : "identical (view nesting is not the bug)");
  return bad == 0;
}

int main() {
  printf("L60 -- flat (TN,TK) vs nested ((F,TN/F),TK) partition_B\n\n");
  bool a = flat_vs_nested<64, 128, 128, 64, 64, 1>("rung 4  (64,128,128) w64x64 F=1  BOX bad=0");
  bool b = flat_vs_nested<64,  64,  64, 32, 32, 2>("int2 ship (64,64,64) w32x32 F=2  BOX ok");
  bool c = flat_vs_nested<64, 128,  64, 64, 64, 2>("rung 5  (64,128, 64) w64x64 F=2  BOX bad");
  printf("\n  verdict: %s\n", (a && b && c) ? "the nesting is irrelevant -- the defect is on the WRITE side"
                                            : "the nesting MATTERS -- plane_map must use SmemLayoutB_MmaView's shape");
  return (a && b && c) ? 0 : 1;
}
