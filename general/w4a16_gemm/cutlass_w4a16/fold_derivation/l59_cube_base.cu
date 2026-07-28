// L59 -- the ONE link in the chain that was never a cute layout, and the shared premise that made every layout
// derivation self-consistent and wrong.
//
// The probe settled it on hardware: at rung 5 `(64,128,64) w64x64 F1=2 F2=4`, PROBE=hi is CLEAN (0/256 columns
// permuted -- the cross-plane composition is right) and PROBE=lo is wrong for exactly half the columns, with a
// CONSTANT delta of +32. Constant and half means a bit permutation of the column index, not a magnitude or scale
// error: within each TN=128 tile the two middle 32-column blocks are swapped, i.e. n's bit 5 and bit 6 exchange.
//
// Everything in the derivation is a cute Layout -- MixGemmEmit, right_inverse(frag.layout()), partition_B, AtLayout,
// the offline's destination walk -- EXCEPT the swzl cube base, which l20 and plane_map both hand-write as
//
//     row = inst*RPI + 16*warp_n + (v/2)*8 + lane/4        RPI = WON*16
//
// The second half, `(v/2)*8 + lane/4`, is Copy_Traits<PPU0010_TSM_LD_SWZL>::LogicalTV verbatim, so it is not the
// suspect. The first half -- which physical rows a given (warp_n, mode-1 index) actually covers -- comes from
// make_tiled_copy_B(SmemCopyAtomB, tiled_mma_s8), and NOTHING has ever checked the hand-written version against it.
//
// It cannot be caught by comparing plane_map against the shipped writer: l52 line 58 runs exactly rung 5's plane-1
// config and they agree byte for byte, because nfold_regroup_gmem contains the SAME hand-written cube base. Two
// implementations of one wrong premise agree. `nfold_regroup_gmem` is box-validated only at (64,64,64) w32x32 F=2.
//
// So: instantiate the real tiled copy and read the cube base off partition_S. cute's answer, no formula.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l59_cube_base.cu -o l59 && ./l59
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <vector>
using namespace cute;

template <int Bits, int BlockK> struct AiuOp {
  static constexpr int BlockCont = BlockK * Bits / 8;
  static constexpr int AiuByte   = BlockCont > 128 ? 128 : BlockCont;
  static constexpr int AiuElem   = AiuByte * 8 / Bits;
  static constexpr int InstNum   = BlockK / AiuElem;
};

// For one plane: compare the hand-written cube base against what the REAL tiled copy partitions.
// `Bits` is the plane's width, `F` its own fold factor. The swzl atom is int8-typed (AiuContElemSize*Bits/8 bytes).
template <int Bits, int TM, int TN, int TK, int WM, int WN, int F>
static bool cube_base(const char* tag, bool verbose) {
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int warpM = (WM > 16) ? WM : 16, warpN = (WN > 16) ? WN : 16;
  constexpr int WOM = TM / warpM, WON = TN / warpN;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                         Tile<Int<WOM*16>, Int<WON*16>, _32>>;

  constexpr int BK = F * TK, Ng = TN / F, RowB = BK * Bits / 8;
  using OB = AiuOp<Bits, BK>;
  using AtomB = Copy_Atom<PPU0010_TSM_LD_SWZL<int8_t, Ng, OB::AiuElem * Bits / 8, true, false, OB::InstNum>, int8_t>;

  // The identity over the PHYSICAL (Ng, RowB) byte tile -- partition_S then reports, per (thread, mode1, mode2),
  // the actual (row, byte) each thread's cube starts at.
  auto idB = make_identity_tensor(make_shape(Int<Ng>{}, Int<RowB>{}));
  auto tiled = make_tiled_copy_B(AtomB{}, MmaS8{});

  constexpr int RPI = WON * 16;
  const int NThr = 32 * WOM * WON;
  bool ok = true;
  int shown = 0;

  auto t0 = tiled.get_thread_slice(0).partition_S(idB);
  if (verbose)
    printf("    partition_S: "), print(t0.layout()), printf("   Ng=%d RowB=%d RPI=%d InstNum=%d AiuElemB=%d\n",
                                                            Ng, RowB, RPI, OB::InstNum, int(OB::AiuElem * Bits / 8));

  for (int t = 0; t < NThr; ++t) {
    const int warp = t / 32, warp_n = warp / WOM;
    // The kernel slices these smem->reg copies at aiu_warp_group_thread_idx = warp_idx*32, so every lane of a warp
    // shares the slice and the lane dimension lives inside the atom's value layout.
    auto ts = tiled.get_thread_slice(warp * 32).partition_S(idB);
    for (int m1 = 0; m1 < int(size<1>(ts)); ++m1) {
      const int got  = int(get<0>(ts(0, m1, 0)));                 // cute's cube base row
      const int hand = m1 * RPI + 16 * warp_n;                    // the hand-written one
      if (got != hand) {
        ok = false;
        if (shown < 8) {
          printf("    warp %d (warp_n=%d) mode1=%d :  cute row %3d   hand %3d   (delta %+d)\n",
                 warp, warp_n, m1, got, hand, hand - got);
          ++shown;
        }
      }
    }
  }
  printf("  %-46s %s\n", tag, ok ? "cube base AGREES with the hand-written formula"
                                 : "<-- MISMATCH: the hand-written cube base is wrong here");
  return ok;
}

int main() {
  printf("L59 -- the swzl cube base: hand-written formula vs the real tiled copy\n\n");

  printf("  A. plane 1 (int2) at the rungs the box has already judged\n");
  // rungs 1-4 are bad=0 on the box, so the formula must AGREE here or the model is broken somewhere else too.
  bool r1 = cube_base<2, 64,  64, 128, 32, 32, 1>("rung 1  (64, 64,128) w32x32 F1=1  BOX bad=0", false);
  bool r2 = cube_base<2, 64, 128, 128, 32, 32, 1>("rung 2  (64,128,128) w32x32 F1=1  BOX bad=0", false);
  bool r3 = cube_base<2, 64, 128, 128, 32, 64, 1>("rung 3  (64,128,128) w32x64 F1=1  BOX bad=0", false);
  bool r4 = cube_base<2, 64, 128, 128, 64, 64, 1>("rung 4  (64,128,128) w64x64 F1=1  BOX bad=0", false);
  // the config nfold_regroup_gmem IS box-validated at (int2's shipping 53.2% row)
  bool rv = cube_base<2, 64,  64,  64, 32, 32, 2>("int2 shipping (64,64,64) w32x32 F1=2  BOX ok", false);

  printf("\n  B. rung 5's plane 1 -- PROBE=lo says half the columns are +32\n");
  bool r5 = cube_base<2, 64, 128,  64, 64, 64, 2>("rung 5  (64,128, 64) w64x64 F1=2  BOX bad", true);

  printf("\n  C. rung 5's plane 2 -- PROBE=hi says this one is CLEAN, so it must agree\n");
  bool p2 = cube_base<1, 64, 128,  64, 64, 64, 4>("rung 5  plane 2 int1 F2=4        BOX ok", true);

  const bool boxok = r1 && r2 && r3 && r4 && rv && p2;
  printf("\n  verdict: box-passing configs %s ; rung 5 plane 1 %s\n",
         boxok ? "all AGREE" : "DISAGREE <-- model broken elsewhere too",
         r5 ? "AGREES (so the cube base is NOT the bug)" : "MISMATCHES <-- this is the bug");
  return (boxok && !r5) ? 0 : 1;
}
