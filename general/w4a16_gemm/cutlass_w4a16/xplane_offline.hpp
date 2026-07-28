#pragma once
// CROSS-PLANE offline placement for the FOLDED high plane of a 2-plane (B-concat) GEMM.
//
// WHY THIS EXISTS, and what is actually wrong without it. With the shipped high-vreg offset hi_vreg0 = kb % P2_DIV,
// a folded plane 2 (P2_DIV == 1) reads only vregs 0 and 2 -- vregs 1 and 3 are NEVER touched, so HALF the tile's high
// bits cannot arrive at all and no placement repairs it. The kernel must use
//     hi_vreg0 = (kb % P2_DIV) + P2_DIV * (ii / MMA_N2)
// and the placement must then move: the composition demands a map differing from plane 2's own single-plane rule in
// 4096 of 8192 entries. Changing only the index measured WORSE on the box than changing neither (bad 15010 -> 29666
// of 32768) -- the two are ONE change.
//
// HOW IT IS DERIVED, and gated. Compose forward and require the identity:
//     delivered position (thread, vreg, code) --[ tile map = the offline placement ]--> logical (n, k)
// for BOTH planes, paired the way MixGemm2Plane_uint2_uint1 pairs them, from its own _E2 lines (l37):
//     line (t, v):  LOW  crumb of lo[v] at code (t%4) + 4*(t/4) [+8]
//                   HIGH bit of hi[hi_vreg0 + 2*(v>>1)] at 8*(v&1) + t [+16]
// THE GATE: at F2=1 this must reproduce plane 2's shipped map exactly, because that configuration measures bad=0 on
// the box. It does -- 0 differ, 0 unset (fold_derivation/l49). Three earlier derivations (l44, l45, l46) had no such
// gate, or gated on their own self-consistency, and all three were wrong; l44's premise is retracted in that file.
//
#include <vector>
#include <cstdint>
#include <cstddef>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"

namespace xplane {

// ONE plane's own map, l20's structure, generalised to DL deliveries per physical row. DL == 1 is l20 verbatim; DL
// > 1 is needed for int2 at Block_K=256, which is the configuration that serves as the GATE (it runs correctly on the
// box, so the derivation must reproduce it). Order=1 -- N-instance outer, delivery inner -- is the chunk ordering that
// satisfies that gate; it was resolved against the gate, not chosen.
template <int Bits, int TM, int TN, int TK, int WM, int WN, int F>
inline std::vector<int> plane_map() {
  using namespace cute;
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  constexpr int WOM = TM / warpM, WON = TN / warpN;
  constexpr int CPW = 32 / Bits, Ng = TN / F, RPI = WON * 16, VEC = 4 * 32 / Bits;
  constexpr int DL = (F * TK * Bits / 8) / 32;
  static_assert(DL >= 1 && DL * 8 * CPW == F * TK, "row must be a whole number of 32B deliveries");
  using Mma = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<32 * 8 / Bits>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>, Int<TK>>{}, Stride<Int<TK>, _1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));
  auto pi = right_inverse(frag.layout());
  std::vector<int> m((size_t)Ng * DL * 8 * CPW, -1);
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int dl = 0; dl < DL; ++dl)
      for (int inst = 0; inst < Ng / RPI; ++inst)
        for (int v = 0; v < 4; ++v) {
          const int row = inst * RPI + 16 * warp_n + (v / 2) * 8 + lane / 4, wd = (v % 2) * 4 + lane % 4;
          for (int j = 0; j < CPW; ++j) {
            const int e = cutlass::MixGemmEmit<Bits>::index(j, v), flat = (inst * DL + dl) * VEC + e;
            if (flat < 0 || flat >= NS) continue;
            auto c = part(pi(flat));
            m[(((size_t)row * DL + dl) * 8 + wd) * CPW + j] = int(get<0>(c)) * TK + int(get<1>(c));
          }
        }
  }
  return m;
}

// The high plane's map, COMPOSED from plane 1's: for every slot the converter reads, record the logical element whose
// high bit plane 1 will pair with it. Gated at F2=1, where it reproduces plane_map<1,...> (the shipped buffer) exactly.
//
// The kernel MUST use hi_vreg0 = (kb % P2_DIV) + P2_DIV * (ii / MMA_N2) for this to be the right map. With the shipped
// hi_vreg0 = kb % P2_DIV, vregs 1 and 3 are never read at all and HALF the tile's high bits cannot arrive -- no
// placement repairs that. Changing only the index (and not the placement) measured WORSE on the box than changing
// neither: 15010 -> 29666 bad of 32768. They are one change.
template <int TM, int TN, int TK, int WM, int WN, int F2>
inline std::vector<int> tile_map_int1() {
  using namespace cute;
  constexpr int InstM = 16, warpM = (WM > InstM) ? WM : InstM, warpN = (WN > 16) ? WN : 16;
  constexpr int WOM = TM / warpM, WON = TN / warpN, RPI = WON * 16;
  constexpr int CPW1 = 16, CPW2 = 32, Ng1 = TN, Ng2 = TN / F2;
  constexpr int DL1 = (TK * 2 / 8) / 32, DL2 = (F2 * TK / 8) / 32;
  constexpr int NI1 = Ng1 / RPI, NI2 = Ng2 / RPI;
  constexpr int P2_DIV = (DL1 && DL2) ? (DL1 / DL2 ? DL1 / DL2 : 1) : 1;

  const auto m1 = plane_map<2, TM, TN, TK, WM, WN, 1>();
  std::vector<int> m((size_t)Ng2 * DL2 * 8 * CPW2, -1);
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
    for (int ii = 0; ii < NI1; ++ii)
      for (int kb = 0; kb < P2_DIV; ++kb)
        for (int v = 0; v < 4; ++v)
          for (int lt = 0; lt < 8; ++lt)
            for (int half = 0; half < 2; ++half) {
              const int j1 = (lt % 4) + 4 * (lt / 4) + 8 * half;
              const int row1 = ii * RPI + 16 * warp_n + (v / 2) * 8 + lane / 4, wd1 = (v % 2) * 4 + lane % 4;
              if (row1 >= Ng1) continue;
              const int e1 = m1[(((size_t)row1 * DL1 + kb % DL1) * 8 + wd1) * CPW1 + j1];
              if (e1 < 0) continue;
              const int base = (kb % P2_DIV) + P2_DIV * (ii / (NI2 ? NI2 : 1));
              const int v2 = base + 2 * (v >> 1), j2 = 8 * (v & 1) + lt + 16 * half;
              if (v2 >= 4) continue;
              const int inst2 = (NI2 > 1) ? (ii % NI2) : 0;
              const int row2 = inst2 * RPI + 16 * warp_n + (v2 / 2) * 8 + lane / 4, wd2 = (v2 % 2) * 4 + lane % 4;
              if (row2 >= Ng2) continue;
              m[(((size_t)row2 * DL2 + (kb / P2_DIV) % DL2) * 8 + wd2) * CPW2 + j2] = e1;
            }
  }
  return m;
}

// Write the folded high-plane buffer. `high_kn` is the raw [K][N] high plane, one code per byte, as the caller reads
// it from the checkpoint -- NOT a preprocessed buffer. Destination addressing is l20's F>1 (plane-major) branch.
template <int TM, int TN, int TK, int WM, int WN, int F2>
inline void place_int1(int8_t* out, const std::vector<uint8_t>& high_kn, int N, int K) {
  constexpr int CPW = 32, R2 = TN / F2, DL = (F2 * TK / 8) / 32;
  static_assert(DL == 1, "the buffer walk below assumes one delivery per folded row");
  const int W_ROW_OFF = 256 / CPW, RUNS = W_ROW_OFF / 8, nrow = N / F2;
  const auto m = tile_map_int1<TM, TN, TK, WM, WN, F2>();
  std::fill(out, out + (size_t)N * K / 8, int8_t(0));
  for (int tn = 0; tn < N / TN; ++tn)
    for (int ki = 0; ki < K / TK; ++ki)
      for (int row = 0; row < R2; ++row)
        for (int wd = 0; wd < 8; ++wd)
          for (int j = 0; j < CPW; ++j) {
            const int loc = m[((size_t)row * 8 + wd) * CPW + j];
            if (loc < 0) continue;
            const int n = tn * TN + loc / TK, k = ki * TK + loc % TK;
            if (!(high_kn[(size_t)k * N + n] & 1)) continue;       // high_kn is [K][N]
            const int kb = ki / RUNS, tt = ki % RUNS;
            const size_t bit = (size_t)((((size_t)kb * nrow + (size_t)tn * R2 + row) * W_ROW_OFF + tt * 8 + wd) * CPW + j);
            out[bit / 8] |= int8_t(1 << (bit % 8));
          }
}

} // namespace xplane
