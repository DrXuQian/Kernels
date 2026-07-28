#pragma once
// CROSS-PLANE offline placement for the FOLDED high plane of a 2-plane (B-concat) GEMM.
//
// WHY THIS EXISTS. Once the two planes fold by different factors they disagree on thread->column by construction: the
// fold divides plane 2's physical row count by F2 and the mma maps threads to PHYSICAL rows. Measured at
// (64,64,128) w32x32 (fold_derivation/l44): thread 0 holds low codes for logical N {0,8,32,40} and high bits for
// {0,1,16,17} -- the bits it needs sit in another thread's registers. No in-kernel offset repairs that, and the box
// showed it as bad=15010/32768 against a bad=0 unfolded control.
//
// The fold is only a LOAD-LAYER device and the offline is derived rather than hand-written, so the fix is to place
// plane 2's bits where the CONVERTER will look for them. The target map is consistent -- conflicts=0, unclaimed=0,
// multiplicity == TM/WM (B is split across the N warps only, so every M-warp legitimately reads the same B) -- at
// every shape probed, so nothing is ruled out (l45/l46).
//
// THE PAIRING, from MixGemm2Plane_uint2_uint1's own _E2 lines (verified in l37):
//     line (t, v):  LOW  two crumbs of lo[v] at code (t%4) + 4*(t/4) [+8] within the vreg
//                   HIGH two bits of hi[2*(v>>1)] at 8*(v&1) + t [+16]
// Inverted, that says which converter line reads plane-2 vreg v2 bit j2, hence which low code it pairs with, hence
// which logical (n,k) belongs at that physical bit.
//
// FRAME. l20's tile_map assumes 8*CPW == TK, i.e. exactly one delivery per physical row per k-tile. A folded plane 2
// satisfies it by construction (F2*TK bits = 256 codes = 8 words x 32). The shipping TK=256 config does NOT satisfy it
// for plane 1 (int2, two deliveries per row), which is why it cannot serve as this file's control; the gate used
// instead is T2, built independently from partition_B, which the tile map reproduces exactly (0/8192 and 0/16384).
#include <vector>
#include <cstdint>
#include <cstddef>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"

namespace xplane {

// tile map: m[(row*8 + wd)*32 + j] = n_local*TK + k_local, over plane 2's Ng = TN/F2 physical rows
template <int TM, int TN, int TK, int WM, int WN, int F2>
inline std::vector<int> tile_map_int1() {
  using namespace cute;
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  constexpr int WOM = TM / warpM, WON = TN / warpN;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                         Tile<Int<WOM*16>, Int<WON*16>, _32>>;
  constexpr int R1 = TN, B1 = TK * 2 / 8, R2 = TN / F2, B2 = F2 * TK / 8;
  constexpr int CPW = 32, RPI = WON * 16;
  static_assert(8 * CPW == F2 * TK, "folded plane 2 must be exactly one delivery per row");

  auto id1 = make_identity_tensor(make_shape(Int<R1>{}, Int<B1>{}));
  auto p1t = MmaS8{}.get_thread_slice(0).partition_B(id1);
  auto p2t = MmaS8{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<R2>{}, Int<B2>{})));
  const int N1 = int(size<1>(p1t.layout())), KB1 = int(size<2>(p1t.layout()));
  const int N2 = int(size<1>(p2t.layout())), KB2 = int(size<2>(p2t.layout()));
  const int P2_DIV = KB2 ? KB1 / KB2 : 1;

  std::vector<int> m((size_t)R2 * 8 * CPW, -1);
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
    auto p1 = MmaS8{}.get_thread_slice(t).partition_B(id1);
    for (int inst2 = 0; inst2 < R2 / RPI; ++inst2)
      for (int v2 = 0; v2 < 4; ++v2) {
        const int row = inst2 * RPI + 16 * warp_n + (v2 / 2) * 8 + lane / 4;   // plane 2's swzl TV, l20's verbatim
        const int wd  = (v2 % 2) * 4 + lane % 4;
        for (int j2 = 0; j2 < CPW; ++j2) {
          const int half = j2 / 16, r = j2 % 16, v1 = 2 * (v2 >> 1) + r / 8, lt = r % 8;
          const int odd = v2 & 1;                    // carries kb-parity when P2_DIV==2, else the MMA_N index
          const int ii  = (P2_DIV == 1) ? odd : inst2;
          const int kb  = (P2_DIV == 1) ? 0   : odd;
          if (ii >= N1 || kb >= KB1) continue;
          const int c1 = 16 * v1 + (lt % 4) + 4 * (lt / 4) + 8 * half;         // plane-1 code in the 64-code chunk
          auto cc = p1(c1 / 4, ii, kb);
          m[((size_t)row * 8 + wd) * CPW + j2] = int(get<0>(cc)) * TK + int(get<1>(cc)) * 4 + (c1 % 4);
        }
      }
  }
  return m;
}

// Write the folded high-plane buffer. `high_kn` is the raw [K][N] high plane, one code per byte, as the caller reads
// it from the checkpoint -- NOT a preprocessed buffer. Destination addressing is l20's F>1 (plane-major) branch.
template <int TM, int TN, int TK, int WM, int WN, int F2>
inline void place_int1(int8_t* out, const std::vector<uint8_t>& high_kn, int N, int K) {
  constexpr int CPW = 32, R2 = TN / F2;
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
