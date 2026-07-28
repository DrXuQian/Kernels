// L46 -- GENERATE plane 2's cross-plane offline placement, using the converter's TRUE pairing.
//
// l45 showed a count-level bijection exists, but paired the thread's i-th low code with its i-th high bit in slot
// order -- necessary, not sufficient. The real pairing is what MixGemm2Plane_uint2_uint1 emits, and it is explicit in
// the _E2 lines (l37 verified it):
//
//   line (t, v), t in [0,8), v in [0,4)
//     LOW : two crumbs of lo[v] -- code index within the 64-code chunk = 16*v + (t%4) + 4*(t/4), and +8
//     HIGH: two bits of hi[2*(v>>1)] at 8*(v&1)+t and +16
//           == bit 64*(v>>1) + 8*(v&1) + t of the 128-bit (4-vreg) high chunk, and +16
//
// So ONE _E2 line consumes 2 low codes and 2 high bits, and the pairing between them is (low half <-> bit, high half
// <-> bit+16). That is the granularity a placement must be derived at.
//
// KERNEL-SIDE INDEXING ASSUMED BELOW (and it is a change, because the shipped one is out of range once plane 2 folds):
//     hi chunk   = cvt_hi(_, ii % MMA_N2)
//     uint32 off = (k_block % P2_DIV) + P2_DIV * (ii / MMA_N2)
// P2_DIV==1 and MMA_N2==MMA_N1 reproduces the shipped expression exactly, so the unfolded path is untouched.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l46_xplane_gen.cu -o l46 && ./l46
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <vector>
#include <map>
#include "unfused_weight_dequantize.hpp"
using namespace cute;

template <int TM, int TN, int TK, int WM, int WN, int F2>
static bool gen(const char* tag, std::vector<int>* out_T2 = nullptr) {
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  constexpr int WOM = TM / warpM, WON = TN / warpN, NTHR = 32 * WOM * WON;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                         Tile<Int<WOM*16>, Int<WON*16>, _32>>;
  constexpr int R1 = TN,      B1 = TK * 2 / 8;          // plane 1 int2, F1 = 1
  constexpr int R2 = TN / F2, B2 = F2 * TK * 1 / 8;     // plane 2 int1, folded by F2

  auto p1t = MmaS8{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<R1>{}, Int<B1>{})));
  auto p2t = MmaS8{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<R2>{}, Int<B2>{})));
  auto pi2 = right_inverse(make_layout(shape<0>(p2t.layout())));
  const int V1 = int(size<0>(p1t.layout())), N1 = int(size<1>(p1t.layout())), KB1 = int(size<2>(p1t.layout()));
  const int V2 = int(size<0>(p2t.layout())), N2 = int(size<1>(p2t.layout())), KB2 = int(size<2>(p2t.layout()));
  const int P2_DIV = KB2 ? KB1 / KB2 : 0;
  if (!KB2 || KB1 % KB2) { printf("  %-26s P2_DIV not integral (%d/%d)\n", tag, KB1, KB2); return false; }

  // T2[plane-2 physical bit] = logical element (n*TK + k) whose HIGH bit belongs there
  std::vector<int> T2((size_t)R2 * B2 * 8, -1);
  std::map<int,int> mult; long conflicts = 0;

  for (int t = 0; t < NTHR; ++t) {
    auto p1 = MmaS8{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<R1>{}, Int<B1>{})));
    auto p2 = MmaS8{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<R2>{}, Int<B2>{})));
    for (int kb = 0; kb < KB1; ++kb)
      for (int ii = 0; ii < N1; ++ii) {
        const int hi_chunk = ii % N2;                      // cvt_hi(_, ii % MMA_N2)
        const int hi_vreg0 = (kb % P2_DIV) + P2_DIV * (ii / N2);   // uint32 offset into the 4-vreg chunk
        const int kb2      = kb / P2_DIV;
        for (int v = 0; v < 4; ++v)
          for (int lt = 0; lt < 8; ++lt)
            for (int half = 0; half < 2; ++half) {
              // ---- low side: which logical (n,k)
              const int code = 16 * v + (lt % 4) + 4 * (lt / 4) + 8 * half;   // within the 64-code chunk
              auto c1 = p1(code / 4, ii, kb);                                  // int8 slot -> (row, byte)
              const int n = int(get<0>(c1));                                   // plane 1 unfolded: row == logical n
              const int k = int(get<1>(c1)) * 4 + (code % 4);
              // ---- high side: which plane-2 physical bit
              const int w   = hi_vreg0 + 2 * (v >> 1);                          // vreg within the chunk
              const int bit = 8 * (v & 1) + lt + 16 * half;                     // bit within that vreg
              if (w >= V2 / 4) { printf("  %-26s high vreg %d out of range (chunk has %d)\n", tag, w, V2/4); return false; }
              // pi = right_inverse(frag.layout()), the same composition l20's tile_map uses. The delivered order is
              // NOT partition_B's slot order; skipping pi made the control buffer differ in every one of its 16384
              // bytes, which is exactly what that diff exists to catch.
              auto c2 = p2(pi2(w * 4 + bit / 8), hi_chunk, kb2);                // int8 slot -> (g, byte)
              const long bp = ((long)int(get<0>(c2)) * B2 + int(get<1>(c2))) * 8 + (bit % 8);
              const int e = n * TK + k;
              if (T2[bp] >= 0 && T2[bp] != e) ++conflicts;
              T2[bp] = e; ++mult[e];
            }
      }
  }
  long unclaimed = 0; for (int e : T2) if (e < 0) ++unclaimed;
  long badmult = 0; for (auto& kv : mult) if (kv.second != WOM) ++badmult;
  const bool ok = (conflicts == 0 && unclaimed == 0 && badmult == 0);
  printf("  %-26s F2=%d MMA_N %d/%d P2_DIV=%d | bits=%zu conflicts=%ld unclaimed=%ld mult!=%d:%ld -> %s\n",
         tag, F2, N1, N2, P2_DIV, T2.size(), conflicts, unclaimed, WOM, badmult,
         ok ? "PLACEMENT DERIVED" : "INCONSISTENT under the true pairing");
  if (out_T2) *out_T2 = T2;
  return ok;
}

// ---------------------------------------------------------------- the tile map, in l20's frame
// l20's tile_map assumes 8*CPW == TK -- exactly ONE delivery per physical row per k-tile. int1 -> TK=256,
// int2 -> 128, int4 -> 64. The FOLDED plane 2 satisfies it (B2*8 = 256 codes/row, CPW=32, 8 words), which is why the
// frame transfers; the shipping TK=256 config does NOT for plane 1 (int2 at 2 deliveries/row), so it cannot serve as
// the control here. The available gate is T2 itself, built independently from partition_B in gen() above: the tile
// map must reproduce it bit for bit.
//
// destination : plane 2's own swzl TV formula, l20's verbatim
// source      : solve which converter line reads this bit, then take ITS low code's logical (n,k) from plane 1
template <int TM, int TN, int TK, int WM, int WN, int F2>
static bool tile_map_vs_T2(const char* tag) {
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  constexpr int WOM = TM / warpM, WON = TN / warpN;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                         Tile<Int<WOM*16>, Int<WON*16>, _32>>;
  constexpr int R1 = TN, B1 = TK * 2 / 8, R2 = TN / F2, B2 = F2 * TK / 8;
  constexpr int CPW2 = 32, RPI = WON * 16;
  static_assert(8 * CPW2 == F2 * TK, "plane 2 must be one delivery per row for l20's frame");

  std::vector<int> T2;
  if (!gen<TM,TN,TK,WM,WN,F2>("(map)", &T2)) return false;

  auto p1t = MmaS8{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<R1>{}, Int<B1>{})));
  auto p2t = MmaS8{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<R2>{}, Int<B2>{})));
  const int N1 = int(size<1>(p1t.layout())), KB1 = int(size<2>(p1t.layout()));
  const int N2 = int(size<1>(p2t.layout())), KB2 = int(size<2>(p2t.layout()));
  const int P2_DIV = KB1 / KB2;

  std::vector<int> m2((size_t)R2 * 8 * CPW2, -1);
  long bad = 0, filled = 0;
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
    auto p1 = MmaS8{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<R1>{}, Int<B1>{})));
    for (int inst2 = 0; inst2 < R2 / RPI; ++inst2)
      for (int v2 = 0; v2 < 4; ++v2) {
        const int row = inst2 * RPI + 16 * warp_n + (v2 / 2) * 8 + lane / 4;
        const int wd  = (v2 % 2) * 4 + lane % 4;
        for (int j2 = 0; j2 < CPW2; ++j2) {
          // which converter line reads plane-2 vreg v2, bit j2?
          //   v2 = hi_vreg0 + 2*(v1>>1),  hi_vreg0 = (kb % P2_DIV) + P2_DIV*(ii / N2)
          //   j2 = 8*(v1&1) + lt + 16*half
          const int half = j2 / 16, r = j2 % 16, v1 = 2 * (v2 >> 1) + r / 8, lt = r % 8;
          const int odd = v2 & 1;                       // carries kb-parity (P2_DIV=2) or ii (P2_DIV=1)
          const int ii  = (P2_DIV == 1) ? odd : inst2;
          const int kb  = (P2_DIV == 1) ? 0   : odd;
          if (ii >= N1 || kb >= KB1) continue;
          const int j1 = (lt % 4) + 4 * (lt / 4) + 8 * half;      // plane-1 code within vreg v1
          auto c1 = p1((16 * v1 + j1) / 4, ii, kb);
          const int elem = int(get<0>(c1)) * TK + int(get<1>(c1)) * 4 + ((16 * v1 + j1) % 4);
          const size_t at = ((size_t)row * 8 + wd) * CPW2 + j2;
          if (m2[at] >= 0 && m2[at] != elem) ++bad; else if (m2[at] < 0) ++filled;
          m2[at] = elem;
        }
      }
  }
  long differ = 0, unset = 0;
  for (size_t i2 = 0; i2 < m2.size(); ++i2) { if (m2[i2] < 0) ++unset; else if (m2[i2] != T2[i2]) ++differ; }
  printf("  %-28s tile-map vs T2: %ld / %zu differ, %ld unset, %ld self-conflicts  %s\n",
         tag, differ, m2.size(), unset, bad,
         (!differ && !unset && !bad) ? "<-- AGREE" : "<-- construction is wrong");
  return !differ && !unset && !bad;
}

int main() {
  printf("L46 -- cross-plane placement generation under the converter's TRUE pairing\n\n");
  printf("  control: the SHIPPING unfolded shape, where the derived map must reproduce today's working placement\n");
  bool c = gen<64, 64,256,32,32, 1>("Q3 (64,64,256) F2=1");
  printf("\n  the folded shapes\n");
  bool a = gen<64, 64,128,32,32, 2>("Q3 (64,64,128) F2=2");
  bool b = gen<32,128,128,32,32, 2>("Q3 (32,128,128) F2=2");
  printf("\n  the tile map in l20's frame, gated on T2 (built independently, from partition_B)\n");
  bool d = tile_map_vs_T2<64, 64,128,32,32, 2>("Q3 (64,64,128) F2=2");
  d = tile_map_vs_T2<32,128,128,32,32, 2>("Q3 (32,128,128) F2=2") && d;
  printf("\n  verdict: %s\n",
         (c && a && b && d) ? "tile map agrees with T2 -- ready to emit the buffer"
                       : "at least one shape is inconsistent; the kernel-side indexing assumed at the top is the suspect");
  return 0;
}
