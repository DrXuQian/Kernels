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
#include <algorithm>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"

namespace xplane {

// (a) THE swzl DELIVERY POSITION, FROM THE REAL OBJECTS. This was the one link in the whole chain that was never a cute
// layout: l20 and plane_map both hand-wrote
//     row = inst*RPI + 16*warp_n + (v/2)*8 + lane/4      wd = (v%2)*4 + lane%4      RPI = WON*16
// and every offline buffer in this work flows through it, which makes it the most load-bearing expression here. It is
// also the reason five rounds of debugging went to the wrong subsystem: it was the only plausible suspect, so it got
// investigated repeatedly, and it was correct every time (fold_derivation/l59 finally proved it against the real
// partition_S). Correct or not, an expression that has to be RE-VERIFIED to be trusted is worse than one that cannot be
// wrong, so both halves now come from the objects that define them:
//   cube base       make_tiled_copy_B(SmemCopyAtomB, tiled_mma_s8).get_thread_slice(warp*32).partition_S(identity)
//   within a cube   Copy_Traits<PPU0010_TSM_LD_SWZL>::LogicalTV, the atom's own descriptive map
// Byte-identity with the previous formula is required by l52 / l58 / l61 / l64, all of which gate against shipped or
// box-validated buffers.
template <int Bits, int TM, int TN, int TK, int WM, int WN, int F>
struct CubeTV {
  static constexpr int warpM = (WM > 16) ? WM : 16, warpN = (WN > 16) ? WN : 16;
  static constexpr int WOM = TM / warpM, WON = TN / warpN;
  static constexpr int BK = F * TK, Ng = TN / F, RowB = BK * Bits / 8;
  static constexpr int AiuByte = RowB > 128 ? 128 : RowB;               // MixGemm_AIU_Operand's own arithmetic
  static constexpr int AiuElem = AiuByte * 8 / Bits;
  static constexpr int InstNum = BK / AiuElem;
  using SInst = cute::PPU0015_16x16x32_S32S8S8S32_TN;
  using MmaS8 = cute::TiledMMA<cute::MMA_Atom<SInst>,
                               cute::Layout<cute::Shape<cute::Int<WOM>, cute::Int<WON>, cute::_1>>,
                               cute::Tile<cute::Int<WOM*16>, cute::Int<WON*16>, cute::_32>>;
  using Op    = cute::PPU0010_TSM_LD_SWZL<int8_t, Ng, AiuElem * Bits / 8, true, false, InstNum>;
  using Tr    = cute::Copy_Traits<Op>;
  static constexpr int WPR = Tr::LogicalWordsPerRow;                    // 32-bit words per cube row

  static int base_row(int warp, int inst) {
    auto id = cute::make_identity_tensor(cute::make_shape(cute::Int<Ng>{}, cute::Int<RowB>{}));
    auto ts = cute::make_tiled_copy_B(cute::Copy_Atom<Op, int8_t>{}, MmaS8{})
                  .get_thread_slice(warp * 32).partition_S(id);
    return int(cute::get<0>(ts(0, inst, 0)));
  }
  static int word(int lane, int v, int dl) {
    return int(typename Tr::LogicalTV{}(cute::make_coord(cute::make_coord(lane % 4, lane / 4),
                                                         cute::make_coord(v % 2, v / 2), dl)));
  }
  static int cube_row(int lane, int v)         { return word(lane, v, 0) / WPR; }
  static int cube_wd (int lane, int v, int dl) { return word(lane, v, dl) % WPR - dl * 8; }
  static constexpr int insts() { return Ng / (WON * 16); }
};

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
  // (e) MmaPermK comes from the SHARED rule the builder uses, not a local restatement of it. The restatement is what
  // silently broke a folded plane: at Block_K=64 with F=2 the composition covered 4160 of 8192 elements.
  constexpr int MmaPermK = cutlass::MixGemmMmaPermK<Bits, TK, F>::value;
  using Mma = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<MmaPermK>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>, Int<TK>>{}, Stride<Int<TK>, _1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));
  auto pi = right_inverse(frag.layout());
  std::vector<int> m((size_t)Ng * DL * 8 * CPW, -1);
  using CTV = CubeTV<Bits, TM, TN, TK, WM, WN, F>;
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32;
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int dl = 0; dl < DL; ++dl)
      for (int inst = 0; inst < Ng / RPI; ++inst)
        for (int v = 0; v < 4; ++v) {
          const int row = CTV::base_row(w, inst) + CTV::cube_row(lane, v), wd = CTV::cube_wd(lane, v, dl);
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
// F1 is plane 1's OWN fold factor. It was hardcoded to 1, which is fine at Block_K 128 and 256 (int2's run is already
// >= 32 B there) but wrong at Block_K=64, where int2 folds by 2 as well -- DL1 = (TK*2/8)/32 evaluated to 16/32 = 0 and
// took the whole map with it. With F1 threaded through, plane 1's physical row is (TN/F1, F1*TK*2/8) and DL1 is 1 again.
// THE HIGH PLANE'S MAP, composed from the low plane's, for ANY (low, high) width pair -- Q3 = int2+int1,
// Q6 = int4+int2, Q5 = int4+int1. Everything that used to be int1-specific is now driven by the two widths:
//     kPairs  = 16/LowBits    half2 pairs per low vreg      (int2 8, int4 4)
//     hstride = 16/HiBits     high codes between the two half2 lanes  (int1 16, int2 8)
//     P2_DIV                  taken from the real fragments, and equal to LowBits/HiBits when both planes are unfolded
// and the converter's pairing is the same closed form the converter itself uses, gated against Q3's shipped constants in
// fold_derivation/l65. Q3's map is required to come out byte-identical (l67).
template <int LowBits, int HiBits, int TM, int TN, int TK, int WM, int WN, int F2, int F1 = 1>
inline std::vector<int> tile_map_hi() {
  using namespace cute;
  constexpr int InstM = 16, warpM = (WM > InstM) ? WM : InstM, warpN = (WN > 16) ? WN : 16;
  constexpr int WOM = TM / warpM, WON = TN / warpN, RPI = WON * 16;
  constexpr int CPW1 = 32 / LowBits, CPW2 = 32 / HiBits, Ng1 = TN / F1, Ng2 = TN / F2;
  constexpr int DL1 = (F1 * TK * LowBits / 8) / 32, DL2 = (F2 * TK * HiBits / 8) / 32;
  constexpr int NI1 = Ng1 / RPI, NI2 = Ng2 / RPI;
  static_assert(DL1 >= 1 && DL2 >= 1, "each plane's physical row must be a whole number of 32 B deliveries");
  // TWO DIFFERENT RATIOS, and conflating them is what broke the first version of this generalisation. They coincide at
  // Block_K=256 (both 2 for Q3), which is exactly the configuration that passed while every other Q3 row changed.
  //   PDcopy = DL1/DL2        COPY STEP ratio: how many low k_blocks share one high copy step. Drives `kb` and `base`.
  //   VR     = LowBits/HiBits VREG ratio INSIDE one delivery: both planes deliver 4 vregs, but the high plane's hold
  //                           32/HiBits codes against the low plane's 32/LowBits, so one low delivery consumes only
  //                           4*HiBits/LowBits of the high vregs and they sit VR apart. Drives v2 and j2.
  // The old int1-only body used DL1/DL2 for the first and a hardcoded 2 for the second -- and 2 is VR for Q3, which is
  // why it was right.
  constexpr int PDcopy = (DL1 / DL2) ? (DL1 / DL2) : 1;
  constexpr int VR     = LowBits / HiBits;
  constexpr int kPairs = 16 / LowBits, hstride = 16 / HiBits;
  // (g) THE PAIRING COMES FROM THE CONVERTER, not from a second statement of it here. These used to be local Layouts
  // saying the same thing the converter said as arithmetic -- one rule, two forms, which is the class that produced both
  // 2-plane defects this session. MixGemm2Plane now owns LoCodeL / HiCodeL / HVregL and both sides call them.
  // (What the original int1-only code hid, worth keeping: its `(lt % 4) + 4 * (lt / 4)` is the IDENTITY on lt in [0,8).)
  using Cvt = cutlass::MixGemm2Plane<LowBits, HiBits>;
  static_assert(Cvt::kPairs == kPairs && Cvt::kVregRatio == VR && Cvt::kHiStride == hstride,
                "the offline and the converter must agree on the pairing shape");

  using CTV1 = CubeTV<LowBits, TM, TN, TK, WM, WN, F1>;
  using CTV2 = CubeTV<HiBits,  TM, TN, TK, WM, WN, F2>;
  const auto m1 = plane_map<LowBits, TM, TN, TK, WM, WN, F1>();
  std::vector<int> m((size_t)Ng2 * DL2 * 8 * CPW2, -1);
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32;
    for (int ii = 0; ii < NI1; ++ii)
      for (int kb = 0; kb < PDcopy; ++kb)
        for (int v = 0; v < 4; ++v)
          for (int lt = 0; lt < kPairs; ++lt)
            for (int half = 0; half < 2; ++half) {
              const int j1 = Cvt::lo_code(lt, half);
              const int row1 = CTV1::base_row(w, ii) + CTV1::cube_row(lane, v), wd1 = CTV1::cube_wd(lane, v, kb % DL1);
              if (row1 >= Ng1) continue;
              const int e1 = m1[(((size_t)row1 * DL1 + kb % DL1) * 8 + wd1) * CPW1 + j1];
              if (e1 < 0) continue;
              const int base = (kb % PDcopy) + PDcopy * (ii / (NI2 ? NI2 : 1));
              // the high vreg and the code inside it, from the converter's own layouts
              const int v2 = base + Cvt::hi_vreg(v), j2 = Cvt::hi_code(lt, v, half);
              if (v2 >= 4) continue;
              const int inst2 = (NI2 > 1) ? (ii % NI2) : 0;
              const int row2 = CTV2::base_row(w, inst2) + CTV2::cube_row(lane, v2),
                        wd2 = CTV2::cube_wd(lane, v2, (kb / PDcopy) % DL2);
              if (row2 >= Ng2) continue;
              m[(((size_t)row2 * DL2 + (kb / PDcopy) % DL2) * 8 + wd2) * CPW2 + j2] = e1;
            }
  }
  return m;
}

// Q3's name, so nothing downstream changes.
template <int TM, int TN, int TK, int WM, int WN, int F2, int F1 = 1>
inline std::vector<int> tile_map_int1() { return tile_map_hi<2, 1, TM, TN, TK, WM, WN, F2, F1>(); }

// BIT-GRANULAR writer, shared by both planes. `q_kn` is the raw [K][N] plane, one code per byte, as the caller reads
// it from the checkpoint -- NOT a preprocessed buffer. Destination addressing is l20's F>1 (plane-major) branch.
//
// WHY A BIT-GRANULAR WRITER IS REQUIRED, and not merely tidier. nfold_regroup_gmem moves whole uint32 words, so every
// word it emits holds ONE logical column -- which is what the mma wants only while cols_per_word == 1. That holds for
// every 32x32 warp tile, and it is why the whole-word packer passed the box at int2's shipping (64,64,64) w32x32 F=2.
// At WN=64 the fragment asks for TWO logical columns inside each word and a whole-word move CANNOT express it; on top
// of that nfold_regroup_gmem groups the folded columns STRIDED (n = g + f*Ng, its line 676) while the kernel's
// SmemLayoutB_MmaView groups them ADJACENT (n = f + P1Fold*g), so the two disagree about which columns even share a
// physical row. Measured: 32768 of 65536 slots misplaced at (64,128,64) w64x64 F=2, sigma = "n -> n+32 for half the
// columns" -- bit for bit the permutation the hardware probe printed (fold_derivation/l61).
//
// l52 called that same configuration BIT-IDENTICAL. Its probe value was (i * 2654435761u >> 5) & 3 and the
// displacement is i += 32*K = 2^14; (M << 14) has 14 low zero bits, so it can neither change bits 5-6 nor carry into
// them -- the misplaced codes are EQUAL and the buffers compare identical. A probe whose period aliases the
// displacement proves nothing. l61 labels each element by the bits of its own index instead, which cannot alias.
template <int Bits, int TM, int TN, int TK, int WM, int WN, int F>
inline void place_from_map(int8_t* out, const std::vector<int>& m, const std::vector<uint8_t>& q_kn, int N, int K) {
  using namespace cute;
  constexpr int CPW = 32 / Bits, R = TN / F, DL = (F * TK * Bits / 8) / 32, MASK = (1 << Bits) - 1;
  static_assert(DL >= 1, "a physical row must be a whole number of 32 B deliveries");
  // (b) THE DESTINATION IS A LAYOUT. l20's two branches used to be two hand-written address expressions inside the
  // innermost loop; they are now one cute layout each, built once, over the coordinate tuple the walk enumerates. The
  // F > 1 branch is plane-major -- super-tile kb becomes a separate plane of N/F rows -- and F == 1 is the
  // interleave-256 one, which is what the deleted five-step pipeline produced.
  const int W_ROW_OFF = 256 / CPW, RUNS = W_ROW_OFF / 8, nrow = N / F;                       // F > 1
  constexpr int kCon = 256, contig = F * TK * Bits / 8, AiuByte = contig > 128 ? 128 : contig;
  constexpr int AiuElem = AiuByte * 8 / Bits, RPS = kCon / AiuElem;                          // F == 1
  static_assert(F > 1 || RPS >= 1, "unfolded: a K-tile's 32B run exceeds the 256-element interleave -- no such config ships");
  const int NT = N / TN, KT = K / TK;
  // (j, wd, row, tn, tt, kb) -> code index. N and K are runtime, so the strides are dynamic; the shape is not a
  // Layout-of-constants but it is still ONE object rather than an expression per call.
  auto dst_fold = make_layout(
      make_shape (Int<CPW>{}, _8{}, Int<R>{}, NT, RUNS ? RUNS : 1, KT / (RUNS ? RUNS : 1)),
      make_stride(_1{}, Int<CPW>{}, W_ROW_OFF * CPW, R * W_ROW_OFF * CPW, 8 * CPW, nrow * W_ROW_OFF * CPW));
  // (j, wd, dl, ki_lo, row, tn, ki_hi) -> code index, with ki = ki_hi*RPS + ki_lo
  auto dst_flat = make_layout(
      make_shape (Int<CPW>{}, _8{}, Int<DL>{}, RPS ? RPS : 1, Int<R>{}, NT, KT / (RPS ? RPS : 1)),
      make_stride(_1{}, Int<CPW>{}, 8 * CPW, AiuElem, kCon, TN * kCon, N * kCon));
  std::fill(out, out + (size_t)N * K * Bits / 8, int8_t(0));
  for (int tn = 0; tn < NT; ++tn)
    for (int ki = 0; ki < KT; ++ki)
      for (int row = 0; row < R; ++row)
        for (int dl = 0; dl < DL; ++dl)
          for (int wd = 0; wd < 8; ++wd)
            for (int j = 0; j < CPW; ++j) {
              const int loc = m[(((size_t)row * DL + dl) * 8 + wd) * CPW + j];
              if (loc < 0) continue;
              const int n = tn * TN + loc / TK, k = ki * TK + loc % TK;
              const int v = q_kn[(size_t)k * N + n] & MASK;          // q_kn is [K][N]
              if (!v) continue;
              const size_t code = (F > 1)
                  ? size_t(dst_fold(j, wd, row, tn, ki % (RUNS ? RUNS : 1), ki / (RUNS ? RUNS : 1)))
                  : size_t(dst_flat(j, wd, dl, ki % (RPS ? RPS : 1), row, tn, ki / (RPS ? RPS : 1)));
              const size_t bit0 = code * Bits;
              for (int b = 0; b < Bits; ++b)
                if ((v >> b) & 1) out[(bit0 + b) / 8] |= int8_t(1 << ((bit0 + b) % 8));
            }
}

// One plane on its own, from its own derived map. This is what plane 1 needs once cols_per_word > 1.
template <int Bits, int TM, int TN, int TK, int WM, int WN, int F>
inline void place_derived(int8_t* out, const std::vector<uint8_t>& q_kn, int N, int K) {
  place_from_map<Bits, TM, TN, TK, WM, WN, F>(out, plane_map<Bits, TM, TN, TK, WM, WN, F>(), q_kn, N, K);
}

// The high plane, from the CROSS-PLANE map, for any width pair.
template <int LowBits, int HiBits, int TM, int TN, int TK, int WM, int WN, int F2, int F1 = 1>
inline void place_hi(int8_t* out, const std::vector<uint8_t>& high_kn, int N, int K) {
  place_from_map<HiBits, TM, TN, TK, WM, WN, F2>(
      out, tile_map_hi<LowBits, HiBits, TM, TN, TK, WM, WN, F2, F1>(), high_kn, N, K);
}

// Q3's name.
template <int TM, int TN, int TK, int WM, int WN, int F2, int F1 = 1>
inline void place_int1(int8_t* out, const std::vector<uint8_t>& high_kn, int N, int K) {
  place_hi<2, 1, TM, TN, TK, WM, WN, F2, F1>(out, high_kn, N, K);
}

} // namespace xplane
