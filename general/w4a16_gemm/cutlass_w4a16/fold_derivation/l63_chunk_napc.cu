// L63 -- the gate for task #12 at Block_K=64, BEFORE touching the collective.
//
// What l62 established: one int2 delivery is 8 mma atoms, and tCrB_mma splits them 1 n x 8 k at TK>=128 but 2 n x 4 k at
// TK=64. The converter already chunks by ATOM (kPerAtom = 4 half2 = 8 fp16 = exactly one mma B atom, kAtoms = 32/4 = 8),
// so no new template parameter is needed. The defect is in the CALL SITE: it loops Chunk over K_ATOM_PER_COPY only,
// which is 4 at TK=64, so 4 of the delivery's 8 atoms are never emitted -- and it writes at `out + 4*ii`, which assumes
// one n-atom per delivery.
//
// The derivation to be gated. tCrB_mma is ((2,2,2),MMA_N,MMA_K):((1,2,4),8*MMA_K,8), and the unchunked path writes
// delivery ii at flat offset 64*ii. So within a delivery, half2 slot s sits at fp16 offset 2s and therefore at
//     n_local = 2s / (8*MMA_K)        k_atom = (2s / 8) % MMA_K
// which, if the low bits of s/4 carry k and the high bits carry n, means
//     Chunk = k_atom + K_ATOM_PER_COPY * n_local          destination n-atom = ii*NAPC + n_local
// Both reduce to the shipped code when NAPC == 1 (Chunk == k_atom, out + 4*ii), so this is a strict generalisation --
// which is exactly why it must be checked at NAPC == 2 rather than inferred from the NAPC == 1 case that works.
//
// Checked here: (a) the Chunk formula against tCrB_mma's own layout, (b) that the chunks PARTITION the 32 slots -- every
// slot kept by exactly one (k_atom, n_local), none missed, (c) that at() rebases into [0,4) within its atom, and
// (d) that the resulting destination equals the unchunked destination for every (t, v). (d) is the real gate: it is the
// unchunked path that the box has already verified numerically.
//
//   nvcc -std=c++17 -Istub_inc -I.. -I../../../../third_party/actlize/include l63_chunk_napc.cu -o l63 && ./l63
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using namespace cute;

template <int Bits, int BlockK> struct AiuOp {
  static constexpr int BlockCont = BlockK * Bits / 8;
  static constexpr int AiuByte   = BlockCont > 128 ? 128 : BlockCont;
  static constexpr int AiuElem   = AiuByte * 8 / Bits;
  static constexpr int InstNum   = BlockK / AiuElem;
};

template <int TM, int TN, int TK, int WM, int WN, int F1>
static bool gate(const char* tag) {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int warpM = (WM > 16) ? WM : 16, warpN = (WN > 16) ? WN : 16;
  constexpr int WOM = TM / warpM, WON = TN / warpN;
  constexpr int MmaPermK = (F1 > 1) ? TK : (32 * 8 / 2);
  using Mma   = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>, Tile<Int<WOM*16>, Int<WON*16>, Int<MmaPermK>>>;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>, Tile<Int<WOM*16>, Int<WON*16>, _32>>;

  auto mma_view = make_layout(make_shape (make_shape(Int<F1>{}, Int<TN/F1>{}), Int<TK>{}),
                             make_stride(make_stride(Int<TK>{}, Int<F1*TK>{}), _1{}));
  auto tCrB_mma = Mma{}.get_thread_slice(0).partition_fragment_B(
      make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), mma_view));

  constexpr int BK1 = F1 * TK, Ng1 = TN / F1, RowB1 = BK1 * 2 / 8;
  using OB1 = AiuOp<2, BK1>;
  using AtomB1 = Copy_Atom<PPU0010_TSM_LD_SWZL<int8_t, Ng1, OB1::AiuElem / 4, true, false, OB1::InstNum>, int8_t>;
  auto tCrB_load = MmaS8{}.get_thread_slice(0).partition_fragment_B(
      make_tensor(make_smem_ptr((int8_t*)nullptr),
                  make_layout(make_shape(Int<Ng1>{}, Int<RowB1>{}), make_stride(Int<RowB1>{}, _1{}))));
  auto copy_view = make_tiled_copy_B(AtomB1{}, MmaS8{}).get_thread_slice(0).retile_D(tCrB_load);

  const int MMA_N = int(size<1>(tCrB_mma)), MMA_K = int(size<2>(tCrB_mma));
  const int CPY_N = int(size<1>(copy_view)), CPY_K = int(size<2>(copy_view));
  const int KAPC = MMA_K / CPY_K, NAPC = MMA_N / CPY_N;
  using Cvt = cutlass::MixGemm2Plane_uint2_uint1<>;
  constexpr int kPerAtom = Cvt::kPerAtom, kAtoms = Cvt::kAtoms;

  int bad_formula = 0, bad_partition = 0, bad_rebase = 0;

  // (a) the Chunk formula, read off tCrB_mma's own layout
  for (int s = 0; s < 32; ++s) {
    const int n_local = (2 * s) / (8 * MMA_K), k_atom = ((2 * s) / 8) % MMA_K;
    if (s / kPerAtom != k_atom + KAPC * n_local) ++bad_formula;
  }

  // (b) the chunks partition the delivery's 32 slots: keep() true exactly once over (k_atom, n_local)
  std::vector<int> hits(32, 0);
  for (int t = 0; t < 8; ++t)
    for (int v = 0; v < 4; ++v) {
      const int sp = Cvt::at_plain(t, v);
      for (int ka = 0; ka < KAPC; ++ka)
        for (int nl = 0; nl < NAPC; ++nl)
          if (sp / kPerAtom == ka + KAPC * nl) ++hits[sp];
    }
  for (int s = 0; s < 32; ++s) if (hits[s] != 1) ++bad_partition;

  // (c) at() rebases into [0, kPerAtom)
  for (int t = 0; t < 8; ++t)
    for (int v = 0; v < 4; ++v) {
      const int r = Cvt::at_plain(t, v) % kPerAtom;
      if (r < 0 || r >= kPerAtom) ++bad_rebase;
    }

  // (d) THE GATE, with every stride read off the real objects. The unchunked path is
  //     cvt_out = make_tensor(tCrB_mma(_,_,k_block*KAPC).data(), cvt_in.layout())
  //     o_p     = cvt_out(_, ii)                       -> base + ii * <cvt_in's mode-1 stride>
  //     h2[at_plain(t,v)]                              -> + at_plain half2
  // so the destination half2, relative to tCrB_mma.data(), is
  //     H = 4*KAPC*k_block + S1*ii + at_plain      with S1 = cvt_in's mode-1 stride in HALF2
  // I first hardcoded S1 = 32 and TK=256 came out self-contradictory (two (k_block, ii) pairs writing the same range).
  // The stride is 64 half2 there, not 32. Ask cute for it; do not write it down.
  //
  // Then, for each mma slot (n-atom j, k-atom ka, half2 p) the CONSUMER wants
  //     H_target = p + 4*MMA_K*j + 4*ka
  // and the chunked path must deliver exactly that value into tCrB_one at (j, p). So DERIVE (ii, Chunk, dest atom) from
  // H_target and check the candidate formulas against it, rather than assuming which of ii / k_block carries N.
  auto cvt_in_l = recast<cutlass::uint2b_t>(tCrB_load(_, _, 0)).layout();
  const int S1 = int(stride<1>(cvt_in_l)) / 2;                  // codes -> half2 (1 code == 1 fp16 out)
  int bad_dest = 0, shown = 0;
  for (int j = 0; j < MMA_N; ++j)
    for (int ka = 0; ka < MMA_K; ++ka)
      for (int p = 0; p < kPerAtom; ++p) {
        const int H = p + 4 * MMA_K * j + 4 * ka;
        // which (k_block, ii, at_plain) wrote it?
        const int kb = ka / KAPC;                                // the copy step that owns this k-atom
        const int rem = H - 4 * KAPC * kb;
        const int ii = (S1 > 0) ? (rem / S1) : 0, sp = (S1 > 0) ? (rem % S1) : rem;
        // the candidate the collective change would use
        const int cand_chunk = (ka % KAPC) + KAPC * (j % NAPC);
        const int cand_atom  = ii * NAPC + (j % NAPC);
        const bool ok = (ii >= 0 && ii < CPY_N) && (sp >= 0 && sp < 32)
                     && (sp / kPerAtom == cand_chunk) && (sp % kPerAtom == p) && (cand_atom == j);
        if (!ok) {
          ++bad_dest;
          if (shown < 4) { printf("      j=%d ka=%2d p=%d : H=%3d kb=%d -> ii=%d sp=%2d (chunk %d, in-atom %d)"
                                  "  cand chunk=%d atom=%d\n", j, ka, p, H, kb, ii, sp, sp / kPerAtom,
                                  sp % kPerAtom, cand_chunk, cand_atom); ++shown; }
        }
      }

  printf("  %-32s MMA_N=%d MMA_K=%d CPY_N=%d CPY_K=%d KAPC=%d NAPC=%d kAtoms=%d S1=%d\n",
         tag, MMA_N, MMA_K, CPY_N, CPY_K, KAPC, NAPC, kAtoms, S1);
  printf("  %-32s Chunk formula %s | partition %s | rebase %s | dest vs unchunked %s\n", "",
         bad_formula ? "WRONG" : "ok", bad_partition ? "NOT a partition" : "exact",
         bad_rebase ? "WRONG" : "ok", bad_dest ? "DIFFERS <-- do not ship" : "IDENTICAL");
  return !(bad_formula || bad_partition || bad_rebase || bad_dest);
}

int main() {
  printf("L63 -- generalising the 2-plane chunk call site to NAPC > 1\n\n");
  bool a = gate<64,  64, 256, 32, 32, 1>("TK=256 w32x32  NAPC=1 SHIPPED");
  bool b = gate<32, 128, 256, 32, 32, 1>("TK=256 w32x32 TN128  NAPC=1");
  bool c = gate<64,  64, 128, 32, 32, 1>("TK=128 w32x32  NAPC=1");
  bool d = gate<64, 128, 128, 32, 64, 1>("TK=128 w32x64  NAPC=1");
  bool e = gate<64, 128,  64, 64, 64, 2>("TK= 64 w64x64  NAPC=2 TARGET");
  bool f = gate<32, 128,  64, 32, 64, 2>("TK= 64 w32x64  NAPC=2");
  bool g = gate<128,128,  64, 64, 64, 2>("TK= 64 w64x64 TM128 NAPC=2");
  const bool ok = a && b && c && d && e && f && g;
  printf("\n  %s\n", ok ? "the generalisation holds at NAPC 1 AND 2 -- safe to change the call site"
                        : "NOT safe -- the derivation is wrong somewhere above");
  return ok ? 0 : 1;
}
