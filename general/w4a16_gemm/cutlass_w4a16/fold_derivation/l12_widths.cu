// L12 -- extend the chain from int1 to int2 and int4, and regress each against the real offline.
//
// So far the chain was validated at ONE bit width. Two things get added here:
//
//   A. L3 for int2 and int4, each checked against an INDEPENDENT model of its own converter -- the mask constants
//      and the explicit sub-vector permutations, read out of the instruction semantics, not my algebra.
//        int2: base = 16*(v&1) + 2*(v>=2); 8 masks at h2[base + off4[m] + 8*level], off4 = {0,1,4,5}, and the
//              mask pair (1<<2m..) | (1<<(16+2m)..) puts crumb m in the LO fp16 and crumb m+8 in the HI one.
//        int4: NOT the same shape. The wide (32-code) converter composes 4x the base-8 one with TWO explicit
//              permutations -- result_ptr[0,2,1,3] = conv(source_ptr[0,1,2,3]), then a swap of 4-element blocks
//              1<->2 and 5<->6. Modelled by following those steps rather than by fitting a formula.
//
//   B. A full-chain regression per width against preprocess_weights_for_mixed_gemm + nfold_regroup_gmem, the same
//      ground truth l7/l10 used. Only FOLD configs are regressed, because there the folded buffer is plane-major
//      and (row, word) lands where the chain expects it. F=1 configs would additionally need a model of
//      load_init_B's interleaved-256 addressing, which is not written yet -- and is stated as such rather than
//      quietly skipped.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l12_widths.cu -o l12 && ./l12
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
#include <set>
using namespace cute;

struct F16Atom {};
namespace cute {
template <> struct MMA_Traits<F16Atom> {
  using ValTypeD=float; using ValTypeA=cutlass::half_t; using ValTypeB=cutlass::half_t; using ValTypeC=float;
  using Shape_MNK=Shape<_16,_16,_16>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
}

// ---------------------------------------------------------------- L3, per width
// int1: e = 64*(v%2) + 4*(v/2) + 2*b0 + 8*b1 + 16*b2 + 32*c + d,  j = b0+2b1+4b2+8c+16d      [validated in l2l3]
static int l3_int1(int j, int v) {
  const int b0=j&1,b1=(j>>1)&1,b2=(j>>2)&1,c=(j>>3)&1,d=(j>>4)&1;
  return 64*(v%2) + 4*(v/2) + 2*b0 + 8*b1 + 16*b2 + 32*c + d;
}
// int2, algebraic form to be checked against the masks below
static int l3_int2(int c, int v) {
  const int m = c & 3, level = (c >> 2) & 1, half = (c >> 3) & 1;
  const int off4 = (m & 1) + 4 * ((m >> 1) & 1);                    // {0,1,4,5}
  const int base = 16 * (v & 1) + 2 * (v >= 2);
  return 2 * (base + off4 + 8 * level) + half;
}
// int2 reference: the eight _E() calls verbatim (h2 offset, level, mask crumb position)
static int l3_int2_ref(int c, int v) {
  struct E { int h2off, level, m; };
  static const E kE[8] = {{0,0,0},{1,0,1},{4,0,2},{5,0,3},{8,1,0},{9,1,1},{12,1,2},{13,1,3}};
  const int base = 16 * (v & 1) + 2 * (v >= 2);
  for (auto& x : kE) {
    // mask (3<<2m) | (3<<(16+2m)) on `reg` (level 0) or `reg>>8` (level 1):
    //   LO fp16 <- crumb (m + 4*level)      HI fp16 <- crumb (m + 4*level + 8)
    if (c == x.m + 4 * x.level)     return 2 * (base + x.h2off) + 0;
    if (c == x.m + 4 * x.level + 8) return 2 * (base + x.h2off) + 1;
  }
  return -1;
}
// int4: follow the wide converter's two permutations literally.
static int l3_int4(int c, int /*unused*/) {
  const int g = c / 8, cin = c % 8;                                 // vreg, code within vreg
  const int G = ((g & 1) << 1) | ((g >> 1) & 1);                    // result_ptr[0,2,1,3] = conv(source_ptr[0,1,2,3])
  const int e8 = 2 * (cin & 3) + ((cin >> 2) & 1);                  // base-8: h[cin&3], LO if cin<4 else HI
  const int idx = 8 * G + e8;
  const int b = idx / 4, r = idx % 4;                               // then swap 4-element blocks 1<->2, 5<->6
  const int nb = (b & ~3) | ((b & 1) << 1) | ((b >> 1) & 1);
  return 4 * nb + r;
}

// ---------------------------------------------------------------- chain, generic over width
template <int Bits, int TM, int TN, int TK, int WM, int WN>
static std::vector<int> chain(int (*l3)(int,int), bool verbose) {
  constexpr int WOM = TM / WM, WON = TN / WN;
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  constexpr int Ng = TN / F, CPW = 32 / Bits, W_ROW = 8;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));
  // pi = frag.layout()^-1, and cute HAS this: right_inverse(Layout). It used to be a hand-rolled loop here (and in
  // seven sibling files); l30_should_have_been_cute.cu verifies the two agree on three shapes including int4's
  // production one. Worth replacing beyond tidiness: l20 is the file scheduled to become the production offline.
  auto pi = right_inverse(frag.layout());
  std::vector<int> out((size_t)Ng * W_ROW * CPW, -1);
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int v = 0; v < 4; ++v) {
      const int row = 16 * warp_n + (v / 2) * 8 + lane / 4, wd = (v % 2) * 4 + lane % 4;
      for (int j = 0; j < CPW; ++j) {
        const int e = l3(Bits == 4 ? (v * 8 + j) : j, v);            // int4's l3 takes a GLOBAL code index
        if (e < 0 || e >= NS) continue;
        auto cc = part(pi(e));
        out[((size_t)row * W_ROW + wd) * CPW + j] = int(get<0>(cc)) * TK + int(get<1>(cc));
      }
    }
  }
  if (verbose) { printf("      fragment "); print(frag.layout()); printf("  NS=%d\n", NS); }
  return out;
}

// ---------------------------------------------------------------- ground truth from the real offline
template <int Bits, int TN, int TK>
static std::vector<int> truth(int NN, int KK) {
  const int Ng = TN / (32 / (TK * Bits / 8) ? (TK*Bits/8 >= 32 ? 1 : 32/(TK*Bits/8)) : 1);
  (void)Ng;
  const size_t nbytes = (size_t)NN * KK * Bits / 8;
  const int nbit = 32 - __builtin_clz(NN * KK) - ((NN*KK & (NN*KK-1)) ? 0 : 1);
  std::vector<int> id(nbytes * 8 / Bits, 0);
  const auto qtc = Bits == 1 ? QuantTypeClass::PACKED_INT1_WEIGHT_ONLY
                 : Bits == 2 ? QuantTypeClass::PACKED_INT2_WEIGHT_ONLY
                             : QuantTypeClass::PACKED_INT4_WEIGHT_ONLY;
  for (int b = 0; b <= nbit; ++b) {
    std::vector<int8_t> packed(nbytes, 0);
    for (int n = 0; n < NN; ++n) for (int k = 0; k < KK; ++k) {
      const size_t i = (size_t)n * KK + k;
      if ((i >> b) & 1) {                       // set code i's bit b -> put a 1 in code i's bit0 slot
        const size_t bitpos = i * Bits;         // codes are Bits wide, little endian within the byte
        packed[bitpos / 8] |= int8_t(1 << (bitpos % 8));
      }
    }
    std::vector<int8_t> pre(nbytes), fold(nbytes);
    preprocess_weights_for_mixed_gemm<false, 256, 0>(pre.data(), packed.data(), {(size_t)KK, (size_t)NN}, qtc);
    legacy::nfold_regroup_gmem(fold.data(), pre.data(), {(size_t)KK, (size_t)NN}, TN, TK, Bits);
    for (size_t c = 0; c < id.size(); ++c) {
      const size_t bitpos = c * Bits;
      if ((fold[bitpos / 8] >> (bitpos % 8)) & 1) id[c] |= (1 << b);
    }
  }
  return id;
}

template <int Bits, int TM, int TN, int TK, int WM, int WN>
static void regress(const char* tag, int NN, int KK, int (*l3)(int,int), const char* note) {
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  constexpr int Ng = TN / F, CPW = 32 / Bits, W_ROW = 8;
  // nfold_regroup_gmem's row segment is kCon/CPW words, NOT 8: it covers kCon=256 CODES, which is 8 words only for
  // int1. int2 gives 16 words and int4 gives 32, i.e. the segment spans SEVERAL 32B AIU runs, selected by the
  // offline's `t` loop (t*(F*WPK), F*WPK = 8). The chain's (row, word) lives inside ONE run, so the run index has
  // to be threaded through. Hardcoding 8 is why int1 passed and the other two did not.
  constexpr int W_ROW_OFF = 256 / CPW;
  printf("  %-30s int%d (%d,%d,%d) w%dx%d F=%d  [%s]  offline row seg = %d words = %d runs\n",
         tag, Bits, TM, TN, TK, WM, WN, F, note, W_ROW_OFF, W_ROW_OFF / 8);
  auto ch = chain<Bits, TM, TN, TK, WM, WN>(l3, true);
  auto gt = truth<Bits, TN, TK>(NN, KK);
  int bad = 0, holes = 0, shown = 0;
  for (int row = 0; row < Ng; ++row) for (int wd = 0; wd < W_ROW; ++wd) for (int j = 0; j < CPW; ++j) {
    const size_t idx = ((size_t)row * W_ROW + wd) * CPW + j;                       // chain: one run
    const size_t off = ((size_t)row * W_ROW_OFF + wd) * CPW + j;                   // offline: run t=0 of the segment
    if (ch[idx] < 0) { ++holes; continue; }
    const int n_gt = gt[off] / KK, k_gt = gt[off] % KK;
    if (ch[idx] / TK != n_gt || ch[idx] % TK != k_gt) {
      ++bad;
      if (shown < 4) { printf("      MISMATCH row%2d w%d c%2d: chain(n%3d,k%3d) truth(n%3d,k%3d)\n",
                              row, wd, j, ch[idx]/TK, ch[idx]%TK, n_gt, k_gt); ++shown; }
    }
  }
  printf("      chain vs the real offline: %d / %d mismatch%s  %s\n\n", bad, Ng*W_ROW*CPW,
         holes ? " (+holes)" : "", bad ? "" : "<-- PASSES");
}

int main() {
  printf("L12 -- int2 and int4: converter models, then full-chain regression\n\n");

  printf("  == A0. the three local L3 models vs the UNIFIED cutlass::MixGemmEmit -- single source of truth\n");
  { int b1 = 0, b2 = 0, b4 = 0;
    for (int v = 0; v < 4; ++v) {
      for (int c = 0; c < 32; ++c) if (l3_int1(c, v) != cutlass::MixGemmEmit<1>::index(c, v)) ++b1;
      for (int c = 0; c < 16; ++c) if (l3_int2(c, v) != cutlass::MixGemmEmit<2>::index(c, v)) ++b2;
    }
    // int4's converter folds the vreg into a global code index; split it back out
    for (int cg = 0; cg < 32; ++cg) if (l3_int4(cg, 0) != cutlass::MixGemmEmit<4>::index(cg % 8, cg / 8)) ++b4;
    printf("      int1 %d/128   int2 %d/64   int4 %d/32 differ -> %s\n", b1, b2, b4,
           (b1 || b2 || b4) ? "DRIFTED" : "ONE MAP, all three widths");
  }

  printf("\n  == A. L3 per width, against an independent model of each converter\n");
  { int bad = 0; std::set<int> seen;
    for (int v = 0; v < 4; ++v) for (int c = 0; c < 16; ++c) {
      if (l3_int2(c, v) != l3_int2_ref(c, v)) ++bad; seen.insert(l3_int2(c, v)); }
    printf("      int2: %d / 64 differ from the mask constants; image size %zu / 64 -> %s\n",
           bad, seen.size(), (!bad && seen.size() == 64) ? "bijection, MODEL OK" : "PROBLEM"); }
  { std::set<int> seen; for (int c = 0; c < 32; ++c) seen.insert(l3_int4(c, 0));
    printf("      int4: image size %zu / 32 -> %s   (map: ", seen.size(),
           seen.size() == 32 ? "bijection, MODEL OK" : "PROBLEM");
    for (int c = 0; c < 8; ++c) printf("%d ", l3_int4(c, 0)); printf("...)\n"); }
  { std::set<int> seen; for (int v = 0; v < 4; ++v) for (int j = 0; j < 32; ++j) seen.insert(l3_int1(j, v));
    printf("      int1: image size %zu / 128 -> %s\n\n", seen.size(),
           seen.size() == 128 ? "bijection, MODEL OK (already validated in l2l3)" : "PROBLEM"); }

  printf("  == B. full-chain regression, FOLD configs only (F=1 needs a load_init_B model, not written)\n");
  regress<1, 32, 128, 128, 32, 32>("int1 fold (already known)", 128, 256, l3_int1, "box: bad=0");
  regress<2, 64,  64,  64, 32, 32>("int2 fold",                  64, 256, l3_int2, "box: bad=0, 53.2%");
  regress<2, 64, 128,  64, 32, 32>("int2 fold wideN",           128, 256, l3_int2, "box: bad=0");
  regress<4, 64,  64,  32, 32, 32>("int4 fold TK=32",            64, 256, l3_int4, "NOT box-verified");
  return 0;
}
