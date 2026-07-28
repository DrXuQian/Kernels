// L13 -- everything so far verified ONE tile. This covers the whole buffer.
//
// l7/l10/l12 all compared plane kb=0, rows [0,Ng), and only run t=0 of each row segment. Three strides were
// therefore never exercised:
//
//   * the N-tile offset      nfold_regroup_gmem's tile_n0 = (r / Ng) * fold_tn
//   * the run-within-segment index   ktile = kb*(kCon/(F*TK)) + t, with the word index t*8 + wd
//     -- int1 has one run per segment (so t never varied), int2 has TWO, int4 has FOUR
//   * the plane stride       kb * (nrow * W_ROW)
//
// A single-tile test cannot see any of them, and all three are exactly the kind of outer index that L12's W_ROW
// bug lived in. So: run the real offline on a buffer with SEVERAL N-tiles and SEVERAL k-tiles, and check every
// bit position against the chain, with the tile indices folded back in.
//
// Also re-checks nfold_place_bits_int1_tk64 over multiple tiles -- l11 used exactly one.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l13_wholebuffer.cu -o l13 && ./l13
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"
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

static int l3_int1(int j, int v) { const int b0=j&1,b1=(j>>1)&1,b2=(j>>2)&1,c=(j>>3)&1,d=(j>>4)&1;
  return 64*(v%2)+4*(v/2)+2*b0+8*b1+16*b2+32*c+d; }
static int l3_int2(int c, int v) { const int m=c&3, lv=(c>>2)&1, hf=(c>>3)&1;
  return 2*(16*(v&1)+2*(v>=2) + (m&1)+4*((m>>1)&1) + 8*lv) + hf; }
static int l3_int4(int c, int)   { const int g=c/8, ci=c%8, G=((g&1)<<1)|((g>>1)&1);
  const int e8=2*(ci&3)+((ci>>2)&1), idx=8*G+e8, b=idx/4, r=idx%4;
  return 4*((b&~3)|((b&1)<<1)|((b>>1)&1)) + r; }

// chain map for ONE tile: index [(row_local*8 + wd)*CPW + j] -> n_local*TK + k_local
template <int Bits, int TM, int TN, int TK, int WM, int WN>
static std::vector<int> chain_tile(int (*l3)(int,int)) {
  constexpr int WOM = TM/WM, WON = TN/WN, contig = TK*Bits/8;
  constexpr int F = contig >= 32 ? 1 : 32/contig, Ng = TN/F, CPW = 32/Bits;
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
  std::vector<int> out((size_t)Ng*8*CPW, -1);
  for (int t=0; t<32*WOM*WON; ++t) {
    const int lane=t%32, w=t/32, warp_n=w/WOM;
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
    for (int v=0; v<4; ++v) {
      const int row=16*warp_n+(v/2)*8+lane/4, wd=(v%2)*4+lane%4;
      for (int j=0; j<CPW; ++j) {
        const int e = l3(Bits==4 ? v*8+j : j, v);
        if (e < 0 || e >= NS) continue;
        auto c = part(pi(e));
        out[((size_t)row*8+wd)*CPW+j] = int(get<0>(c))*TK + int(get<1>(c));
      }
    }
  }
  return out;
}

template <int Bits, int TM, int TN, int TK, int WM, int WN>
static bool whole(const char* tag, int NN, int KK) {
  constexpr int contig = TK*Bits/8, F = contig >= 32 ? 1 : 32/contig;
  constexpr int Ng = TN/F, CPW = 32/Bits;
  const int W_ROW_OFF = 256/CPW, RUNS = W_ROW_OFF/8;      // runs per row segment == kCon/(F*TK) == Bits
  const size_t ncodes = (size_t)NN*KK, nbytes = ncodes*Bits/8;
  const int nrow = NN/F, nkb = int((size_t)KK*F/256);
  const int ntile_n = NN/TN, nki = KK/TK;
  printf("  %-22s int%d tile(%d,%d,%d) w%dx%d | buffer %dx%d = %d N-tiles x %d k-tiles, %d planes, %d run(s)/seg\n",
         tag, Bits, TM, TN, TK, WM, WN, NN, KK, ntile_n, nki, nkb, RUNS);

  // ground truth over the WHOLE buffer, binary label
  int nbit = 0; while ((1u << nbit) < ncodes) ++nbit;
  std::vector<int> id(ncodes, 0);
  const auto qtc = Bits==1 ? QuantTypeClass::PACKED_INT1_WEIGHT_ONLY
                 : Bits==2 ? QuantTypeClass::PACKED_INT2_WEIGHT_ONLY
                           : QuantTypeClass::PACKED_INT4_WEIGHT_ONLY;
  for (int b = 0; b < nbit; ++b) {
    std::vector<int8_t> packed(nbytes, 0);
    for (size_t i = 0; i < ncodes; ++i) if ((i >> b) & 1) {
      const size_t bp = i * Bits; packed[bp/8] |= int8_t(1 << (bp%8));
    }
    std::vector<int8_t> pre(nbytes), fold(nbytes);
    preprocess_weights_for_mixed_gemm<false,256,0>(pre.data(), packed.data(), {(size_t)KK,(size_t)NN}, qtc);
    legacy::nfold_regroup_gmem(fold.data(), pre.data(), {(size_t)KK,(size_t)NN}, TN, TK, Bits);
    for (size_t c = 0; c < ncodes; ++c) { const size_t bp = c*Bits;
      if ((fold[bp/8] >> (bp%8)) & 1) id[c] |= (1 << b); }
  }
  std::set<int> uq(id.begin(), id.end());
  if (uq.size() != ncodes) { printf("      ground truth is not a bijection (%zu/%zu) -- abort\n", uq.size(), ncodes); return false; }

  auto tile = chain_tile<Bits,TM,TN,TK,WM,WN>(Bits==1 ? l3_int1 : Bits==2 ? l3_int2 : l3_int4);
  long bad = 0, tot = 0; int shown = 0;
  std::set<int> bad_tn, bad_ki;
  for (int kb = 0; kb < nkb; ++kb)
    for (int r = 0; r < nrow; ++r)
      for (int t = 0; t < RUNS; ++t)
        for (int wd = 0; wd < 8; ++wd)
          for (int j = 0; j < CPW; ++j) {
            const int tn = r / Ng, rl = r % Ng, ki = kb * RUNS + t;
            const int loc = tile[((size_t)rl*8+wd)*CPW+j];
            if (loc < 0) continue;
            const int n_exp = tn * TN + loc / TK, k_exp = ki * TK + loc % TK;
            const int gt = id[((size_t)(kb*nrow + r)*W_ROW_OFF + t*8 + wd)*CPW + j];
            ++tot;
            if (gt != n_exp * KK + k_exp) {
              ++bad; bad_tn.insert(tn); bad_ki.insert(ki);
              if (shown < 4) { printf("      MISMATCH kb%d r%3d t%d w%d c%2d: chain(n%4d,k%4d) truth(n%4d,k%4d)\n",
                                      kb, r, t, wd, j, n_exp, k_exp, gt/KK, gt%KK); ++shown; }
            }
          }
  printf("      whole buffer: %ld / %ld mismatch  %s\n", bad, tot, bad ? "" : "<-- PASSES");
  if (bad) { printf("      failing N-tiles:"); for (int x : bad_tn) printf(" %d", x);
             printf("   failing k-tiles:"); for (int x : bad_ki) printf(" %d", x); printf("\n"); }
  printf("\n");
  return bad == 0;
}

int main() {
  printf("L13 -- whole-buffer regression: N-tile stride, run-in-segment index, plane stride\n\n");
  bool ok = true;
  // ISOLATION LADDER. Vary one thing at a time to separate "N-tile stride" from "K > 256".
  //   interleave_column_major_tensor_ppu writes  dst[nt*(vrpt*N) + c*vrpt + ti],  nt = vr/vrpt, vrpt = 256/CPW.
  //   nfold_regroup_gmem reads                   src[n_log*WPN + ktile*WPK + w],  WPN = K/CPW.
  //   Those agree only when nvr == vrpt, i.e. K/CPW == 256/CPW, i.e. K == 256 -- exactly one k-supertile. Every
  //   regression so far used K=256, so it never tested the composition.
  printf("  == isolation: one N-tile, K at and above 256\n");
  ok &= whole<1, 32, 128, 128, 32, 32>("int1 1tile K=256", 128, 256);   // the config every earlier test used
  ok &= whole<1, 32, 128, 128, 32, 32>("int1 1tile K=512", 128, 512);   // same tiling, K doubled
  printf("  == and then several N-tiles\n");
  ok &= whole<1, 32, 128, 128, 32, 32>("int1 2tile K=256", 256, 256);
  ok &= whole<1, 32, 128, 128, 32, 32>("int1 fold", 256, 512);   // 2 N-tiles, 4 k-tiles, t never varies for int1
  ok &= whole<2, 64,  64,  64, 32, 32>("int2 fold", 256, 512);   // 4 N-tiles, 8 k-tiles, TWO runs per segment
  ok &= whole<2, 64, 128,  64, 32, 32>("int2 wideN", 256, 512);
  ok &= whole<4, 64,  64,  32, 32, 32>("int4 fold", 256, 512);   // FOUR runs per segment

  // the bit-granular packer, over several tiles (l11 used exactly one)
  printf("  == nfold_place_bits_int1_tk64 over multiple tiles\n");
  { const int NN = 256, KK = 256, TN = 128, TK = 64;
    const size_t ncodes = (size_t)NN*KK, nbytes = ncodes/8;
    int nbit = 0; while ((1u << nbit) < ncodes) ++nbit;
    std::vector<int> id(ncodes, 0);
    for (int b = 0; b < nbit; ++b) {
      std::vector<int8_t> in(nbytes,0), out(nbytes,0);
      for (size_t i = 0; i < ncodes; ++i) if ((i>>b)&1) in[i/8] |= int8_t(1<<(i%8));
      legacy::nfold_place_bits_int1_tk64(out.data(), in.data(), NN, KK, TN, TK);
      for (size_t p = 0; p < ncodes; ++p) if ((out[p/8]>>(p%8))&1) id[p] |= (1<<b);
    }
    std::set<int> uq(id.begin(), id.end());
    printf("      %dx%d (%d N-tiles x %d k-tiles): %zu distinct / %zu -> %s\n", NN, KK, NN/TN, KK/TK,
           uq.size(), ncodes, uq.size()==ncodes ? "bijection" : "NOT a bijection");
    // and it must agree with the chain on every tile
    auto tile = chain_tile<1, 64, 128, 64, 32, 64>(l3_int1);
    const int F = 4, Ng = TN/F, CPW = 32, nrow = NN/F;
    long bad = 0; std::set<int> btn, bki;
    for (int kb = 0; kb < KK/TK; ++kb) for (int r = 0; r < nrow; ++r)
      for (int wd = 0; wd < 8; ++wd) for (int j = 0; j < CPW; ++j) {
        const int tn = r/Ng, rl = r%Ng, loc = tile[((size_t)rl*8+wd)*CPW+j];
        if (loc < 0) continue;
        const int n_exp = tn*TN + loc/TK, k_exp = kb*TK + loc%TK;
        const int gt = id[((size_t)(kb*nrow + r)*8 + wd)*CPW + j];
        if (gt != n_exp*KK + k_exp) { ++bad; btn.insert(tn); bki.insert(kb); }
      }
    printf("      packer vs chain over all tiles: %ld mismatch %s\n", bad, bad ? "" : "<-- PASSES");
    if (bad) { printf("      failing N-tiles:"); for (int x : btn) printf(" %d", x);
               printf("   k-tiles:"); for (int x : bki) printf(" %d", x); printf("\n"); }
    ok &= (bad == 0);
  }
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
