// L16 (D1) -- verify the chain on the F=1 (unfolded) path. This is the biggest remaining hole: int4's PRODUCTION
// config (TN=64, TK=64, the 55.9% MFU one) runs F=1, and everything verified so far was a fold config.
//
// F=1 differs from the fold path in three ways, each of which had to be read out of the collective:
//
//  1. gmem is the interleave-256 buffer ITSELF -- no nfold repack. load_init_B's kCon branch builds
//         make_shape(N, (kCon, K/kCon), L)   make_stride(kCon, (1, kCon*N), N*K*bits/8)
//     so code (n, k) with k = nt*kCon + k_in sits at  (nt*N + n)*kCon + k_in.  That IS interleave-256.
//  2. Ng == TN (no fold), but L2 only reaches warpOnN*16 rows per copy instance. int4 TN=64 at WN=32 gives 32,
//     so the tile needs TWO instances at coord_h 0 and 32.
//  3. The fragment is filled by NumIterations converter calls. l9_cvtin measured the chunk offsets as 0 and
//     CPY_VEC (0/32 for int4, 0/64 for int2, 0/128 for int1) -- consecutive halves, not strided.
//
// The instance -> row-range correspondence is the one thing left as a HYPOTHESIS: instance i covers rows
// [i*RPI, (i+1)*RPI). The program reports which n each fragment half demands so the hypothesis is checked against
// the data rather than assumed, and then regresses the whole thing against the real offline.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l16_f1path.cu -o l16 && ./l16
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "unfused_weight_dequantize.hpp"
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

template <int Bits, int TM, int TN, int TK, int WM, int WN>
static bool f1(const char* tag, int NN, int KK) {
  constexpr int WOM = TM/WM, WON = TN/WN, CPW = 32/Bits;
  constexpr int contig = TK*Bits/8;
  static_assert(contig >= 32, "this file is the F=1 path only");
  constexpr int Ng = TN;                                   // no fold
  constexpr int AiuByte = contig > 128 ? 128 : contig, AiuElem = AiuByte*8/Bits;
  constexpr int W_ROW = AiuByte/4;                         // uint32 words per row per AIU run
  constexpr int RPI = WON*16;                              // rows one copy instance reaches (L2)
  constexpr int NINST = Ng/RPI, VEC = 4*32/Bits;
  const int kCon = 256, RPS = kCon/AiuElem;                // AIU runs per k-supertile
  printf("  %-24s int%d tile(%d,%d,%d) w%dx%d | Ng=%d AiuElem=%d W_ROW=%d instances=%d VEC=%d runs/super=%d\n",
         tag, Bits, TM, TN, TK, WM, WN, Ng, AiuElem, W_ROW, NINST, VEC, RPS);
  if (W_ROW != 8) { printf("      cube is %dB wide (%d slices) -- L2 covers one 32B slice only, SKIPPED\n",
                           AiuByte, AiuByte/32); return true; }

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

  // CHECK THE HYPOTHESIS FIRST: which n does each fragment half demand?
  { auto part = Mma{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
    for (int i = 0; i < NINST; ++i) {
      std::set<int> ns;
      for (int e = i*VEC; e < (i+1)*VEC; ++e) ns.insert(int(get<0>(part(pi(e)))));
      printf("      fragment half %d (flat %3d..%3d) demands n in {", i, i*VEC, (i+1)*VEC-1);
      int c = 0; for (int n : ns) { if (c++ < 6) printf("%d,", n); }
      printf("%s}  -> expected rows [%d,%d)\n", ns.size()>6?"..":"", i*RPI, (i+1)*RPI);
    }
  }

  // ground truth: preprocess ONLY (no nfold) -- this is the shipped unfolded path
  const size_t ncodes = (size_t)NN*KK, nbytes = ncodes*Bits/8;
  int nbit = 0; while ((1u<<nbit) < ncodes) ++nbit;
  std::vector<int> id(ncodes, 0);
  const auto qtc = Bits==1 ? QuantTypeClass::PACKED_INT1_WEIGHT_ONLY
                 : Bits==2 ? QuantTypeClass::PACKED_INT2_WEIGHT_ONLY
                           : QuantTypeClass::PACKED_INT4_WEIGHT_ONLY;
  for (int b = 0; b < nbit; ++b) {
    std::vector<int8_t> packed(nbytes,0), pre(nbytes);
    for (size_t i = 0; i < ncodes; ++i) if ((i>>b)&1) { const size_t bp=i*Bits; packed[bp/8] |= int8_t(1<<(bp%8)); }
    preprocess_weights_for_mixed_gemm<false,256,0>(pre.data(), packed.data(), {(size_t)KK,(size_t)NN}, qtc);
    // int4 adds +8 (sets bit 3); bit 0 still carries the label, so read bit 0 of each code
    for (size_t c = 0; c < ncodes; ++c) { const size_t bp = c*Bits; if ((pre[bp/8]>>(bp%8))&1) id[c] |= (1<<b); }
  }
  std::set<int> uq(id.begin(), id.end());
  if (uq.size() != ncodes) { printf("      ground truth not a bijection (%zu/%zu) -- abort\n", uq.size(), ncodes); return false; }

  long bad = 0, tot = 0; int shown = 0;
  for (int tn = 0; tn < NN/TN; ++tn)
    for (int ki = 0; ki < KK/TK; ++ki)
      for (int i = 0; i < NINST; ++i)
        for (int t = 0; t < 32*WOM*WON; ++t) {
          const int lane = t%32, w = t/32, warp_n = w/WOM;
          auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
          for (int v = 0; v < 4; ++v) {
            const int row = i*RPI + 16*warp_n + (v/2)*8 + lane/4;
            const int wd  = (v%2)*4 + lane%4;
            for (int jj = 0; jj < CPW; ++jj) {
              // int4's l3 takes a GLOBAL code index because its converter permutes across all four vregs;
              // int1's and int2's take (code-within-vreg, vreg). Writing v*4+jj for int2 was my bug: it gave
              // 24576/32768 with the k off by 32, which is what a wrong code index inside the right word looks like.
              const int e_local = (Bits==4) ? l3_int4(v*8+jj, v)
                                : (Bits==2) ? l3_int2(jj, v) : l3_int1(jj, v);
              const int flat = i*VEC + e_local;
              if (flat < 0 || flat >= NS) continue;
              auto c = part(pi(flat));
              const int n_exp = tn*TN + int(get<0>(c)), k_exp = ki*TK + int(get<1>(c));
              const size_t off = ((size_t)(ki/RPS)*NN + tn*TN + row)*kCon + (ki%RPS)*AiuElem + wd*CPW + jj;
              ++tot;
              if (id[off] != n_exp*KK + k_exp) {
                ++bad;
                if (shown < 4) { printf("      MISMATCH tn%d ki%d i%d r%2d w%d c%d: chain(n%3d,k%3d) truth(n%3d,k%3d)\n",
                                        tn, ki, i, row, wd, jj, n_exp, k_exp, id[off]/KK, id[off]%KK); ++shown; }
              }
            }
          }
        }
  printf("      chain vs the real unfolded offline: %ld / %ld mismatch  %s\n\n", bad, tot, bad ? "" : "<-- PASSES");
  return bad == 0;
}

int main() {
  printf("L16 (D1) -- the F=1 / unfolded path, which nothing had verified\n\n");
  bool ok = true;
  ok &= f1<4, 64, 64,  64, 32, 32>("int4 PRODUCTION", 64, 256);
  ok &= f1<4, 64, 64,  64, 32, 32>("int4 multi-tile", 128, 512);
  ok &= f1<2, 64, 64, 128, 32, 32>("int2 unfolded", 64, 256);
  ok &= f1<1, 64, 64, 256, 32, 32>("int1 unfolded", 64, 256);
  printf("  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
