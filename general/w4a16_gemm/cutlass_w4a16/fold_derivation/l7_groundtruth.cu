// L7/1 -- GROUND TRUTH, by running the shipped offline instead of reading it.
//
// The whole L6 dispute is "does one 32-bit word of the folded gmem buffer hold one logical column, or several?"
// Reading the five steps says one. A comment in unfused_weight_dequantize.hpp says several, "proven from the
// validated int2@TK128 config". Both cannot be right, and neither is worth another paragraph of argument.
//
// So: call the real preprocess_weights_for_mixed_gemm + nfold_regroup_gmem and recover their map exactly.
// int1 codes carry one bit, so a position cannot be tagged with an id -- but the pipeline is a BIT PERMUTATION,
// so a binary label works: run it once per bit of the logical index, and OR the results back together. 15 runs
// recover the full map for a 128x256 weight.
//
// Then compose L2 o L3 o L4 against it. If the chain is right, the composition is the identity on (n, k).
//
// RESULT -- the chain is now CALIBRATED: 0 / 16384 mismatch, once pi is inserted.
//
// Two things this settled that argument could not:
//
//   1. The shipped pipeline is SINGLE-column per 32-bit word (512 words, all 1 column). So the claim in
//      unfused_weight_dequantize.hpp:791 that it "already interleaves several N columns into one 32B contiguous
//      run AT CRUMB LEVEL" is WRONG, and L6's contradiction was real. The truth is a clean closed form:
//          n = row + Ng*(wd / WPK)
//          k = 8*(j % 16) + (j / 16) + 2*(wd % WPK)          WPK = TK*Bits/32
//      which is exactly the five steps composed (add_bias_and_interleave_int1s' split-at-16 is the 8*(j%16) term).
//
//   2. The broken leg was L4, the one link I argued instead of measured -- "converter output element e lands at
//      fragment flat index e". It does not. Solving pi = part^-1 o truth gives a SINGLE fixed map, identical for
//      all 128 threads, GF(2)-linear, pi(0)=0, and on the element-index basis:
//          e bits 0,1,2 -> fixed        e bit3 -> 16   e bit4 -> 32   e bit5 -> 64   e bit6 -> 8
//      i.e. the high four bits rotate by one. Structurally, with fragment flat = atom_val + 64*mma_n:
//          atom_val = (e & 7) | ((e>>6 & 1) << 3) | ((e>>3 & 3) << 4)      mma_n = (e>>5) & 1
//      So mma_n is fed by the converter's r8 LEVEL bit, not by its vreg parity as flat indexing would imply.
//
// CAVEAT ON TRANSFERRING pi. It was calibrated at MMA_N=2. The WN=64/TK=64 target has MMA_N=4, so the fragment's
// mode split is different (atom 5 bits + 2 mma_n bits, not 6 + 1) and which element bits feed mma_n there is NOT
// determined by this calibration. Either derive pi structurally from the cvt_in/cvt_out layouts, or calibrate a
// second point. Generating the target placement from the MMA_N=2 pi would be exactly the kind of unverified
// transfer that produced the last three wrong answers.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l7_groundtruth.cu -o l7 && ./l7
//   NO_PI=1 ./l7    # the 14336/16384 failure, for contrast
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"
#include <cstdio>
#include <vector>
#include <map>
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

static const int NN = 128, KK = 256;          // smallest shape the pipeline allows (interleave-256 needs K>=256)
static const int TN = 128, TK = 128, WM = 32, WN = 32;   // the box-verified int1 F=2 config
static const int Bits = 1, F = 32 / (TK * Bits / 8);     // = 2
static const int Ng = TN / F, W_ROW = 8, CPW = 32;

int main() {
  const size_t nbytes = (size_t)NN * KK / 8;
  const int LOGBITS = 15;                                // NN*KK = 32768
  std::vector<int> id(nbytes * 8, 0);

  for (int b = 0; b < LOGBITS; ++b) {
    // logical code at (n,k) = bit b of (n*KK + k); packed as the test does: [N][K], k contiguous, 8 codes/byte
    std::vector<int8_t> packed(nbytes, 0);
    for (int n = 0; n < NN; ++n)
      for (int k = 0; k < KK; ++k) {
        const size_t idx = (size_t)n * KK + k;
        if ((idx >> b) & 1) packed[idx / 8] |= int8_t(1 << (idx % 8));
      }
    std::vector<int8_t> pre(nbytes), fold(nbytes);
    preprocess_weights_for_mixed_gemm<false, 256, 0>(
        pre.data(), packed.data(), {(size_t)KK, (size_t)NN}, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);
    legacy::nfold_regroup_gmem(fold.data(), pre.data(), {(size_t)KK, (size_t)NN}, TN, TK, Bits);
    for (size_t p = 0; p < nbytes * 8; ++p)
      if ((fold[p / 8] >> (p % 8)) & 1) id[p] |= (1 << b);
  }

  // sanity: the recovered map must be a bijection
  std::set<int> uniq(id.begin(), id.end());
  printf("GROUND TRUTH from the real preprocess + legacy::nfold_regroup_gmem(N=%d K=%d, int1, TN=%d TK=%d F=%d)\n\n",
         NN, KK, TN, TK, F);
  printf("  recovered %zu distinct logical ids over %zu bit positions -> %s\n\n",
         uniq.size(), nbytes * 8, uniq.size() == nbytes * 8 ? "bijection, map is trustworthy" : "NOT a bijection");

  // ---- THE QUESTION: how many logical columns share one 32-bit word?
  // folded layout (nfold_regroup_gmem): dst_w = kb*(nrow*W_ROW) + r*W_ROW + t*(F*WPK) + f*WPK + w
  const int nrow = NN / F;
  std::map<int,int> hist;
  for (int wd = 0; wd < nrow * W_ROW; ++wd) {
    std::set<int> ns;
    for (int j = 0; j < CPW; ++j) ns.insert(id[(size_t)wd * CPW + j] / KK);
    hist[(int)ns.size()]++;
  }
  printf("  columns per 32-bit word, over the whole folded buffer:\n");
  for (auto& [c, cnt] : hist) printf("      %d column(s) : %d words\n", c, cnt);
  printf("  => the shipped pipeline is %s\n\n",
         hist.size() == 1 && hist.begin()->first == 1 ? "SINGLE-column per word (my step-by-step reading)"
                                                      : "MULTI-column per word (the code comment's claim)");

  // ---- COMPOSE L2 o L3 o L4 and compare against ground truth, on plane kb=0
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<32/WM>,Int<TN/WN>,_1>>,
                       Tile<Int<(32/WM)*16>, Int<(TN/WN)*16>, Int<TK>>>;
  auto l3 = [](int j, int v) {
    const int b0=j&1,b1=(j>>1)&1,b2=(j>>2)&1,c=(j>>3)&1,d=(j>>4)&1;
    return 64*(v%2) + 4*(v/2) + 2*b0 + 8*b1 + 16*b2 + 32*c + d;
  };
  // pi, SOLVED below and then fed back here: the converter's element index does not land at the fragment's flat
  // index. Its high four bits are rotated by one. Verified as a single fixed GF(2)-linear map over all 128 threads
  // x 128 positions, against the box-verified offline.
  auto pi = [](int e) { const int hi = (e >> 3) & 0xF; return (e & 7) | ((((hi << 1) | (hi >> 3)) & 0xF) << 3); };
  const bool USE_PI = getenv("NO_PI") == nullptr;
  int bad = 0, tot = 0, shown = 0;
  std::map<std::pair<int,int>,int> dn_hist, dk_hist;
  for (int t = 0; t < 32 * (32/WM) * (TN/WN); ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / (32/WM);
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int v = 0; v < 4; ++v) {
      const int row = 16 * warp_n + (v / 2) * 8 + lane / 4;
      const int wd  = (v % 2) * 4 + lane % 4;
      for (int j = 0; j < CPW; ++j) {
        const int e0 = l3(j, v);
        auto c = part(USE_PI ? pi(e0) : e0);
        const int n_chain = int(get<0>(c)), k_chain = int(get<1>(c));
        const int gt = id[((size_t)row * W_ROW + wd) * CPW + j];     // plane kb=0
        const int n_gt = gt / KK, k_gt = gt % KK;
        ++tot;
        if (n_gt != n_chain || k_gt != k_chain) {
          ++bad;
          dn_hist[{n_chain, n_gt}]++; dk_hist[{k_chain, k_gt}]++;
          if (shown < 6) { printf("      MISMATCH row%2d word%d bit%2d : chain (n%3d,k%3d)  truth (n%3d,k%3d)\n",
                                  row, wd, j, n_chain, k_chain, n_gt, k_gt); ++shown; }
        }
      }
    }
  }
  printf("  L2 o L3 o L4%s vs ground truth: %d / %d mismatch %s\n\n",
         USE_PI ? " o pi" : " (NO_PI=1, raw)", bad, tot, bad ? "" : "<-- CALIBRATED");

  // ---- what k range does plane kb=0 actually hold? (if it is not [0,TK) the comparison domain is wrong)
  { std::set<int> ks, ns;
    for (int wd = 0; wd < Ng * W_ROW; ++wd) for (int j = 0; j < CPW; ++j) {
      ks.insert(id[(size_t)wd * CPW + j] % KK); ns.insert(id[(size_t)wd * CPW + j] / KK); }
    printf("  plane kb=0 holds %zu distinct k in [%d,%d] and %zu distinct n in [%d,%d]\n",
           ks.size(), *ks.begin(), *ks.rbegin(), ns.size(), *ns.begin(), *ns.rbegin()); }

  // ---- truth for row 0, all 8 words
  printf("\n  TRUTH, row 0: (n, k) per bit\n");
  for (int wd = 0; wd < W_ROW; ++wd) {
    printf("      word %d: ", wd);
    for (int j = 0; j < 8; ++j) { int g = id[((size_t)0*W_ROW+wd)*CPW+j]; printf("(%d,%d)", g/KK, g%KK); }
    printf(" ... bit31=(%d,%d)\n", id[((size_t)0*W_ROW+wd)*CPW+31]/KK, id[((size_t)0*W_ROW+wd)*CPW+31]%KK);
  }

  // ---- SOLVE for the permutation sigma = truth o chain^-1 on (n,k), and test it for bit-linearity
  printf("\n  sigma = truth o chain^-1, as a map on (n,k). Both are bijections, so sigma is well defined.\n");
  std::map<int,int> sigma;                                  // (n_c*TK + k_c) -> (n_t*TK_or_KK...)
  bool welldef = true;
  for (int t = 0; t < 32 * (32/WM) * (TN/WN); ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / (32/WM);
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int v = 0; v < 4; ++v) {
      const int row = 16 * warp_n + (v / 2) * 8 + lane / 4;
      const int wd  = (v % 2) * 4 + lane % 4;
      for (int j = 0; j < CPW; ++j) {
        auto c = part(l3(j, v));
        const int key = int(get<0>(c)) * KK + int(get<1>(c));
        const int val = id[((size_t)row * W_ROW + wd) * CPW + j];
        auto it = sigma.find(key);
        if (it == sigma.end()) sigma[key] = val; else if (it->second != val) welldef = false;
      }
    }
  }
  printf("      well defined: %s   entries: %zu\n", welldef ? "yes" : "NO", sigma.size());
  // is sigma bit-linear over GF(2)? test on the basis
  {
    bool linear = true;
    auto S = [&](int x) { auto it = sigma.find(x); return it == sigma.end() ? -1 : it->second; };
    const int s0 = S(0);
    for (auto& [x, y] : sigma) {
      // linear (affine) means sigma(a ^ b) == sigma(a) ^ sigma(b) ^ sigma(0)
      for (int bit = 0; bit < 15 && linear; ++bit) {
        const int a = 1 << bit;
        if (S(a) < 0 || S(x ^ a) < 0) continue;
        if (S(x ^ a) != (S(x) ^ S(a) ^ s0)) linear = false;
      }
    }
    printf("      GF(2)-affine: %s  (if yes, the fault is a fixed bit permutation -- findable by inspection)\n",
           linear ? "YES" : "no");
    printf("      sigma on the basis vectors (n_c,k_c) -> (n_t,k_t):\n");
    for (int bit = 0; bit < 15; ++bit) {
      const int a = 1 << bit, r = S(a);
      if (r < 0) continue;
      printf("        (n%3d,k%3d) -> (n%3d,k%3d)%s\n", a / KK, a % KK, r / KK, r % KK,
             (a == r) ? "   [fixed]" : "");
    }
    printf("        sigma(0,0) = (n%d,k%d)\n", s0 / KK, s0 % KK);
  }

  // ---- SOLVE for pi: which fragment index does converter element e ACTUALLY land at?
  // L3 is validated against the mask constants, so e is right. If L4's assumption (e == partition_B's linear
  // index) is what is broken, then the truth is (n,k) = part(pi(e)) for some fixed pi. Solve pi = part^-1 o truth.
  printf("\n  pi = part^-1 o truth, i.e. where converter element e really lands in the fragment.\n");
  {
    std::map<int,int> pi;                       // e -> fragment index
    bool welldef = true, same_across_threads = true;
    for (int t = 0; t < 32 * (32/WM) * (TN/WN); ++t) {
      const int lane = t % 32, w = t / 32, warp_n = w / (32/WM);
      auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
      std::map<int,int> inv;                    // (n,k) -> fragment index, for THIS thread
      for (int i = 0; i < int(size(part)); ++i) inv[int(get<0>(part(i))) * KK + int(get<1>(part(i)))] = i;
      for (int v = 0; v < 4; ++v) {
        const int row = 16 * warp_n + (v / 2) * 8 + lane / 4;
        const int wd  = (v % 2) * 4 + lane % 4;
        for (int j = 0; j < CPW; ++j) {
          const int e = l3(j, v);
          const int gt = id[((size_t)row * W_ROW + wd) * CPW + j];
          auto it = inv.find(gt);
          if (it == inv.end()) { same_across_threads = false; continue; }
          auto jt = pi.find(e);
          if (jt == pi.end()) pi[e] = it->second;
          else if (jt->second != it->second) welldef = false;
        }
      }
    }
    printf("      pi is a single fixed map for every thread: %s   entries %zu%s\n",
           welldef ? "YES" : "no", pi.size(), same_across_threads ? "" : "  (some truth (n,k) not in the fragment!)");
    if (welldef && pi.size() == 128) {
      bool ident = true, linear = true;
      for (auto& [e, f] : pi) if (e != f) ident = false;
      auto P = [&](int x) { auto it = pi.find(x); return it == pi.end() ? -1 : it->second; };
      const int p0 = P(0);
      for (auto& [x, y] : pi)
        for (int b = 0; b < 7 && linear; ++b) {
          const int a = 1 << b;
          if (P(a) < 0 || P(x ^ a) < 0) continue;
          if (P(x ^ a) != (P(x) ^ P(a) ^ p0)) linear = false;
        }
      printf("      identity: %s    GF(2)-affine: %s   pi(0) = %d\n",
             ident ? "yes" : "NO", linear ? "YES" : "no", p0);
      printf("      pi on the element-index basis (element bit -> fragment bits):\n");
      for (int b = 0; b < 7; ++b) printf("        e bit%d (=%3d) -> %3d%s\n", b, 1 << b, P(1 << b),
                                         P(1 << b) == (1 << b) ? "   [fixed]" : "");
    }
  }
  if (bad) {
    printf("      how n is wrong (chain -> truth), first 8 distinct pairs:\n        ");
    int c = 0; for (auto& [p, cnt] : dn_hist) { if (c++ < 8) printf("%d->%d (x%d)  ", p.first, p.second, cnt); }
    printf("\n      how k is wrong, first 8:\n        ");
    c = 0; for (auto& [p, cnt] : dk_hist) { if (c++ < 8) printf("%d->%d (x%d)  ", p.first, p.second, cnt); }
    printf("\n");
  }
  return 0;
}
