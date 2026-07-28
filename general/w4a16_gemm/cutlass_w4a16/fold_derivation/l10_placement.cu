// L10 (the "L7 packer" step) -- generate the offline placement for a config the shipped offline has never seen.
//
// The chain, every link of which is now either measured or derived:
//     (lane, v)          --L2--> row_local = (v/2)*8 + lane/4 ,  word = (v%2)*4 + lane%4     [0 mismatch vs the sim]
//     row                =       16*warp_n + row_local
//     (bit j, v)         --L3--> converter element e                                         [0 mismatch vs the masks]
//     e                  --pi--> fragment LOGICAL index                                      [= frag.layout()^-1, l8]
//     logical index      --L4--> (n, k)                                                      [cute partition_B]
//
// pi comes from partition_fragment_B on sB built with the COLLECTIVE's stride, make_stride(Int<TKe>{}, _1{}).
// Getting that stride wrong is what made pi look like the identity in l8's first run.
//
// ORDER OF BUSINESS, deliberately:
//   1. REGRESS on the box-verified F=2 config against ground truth from the real offline. Must be 0/16384. If this
//      fails, nothing below is worth reading.
//   2. Only then generate the WN=64 / TK=64 target, check it is a bijection, and fit a closed form.
//   3. VERIFY the fitted closed form against the table it was fitted to. A fit that is not verified is a guess.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l10_placement.cu -o l10 && ./l10
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

struct Cell { int n, k; };

// Build the chain's placement for one config. Returns phys[(row*W_ROW + word)*CPW + bit] = (n,k).
template <int Bits, int TM, int TN, int TK, int WM, int WN>
static std::vector<Cell> chain_map(bool verbose) {
  constexpr int WOM = TM / WM, WON = TN / WN;
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  constexpr int Ng = TN / F, CPW = 32 / Bits, W_ROW = 8;

  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  // the collective's SmemLayoutB_MmaView -- ROW major. This stride is load bearing (see l8).
  auto sB  = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                         make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));

  // pi = frag.layout()^-1 : flat offset (where the converter writes) -> fragment logical index
  // cute's right_inverse, not a hand-rolled loop (l30 verifies they agree).
  auto pi = right_inverse(frag.layout());

  auto l3 = [](int j, int v) {
    const int b0=j&1,b1=(j>>1)&1,b2=(j>>2)&1,c=(j>>3)&1,d=(j>>4)&1;
    return 64*(v%2) + 4*(v/2) + 2*b0 + 8*b1 + 16*b2 + 32*c + d;
  };

  std::vector<Cell> phys((size_t)Ng * W_ROW * CPW, Cell{-1,-1});
  for (int t = 0; t < 32 * WOM * WON; ++t) {
    const int lane = t % 32, w = t / 32, warp_n = w / WOM;
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int v = 0; v < 4; ++v) {
      const int row = 16 * warp_n + (v / 2) * 8 + lane / 4;
      const int wd  = (v % 2) * 4 + lane % 4;
      for (int j = 0; j < CPW; ++j) {
        auto c = part(pi(l3(j, v)));
        phys[((size_t)row * W_ROW + wd) * CPW + j] = Cell{int(get<0>(c)), int(get<1>(c))};
      }
    }
  }
  if (verbose) { printf("      fragment "); print(frag.layout()); printf("   pi basis: ");
                 for (int b = 0; (1<<b) < NS; ++b) printf("bit%d->%d ", b, pi(1<<b)); printf("\n"); }
  return phys;
}

// ---------------- step 1: regression against the real offline, on the box-verified F=2 config
static bool regress() {
  const int NN = 128, KK = 256, TN = 128, TK = 128, Bits = 1, F = 2, Ng = TN / F, CPW = 32, W_ROW = 8;
  const size_t nbytes = (size_t)NN * KK / 8;
  std::vector<int> id(nbytes * 8, 0);
  for (int b = 0; b < 15; ++b) {
    std::vector<int8_t> packed(nbytes, 0);
    for (int n = 0; n < NN; ++n) for (int k = 0; k < KK; ++k) {
      const size_t idx = (size_t)n * KK + k;
      if ((idx >> b) & 1) packed[idx / 8] |= int8_t(1 << (idx % 8));
    }
    std::vector<int8_t> pre(nbytes), fold(nbytes);
    preprocess_weights_for_mixed_gemm<false, 256, 0>(
        pre.data(), packed.data(), {(size_t)KK, (size_t)NN}, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);
    legacy::nfold_regroup_gmem(fold.data(), pre.data(), {(size_t)KK, (size_t)NN}, TN, TK, Bits);
    for (size_t p = 0; p < nbytes * 8; ++p) if ((fold[p/8] >> (p%8)) & 1) id[p] |= (1 << b);
  }
  auto chain = chain_map<1, 32, 128, 128, 32, 32>(true);
  int bad = 0;
  for (int row = 0; row < Ng; ++row) for (int wd = 0; wd < W_ROW; ++wd) for (int j = 0; j < CPW; ++j) {
    const size_t idx = ((size_t)row * W_ROW + wd) * CPW + j;
    if (chain[idx].n != id[idx] / KK || chain[idx].k != id[idx] % KK) ++bad;
  }
  printf("      chain vs the real offline: %d / %d mismatch  %s\n\n", bad, Ng * W_ROW * CPW,
         bad ? "<-- STOP, do not trust anything below" : "<-- REGRESSION PASSES");
  return bad == 0;
}

// ---------------- step 2/3: the target, its structure, and a verified closed form
template <int Bits, int TM, int TN, int TK, int WM, int WN>
static void emit(const char* tag) {
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  constexpr int Ng = TN / F, CPW = 32 / Bits, W_ROW = 8, WPK = TK * Bits / 32;
  printf("  %s: int%d (%d,%d,%d) warp %dx%d | F=%d Ng=%d WPK=%d\n", tag, Bits, TM, TN, TK, WM, WN, F, Ng, WPK);
  auto phys = chain_map<Bits, TM, TN, TK, WM, WN>(true);
  if (phys.empty()) return;

  std::set<int> seen;
  int holes = 0;
  for (auto& c : phys) { if (c.n < 0) ++holes; else seen.insert(c.n * TK + c.k); }
  printf("      bijection: %zu distinct (n,k) over %zu positions, %d holes -> %s\n",
         seen.size(), phys.size(), holes, (seen.size() == phys.size() && !holes) ? "yes" : "NO");

  std::map<int,int> hist;
  for (int row = 0; row < Ng; ++row) for (int wd = 0; wd < W_ROW; ++wd) {
    std::set<int> ns;
    for (int j = 0; j < CPW; ++j) ns.insert(phys[((size_t)row*W_ROW+wd)*CPW+j].n);
    hist[(int)ns.size()]++;
  }
  printf("      columns per 32-bit word: ");
  for (auto& [c, n] : hist) printf("%d cols x %d words   ", c, n);
  printf("%s\n", hist.size() == 1 && hist.begin()->first == 1
         ? "(a whole-uint32 packer suffices)" : "(needs a BIT-granular packer)");

  printf("      row 0: ");
  for (int wd = 0; wd < W_ROW; ++wd) {
    printf("w%d{", wd);
    std::set<int> ns; for (int j = 0; j < CPW; ++j) ns.insert(phys[((size_t)0*W_ROW+wd)*CPW+j].n);
    for (int n : ns) printf("n%d,", n);
    printf("} ");
  }
  printf("\n      row 0 word 0, k per bit: ");
  for (int j = 0; j < 16; ++j) printf("%d ", phys[j].k);
  printf("...\n");

  // FIT a GENERAL GF(2)-affine form, rather than guessing a shape. Index bits are (row, wd, j) concatenated;
  // the value is n*TK + k. If the map is affine, its image on the basis IS the exact closed form, and it is then
  // verified over every position -- a fit that is not verified is a guess.
  const int RB = 32 - __builtin_clz(Ng) - 1, WB = 3, JB = 32 - __builtin_clz(CPW) - 1;   // bit counts
  auto pack = [&](int row, int wd, int j) { return (row) | (wd << RB) | (j << (RB + WB)); };
  auto at   = [&](int x) {
    const int row = x & (Ng - 1), wd = (x >> RB) & 7, j = (x >> (RB + WB)) & (CPW - 1);
    const Cell& c = phys[((size_t)row * W_ROW + wd) * CPW + j];
    return c.n * TK + c.k;
  };
  const int NBITS = RB + WB + JB, NX = 1 << NBITS;
  const int f0 = at(0);
  bool affine = true;
  for (int x = 0; x < NX && affine; ++x)
    for (int b = 0; b < NBITS && affine; ++b)
      if (at(x ^ (1 << b)) != (at(x) ^ at(1 << b) ^ f0)) affine = false;
  printf("      GF(2)-affine in (row, wd, j): %s%s\n", affine ? "YES" : "no",
         affine ? "  -> the basis below is the exact closed form" : "  -> emit the table, not a formula");
  if (affine) {
    printf("      const term: (n%d,k%d)\n", f0 / TK, f0 % TK);
    const char* names[3] = {"row", "wd ", "j  "};
    const int  counts[3] = {RB, WB, JB};
    int base = 0;
    for (int g = 0; g < 3; ++g) {
      for (int b = 0; b < counts[g]; ++b) {
        const int v = at(1 << (base + b)) ^ f0;
        printf("        %s bit%d (=%3d) -> n += %3d , k += %3d\n", names[g], b, 1 << b, v / TK, v % TK);
      }
      base += counts[g];
    }
    // and verify by reconstructing every entry from the basis alone
    int bad2 = 0;
    for (int x = 0; x < NX; ++x) {
      int acc = f0;
      for (int b = 0; b < NBITS; ++b) if ((x >> b) & 1) acc ^= (at(1 << b) ^ f0);
      if (acc != at(x)) ++bad2;
    }
    printf("      reconstructed from the basis over all %d positions: %d mismatch%s\n", NX, bad2,
           bad2 ? "" : "  <-- CLOSED FORM EXACT");
  }
  printf("\n");
}

int main() {
  printf("L10 -- placement generation, regression first\n\n");
  printf("  == STEP 1: regress the chain on the box-verified int1 F=2 config (TM32/TN128/TK128, WN=32)\n");
  if (!regress()) return 1;
  printf("  == STEP 2: the same config, structure -- must come out single-column per word\n");
  emit<1, 32, 128, 128, 32, 32>("calib ");
  printf("  == STEP 3: the WN=64 target that fixes over-delivery at TK=64\n");
  emit<1, 64, 128,  64, 32, 64>("target");
  return 0;
}
