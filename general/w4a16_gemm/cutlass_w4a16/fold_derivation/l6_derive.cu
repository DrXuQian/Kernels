// L6 -- compose L2 o L3 o L4 and invert: the offline placement table, DERIVED.
//
// For every physical bit position in a B tile, which logical (n, k) must sit there?
//
//     (lane, v)      --L2-->  (row_local, run-word)     row_local = (v/2)*8 + lane/4, word = (v%2)*4 + lane%4
//     row            =        16*warp_n + row_local     (forced by counting: warpOnN groups of 16 rows == Ng)
//     (bit j, v)     --L3-->  converter element e
//     e              --L4-->  (n, k)                    partition_B's linear index, on the LOGICAL (TN,TK) view
//
// Both configs of interest are `tight` (delivery == slots == 128), so one converter call covers the whole
// fragment and e is the flat fragment index with no k_block offset. That is why this composition is short.
//
// THREE INDEPENDENT CHECKS, none of which is a restatement of the chain:
//   1. BIJECTION. Ng*8*32 physical bits must map one-to-one onto TN*TK logical codes. Any error in any leg shows
//      up as a collision or a hole.
//   2. WARP-M CONSISTENCY. B is duplicated across the M warps, so two threads differing only in warp_m must be
//      handed the same physical data AND demand the same (n,k). A contradiction here would mean row_local or the
//      warp_n extraction is wrong.
//   3. THE cols_per_word PREDICTION. l5_slots measured cols_per_word = WN/32 by counting. This map says WHICH
//      columns share a word. At WN=32 every word must come out single-column -- which is exactly the property
//      the shipped whole-uint32 packer relies on, so reproducing it validates the chain against a box-verified
//      artefact without re-implementing the 5-step relayout.
//
// RESULT: checks 1 and 2 PASS, check 3 FAILS -- and the failure is a real falsification, not a bad prediction.
//
// The derived map says one 32-bit word carries F logical columns, alternating every TWO bits between n and
// n + Ng. But the shipped packer provably carries ONE column per word: nfold_regroup_gmem is
// dst[dst_w] = src[src_w] with src_w = n_log*WPN + ktile*WPK + w -- a whole-word move indexed by a single n_log --
// and the 5-step relayout ahead of it never mixes columns inside a word either (subbyte_transpose twice preserves
// the n axis, permute_B_rows permutes K, interleave-256 relocates whole uint32 vecs). And that packer measures
// bad=0 on ppu001 at (32,128,128), exactly the config this file says needs two columns per word.
//
// So a leg of the composition is wrong even though each leg was checked in isolation. Where the two columns come
// from is worth recording, because it localises the suspect: it is L3 o L4, NOT L2. Within one vreg, j's bit 1
// enters the element index at weight 8 (the converter's 8*b1 term), element bit 3 selects mma_n, and one MMA_N
// step advances n by size(PermN) = 64 = Ng. L2 only decides WHICH (row, word) a vreg is -- it cannot change how a
// vreg's own 32 bits spread over columns.
//
// Note also that this reproduces, independently, the placement recorded in nfold_column_pairs_ppu's comment
// ("(crumb>>1)&1 == 1 -> logical column n + TN/F", from the old scratchpad/derive2.cu) -- and THAT function was
// measured wrong on the box and superseded. Arriving at the same answer twice by two routes says the shared
// premise is what is broken.
//
// Things ruled out while localising it:
//   * the bad=0 is not weak. test_fold_int2 sets M = K and A = full identity, so `bad` covers every (n,k) of the
//     tile -- 65536/65536 at the default 256x256. The shipped placement really is right.
//   * MmaPermK really is TileShape.K on the fold path (ppu_mma_builder.inl:587), so the fragment shape used here
//     is the one the kernel builds.
//   * the whole-word argument survives the interleave-256 objection: even though nfold_regroup_gmem indexes its
//     input as if it were n-major (which the interleaved buffer is not), interleave-256 relocates whole uint32
//     vecs, so every word of the 5-step output holds 32 codes of ONE column no matter where it lands. A
//     whole-word regroup of single-column words yields single-column words.
//   * L2 is not implicated: for a FIXED vreg, j alone drives the two columns, and L2 only says which (row, word)
//     a vreg is.
// What is left is the JOIN between L3 and L4 -- that converter output element e lands at fragment flat index e.
// That is the one step I argued (from cvt_out = make_tensor(tCrB_mma(...).data(), cvt_in.layout()) being compact)
// rather than measured, and it is now the prime suspect.
//
// NEXT, and it is the calibration that has been deferred twice: model the shipped int1 pipeline
// (add_bias_and_interleave_int1s + permute_B_rows + interleave-256 + nfold_regroup_gmem) to get the ground-truth
// (n,k) -> (row, word, bit) map, then solve for whichever leg has to change to reproduce it. pipe_truth.py does
// exactly this for int2 and is the template.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l6_derive.cu -o l6 && ./l6
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <map>
#include <set>
#include <vector>
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

// L2, as the validated layout (l2l3_layouts.cu): word_in_cube = 64*(v/2) + 8*(lane/4) + 4*(v%2) + lane%4
static inline void l2(int lane, int v, int& row_local, int& word) {
  row_local = (v / 2) * 8 + lane / 4;
  word      = (v % 2) * 4 + lane % 4;
}
// L3, the int1 converter as the validated layout: e = 64*v0 + 4*v1 + 2*b0 + 8*b1 + 16*b2 + 32*c + d
static inline int l3_int1(int j, int v) {
  const int b0 = j & 1, b1 = (j >> 1) & 1, b2 = (j >> 2) & 1, c = (j >> 3) & 1, d = (j >> 4) & 1;
  return 64 * (v % 2) + 4 * (v / 2) + 2 * b0 + 8 * b1 + 16 * b2 + 32 * c + d;
}

struct Cell { int n, k; };

template <int Bits, int TM, int TN, int TK, int WM, int WN>
bool derive(const char* tag, bool dump) {
  constexpr int WOM = TM / WM, WON = TN / WN;
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  constexpr int Ng = TN / F;
  constexpr int CPW = 32 / Bits;                       // codes per 32-bit word
  constexpr int W_ROW = 8;                             // a 32B run is 8 uint32

  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  const int slots = int(size(Mma{}.get_thread_slice(0).partition_B(
      make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})))));
  const int delivery = 16 * 8 / Bits;

  printf("  %s: int%d (%d,%d,%d) warp %dx%d | F=%d Ng=%d | slots=%d deliv=%d %s\n",
         tag, Bits, TM, TN, TK, WM, WN, F, Ng, slots, delivery,
         slots == delivery ? "(tight -> one converter call)" : "(NOT tight -- L6 needs k_block bookkeeping)");
  if (slots != delivery) { printf("      skipped\n\n"); return true; }

  // physical (row, word, bit) -> logical (n,k)
  std::map<long, Cell> phys;                            // key = (row*W_ROW + word)*CPW + bit
  std::map<long, long> seen_logical;                    // (n,k) -> physical key, for the bijection check
  int collisions = 0, warpm_conflicts = 0;
  std::map<long, Cell> by_thread_check;

  const int nthreads = 32 * WOM * WON;
  for (int t = 0; t < nthreads; ++t) {
    const int lane = t % 32, w = t / 32;
    const int warp_m = w % WOM, warp_n = w / WOM;       // Layout<Shape<WOM,WON,_1>> is m-major
    auto part = Mma{}.get_thread_slice(t).partition_B(
        make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    for (int v = 0; v < 4; ++v) {
      int rl, wd; l2(lane, v, rl, wd);
      const int row = 16 * warp_n + rl;
      for (int j = 0; j < CPW; ++j) {
        const int e = l3_int1(j, v);
        auto c = part(e);
        const Cell cell{int(get<0>(c)), int(get<1>(c))};
        const long key = ((long)row * W_ROW + wd) * CPW + j;
        auto it = phys.find(key);
        if (it == phys.end()) {
          phys[key] = cell;
        } else if (it->second.n != cell.n || it->second.k != cell.k) {
          // two threads read the same physical bit but want different logical codes
          if (warp_m != 0) ++warpm_conflicts; else ++collisions;
        }
        const long lkey = (long)cell.n * TK + cell.k;
        auto jt = seen_logical.find(lkey);
        if (jt == seen_logical.end()) seen_logical[lkey] = key;
        else if (jt->second != key) ++collisions;
      }
    }
  }

  const long want = (long)Ng * W_ROW * CPW;
  printf("      physical positions filled %zu / %ld    distinct logical (n,k) reached %zu / %d\n",
         phys.size(), want, seen_logical.size(), TN * TK);
  printf("      CHECK 1 bijection      : %s\n",
         (phys.size() == (size_t)want && seen_logical.size() == (size_t)(TN * TK) && collisions == 0)
             ? "PASS" : "FAIL");
  printf("      CHECK 2 warp-M agree   : %s (%d contradictions)\n", warpm_conflicts == 0 ? "PASS" : "FAIL",
         warpm_conflicts);

  // CHECK 3: how many distinct n share one 32-bit word?
  std::map<int, int> hist;
  int maxcols = 0;
  for (int row = 0; row < Ng; ++row)
    for (int wd = 0; wd < W_ROW; ++wd) {
      std::set<int> ns;
      for (int j = 0; j < CPW; ++j) {
        auto it = phys.find(((long)row * W_ROW + wd) * CPW + j);
        if (it != phys.end()) ns.insert(it->second.n);
      }
      hist[(int)ns.size()]++;
      if ((int)ns.size() > maxcols) maxcols = (int)ns.size();
    }
  printf("      CHECK 3 cols per word  : ");
  for (auto& [c, cnt] : hist) printf("%d cols x %d words   ", c, cnt);
  printf("| shipped packer can only do 1 -> %s\n", maxcols == 1 ? "consistent" : "CONTRADICTION");

  if (dump) {
    printf("      row 0, its 8 words -- (n,k) of bits 0..7, and the column set of each word:\n");
    for (int wd = 0; wd < W_ROW; ++wd) {
      std::set<int> ns; std::set<int> ks;
      for (int j = 0; j < CPW; ++j) {
        auto it = phys.find(((long)0 * W_ROW + wd) * CPW + j);
        if (it != phys.end()) { ns.insert(it->second.n); ks.insert(it->second.k); }
      }
      printf("        word %d: n={", wd);
      for (int n : ns) printf("%d,", n);
      printf("} k has %zu values | bits0-7 -> ", ks.size());
      for (int j = 0; j < 8; ++j) {
        auto it = phys.find(((long)0 * W_ROW + wd) * CPW + j);
        if (it != phys.end()) printf("(%d,%d)", it->second.n, it->second.k);
      }
      printf("\n");
    }
  }
  printf("\n");
  return phys.size() == (size_t)want && seen_logical.size() == (size_t)(TN * TK)
         && collisions == 0 && warpm_conflicts == 0;   // check 3 is reported, not required: it is the open falsification
}

int main() {
  printf("L6 -- derived offline placement, and three checks on the composition\n\n");
  bool ok = true;
  printf("  == CALIBRATION: the box-verified F=2 winner. cols_per_word must come out 1, which is exactly what\n"
         "     the shipped whole-uint32 packer can express.\n");
  ok &= derive<1, 32, 128, 128, 32, 32>("calib", true);
  printf("  == TARGET: int1 at TK=64, at the WN=64 that fixes over-delivery.\n");
  ok &= derive<1, 64, 128,  64, 32, 64>("target", true);
  printf("  all checks: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
