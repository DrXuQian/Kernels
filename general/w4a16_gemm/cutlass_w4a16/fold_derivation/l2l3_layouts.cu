// L2 + L3 -- the two hand-written index chains, expressed as cute Layouts and checked against the real thing.
//
// These are the maps that every fold bug has been hiding in, because each is currently open-coded in a different
// file. Both turn out to be plain linear layouts, so cute can hold them and they compose.
//
//   L2  swzl delivery :  (lane, vreg)  ->  logical 32-bit word index inside the cube
//   L3  int1 converter:  (bit, vreg)   ->  fp16 element index inside the 128-element output
//   L4  mma fragment  :  fp16 element index  ->  logical (n, k)
//
// Each is validated against an INDEPENDENT brute-force model, not against a restatement of itself:
//   L2 vs ppu_tsm_ld_swzl_sim's SWAP=true arithmetic, swizzle terms stripped (they cancel with the AIU write).
//   L3 vs the instruction semantics of the 16 _E() macros -- lop3(reg, MASK, 0x64006400, 0xEA) = (reg&MASK)|magic,
//      then fma(2^-b, -2^(10-b)), which yields exactly the masked bit. So the mask constants alone say which
//      input bit reaches which output fp16, with no appeal to my algebra.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l2l3_layouts.cu -o l2l3 && ./l2l3
#include "cute/tensor.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <set>
using namespace cute;

// ---------------------------------------------------------------------- L2: swzl delivery
// leg1 established, for SWAP=true and with the write/read swizzles cancelled:
//     row      = (v/2)*8 + lane/4        run-word = (v%2)*4 + lane%4
//     word     = row*8 + run-word        (a row's 32B run is 8 uint32)
// Substituting: word = 64*(v/2) + 8*(lane/4) + 4*(v%2) + (lane%4). Linear in the bits of (lane, v):
using SwzlDelivery = Layout<Shape <Shape<_4, _8>, Shape<_2, _2>>,     // ((lane%4, lane/4), (v%2, v/2))
                            Stride<Stride<_1, _8>, Stride<_4,_64>>>;

static int swzl_ref(int lane, int v) {                 // independent: the sim's own arithmetic
  int lane_row_idx  = lane / 4;                        // coord_h = 0
  int vreg_row_idx  = (v / 2) * 8 + lane_row_idx;      // SWAP=true
  int vreg_line_idx = vreg_row_idx / 4;
  int vreg_vec_idx  = (vreg_row_idx % 4) * 2 + (v % 2);// SWAP=true
  return vreg_line_idx * 32 + vreg_vec_idx * 4 + (lane % 4);   // swizzle terms dropped
}

// ---------------------------------------------------------------------- L3: int1 converter emission
// The 16 _E() calls, verbatim from fast_numeric_conversion_for_mix_gemm.h. Each is
//     _E(h2[base + H2OFF], SRC, MASK, ...)   with MASK = (1<<m) | (1<<(m+16))
// and SRC being either `reg` (level 0) or `reg >> 8` (level 1). lop3 keeps the masked bits and the fma turns
// each into its plain value, so:
//     fp16 element 2*(base+H2OFF) + 0  <-  reg bit (m + 8*level)
//     fp16 element 2*(base+H2OFF) + 1  <-  reg bit (m + 8*level + 16)
struct Emit { int h2off, level, m; };
static const Emit kEmit[16] = {
  {0,0,0},{1,0,1},{4,0,2},{5,0,3},{8,0,4},{9,0,5},{12,0,6},{13,0,7},          // from reg
  {16,1,0},{17,1,1},{20,1,2},{21,1,3},{24,1,4},{25,1,5},{28,1,6},{29,1,7},    // from reg >> 8
};
static int conv_ref(int bit, int v) {                  // independent: mask constants -> element index
  const int base = 32 * (v & 1) + 2 * (v >= 2);
  for (int e = 0; e < 16; ++e) {
    const Emit& x = kEmit[e];
    if (bit == x.m + 8 * x.level)      return 2 * (base + x.h2off) + 0;
    if (bit == x.m + 8 * x.level + 16) return 2 * (base + x.h2off) + 1;
  }
  return -1;
}
// Algebraically: off8[b] = b0 + 4*b1 + 8*b2 (checked below), so with bit = b0 + 2*b1 + 4*b2 + 8*c + 16*d,
//     element = 64*v0 + 4*v1 + 2*b0 + 8*b1 + 16*b2 + 32*c + d
using Int1Emit = Layout<Shape <Shape<_2,_2,_2,_2,_2>, Shape<_2,_2>>,   // ((b0,b1,b2,c,d), (v%2, v/2))
                        Stride<Stride<_2,_8,_16,_32,_1>, Stride<_64,_4>>>;

// ---------------------------------------------------------------------- L4: fragment element -> (n, k)
// The converter writes through
//     cvt_out = make_tensor(tCrB_mma(_,_, k_block*K_ATOM_PER_COPY).data(), cvt_in.layout())
// i.e. CONSECUTIVE elements from a pointer into the register fragment. A register fragment is compact, so
// converter element e lands at fragment flat index  k_block*K_ATOM_PER_COPY*size<0>*size<1> + e. And the flat
// index is exactly the linear index of partition_B, so cute answers L4 directly -- no hand-written chain at all.
struct StubMma {};
namespace cute {
template <> struct MMA_Traits<StubMma> {
  using ValTypeD=float; using ValTypeA=cutlass::half_t; using ValTypeB=cutlass::half_t; using ValTypeC=float;
  using Shape_MNK=Shape<_16,_16,_16>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
}
template <int TN, int TK>
static void l4_report(const char* tag) {
  using Mma = TiledMMA<MMA_Atom<StubMma>, Layout<Shape<_2,_2,_1>>, Tile<_32,_32,Int<TK>>>;
  auto part = Mma{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
  const int slots = int(size(part));
  std::set<std::pair<int,int>> uniq;
  for (int f = 0; f < slots; ++f) uniq.insert({get<0>(part(f)), get<1>(part(f))});
  printf("      %-16s slots=%3d  distinct (n,k)=%3zu  flat 0..7 -> ", tag, slots, uniq.size());
  for (int f = 0; f < 8 && f < slots; ++f) printf("(%d,%d)", get<0>(part(f)), get<1>(part(f)));
  printf("\n");
}

int main() {
  printf("L2 + L3 + L4 -- the hand-written index chains, as cute Layouts\n\n");

  printf("  L2 SwzlDelivery = "); print(SwzlDelivery{}); printf("\n");
  // SINGLE SOURCE OF TRUTH: the same layout now lives in the copy atom's traits as LogicalTV. If these ever
  // disagree, one of the two has been edited alone -- the failure mode that member exists to prevent.
  {
    using CT = Copy_Traits<PPU0010_TSM_LD_SWZL<cutlass::uint1b_t, 64, 256, true, false, 1>>;
    printf("      Copy_Traits<...>::LogicalTV = "); print(typename CT::LogicalTV{});
    printf("   (WordsPerRow=%d, valid=%s)\n", CT::LogicalWordsPerRow, CT::LogicalTVValid ? "yes" : "no");
    int dis = 0;
    for (int lane = 0; lane < 32; ++lane) for (int v = 0; v < 4; ++v) {
      auto crd = make_coord(make_coord(lane % 4, lane / 4), make_coord(v % 2, v / 2), 0);   // slice 0
      auto crd2 = make_coord(make_coord(lane % 4, lane / 4), make_coord(v % 2, v / 2));
      if (SwzlDelivery{}(crd2) != typename CT::LogicalTV{}(crd)) ++dis;
    }
    printf("      local copy vs the traits member: %d / 128 differ -> %s\n", dis, dis ? "DRIFTED" : "same map");
  }
  int bad2 = 0, seen2[128] = {0};
  for (int lane = 0; lane < 32; ++lane)
    for (int v = 0; v < 4; ++v) {
      int got = SwzlDelivery{}(make_coord(make_coord(lane % 4, lane / 4), make_coord(v % 2, v / 2)));
      int exp = swzl_ref(lane, v);
      if (got != exp) { if (bad2 < 4) printf("      MISMATCH lane%d v%d: layout %d vs sim %d\n", lane, v, got, exp); ++bad2; }
      if (got >= 0 && got < 128) seen2[got]++;
    }
  int cov2 = 0; for (int i = 0; i < 128; ++i) cov2 += (seen2[i] == 1);
  printf("      vs ppu_tsm_ld_swzl_sim (SWAP=true, swizzles cancelled): %d mismatches / 128\n", bad2);
  printf("      bijective onto the 128 words of a 16-row cube: %s\n\n", cov2 == 128 ? "yes" : "NO");

  // the off8 claim L3 rests on
  const int off8[8] = {0,1,4,5,8,9,12,13};
  int bado = 0;
  for (int b = 0; b < 8; ++b) if (off8[b] != (b&1) + 4*((b>>1)&1) + 8*((b>>2)&1)) ++bado;
  printf("  off8[b] == b0 + 4*b1 + 8*b2 (what makes the converter linear): %s\n", bado ? "NO" : "yes, all 8");

  printf("  L3 Int1Emit     = "); print(Int1Emit{}); printf("\n");
  int bad3 = 0, seen3[128] = {0};
  for (int v = 0; v < 4; ++v)
    for (int bit = 0; bit < 32; ++bit) {
      int b0 = bit & 1, b1 = (bit >> 1) & 1, b2 = (bit >> 2) & 1, c = (bit >> 3) & 1, d = (bit >> 4) & 1;
      int got = Int1Emit{}(make_coord(make_coord(b0, b1, b2, c, d), make_coord(v % 2, v / 2)));
      int exp = conv_ref(bit, v);
      if (got != exp) { if (bad3 < 4) printf("      MISMATCH v%d bit%2d: layout %d vs converter %d\n", v, bit, got, exp); ++bad3; }
      if (got >= 0 && got < 128) seen3[got]++;
    }
  int cov3 = 0; for (int i = 0; i < 128; ++i) cov3 += (seen3[i] == 1);
  printf("      vs the 16 _E() mask constants: %d mismatches / 128\n", bad3);
  printf("      bijective onto the 128 fp16 the converter emits: %s\n", cov3 == 128 ? "yes" : "NO");

  printf("\n  L4 fragment element -> (n,k): cute's own partition_B, indexed linearly. Nothing to hand-write.\n");
  l4_report<128,128>("int1 F=2 winner");
  l4_report<128, 64>("int1 F=4 target");
  l4_report< 64, 64>("int2 F=2");

  printf("\n  So the whole chain is now three composable maps plus cute:\n"
         "     (lane,v)   --L2-->  physical word in the cube\n"
         "     (bit,v)    --L3-->  converter output element\n"
         "     element    --L4-->  logical (n,k)\n"
         "  and the offline is the INVERSE: place logical (n,k) at the physical (word,bit) that reaches it.\n"
         "  L5 is the composition; the one piece it still needs from the collective is which copy-atom\n"
         "  instance (coord_h / slice) supplies each group of 4 vregs when a fragment takes several.\n");
  return (bad2 || bad3 || bado || cov2 != 128 || cov3 != 128) ? 1 : 0;
}
