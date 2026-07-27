// L30 -- an audit prompted by the right question: "don't turn a cute layout into a non-cute one." The scale-fragment
// change did exactly that and was reverted. This file checks the OTHER direction across everything built here: places
// where cute already provides the operation and I hand-rolled it anyway. Two found, both verified below.
//
// (A) THE INVERSE PERMUTATION pi = frag.layout()^-1. Eight files compute it as
//         for (int f = 0; f < NS; ++f) { int o = frag.layout()(f); if (o >= 0 && o < NS) pi[o] = f; }
//     cute has right_inverse(Layout) (layout.hpp:1251). This one matters beyond tidiness: l20_derived_offline.cu is
//     the file scheduled to BECOME the production offline, so a hand-rolled inverse would ship.
//
// (B) MixGemmEmit AS A LAYOUT. Its index() is a closed-form loop, but the formula
//         e = c_{nb-1} + 2*c0 + 8*c1 + 16*c2 + ... + 2*CPW*(v&1) + 4*(v>>1)
//     is a sum of per-bit weights, i.e. EXACTLY a cute Layout over (code bits, vreg bits). Expressed that way it
//     composes with the other layouts in the chain (pi, LogicalTV, partition_B) instead of being a special case that
//     every consumer has to call as a function.
//
// Neither is a regression -- they were never cute to begin with. But both are the same lesson as the reverted change,
// pointing the other way: prefer the algebra cute already has over index arithmetic that happens to work.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l30_should_have_been_cute.cu -o l30 && ./l30
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using namespace cute;
using cutlass::MixGemmEmit;

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

// ---------------- (A) hand-rolled inverse vs cute's right_inverse ----------------
template <int TN, int TK, int WM, int WN>
static bool check_inverse(const char* tag) {
  constexpr int WOM = 32/WM, WON = TN/WN;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));
  std::vector<int> hand(NS, -1);
  for (int f = 0; f < NS; ++f) { const int o = frag.layout()(f); if (o >= 0 && o < NS) hand[o] = f; }
  auto inv = right_inverse(frag.layout());
  long bad = 0;
  for (int o = 0; o < NS; ++o) {
    const int c = (o < int(size(inv))) ? int(inv(o)) : -1;
    if (c != hand[o]) ++bad;
  }
  printf("  %-26s TN=%3d TK=%3d w%dx%-3d | NS=%3d  right_inverse size %3d | %ld / %d differ %s\n",
         tag, TN, TK, WM, WN, NS, int(size(inv)), bad, NS,
         bad ? "<-- NOT a drop-in" : "<-- right_inverse IS the hand-rolled loop");
  return bad == 0;
}

// ---------------- (B) MixGemmEmit as a cute Layout ----------------
// e = sum of per-bit weights, so: shape ((2,...,2) over nb code bits, (2,2) over vreg bits)
//     code bit 0 -> 2 ; code bit i (1..nb-2) -> 4<<i ; code bit nb-1 (the half2 lo/hi bit) -> 1
//     vreg bit 0 -> 2*CodesPerVreg ; vreg bit 1 -> 4
template <int Bits> static bool check_layout(const char* tag);
template <> bool check_layout<4>(const char* tag) {
  auto L = make_layout(make_shape (make_shape(_2{},_2{},_2{}), make_shape(_2{},_2{})),
                       make_stride(make_stride(_2{},_8{},_1{}), make_stride(_16{},_4{})));
  long bad = 0;
  for (int v = 0; v < 4; ++v) for (int c = 0; c < 8; ++c)
    if (int(L(make_coord(make_coord(c&1,(c>>1)&1,(c>>2)&1), make_coord(v&1,(v>>1)&1)))) != MixGemmEmit<4>::index(c,v)) ++bad;
  printf("  %-26s int4 | shape ((2,2,2),(2,2)) stride ((2,8,1),(16,4)) | %ld / 32 differ %s\n",
         tag, bad, bad ? "" : "<-- MixGemmEmit IS a cute Layout");
  return bad == 0;
}
template <> bool check_layout<2>(const char* tag) {
  auto L = make_layout(make_shape (make_shape(_2{},_2{},_2{},_2{}), make_shape(_2{},_2{})),
                       make_stride(make_stride(_2{},_8{},_16{},_1{}), make_stride(_32{},_4{})));
  long bad = 0;
  for (int v = 0; v < 4; ++v) for (int c = 0; c < 16; ++c)
    if (int(L(make_coord(make_coord(c&1,(c>>1)&1,(c>>2)&1,(c>>3)&1), make_coord(v&1,(v>>1)&1)))) != MixGemmEmit<2>::index(c,v)) ++bad;
  printf("  %-26s int2 | shape ((2,2,2,2),(2,2)) stride ((2,8,16,1),(32,4)) | %ld / 64 differ %s\n",
         tag, bad, bad ? "" : "<-- MixGemmEmit IS a cute Layout");
  return bad == 0;
}
template <> bool check_layout<1>(const char* tag) {
  auto L = make_layout(make_shape (make_shape(_2{},_2{},_2{},_2{},_2{}), make_shape(_2{},_2{})),
                       make_stride(make_stride(_2{},_8{},_16{},_32{},_1{}), make_stride(_64{},_4{})));
  long bad = 0;
  for (int v = 0; v < 4; ++v) for (int c = 0; c < 32; ++c)
    if (int(L(make_coord(make_coord(c&1,(c>>1)&1,(c>>2)&1,(c>>3)&1,(c>>4)&1), make_coord(v&1,(v>>1)&1)))) != MixGemmEmit<1>::index(c,v)) ++bad;
  printf("  %-26s int1 | shape ((2,2,2,2,2),(2,2)) stride ((2,8,16,32,1),(64,4)) | %ld / 128 differ %s\n",
         tag, bad, bad ? "" : "<-- MixGemmEmit IS a cute Layout");
  return bad == 0;
}

int main() {
  printf("L30 -- where cute already had the operation and I hand-rolled it\n\n");
  bool ok = true;
  printf("  == (A) pi = frag.layout()^-1, hand-rolled in 8 files incl. the future production offline\n");
  ok &= check_inverse<128, 64, 32, 64>("l20/l19/l13/l16/... loop");
  ok &= check_inverse<128,128, 32, 32>("same, TK=128");
  ok &= check_inverse< 64, 64, 32, 32>("same, int4 production shape");
  printf("\n  == (B) MixGemmEmit's closed form is a sum of per-bit weights, i.e. a Layout\n");
  ok &= check_layout<4>("closed form vs Layout");
  ok &= check_layout<2>("closed form vs Layout");
  ok &= check_layout<1>("closed form vs Layout");
  printf("\n  Neither was ever cute, so neither is a regression -- but both are the reverted change's lesson pointing\n");
  printf("  the other way: prefer the algebra cute already has over index arithmetic that happens to work.\n");
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
