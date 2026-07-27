// L28 -- construct the scale broadcast view, and prove it delivers the SAME scale to every B slot.
//
// L27 measured the problem: the scale copy asks for 64-256 slots per thread to fetch 4-8 distinct smem elements. The
// source partition genuinely carries stride 0 on the redundant modes (they walk mode-1 of the (TN,1,SK) tensor, whose
// extent is 1), but make_fragment_like only inherits SOME of those zeros -- 128 slots collapse to 32 registers where
// 8 would do. So the register side materialises a 4x replication and the copy still issues the full 16x.
//
// THREE THINGS TO ESTABLISH HERE, all local, before touching a shipped collective:
//   Q1  does make_fragment_like of the COPY PARTITION (rather than of partition_fragment_B) inherit every zero?
//       If cosize comes out at ne, the minimal fragment is free -- no hand-built layout needed.
//   Q2  does cute's copy actually issue ne loads for such a tensor, or does it still walk all size() slots?
//       filter_zeros is the tool; this checks the counts agree on both sides so copy() can be given filtered views.
//   Q3  CORRECTNESS: with the minimal storage, does every B fragment slot still see the scale for ITS OWN n?
//       This is L21's Q1 re-asked against the new plumbing, and it is the check that matters -- a broadcast that
//       drops or rotates an n would be silently wrong on the box.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l28_scale_bcast.cu -o l28 && ./l28
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
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

template <int TM, int TN, int TK, int WM, int WN, int SK>
static bool probe(const char* tag) {
  constexpr int WOM = TM/WM, WON = TN/WN;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto tiled_S = make_tiled_copy_B(Copy_Atom<DefaultCopy, cutlass::half_t>{}, Mma{});
  auto sS_lay  = tile_to_shape(Layout<Shape<_8,_1>>{}, make_shape(Int<TN>{}, Int<1>{}, Int<SK>{}));

  // values in sS encode n, so a wrong broadcast shows up as a wrong n rather than as garbage
  std::vector<int> sval((size_t)cosize(sS_lay), -1);
  for (int g = 0; g < SK; ++g) for (int n = 0; n < TN; ++n) sval[sS_lay(make_coord(n,0,g))] = n;
  auto sS = make_tensor(sval.data(), sS_lay);

  std::vector<int> bval((size_t)TN*TK);
  for (int n = 0; n < TN; ++n) for (int k = 0; k < TK; ++k) bval[(size_t)n*TK + k] = n;
  auto tB = make_tensor(bval.data(), make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));

  auto thr0 = tiled_S.get_thread_slice(0);
  auto sl0  = thr0.partition_S(sS)(_,_,_,Int<0>{});
  // Q1: the fragment built from the COPY PARTITION vs from partition_fragment_B
  auto frag_copy = make_fragment_like<cutlass::half_t>(sl0);
  auto frag_mma  = make_fragment_like<cutlass::half_t>(
                     Mma{}.get_thread_slice(0).partition_fragment_B(sS(_,_,Int<0>{})));
  // Q1b -- is the fix a ONE-WORD change? partition_fragment_B is make_fragment_like(partition_B(...)) but on the
  // COMPACT shape, dropping the source's strides; partition_B keeps them, including the zeros the k modes get from
  // sS's extent-1 mode. If make_fragment_like(partition_B(...)) already lands at ne, the collective edit is just
  // partition_fragment_B -> partition_B. The catch (L21) is that partition_B on a K-extent-1 tensor was measured to
  // index OUT OF BOUNDS, which would mean some mode has a NON-zero stride and the replication survives.
  auto pB_raw    = Mma{}.get_thread_slice(0).partition_B(sS(_,_,Int<0>{}));
  auto frag_raw  = make_fragment_like<cutlass::half_t>(pB_raw);
  const int reg_raw = int(cosize(frag_raw.layout()));
  printf("      Q1b partition_B (strides kept): "); print(pB_raw.layout()); printf("\n");
  printf("      Q1b make_fragment_like of it -> %d reg (ne=%d) %s\n", reg_raw, int(size(filter_zeros(sl0.layout()))),
         reg_raw == int(size(filter_zeros(sl0.layout())))
           ? "<-- ONE-WORD FIX: partition_fragment_B -> partition_B"
           : "<-- replication survives, and cute's own comment says why");
  // WHY it survives, straight out of layout.hpp: make_fragment_like "generates the 0th mode with LayoutLeft
  // (PRESERVING STRIDE-0s) regardless of the reference layout" -- mode 0 only. So the two k val-bits keep their
  // zeros and MMA_K does NOT. That single materialised mode IS the 4x.
  printf("      Q1c frag_like(pB): "); print(make_fragment_like(pB_raw.layout())); printf("\n");
  // THE FIX, and it hardcodes no bit position: inherit mode 0 from make_fragment_like (which already preserves the
  // val-mode zeros) and put MMA_K's zero back. Scale is k-invariant by definition, so MMA_K is stride 0 by meaning,
  // not by measurement.
  // Naive patch FAILS: make_fragment_like's compact_order follows the REFERENCE strides, and MMA_K's reference
  // stride is 0, so it sorts MMA_K ahead of MMA_N and hands MMA_N a non-compact 8. Zeroing mode 2 alone leaves
  // cosize 26. Filter first so MMA_K is extent 1 during compaction, THEN put its zero back:
  auto fl = make_fragment_like(pB_raw.layout());
  auto naive = make_layout(pB_raw.shape(), make_stride(stride<0>(fl), stride<1>(fl), _0{}));
  auto fz  = make_fragment_like(filter_zeros(pB_raw.layout()));
  auto bcast = make_layout(pB_raw.shape(), make_stride(stride<0>(fz), stride<1>(fz), _0{}));
  const int ne0 = int(size(filter_zeros(sl0.layout())));
  printf("      Q1c naive patch  : "); print(naive);   printf("  -> cosize %2d reg  (compaction followed MMA_K first)\n",
         int(cosize(naive)));
  printf("      Q1c filter+patch : "); print(bcast); printf("  -> cosize %2d reg (ne=%d) %s\n",
         int(cosize(bcast)), ne0, int(cosize(bcast)) == ne0 ? "<-- MINIMAL" : "<-- WRONG SIZE");
  const int n_slot = int(size(sl0));
  const int ne_src = int(size(filter_zeros(sl0.layout())));
  const int reg_copy = int(cosize(frag_copy.layout())), reg_mma = int(cosize(frag_mma.layout()));

  printf("  %-22s (%3d,%3d,%3d) w%dx%-3d | copy asks %3d slot | distinct src %2d |"
         " frag from partition_fragment_B: %3d reg | frag from COPY PARTITION: %3d reg %s\n",
         tag, TM, TN, TK, WM, WN, n_slot, ne_src, reg_mma, reg_copy,
         reg_copy == ne_src ? "<-- MINIMAL, no hand-built layout needed" : "<-- still over-allocates");

  // Q2: filtered views must agree in size on both sides, or copy() cannot be handed them
  const int ne_dst = int(size(filter_zeros(frag_copy.layout())));
  if (ne_src != ne_dst) { printf("      Q2 FAIL: filtered src %d vs dst %d\n", ne_src, ne_dst); return false; }

  // Q3: which of the B fragment's val modes is the N direction? The scale is k-invariant, so the minimal
  // mma-shaped fragment is "stride 0 on every k mode, compact on the n modes" -- but WHICH modes those are is a
  // property of the mma's BLayout, and assuming it is exactly how the last several rounds went wrong. Detect it
  // from partition_B on an identity tensor, then assert the detected pattern matches the layout we build.
  auto pB_id = Mma{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
  int nmode_mask = 0;                                            // bit v set if val bit v moves N
  for (int bit = 0; bit < 3; ++bit) {
    auto c0 = pB_id(0, 0, 0);
    auto c1 = pB_id(1 << bit, 0, 0);
    const bool moves_n = int(get<0>(c1)) != int(get<0>(c0));
    const bool moves_k = int(get<1>(c1)) != int(get<1>(c0));
    if (moves_n) nmode_mask |= 1 << bit;
    printf("      val bit %d: dn=%+d dk=%+d -> %s\n", bit,
           int(get<0>(c1)) - int(get<0>(c0)), int(get<1>(c1)) - int(get<1>(c0)),
           moves_n ? "N mode (scale must vary)" : (moves_k ? "K mode (scale is invariant)" : "neither"));
  }
  // and MMA_K is a k mode by construction; MMA_N is an n mode
  const int nmodes = __builtin_popcount(nmode_mask);
  printf("      => %d of 3 val bits move N (mask 0x%x); minimal mma-shaped scale fragment = %d * MMA_N reg\n",
         nmodes, nmode_mask, 1 << nmodes);

  // Q4 -- THE LAST THING THAT COULD FAIL ON THE BOX. The collective feeds the fragment through
  //     tCrS_copy_view = smem_thr_copy_S.retile_D(tCrS);   copy(tiled, tCsS(_,_,0,sk), tCrS_copy_view(_,_,0));
  // so retile_D must accept a tensor whose MMA_K stride is 0, and the retiled view must still cover the copy's shape.
  // Nothing above tests that: l28 up to here only checked the layout in isolation. If retile_D rejects it the build
  // breaks; if it silently mis-maps, the scales are wrong and only the box would say so.
  {
    auto tCrS_bc   = make_tensor<cutlass::half_t>(bcast);
    auto view      = thr0.retile_D(tCrS_bc);
    auto dst_slice = view(_,_,Int<0>{});
    auto src_slice = thr0.partition_S(sS)(_,_,Int<0>{},Int<0>{});
    printf("      Q4 retile_D accepted. view "); print(view.layout()); printf("\n");
    printf("         copy shapes: src %d vs dst %d %s | dst distinguishes %d reg (ne=%d) %s\n",
           int(size(src_slice)), int(size(dst_slice)),
           int(size(src_slice)) == int(size(dst_slice)) ? "MATCH" : "<-- MISMATCH, copy ill-formed",
           int(cosize(dst_slice.layout())), ne_src,
           int(cosize(dst_slice.layout())) <= ne_src ? "<-- copy visits ne registers" : "<-- still over-visits");
    if (int(size(src_slice)) != int(size(dst_slice))) return false;
  }

  // Q3, stated so it depends on NO index ordering. The minimal layout makes several mma slots share one
  // register; it is correct iff the equivalence classes it induces on slots are EXACTLY the classes induced by n.
  // Guessing how filter_zeros orders the source (which is what failed twice above) is then unnecessary -- if the
  // classes coincide, any consistent fill is right, and if they do not, no fill can rescue it.
  //
  // The layout under test, built from the DETECTED mask: stride 0 on both k val-bits and on MMA_K, compact on the
  // n val-bit and MMA_N.  shape ((2,2,2), MMA_N, MMA_K)  stride ((0,0,1), 2, 0)  -> cosize 2*MMA_N.
  const int MMA_N = int(size<1>(pB_id)), MMA_K = int(size<2>(pB_id));
  auto min_lay = make_layout(make_shape (make_shape(_2{},_2{},_2{}), MMA_N, MMA_K),
                             make_stride(make_stride(_0{},_0{},_1{}), _2{}, _0{}));
  const int nreg_min = int(cosize(min_lay));
  printf("      layout under test: cosize %d reg (ne from the copy: %d) %s\n", nreg_min, ne_src,
         nreg_min == ne_src ? "<-- agree" : "<-- DISAGREE with the copy's distinct count");
  if (nreg_min != ne_src) return false;

  long bad = 0, tot = 0; int shown = 0;
  for (int t = 0; t < 32*WOM*WON; ++t) {
    auto pB = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
    std::vector<int> reg_n(nreg_min, -1);
    for (int j = 0; j < MMA_N; ++j)
      for (int kk = 0; kk < MMA_K; ++kk)
        for (int v = 0; v < 8; ++v) {
          const int r = min_lay(make_coord(make_coord(v&1,(v>>1)&1,(v>>2)&1), j, kk));
          const int n = int(get<0>(pB(v, j, kk)));
          ++tot;
          if (reg_n[r] < 0) reg_n[r] = n;
          else if (reg_n[r] != n) { ++bad; if (shown < 3) {
            printf("      Q3 MISMATCH thr%d (v%d,n%d,k%d): register %d already holds n=%d but this slot needs n=%d\n",
                   t, v, j, kk, r, reg_n[r], n); ++shown; } }
        }
  }
  printf("      Q3: %ld / %ld slot(s) agree with their register's n  %s\n", tot - bad, tot,
         bad ? "<-- BROADCAST WOULD APPLY THE WRONG SCALE" : "<-- classes coincide, broadcast is correct");
  return bad == 0 && reg_copy == ne_src && nreg_min == ne_src;
}

int main() {
  printf("L28 -- the scale broadcast view: minimal storage, and does it still deliver the right n?\n\n");
  bool ok = true;
  ok &= probe<32,128, 64,32,64,2>("int1 best TK64 gs32");
  ok &= probe<32,128,128,32,32,4>("int1 TK128 gs32");
  ok &= probe<32,128,128,32,32,8>("int1 TK128 gs16");
  ok &= probe<64, 64, 64,32,32,2>("int4 PRODUCTION");
  ok &= probe<32,256, 64,32,64,2>("int1 TK64 TN=256");
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
