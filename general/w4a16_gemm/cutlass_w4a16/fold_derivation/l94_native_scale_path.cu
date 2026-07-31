// L94 -- THE NATIVE SCALE TILE AS A CUTE LAYOUT, derived from the real objects, verified locally. Plan #20 option E.
//
// The design this gates: cp.async the gguf's OWN scale bytes into smem and decode in registers -- no offline widening
// (which would stop the device holding the gguf's bytes), no giving up cp.async (which llama.cpp's loader-side decode
// would cost us, since cp.async cannot do arithmetic between gmem and smem).
//
// THE OBSERVATION THAT MAKES IT CHEAP: a Q4_K superblock's 12 scale bytes are per COLUMN and cover all 8 of that
// column's groups. So a lane that owns column n reads 3 uint32 ONCE per k-tile and has every group's sc and mn in
// registers. The FINE path's per-group s2r read disappears entirely -- 16 reads per k-tile become 1 -- instead of
// merely halving as an interleaved (sc,mn) slot would.
//
// EVERYTHING HERE IS READ OFF AN OBJECT, not written down:
//   * which lane owns which n: from make_tiled_copy_B(SmemCopyAtomScale, TiledMma) partitioning an identity tensor,
//     i.e. the same map l90 printed as src_tv. Never inferred from the fragment's shape.
//   * where a group's bits live: from gguf_scale_layout.hpp's Field Layouts (byte_of / shift_of), the ones l91 gated on
//     4096 real superblocks.
//   * the placement: ONE cute Layout, (n, byte) -> offset, composed with the above. No hand-written index arithmetic.
//
// THREE ANSWERS IT PRODUCES, all local:
//   (1) does the decode reproduce today's fp16 planes exactly, when driven through the PLACED bytes
//   (2) how many lanes land on each bank -- today's 1.02 conflicts per scale read come from a 512 B thread stride being
//       a multiple of the 128 B bank period; a 12-byte column stride is 3 words, and gcd(3,32) == 1, so the claim to
//       check is that the conflict disappears for free
//   (3) how many bytes each lane must hold across the k-tile (the register cost, the one real downside)
#include <cstdio>
#include <cstdint>
#include <vector>
#include <utility>
#include <set>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/numeric_types.h"
#include "gguf_scale_decode.hpp"
using namespace cute;
using cutlass::half_t;
using gguf_scale::KType;
using Tr = gguf_scale::Traits<KType::Q4_K>;

// THE MMA, as l21's stub. The CollectiveBuilder cannot be used here: naming it is fine (l90 does) but CALLING
// partition_S/make_tiled_copy_B on its types needs -D__HGGCCC__ so CUTLASS_DEVICE lands in device code, and then cute's
// namespace-scope `_` is not device-visible under nvcc. l21 established this stub -- same Shape_MNK, same B layout, same
// Tile<> the builder uses -- and the whole fold derivation was built and later box-validated on it.
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
// the decode-band config: TileShape (16,128,256), warp 16x16, gs=32 -> Scale_TileK = 8 = ONE Q4_K superblock per k-tile
static constexpr int kTN_ = 128, kSK_ = 8, kWON = 8;
using Mma             = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<_1,Int<kWON>,_1>>, Tile<_16,Int<kWON*16>,Int<256>>>;
using SmemLayoutScale = decltype(tile_to_shape(Layout<Shape<_8,_1>>{},
                                               make_shape(Int<kTN_>{}, Int<kSK_>{}, Int<2>{})));
using SmemCopyAtomS   = Copy_Atom<DefaultCopy, half_t>;
using TMma            = Mma;

static constexpr int kTN  = kTN_;
static constexpr int kSK  = kSK_;
static constexpr int kThr = size(TMma{});
static constexpr int kBB  = Tr::kBlockBytes;            // 12 for Q4_K
static constexpr int kG   = Tr::kGroups;                // 8

// THE NATIVE TILE, as a Layout: (n, byte) -> byte offset. Column stride is the block's own size, so a column's bytes
// are contiguous and a lane reads them as 3 uint32. 12 bytes = 3 words and gcd(3,32) == 1, which is the whole reason to
// prefer the unpadded stride -- check (2) below is what decides it rather than this comment.
using NativeScaleTile = Layout<Shape<Int<kTN>, Int<kBB>>, Stride<Int<kBB>, _1>>;

int main() {
  std::printf("== l94: the native scale tile as a cute Layout, option E ==\n");
  std::printf("  ScaleTile=(%d,%d) threads=%d | Q4_K block=%d B, %d groups, gs=%d\n",
              kTN, kSK, kThr, kBB, kG, Tr::kGroupSize);
  print("  today's SmemLayoutScale : "); print(SmemLayoutScale{}); printf("\n");
  print("  native tile             : "); print(NativeScaleTile{}); printf("\n");

  // ---- which lane owns which n, READ OFF THE PARTITIONING (never inferred from the fragment shape)
  auto sS      = make_tensor(make_smem_ptr(static_cast<half_t*>(nullptr)), SmemLayoutScale{});
  auto tiled_s = make_tiled_copy_B(SmemCopyAtomS{}, TMma{});
  auto cS      = make_identity_tensor(shape(sS));
  std::vector<std::vector<int>> lane_n(kThr);
  int vals_per_thread = 0;
  for (int t = 0; t < kThr; ++t) {
    auto thr  = tiled_s.get_thread_slice(t);
    auto tCsC = thr.partition_S(cS);                       // (CPY, CPY_N, CPY_K, groups)
    int const nv = int(size<0>(tCsC)) * int(size<1>(tCsC));
    vals_per_thread = nv;
    for (int v = 0; v < nv; ++v) {
      auto c = tCsC(v % int(size<0>(tCsC)), v / int(size<0>(tCsC)), 0, 0);
      lane_n[t].push_back(int(get<0>(c)));                 // the column this (lane, value) reads
    }
  }
  std::printf("  lane -> n: %d values per thread; warp0 lane0 n=", vals_per_thread);
  for (int v = 0; v < vals_per_thread; ++v) std::printf("%d%s", lane_n[0][v], v + 1 < vals_per_thread ? "," : "\n");

  // ---- (2) BANKS, counted as DISTINCT ADDRESSES per bank, not lanes per bank. Several lanes hitting the SAME address
  // is a broadcast, not a conflict, and the scale fragment is deliberately k-broadcast (task #1's stride-0 view), so
  // lanes DO share addresses. My first version of this check counted lanes and would have reported a 4-way "conflict"
  // that is really a 4-way broadcast -- the metric has to be distinct addresses or it measures the wrong thing.
  {
    std::set<int> today_w[32], native_w[32];
    std::set<int> today_addr, native_addr;
    for (int t = 0; t < 32 && t < kThr; ++t) {
      auto thr  = tiled_s.get_thread_slice(t);
      auto tCsC = thr.partition_S(cS);
      auto c    = tCsC(0, 0, 0, 0);
      int const n = int(get<0>(c)), g = int(get<1>(c));
      int const tw = (int(SmemLayoutScale{}(n, g, 0)) * 2) / 4;      // halfs -> bytes -> words
      today_w[tw & 31].insert(tw);
      today_addr.insert(tw);
      for (int w = 0; w < (kBB + 3) / 4; ++w) {                       // the native read is 3 words of one column
        int const nw = (kBB * n) / 4 + w;
        native_w[nw & 31].insert(nw);
        native_addr.insert(nw);
      }
    }
    auto worst = [](std::set<int> const* h) {
      int mx = 0, used = 0;
      for (int b = 0; b < 32; ++b) { if (!h[b].empty()) ++used; if (int(h[b].size()) > mx) mx = int(h[b].size()); }
      return std::pair<int,int>{mx, used};
    };
    auto a = worst(today_w), b = worst(native_w);
    std::printf("  (2) DISTINCT addresses per bank over warp0: today %d-way on %d banks (%zu addrs) | native %d-way on %d banks (%zu addrs)\n",
                a.first, a.second, today_addr.size(), b.first, b.second, native_addr.size());
    std::printf("      -> %s\n", b.first < a.first
                ? "the conflict shrinks; the native column stride is 3 words and gcd(3,32)==1, so consecutive columns spread"
                : "NO improvement -- do not claim one");
  }

  // ---- (1) THE DECODE, driven through the PLACED bytes. Synthetic full-range codes: every sc and mn in [0,64) so no
  // field can be missed, and NON-dyadic d so the fp16 rounding path is actually entered (l93's first version used
  // powers of two, which made every product exact and tested nothing).
  std::vector<uint8_t> tile((size_t)kTN * kBB, 0);
  std::vector<half_t>  dv(kTN), dmv(kTN);
  std::vector<int>     want_sc((size_t)kTN * kG), want_mn((size_t)kTN * kG);
  for (int n = 0; n < kTN; ++n) {
    dv[n]  = half_t(0.0123f + 0.0007f * float(n % 11));
    dmv[n] = half_t(0.00789f + 0.0011f * float(n % 7));
    // pack the codes the way the gguf does, THROUGH the Field Layouts -- the same byte_of/shift_of the decode reads,
    // so a wrong map here would have to be wrong in the same direction there to hide, and l91 already pinned them
    // against get_scale_min_k4 on 4096 real blocks.
    for (int g = 0; g < kG; ++g) {
      int const sc = (n * kG + g) % 64, mn = (n * 11 + g * 37) % 64;
      want_sc[(size_t)n * kG + g] = sc;
      want_mn[(size_t)n * kG + g] = mn;
      uint8_t* col = tile.data() + (size_t)NativeScaleTile{}(n, 0);
      col[Tr::ScLo::byte_of(g)] |= uint8_t((sc & 0xF) << Tr::ScLo::shift_of(g));
      col[Tr::ScHi::byte_of(g)] |= uint8_t((sc >> 4)  << Tr::ScHi::shift_of(g));
      col[Tr::MnLo::byte_of(g)] |= uint8_t((mn & 0xF) << Tr::MnLo::shift_of(g));
      col[Tr::MnHi::byte_of(g)] |= uint8_t((mn >> 4)  << Tr::MnHi::shift_of(g));
    }
  }
  int bad_code = 0, bad_s = 0, bad_z = 0, nz = 0;
  for (int n = 0; n < kTN; ++n) {
    uint8_t const* col = tile.data() + (size_t)NativeScaleTile{}(n, 0);
    for (int g = 0; g < kG; ++g) {
      int const sc = gguf_scale::scale_of<KType::Q4_K>(col, g);
      int const mn = gguf_scale::min_of  <KType::Q4_K>(col, g);
      if (sc != want_sc[(size_t)n * kG + g] || mn != want_mn[(size_t)n * kG + g]) {
        if (bad_code < 4) std::printf("    [code] n=%d g=%d sc=%d/%d mn=%d/%d\n",
                                      n, g, sc, want_sc[(size_t)n*kG+g], mn, want_mn[(size_t)n*kG+g]);
        ++bad_code; continue;
      }
      // and against today's fp16 planes: the exact product rounded once, which is what the offline writes
      float const ref_s = float(half_t(float(dv[n])  * float(sc)));
      float const ref_z = float(half_t(-float(dmv[n]) * float(mn)));
      auto const gsz = gguf_scale::make_group_scale<KType::Q4_K>(sc, mn, dv[n], dmv[n]);
      if (float(gsz.scale) != ref_s) ++bad_s;
      if (float(gsz.zero)  != ref_z) ++bad_z;
      if (ref_s != 0.f) ++nz;
    }
  }
  std::printf("  (1) %d (n,group) pairs | codes %d bad | scale %d bad | zero %d bad | non-zero refs %d%s\n",
              kTN * kG, bad_code, bad_s, bad_z, nz, nz == 0 ? "  <-- VACUOUS" : "");

  // ---- (3) the register cost, and the read count that is the whole point
  std::printf("  (3) per lane per k-tile: %d B of codes (%d uint32) + 1 half2 of (d,dmin) held across %d groups\n",
              kBB, (kBB + 3) / 4, kG);
  std::printf("      s2r reads per k-tile: today %d (%d groups x scale+zero) -> native %d\n",
              2 * kSK, kSK, (kBB + 3) / 4);
  std::printf("      smem bytes per (group,column): today 4.0 -> native %.2f\n", double(kBB) / kG + 4.0 / kG);

  int const fail = bad_code + bad_s + bad_z + (nz == 0 ? 1 : 0);
  std::printf("== %s: %d ==\n", fail ? "FAIL" : "PASS", fail);
  return fail ? 1 : 0;
}
