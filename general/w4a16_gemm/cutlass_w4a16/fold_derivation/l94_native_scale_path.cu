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
#include <algorithm>
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
using Mma             = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<_1,Int<kWON>,_1>>, Tile<_16,Int<kWON*16>,_64>>;
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

// ARRANGEMENT B, and the reason to prefer it: a Q4_K block's FIRST 16 BYTES are exactly d(2) + dmin(2) + scales(12).
// So one contiguous 16 B per (superblock, column) carries everything a lane needs, is 16 B aligned (one cp.async, one
// LDS.128), wastes nothing (16/8 groups = 2.0 B per group per column, the same as 12+4 stored apart) and removes the
// (d,dmin) plane, its tile and ptr_Z outright. The open question is banks: stride 16 B is 4 words, so columns n and n+8
// alias, whereas 12 B is 3 words and gcd(3,32) == 1. Measured below rather than argued.
static constexpr int kBB16 = 16;
using NativeScaleTile16 = Layout<Shape<Int<kTN>, Int<kBB16>>, Stride<Int<kBB16>, _1>>;

static int g_fail_extra = 0;

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
    std::set<int> today_w[32], native_w[32], n16_w[32];
    std::set<int> today_addr, native_addr, n16_addr;
    for (int t = 0; t < 32 && t < kThr; ++t) {
      auto thr  = tiled_s.get_thread_slice(t);
      auto tCsC = thr.partition_S(cS);
      auto c    = tCsC(0, 0, 0, 0);
      int const n = int(get<0>(c)), g = int(get<1>(c));
      int const tw = (int(SmemLayoutScale{}(n, g, 0)) * 2) / 4;      // halfs -> bytes -> words
      today_w[tw & 31].insert(tw);
      today_addr.insert(tw);
      for (int w = 0; w < (kBB + 3) / 4; ++w) {                       // arrangement A: 3 words of codes
        int const nw = (kBB * n) / 4 + w;
        native_w[nw & 31].insert(nw);
        native_addr.insert(nw);
      }
      for (int w = 0; w < kBB16 / 4; ++w) {                            // arrangement B: 4 words, d+dmin+codes
        int const nw = (kBB16 * n) / 4 + w;
        n16_w[nw & 31].insert(nw);
        n16_addr.insert(nw);
      }
    }
    auto worst = [](std::set<int> const* h) {
      int mx = 0, used = 0;
      for (int b = 0; b < 32; ++b) { if (!h[b].empty()) ++used; if (int(h[b].size()) > mx) mx = int(h[b].size()); }
      return std::pair<int,int>{mx, used};
    };
    auto a = worst(today_w), b = worst(native_w), c = worst(n16_w);
    std::printf("  (2) DISTINCT addresses per bank over warp0 (a conflict, unlike shared addresses which broadcast):\n");
    std::printf("      today                        %d-way on %2d banks (%2zu addrs)\n", a.first, a.second, today_addr.size());
    std::printf("      A: 12 B codes only           %d-way on %2d banks (%2zu addrs)  + a separate (d,dmin) plane\n",
                b.first, b.second, native_addr.size());
    std::printf("      B: 16 B = d+dmin+codes       %d-way on %2d banks (%2zu addrs)  no second plane, 16 B aligned\n",
                c.first, c.second, n16_addr.size());
    std::printf("      distinct columns a warp0 lane set touches: %zu\n", native_addr.size() / ((kBB + 3) / 4));
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


  // ---------------------------------------------------------------------------------------------------------------
  // (4) THE REORDERED 16 B, made SEPARABLE, and why it has to be reordered at all.
  //
  // In the native packing the two halves are not separable: get_scale_min_k4 takes sc[4..7] from bytes 8-11's low
  // nibbles PLUS bytes 0-3's top 2 bits, and mn[4..7] from bytes 8-11's high nibbles PLUS bytes 4-7's top 2 bits. So
  // groups 0-3 need bytes 0-7 and groups 4-7 need ALL twelve -- a k-tile covering half a superblock (TileK=128) would
  // have to read the whole block. Since the offline order is ours to choose, make each half self-contained:
  //
  //     byte 0-1  d          byte 2-3  dmin        byte 4-9  half0 (groups 0-3)     byte 10-15  half1 (groups 4-7)
  //
  // A half is 4 sc + 4 mn as 6-bit fields = 48 bits = 6 bytes exactly, so nothing grows: still 16 B per (superblock,
  // column) = 2.0 B per group per column. Bit position as ONE Layout over (i, h, which), i = g%4, h = g/4,
  // which = 0 for sc and 1 for mn:  bit = 32 + 6*i + 48*h + 24*which.
  // THE SHIPPED functions, not local copies: gguf_scale_decode.hpp owns PackBits, packed_put and packed_code, so this
  // gate covers the code the mainloop will call. A second implementation here would be the exact failure this file
  // keeps recording -- one relation in two places.
  using gguf_scale::PackBits; using gguf_scale::kPackBitBase;
  auto bit_of = [](int g, int which) { return gguf_scale::packed_bit_of(g, which); };
  print("  (4) PackBits (i,h,which)->bit : "); print(PackBits{}); printf("  base=%d\n", kPackBitBase);
  {
    // native block -> reference sc/mn (the l91-gated path) -> new 16 B -> decode -> must equal the reference
    int bad_rt = 0, span_bad = 0;
    for (int n = 0; n < kTN; ++n) {
      uint8_t const* col = tile.data() + (size_t)NativeScaleTile{}(n, 0);
      uint8_t nw[16] = {};
      *reinterpret_cast<uint16_t*>(nw + 0) = dv[n].raw();
      *reinterpret_cast<uint16_t*>(nw + 2) = dmv[n].raw();
      for (int g = 0; g < kG; ++g) {
        gguf_scale::packed_put(nw, g, 0, gguf_scale::scale_of<KType::Q4_K>(col, g));
        gguf_scale::packed_put(nw, g, 1, gguf_scale::min_of  <KType::Q4_K>(col, g));
      }
      for (int g = 0; g < kG; ++g) {
        if (gguf_scale::packed_code(nw, g, 0) != gguf_scale::scale_of<KType::Q4_K>(col, g) ||
            gguf_scale::packed_code(nw, g, 1) != gguf_scale::min_of  <KType::Q4_K>(col, g)) {
          if (bad_rt < 4) std::printf("    [reorder] n=%d g=%d round trip differs\n", n, g);
          ++bad_rt;
        }
      }
      // SEPARABILITY, checked rather than asserted: every bit of half h must lie inside that half's 6 bytes.
      for (int g = 0; g < kG; ++g)
        for (int w = 0; w < 2; ++w) {
          int const lo = bit_of(g, w) >> 3, hi = (bit_of(g, w) + 5) >> 3, h = g / 4;
          if (lo < 4 + 6 * h || hi > 4 + 6 * h + 5) ++span_bad;
        }
    }
    std::printf("      round trip over %d columns x %d groups -> %d bad | bits outside their own half: %d\n",
                kTN, kG, bad_rt, span_bad);
    g_fail_extra += bad_rt + span_bad;
  }
  // Sub-reads and their alignment, per TileK. A half is 6 B at byte 4 or 10, so it is 4B+2B or 2B+4B, every piece
  // naturally aligned; padding a half to 8 B would make it one load but cost 20 B per superblock, +25%.
  for (int TK : {256, 128, 64}) {
    int const groups_per_tile = TK / Tr::kGroupSize;
    int const halves = (groups_per_tile + 3) / 4;
    int const bytes  = (TK >= 256) ? 16 : (4 + 6 * halves);
    std::printf("      TileK=%3d: %d group(s)/tile, read %2d B (%s) -> %.2f B per (group,col)%s\n",
                TK, groups_per_tile, bytes,
                TK >= 256 ? "one LDS.128, 16 B aligned" : "LDS.32 at 4 + LDS.16 at 8 (or 10/12), all aligned",
                double(bytes) / groups_per_tile,
                double(bytes) / groups_per_tile > 4.0 ? "   <-- WORSE than fp16, gate this TileK off" : "");
  }


  // ---------------------------------------------------------------------------------------------------------------
  // (5) THE OTHER FORMATS. 16 B is a COINCIDENCE of Q4_K's header (d+dmin+12 = 16); it does not carry. Everything below
  // is computed from Traits<T>, so a format's numbers cannot be inherited from Q4_K by accident.
  auto format_row = [](char const* name, int G, int sbits, int mbits, bool has_min, int gs_) {
    double const code_bytes = double(G) * double(sbits + mbits) / 8.0;      // all groups' codes, one column
    double const hdr        = 2.0 + (has_min ? 2.0 : 0.0);                  // d, and dmin where there is one
    double const total      = code_bytes + hdr;
    double const per_group  = total / double(G);
    double const fp16       = has_min ? 4.0 : 2.0;                          // today: scale (+ zero) as fp16 per group
    bool   const fits16     = total <= 16.0;
    // a TileK=128 tile covers 128/gs groups; is that many codes a whole number of bytes?
    int const gpt = 128 / gs_;
    double const half_bytes = double(gpt) * double(sbits + mbits) / 8.0;
    bool const half_whole   = (double(int(half_bytes)) == half_bytes);
    std::printf("      %-5s G=%2d %d+%d bit  codes %5.1f B + hdr %.0f = %5.1f B/superblock/col -> %.3f B/(group,col)"
                "  vs fp16 %.1f = %.2fx | %s | TileK=128 half %5.1f B %s\n",
                name, G, sbits, mbits, code_bytes, hdr, total, per_group, fp16, fp16 / per_group,
                fits16 ? "fits one 16 B read" : "does NOT fit 16 B -> split d out, keep codes aligned",
                half_bytes, half_whole ? "(whole bytes)" : "(NOT whole bytes -- needs its own arrangement)");
  };
  std::printf("  (5) per-format, from Traits:\n");
  format_row("Q4_K", gguf_scale::Traits<KType::Q4_K>::kGroups, gguf_scale::Traits<KType::Q4_K>::kScaleBits,
             gguf_scale::Traits<KType::Q4_K>::kMinBits, gguf_scale::Traits<KType::Q4_K>::kHasMin,
             gguf_scale::Traits<KType::Q4_K>::kGroupSize);
  format_row("Q3_K", gguf_scale::Traits<KType::Q3_K>::kGroups, gguf_scale::Traits<KType::Q3_K>::kScaleBits,
             gguf_scale::Traits<KType::Q3_K>::kMinBits, gguf_scale::Traits<KType::Q3_K>::kHasMin,
             gguf_scale::Traits<KType::Q3_K>::kGroupSize);
  format_row("Q2_K", gguf_scale::Traits<KType::Q2_K>::kGroups, gguf_scale::Traits<KType::Q2_K>::kScaleBits,
             gguf_scale::Traits<KType::Q2_K>::kMinBits, gguf_scale::Traits<KType::Q2_K>::kHasMin,
             gguf_scale::Traits<KType::Q2_K>::kGroupSize);
  format_row("Q6_K", gguf_scale::Traits<KType::Q6_K>::kGroups, gguf_scale::Traits<KType::Q6_K>::kScaleBits,
             gguf_scale::Traits<KType::Q6_K>::kMinBits, gguf_scale::Traits<KType::Q6_K>::kHasMin,
             gguf_scale::Traits<KType::Q6_K>::kGroupSize);


  // ---------------------------------------------------------------------------------------------------------------
  // (6) THE GMEM SIDE, the one piece I called high risk. Today's scale g2s is
  //     make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>, Layout<(TN/8, SK)>, Layout<(_8,_1)>)
  // i.e. 128 threads x 8 halfs = 16 B each, covering TN*SK*2 = 2048 B.
  //
  // The native tile is TN columns x 16 B = 2048 B -- THE SAME BYTE COUNT, because the Z tile disappears (2.0 vs 4.0 B
  // per (group,col)). So the copy stays one uint128 per thread and only the shape changes: one thread per COLUMN.
  //     Layout<(TN, _1)> threads, Layout<(_1, _16)> values, element uint8_t
  // Then thread t reads bytes [16t, 16t+16) -- 16 B aligned, and consecutive threads read consecutive chunks, i.e. one
  // fully coalesced 2048 B burst. Today's map is strided instead (thread t covers N-offset 8*(t%16), K-offset t/16).
  //
  // The atom here is DefaultCopy, not the PPU cp.async: the check is make_tiled_copy's ALGEBRA -- who reads which bytes
  // -- which does not depend on the instruction. What the uint128 atom requires is 16 bytes per thread, and that is
  // exactly what the value layout gives.
  {
    auto gtc  = make_tiled_copy(Copy_Atom<DefaultCopy, uint8_t>{},
                                Layout<Shape<Int<kTN>, _1>>{},
                                Layout<Shape<_1, Int<kBB16>>>{});
    auto cN   = make_identity_tensor(make_shape(Int<kTN>{}, Int<kBB16>{}));
    int  cover[128 * 16] = {};
    int  bad_align = 0, bad_contig = 0, bad_coal = 0;
    int const nthr = int(size(gtc));
    for (int t = 0; t < nthr; ++t) {
      auto sl = gtc.get_thread_slice(t).partition_S(cN);
      int first = -1, prev = -2, cnt = 0;
      for (int i = 0; i < int(size(sl)); ++i) {
        auto c   = sl(i);
        int  off = int(get<0>(c)) * kBB16 + int(get<1>(c));
        if (off < 0 || off >= kTN * kBB16) { ++bad_contig; continue; }
        ++cover[off];
        if (first < 0) first = off;
        if (prev >= 0 && off != prev + 1) ++bad_contig;      // a thread's bytes must be one contiguous run
        prev = off; ++cnt;
      }
      if (cnt != kBB16) ++bad_contig;                        // exactly 16 B per thread (what uint128 needs)
      if (first % kBB16 != 0) ++bad_align;                   // 16 B aligned
      if (first != t * kBB16) ++bad_coal;                    // consecutive threads -> consecutive chunks
    }
    int gaps = 0, dups = 0;
    for (int i = 0; i < kTN * kBB16; ++i) { if (cover[i] == 0) ++gaps; if (cover[i] > 1) ++dups; }
    std::printf("  (6) gmem g2s, native shape: %d threads x %d B = %d B (today's S tile is %d B)\n",
                nthr, kBB16, nthr * kBB16, kTN * kSK * 2);
    std::printf("      contiguity %d bad | 16 B alignment %d bad | coalescing (t -> 16t) %d bad | gaps %d | overlaps %d\n",
                bad_contig, bad_align, bad_coal, gaps, dups);
    g_fail_extra += bad_contig + bad_align + bad_coal + gaps + dups;
  }


  // ---------------------------------------------------------------------------------------------------------------
  // (7) THE SCATTER MAP: value -> which of MY columns. This is the last unknown before the transform arms can be
  // written, and it decides the register cost: a lane must hold one 16 B unit per DISTINCT column its scale fragment
  // touches, so what matters is that count and whether it is the same for every lane.
  //
  // Derived from partition_S of an identity tensor -- never from the fragment's shape. l94 (1)-(6) all rest on the same
  // source, and l95 proved that source is the collective's own object.
  {
    int  ndist_min = 1 << 30, ndist_max = 0;
    bool same_pattern = true;
    std::vector<int> ref_slot;
    for (int t = 0; t < kThr; ++t) {
      auto thr  = tiled_s.get_thread_slice(t);
      auto tCsC = thr.partition_S(cS);
      int const n0 = int(size<0>(tCsC)), n1 = int(size<1>(tCsC));
      std::vector<int> cols, slot;
      for (int v = 0; v < n0 * n1; ++v) {
        auto c = tCsC(v % n0, v / n0, 0, 0);
        int const n = int(get<0>(c));
        int k = -1;
        for (int i = 0; i < int(cols.size()); ++i) if (cols[i] == n) { k = i; break; }
        if (k < 0) { k = int(cols.size()); cols.push_back(n); }
        slot.push_back(k);
      }
      ndist_min = std::min(ndist_min, int(cols.size()));
      ndist_max = std::max(ndist_max, int(cols.size()));
      if (t == 0) ref_slot = slot;
      else if (slot != ref_slot) same_pattern = false;
    }
    std::printf("  (7) distinct columns per lane: min %d max %d | v->slot pattern identical across all %d lanes: %s\n",
                ndist_min, ndist_max, kThr, same_pattern ? "YES" : "NO");
    std::printf("      registers per lane: %d units x 16 B = %d uint32 (plus the fp16 d/dmin inside them)\n",
                ndist_max, ndist_max * 4);
    std::printf("      v->slot, first 24 of %zu: ", ref_slot.size());
    for (int i = 0; i < 24 && i < int(ref_slot.size()); ++i) std::printf("%d", ref_slot[i]);
    std::printf("\n");
    // Is the pattern expressible as a LAYOUT rather than a table? If slot(v) is periodic with a power-of-two period and
    // linear in the value index, one Layout covers it and nothing has to be stored.
    int period = 0;
    for (int P = 1; P <= int(ref_slot.size()); ++P) {
      bool ok = true;
      for (int i = 0; i + P < int(ref_slot.size()); ++i) if (ref_slot[i] != ref_slot[i + P]) { ok = false; break; }
      if (ok) { period = P; break; }
    }
    std::printf("      slot(v) period: %d %s\n", period,
                (period && (period & (period - 1)) == 0) ? "(a power of two -> expressible as one Layout, no table)"
                                                         : "(NOT a power of two -- needs a stored map)");
    if (!same_pattern || ndist_min != ndist_max) ++g_fail_extra;
  }

  int const fail = bad_code + bad_s + bad_z + (nz == 0 ? 1 : 0) + g_fail_extra;
  std::printf("== %s: %d ==\n", fail ? "FAIL" : "PASS", fail);
  return fail ? 1 : 0;
}
