// WHAT FORMAT IS THIS WEIGHT BUFFER IN? -- the cute Layout is the answer, and the record carries it.
//
// A packed weight buffer is not self-describing: the same bytes mean different things depending on which
// offline pass produced them, and each offline format implies a different way of reading it. Getting that
// wrong is silent -- it produces numbers, just not the right ones. So the layout gets three things here:
//
//   1. A DEFINITION as a cute Layout mapping the logical coordinate (n, k) to an element index in the packed
//      buffer. This is the authority. Adding an offline format means adding one Layout, not editing address
//      arithmetic in a kernel.
//
//   2. AN IMPLEMENTATION as the four byte strides the GEMV's iterator wants, DERIVED by evaluating that same
//      Layout at four points on the HOST and passed down in KernelArgs. This is the same posture the existing
//      mixed-input GEMM takes -- cute types define the layout, make_cute_packed_stride builds the strides on
//      the host, and the kernel consumes them out of Params. There is deliberately no second, closed-form
//      definition of the format in this header; the gate's independent oracle lives in the test instead.
//
//      CUTE STAYS ON THE HOST, and that is deliberate rather than incidental. cute/config.hpp gates
//      CUTE_HOST_DEVICE on __HGGCCC__, so under plain nvcc every cute entity is `inline` and host-only --
//      which is why every local harness in fold_derivation/ uses cute host-side. Evaluating a Layout inside
//      the kernel therefore both needed a compiler-specific define AND put address arithmetic on the critical
//      path for no gain: the addresses are affine in (column, thread, iteration), so four host scalars carry
//      the whole format. (Cost of learning this the other way: the device silently returned 0 for every
//      offset, because a __host__ __device__ caller reaching a __host__ callee is an nvcc WARNING, and the
//      build filter only grepped for "error".)
//
//   3. A RECORD, WeightFormatRecord, written next to the weight by the offline pass and checked at load. Its
//      layout string is produced by printing the very Layout object the kernel indexes with, so the record
//      cannot describe something other than what is read. A buffer whose record does not match the
//      instantiation is REFUSED rather than computed -- the same net as gemv_fail_count().
//
// A note on flat k: cute decomposes an integer coordinate against a hierarchical shape, so TileK's
// (n, (kin, kt)) is addressed as L(n, k) with a flat k and the tiling stays inside the layout. Verified.
#pragma once

#include "gemv_common.hpp"

#include <cstring>
#include <cstdio>

#include "cute/tensor.hpp"

// Host-side only, but included unconditionally: nvcc's DEVICE pass parses the whole translation unit, so
// hiding the record helpers behind #if !defined(__CUDA_ARCH__) makes launch_gemv fail to parse in that pass
// even though it is a host function.
#include <sstream>
#include <string>

namespace ppu_gemv {

// ---------------------------------------------------------------------------------------------------
// (1) THE DEFINITION.

// Native [N][K], K contiguous: (n, k) -> n*K + k. GGUF / GPTQ native, no offline pass.
inline auto wlayout_native(int N, int K) {
  (void)N;
  return cute::make_layout(cute::make_shape(N, K), cute::make_stride(K, cute::_1{}));
}

// TileK<TS> [K/TS][N][TS]: (n, (kin, kt)) -> n*TS + kin + kt*(N*TS).
// This is cutlass::layout::ColumnMajorInterleaved<TS> -- the buffer the prefill AIU collective reads for
// int4, so decode and prefill share one copy of the weights at that width.
template <int TS>
inline auto wlayout_tilek(int N, int K) {
  return cute::make_layout(
      cute::make_shape(N, cute::make_shape(cute::Int<TS>{}, K / TS)),
      cute::make_stride(cute::Int<TS>{}, cute::make_stride(cute::_1{}, N * TS)));
}

template <WLayout Kind, int TS>
inline auto wlayout_of(int N, int K) {
  if constexpr (Kind == WLayout::Native) return wlayout_native(N, K);
  else                                   return wlayout_tilek<TS>(N, K);
}


// ---------------------------------------------------------------------------------------------------
// (2) THE IMPLEMENTATION. Byte offsets for the iterator, in two interchangeable forms.
//
// The triple is only enough when the layout is AFFINE at the granularities the iterator steps by: k in
// units of CtaK and n in units of 1. Native and TileK both are (TileK because CtaK is a whole number of
// k-tiles). A cube-folded offline layout is NOT -- its within-cube permutation scatters a contiguous
// logical-k run -- which is precisely why it needs a different compute path and therefore has to be
// recorded rather than assumed. kAffine states which case a layout is in.
// The four byte strides that describe an affine weight layout to the iterator. The address of (column
// n0+col, thread tid, iteration iter) is
//     n0*col + (tid/TPT)*thr_major + (tid%TPT)*thr_minor + iter*iter_step
// with TPT a compile-time constant (TileSizeK/StepK for a tiled layout, Threads for a flat one, where the
// major term vanishes because tid/TPT is 0). One form covers both layouts.
struct WStrides {
  int64_t col       = 0;
  int64_t thr_minor = 0;
  int64_t thr_major = 0;
  int64_t iter      = 0;
};

template <WLayout Kind, int Bits, int StepK, int Threads, int TileSizeK>
struct WFormatTraits {
  static constexpr bool kAffine = (Kind == WLayout::Native || Kind == WLayout::TileK);
  static constexpr int kStepBytes = StepK * Bits / 8;
  // Threads covering one k-tile. Native is flat, so the whole CTA is "one tile" and the major term is dead.
  static constexpr int kTPT = (Kind == WLayout::TileK) ? (TileSizeK / StepK) : Threads;
  static_assert(StepK * Bits % 8 == 0, "StepK*Bits must be a whole number of bytes");
  static_assert(Kind != WLayout::TileK || TileSizeK % StepK == 0,
                "TileSizeK must be a whole multiple of StepK");
  static_assert(Kind != WLayout::TileK || Threads % (TileSizeK / StepK) == 0,
                "Threads must cover a whole number of k-tiles");

  // ---- derived from the Layout: four evaluations, no second arithmetic ----
  static int64_t elem_bytes(int64_t idx) { return idx * Bits / 8; }

  static int64_t base(int n, int tid, int64_t N, int64_t K) {
    auto L = wlayout_of<Kind, TileSizeK>(int(N), int(K));
    return elem_bytes(int64_t(L(n, tid * StepK)));
  }
  static int64_t step(int64_t N, int64_t K) {
    auto L = wlayout_of<Kind, TileSizeK>(int(N), int(K));
    return elem_bytes(int64_t(L(0, Threads * StepK)) - int64_t(L(0, 0)));
  }
  static int64_t stride(int64_t N, int64_t K) {
    auto L = wlayout_of<Kind, TileSizeK>(int(N), int(K));
    return elem_bytes(int64_t(L(1, 0)) - int64_t(L(0, 0)));
  }
  // The four strides the kernel runs on: each one evaluation of the Layout.
  static WStrides strides(int64_t N, int64_t K) {
    auto L = wlayout_of<Kind, TileSizeK>(int(N), int(K));
    int64_t const o = int64_t(L(0, 0));
    WStrides w;
    w.col       = elem_bytes(int64_t(L(1, 0)) - o);
    w.thr_minor = elem_bytes(int64_t(L(0, StepK)) - o);
    w.thr_major = (Kind == WLayout::TileK) ? elem_bytes(int64_t(L(0, kTPT * StepK)) - o) : 0;
    w.iter      = elem_bytes(int64_t(L(0, Threads * StepK)) - o);
    return w;
  }

};

// ---------------------------------------------------------------------------------------------------
// (3) THE RECORD. Written by the offline pass, checked at load.
struct WeightFormatRecord {
  char     magic[8];        // "PPUWFMT"
  int32_t  version;
  int32_t  format;          // WFormat
  int32_t  lo_bits, hi_bits;
  int32_t  layout;          // WLayout
  int32_t  tile_k;          // TileSizeK, 0 when not tiled
  int32_t  n, k;
  int32_t  group_size;
  int32_t  quant;           // QuantOp
  int32_t  num_experts;     // 0 = dense
  int32_t  a_bf16;
  // Cube-folded offline parameters. Zero when the layout is not cube-folded; non-zero means the buffer was
  // produced for a SPECIFIC prefill tile shape and cannot be read by a layout-agnostic consumer.
  int32_t  tm, tn, tk, wm, wn, fold;
  int32_t  reserved[6];
  char     layout_str_lo[192];   // printed from the Layout object the kernel indexes with
  char     layout_str_hi[192];
};

static constexpr int32_t kWeightFormatVersion = 1;

inline void wfmt_set_magic(WeightFormatRecord& r) { std::memcpy(r.magic, "PPUWFMT", 8); }
inline bool wfmt_magic_ok(WeightFormatRecord const& r) { return std::memcmp(r.magic, "PPUWFMT", 8) == 0; }

// Print a cute Layout into a fixed buffer. The record's description is therefore READ OFF the object rather
// than written down beside it -- the distinction this project keeps paying for.
template <class Layout>
inline void wfmt_print_layout(char* dst, size_t cap, Layout const& l) {
  std::ostringstream os;
  os << l;
  std::string const s = os.str();
  std::snprintf(dst, cap, "%s", s.c_str());
}

// Build the record for one instantiation + problem. Everything the reader needs to decide whether it may
// read the buffer at all.
template <typename Details>
inline WeightFormatRecord wfmt_record(int n, int k, int group_size, QuantOp quant, int num_experts) {
  WeightFormatRecord r{};
  wfmt_set_magic(r);
  r.version = kWeightFormatVersion;
  r.format  = int32_t(Details::kFormat);
  r.lo_bits = Details::kLoBits;
  r.hi_bits = Details::kHiBits;
  r.layout  = int32_t(Details::kLayout);
  r.tile_k  = (Details::kLayout == WLayout::TileK) ? Details::kTileSizeK : 0;
  r.n = n; r.k = k; r.group_size = group_size;
  r.quant = int32_t(quant);
  r.num_experts = num_experts;
  r.a_bf16 = Details::ADetails::kIsBF16 ? 1 : 0;
  {
    auto Llo = wlayout_of<Details::kLayout, Details::kTileSizeK>(n, k);
    wfmt_print_layout(r.layout_str_lo, sizeof(r.layout_str_lo), Llo);
    if (Details::kTwoPlane) {
      // Same shape, different element width: the plane's bit width lives in lo_bits/hi_bits, not in the
      // layout, because the layout indexes ELEMENTS.
      wfmt_print_layout(r.layout_str_hi, sizeof(r.layout_str_hi), Llo);
    }
  }
  return r;
}

// Does this buffer's record describe something this instantiation can read?
template <typename Details>
inline bool wfmt_matches(WeightFormatRecord const& r, int n, int k, int group_size, QuantOp quant,
                         char const** why = nullptr) {
  auto fail = [&](char const* m) { if (why) *why = m; return false; };
  if (!wfmt_magic_ok(r))                            return fail("bad magic");
  if (r.version != kWeightFormatVersion)            return fail("record version");
  if (r.format != int32_t(Details::kFormat))        return fail("weight format");
  if (r.lo_bits != Details::kLoBits)                return fail("low-plane width");
  if (r.hi_bits != Details::kHiBits)                return fail("high-plane width");
  if (r.layout != int32_t(Details::kLayout))        return fail("weight layout");
  if (r.layout == int32_t(WLayout::TileK) && r.tile_k != Details::kTileSizeK) return fail("tile K");
  if (r.n != n || r.k != k)                         return fail("problem shape");
  if (r.group_size != group_size)                   return fail("group size");
  if (r.quant != int32_t(quant))                    return fail("quant op");
  if (r.a_bf16 != (Details::ADetails::kIsBF16 ? 1 : 0)) return fail("activation type");
  // A cube-folded buffer was produced for a specific prefill tile shape; its within-cube permutation
  // scatters a contiguous logical-k run, so an affine consumer must not touch it.
  if (r.fold != 0 || r.tm != 0)                     return fail("cube-folded offline: not affine in k");
  {
    auto L = wlayout_of<Details::kLayout, Details::kTileSizeK>(n, k);
    char buf[192];
    wfmt_print_layout(buf, sizeof(buf), L);
    if (std::strncmp(buf, r.layout_str_lo, sizeof(buf)) != 0) return fail("layout string");
  }
  if (why) *why = "";
  return true;
}

inline bool wfmt_write(char const* path, WeightFormatRecord const& r) {
  std::FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  bool const ok = std::fwrite(&r, sizeof(r), 1, f) == 1;
  std::fclose(f);
  return ok;
}

inline bool wfmt_read(char const* path, WeightFormatRecord& r) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  bool const ok = std::fread(&r, sizeof(r), 1, f) == 1;
  std::fclose(f);
  return ok && wfmt_magic_ok(r);
}

inline void wfmt_print(WeightFormatRecord const& r) {
  std::printf("  weight format record v%d: %s / %s, planes %d+%d bits, gs=%d, %s, n=%d k=%d",
              r.version, name_of(WFormat(r.format)), name_of(WLayout(r.layout)),
              r.lo_bits, r.hi_bits, r.group_size, name_of(QuantOp(r.quant)), r.n, r.k);
  if (r.tile_k) std::printf(", TS=%d", r.tile_k);
  if (r.num_experts) std::printf(", experts=%d", r.num_experts);
  if (r.fold) std::printf(", CUBE-FOLDED tm=%d tn=%d tk=%d wm=%d wn=%d F=%d", r.tm, r.tn, r.tk, r.wm, r.wn, r.fold);
  std::printf("\n    layout: %s\n", r.layout_str_lo[0] ? r.layout_str_lo : "(none)");
}

}  // namespace ppu_gemv
