// Gate + bench for the low-bit weight-only GEMV.
//
// TWO ORACLES, both independent of the kernel:
//
//  1. CONVERTER. RawConverter (magic-number, slot-ordered) is compared elementwise against RefRawConverter
//     (per-element extraction, identity mapper) THROUGH the mapper, so a permuted output cannot pass. When
//     they disagree the probe re-runs one-hot and prints the permutation the hardware actually produced,
//     which is the difference between "wrong" and "wrong in this specific way".
//
//  2. FULL KERNEL, IN EXACT MODE. Activations in {-2..2}, scale 2^-2, integer codes: every product and every
//     partial sum is exactly representable in fp16, so the comparison is BIT-EXACT rather than a tolerance.
//     A tolerance would have accepted several of the bugs this project has already shipped and caught later.
//     A random mode with a relative bound runs alongside it to cover the magnitudes real weights have.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

// ITERATION COST. The full matrix instantiates GS{0,16,32,64,128} x 3 quant ops x CtaM 1-4 x bias x grouped
// for every (format, layout, CtaN, Chunk) row, which is a ~10 minute nvcc build. That is the right cost before
// a commit and the wrong one while iterating, so GEMV_GATE_FAST narrows the group-size axis to the two that
// carry most of the coverage. Build the FULL matrix before trusting a result.
#if defined(GEMV_GATE_FAST)
#define GEMV_GS_LIST(EMIT) EMIT(0) EMIT(32)
#endif
#include "gemv_lowbit/gemv_launcher.hpp"
#include "gemv_lowbit/gemv_rt.hpp"      // one source, both runtimes -- see the header for why

using namespace ppu_gemv;

static int g_pass = 0, g_fail = 0, g_skip = 0;

// Which group sizes this BUILD instantiated. A fast build must SKIP the rows it cannot serve, not let them be
// refused and counted as failures: an output whose 26 red lines are all expected is one that gets misread the
// first time it has a real one.
static bool gs_instantiated(int gs) {
#if defined(GEMV_GATE_FAST)
  return gs == 0 || gs == 32;
#else
  return true;
#endif
}

// ---------------------------------------------------------------------------------------------------
// Host packing. q[n*K + k] holds the logical code of column n, row k. Both layouts are expressed as a bit
// position so there is exactly one place the convention lives.
static std::vector<uint8_t> pack_plane(WLayout lay, int bits, int TS,
                                       std::vector<uint8_t> const& q, int N, int K) {
  std::vector<uint8_t> out(size_t(N) * K * bits / 8, 0);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      size_t bitpos = (lay == WLayout::Native)
          ? (size_t(n) * K + size_t(k)) * bits
          : ((size_t(k / TS) * N * TS) + size_t(n) * TS + size_t(k % TS)) * bits;
      uint32_t const v = q[size_t(n) * K + k] & ((1u << bits) - 1u);
      // bits divides 8 for every supported width, so an element never straddles a byte.
      out[bitpos >> 3] |= uint8_t(v << (bitpos & 7));
    }
  return out;
}

// ---------------------------------------------------------------------------------------------------
// Converter gate.
template <typename ADetails, int Bits, bool SubOffset, int N>
__global__ void cvt_probe(uint32_t const* src, float* fast_out, float* ref_out, int* map_out) {
  using Fast = RawConverter<ADetails, Bits, SubOffset>;
  using Ref  = RefRawConverter<ADetails, Bits, SubOffset>;
  using T = typename ADetails::Type;
  T f[N], r[N];
  Fast::template convert<N>(src, f);
  Ref::template convert<N>(src, r);
  for (int i = 0; i < N; ++i) {
    fast_out[i] = MathWrapper<ADetails>::to_float(f[i]);
    ref_out[i]  = MathWrapper<ADetails>::to_float(r[i]);
    map_out[i]  = Fast::mapper(i);
  }
}

template <int Bits, bool SubOffset>
static void gate_converter(const char* tag) {
  constexpr int N = (32 / Bits) * 4;   // four source words
  std::mt19937 gen(0xC0FFEE ^ (Bits * 131 + int(SubOffset)));
  std::vector<uint32_t> h_src(N * Bits / 32);
  uint32_t *d_src; float *d_f, *d_r; int* d_m;
  d_src = (uint32_t*)rt_malloc(h_src.size() * 4);
  d_f = decltype(d_f)(rt_malloc(N * 4)); d_r = decltype(d_r)(rt_malloc(N * 4)); d_m = decltype(d_m)(rt_malloc(N * 4));

  bool ok = true;
  int bad_first = -1;
  std::vector<float> h_f(N), h_r(N);
  std::vector<int> h_m(N);
  for (int trial = 0; trial < 64 && ok; ++trial) {
    for (auto& w : h_src) w = gen();
    rt_h2d(d_src, h_src.data(), h_src.size() * 4);
    cvt_probe<FP16DetailsA, Bits, SubOffset, N><<<1, 1>>>(d_src, d_f, d_r, d_m);
    rt_sync("probe");
    rt_d2h(h_f.data(), d_f, N * 4);
    rt_d2h(h_r.data(), d_r, N * 4);
    rt_d2h(h_m.data(), d_m, N * 4);
    for (int l = 0; l < N; ++l) {
      // ref is in logical order; fast is in slot order. Compare THROUGH the mapper.
      if (h_f[h_m[l]] != h_r[l]) { ok = false; bad_first = l; break; }
    }
  }

  if (ok) {
    std::printf("  [ok]   converter %-14s N=%3d  fast==ref through mapper\n", tag, N);
    ++g_pass;
  } else {
    std::printf("  [FAIL] converter %-14s N=%3d  first mismatch at logical %d "
                "(mapper->%d: fast %.1f vs ref %.1f)\n",
                tag, N, bad_first, h_m[bad_first], h_f[h_m[bad_first]], h_r[bad_first]);
    // Diagnostic: one-hot each logical element and report where it actually landed.
    std::printf("         discovered permutation (logical -> slot):");
    for (int l = 0; l < N && l < 32; ++l) {
      std::fill(h_src.begin(), h_src.end(), 0u);
      int const w = l / (32 / Bits), b = (l % (32 / Bits)) * Bits;
      h_src[w] = 1u << b;
      rt_h2d(d_src, h_src.data(), h_src.size() * 4);
      cvt_probe<FP16DetailsA, Bits, SubOffset, N><<<1, 1>>>(d_src, d_f, d_r, d_m);
      rt_sync("probe");
      rt_d2h(h_f.data(), d_f, N * 4);
      float base = SubOffset ? 0.f : 1024.f;
      int found = -1;
      for (int s = 0; s < N; ++s) if (h_f[s] == base + 1.f) { found = s; break; }
      std::printf(" %d->%d", l, found);
    }
    std::printf("\n");
    ++g_fail;
  }
  rt_free(d_src); rt_free(d_f); rt_free(d_r); rt_free(d_m);
}

// ---------------------------------------------------------------------------------------------------
// One end-to-end case.
struct CaseSpec {
  int rows;         // dense: m. grouped: rows per expert (uniform) unless ragged.
  int N, K;
  QuantOp qop;
  int gs;
  int L;            // 0 => dense
  bool ragged;
  bool exact;
  bool bias;
};

template <typename Details, int CtaN, int Chunk>
static bool run_case(CaseSpec const& c, uint32_t seed) {
  if (!gs_instantiated(c.gs)) { ++g_skip; return true; }
  constexpr int LoBits = Details::kLoBits;
  constexpr int HiBits = Details::kHiBits;
  constexpr bool TwoPlane = Details::kTwoPlane;
  constexpr int TS = Details::kTileSizeK;
  constexpr WLayout Lay = Details::kLayout;

  int const L = c.L;
  int const experts = L > 0 ? L : 1;
  std::vector<int> rows_per(experts, c.rows);
  if (c.ragged) for (int e = 0; e < experts; ++e) rows_per[e] = 1 + (e * 3) % std::max(1, c.rows);
  std::vector<int> offs(experts + 1, 0);
  for (int e = 0; e < experts; ++e) offs[e + 1] = offs[e] + rows_per[e];
  int const total_rows = offs[experts];
  int const max_rows = *std::max_element(rows_per.begin(), rows_per.end());

  int const N = c.N, K = c.K;
  int const gs = c.gs;
  int const sk = (gs == 0) ? 1 : K / gs;

  std::mt19937 gen(seed);
  // ---- logical codes ----
  //
  // EXACT MODE HAS A REPRESENTABILITY BUDGET, and it is the CODE RANGE that has to give. Each thread sums
  // depth = K/Threads products into one output, so the largest partial is depth*max|A|*(q_max + |z|/s) in
  // units of s; fp16 holds integers exactly only to 2^11, so q_max is capped at 2048/depth - |z|/s. With
  // A in {-1,+1} and z/s in {-4,0,4} that is 124 at depth 16 and 60 at depth 32 -- full range for every
  // format except int8, whose top bits are covered by the converter gate's random full-word probe instead.
  // Reporting the cap rather than silently shrinking the test is the point: a "BIT-EXACT" line that quietly
  // tested a third of the value range would be worse than a tolerance.
  int const depth = K / Details::kThreads;
  int const q_full = (1 << (LoBits + HiBits)) - 1;
  int const q_cap  = c.exact ? std::max(1, 2048 / depth - 4) : q_full;
  int const q_max  = std::min(q_full, q_cap);
  bool const capped = q_max < q_full;
  std::vector<std::vector<uint8_t>> qlo(experts), qhi(experts);
  for (int e = 0; e < experts; ++e) {
    qlo[e].resize(size_t(N) * K);
    if (TwoPlane) qhi[e].resize(size_t(N) * K);
    for (size_t i = 0; i < qlo[e].size(); ++i) {
      int const q = int(gen() % unsigned(q_max + 1));   // combined code, then split across the planes
      qlo[e][i] = uint8_t(q & ((1 << LoBits) - 1));
      if (TwoPlane) qhi[e][i] = uint8_t(q >> LoBits);
    }
  }
  // ---- activations, scales, zeros, bias ----
  std::vector<float> A(size_t(total_rows) * K), S(size_t(experts) * sk * N), Z(size_t(experts) * sk * N, 0.f),
                     Bs(N, 0.f);
  if (c.exact) {
    for (auto& v : A) v = (gen() & 1u) ? 1.f : -1.f;             // {-1,+1}: no zeros, so every k contributes
    for (auto& v : S) v = 0.25f;                                // 2^-2
    if (has_zero(c.qop)) for (auto& v : Z) v = float(int(gen() % 3) - 1);   // {-1,0,1}
    if (c.bias) for (auto& v : Bs) v = float(int(gen() % 5) - 2);
    (void)0;
  } else {
    std::uniform_real_distribution<float> da(-1.f, 1.f), ds(0.005f, 0.02f), dz(-0.1f, 0.1f);
    for (auto& v : A) v = da(gen);
    for (auto& v : S) v = ds(gen);
    if (has_zero(c.qop)) for (auto& v : Z) v = dz(gen);
    if (c.bias) for (auto& v : Bs) v = da(gen);
  }
  // fp16 round-trip so the golden sees exactly what the kernel sees
  auto h16 = [](float v) { return __half2float(__float2half(v)); };
  for (auto& v : A) v = h16(v);
  for (auto& v : S) v = h16(v);
  for (auto& v : Z) v = h16(v);
  for (auto& v : Bs) v = h16(v);

  // ---- golden ----
  std::vector<float> G(size_t(total_rows) * N, 0.f);
  for (int e = 0; e < experts; ++e)
    for (int r = 0; r < rows_per[e]; ++r) {
      int const row = offs[e] + r;
      for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
          int q = qlo[e][size_t(n) * K + k];
          if (TwoPlane) q += int(qhi[e][size_t(n) * K + k]) << LoBits;
          int const g = (gs == 0) ? 0 : k / gs;
          float const s = S[(size_t(e) * sk + g) * N + n];
          float const z = has_zero(c.qop) ? Z[(size_t(e) * sk + g) * N + n] : 0.f;
          acc += A[size_t(row) * K + k] * (float(q) * s + z);
        }
        G[size_t(row) * N + n] = acc + (c.bias ? Bs[n] : 0.f);
      }
    }

  // ---- device buffers ----
  auto to_half = [](std::vector<float> const& v) {
    std::vector<__half> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = __float2half(v[i]);
    return o;
  };
  std::vector<uint8_t> pk_lo, pk_hi;
  for (int e = 0; e < experts; ++e) {
    auto p = pack_plane(Lay, LoBits, TS, qlo[e], N, K);
    pk_lo.insert(pk_lo.end(), p.begin(), p.end());
    if (TwoPlane) {
      auto ph = pack_plane(Lay, HiBits, TS, qhi[e], N, K);
      pk_hi.insert(pk_hi.end(), ph.begin(), ph.end());
    }
  }
  auto hA = to_half(A), hS = to_half(S), hZ = to_half(Z), hB = to_half(Bs);

  void *dA, *dS, *dZ = nullptr, *dB = nullptr, *dW, *dWh = nullptr, *dO;
  int* dOff = nullptr;
  dA = rt_malloc(hA.size() * 2);   rt_h2d(dA, hA.data(), hA.size() * 2);
  dS = rt_malloc(hS.size() * 2);   rt_h2d(dS, hS.data(), hS.size() * 2);
  if (has_zero(c.qop)) { dZ = rt_malloc(hZ.size() * 2); rt_h2d(dZ, hZ.data(), hZ.size() * 2); }
  if (c.bias)          { dB = rt_malloc(hB.size() * 2); rt_h2d(dB, hB.data(), hB.size() * 2); }
  dW = rt_malloc(pk_lo.size());    rt_h2d(dW, pk_lo.data(), pk_lo.size());
  if (TwoPlane) { dWh = rt_malloc(pk_hi.size()); rt_h2d(dWh, pk_hi.data(), pk_hi.size()); }
  dO = rt_malloc(size_t(total_rows) * N * 2);
  rt_memset0(dO, size_t(total_rows) * N * 2);
  if (L > 0) { dOff = (int*)rt_malloc(offs.size() * 4); rt_h2d(dOff, offs.data(), offs.size() * 4); }

  Params p;
  p.act = dA; p.weight = dW; p.weight_hi = dWh; p.scales = dS; p.zeros = dZ; p.bias = dB; p.out = dO;
  p.alpha = 1.f;
  p.m = total_rows; p.n = N; p.k = K; p.groupsize = gs;
  p.format = Details::kFormat; p.quant = c.qop; p.layout = Lay; p.is_bf16 = false;
  if (L > 0) {
    p.num_experts = L; p.row_offsets = dOff; p.max_rows = max_rows;
    p.w_bytes_per_expert = int64_t(pk_lo.size()) / experts;
    p.w_hi_bytes_per_expert = TwoPlane ? int64_t(pk_hi.size()) / experts : 0;
    p.scale_elems_per_expert = int64_t(sk) * N;
  }

  int const fail0 = gemv_fail_count();
  bool const launched = launch_gemv<Details, CtaN, Chunk>(p, 0);
  rt_sync("probe");

  bool ok = launched && gemv_fail_count() == fail0;
  double max_rel = 0.0; int bad = 0; int bad_at = -1;
  if (ok) {
    std::vector<__half> hO(size_t(total_rows) * N);
    rt_d2h(hO.data(), dO, hO.size() * 2);
    double maxg = 1e-9;
    for (auto v : G) maxg = std::max(maxg, double(std::fabs(v)));
    for (size_t i = 0; i < hO.size(); ++i) {
      double const got = __half2float(hO[i]);
      double const want = double(__half2float(__float2half(G[i])));
      if (c.exact) {
        if (got != want) { if (bad == 0) bad_at = int(i); ++bad; }
      } else {
        // NaN would slip through here: std::max(max_rel, NaN) returns max_rel and `NaN > 4e-3` is false. The
        // exact branch above is immune only by accident -- `got != want` is TRUE for NaN.
        if (!std::isfinite(got)) { if (bad == 0) bad_at = int(i); ++bad; continue; }
        double const rel = std::fabs(got - want) / maxg;
        max_rel = std::max(max_rel, rel);
        if (rel > 4e-3) { if (bad == 0) bad_at = int(i); ++bad; }
      }
    }
    ok = (bad == 0);
  }

  char tag[256];
  std::snprintf(tag, sizeof(tag), "%-8s %-6s CtaN=%d Ch=%d %-14s gs=%-4d rows=%-3d N=%d K=%d %s%s",
                Details::format_name(), name_of(Lay), CtaN, Chunk, name_of(c.qop), gs,
                c.rows, N, K, L > 0 ? (c.ragged ? "MoE-ragged " : "MoE ") : "dense ",
                c.exact ? "exact" : "rand");
  char cap_note[48] = "";
  if (capped) std::snprintf(cap_note, sizeof(cap_note), " [codes<=%d of %d]", q_max, q_full);
  if (ok) {
    if (c.exact) std::printf("  [ok]   %s  BIT-EXACT%s\n", tag, cap_note);
    else         std::printf("  [ok]   %s  max_rel=%.2e\n", tag, max_rel);
    ++g_pass;
  } else if (!launched || gemv_fail_count() != fail0) {
    std::printf("  [FAIL] %s  LAUNCH REFUSED\n", tag);
    ++g_fail;
  } else {
    std::printf("  [FAIL] %s  %d bad, first at %d (row %d col %d), max_rel=%.2e\n",
                tag, bad, bad_at, bad_at / N, bad_at % N, max_rel);
    ++g_fail;
  }

  rt_free(dA); rt_free(dS); rt_free(dW); rt_free(dO);
  if (dZ) rt_free(dZ);
  if (dB) rt_free(dB);
  if (dWh) rt_free(dWh);
  if (dOff) rt_free(dOff);
  return ok;
}

// ---------------------------------------------------------------------------------------------------
// LAYOUT IDENTITY GATE. The iterator reads address (base + iter*step + col*stride); the format's DEFINITION
// is the cute Layout. This checks they are the same object at every point the kernel touches, for every
// (layout, width, StepK, Threads, TS) actually instantiated. Without it the closed form is a SECOND
// definition of the format, free to drift from the one the record describes.
template <WLayout Lay, int Bits, int StepK, int Threads, int TS>
static void gate_layout(const char* tag) {
  using Tr = WFormatTraits<Lay, Bits, StepK, Threads, TS>;
  constexpr int TPT = Tr::kTPT;
  int bad = 0, checked = 0, stride_bad = 0;
  int first_n = -1, first_tid = -1, first_iter = -1, first_col = -1;
  int64_t got0 = 0, want0 = 0;
  for (int N : {64, 128, 2048})
    for (int K : {Threads * StepK, Threads * StepK * 4}) {
      int const CtaK = Threads * StepK;
      WStrides const w = Tr::strides(N, K);
      for (int n0 = 0; n0 + 8 <= N; n0 += std::max(1, N / 5))
        for (int tid = 0; tid < Threads; tid += std::max(1, Threads / 8))
          for (int iter = 0; iter < K / CtaK; ++iter)
            for (int col = 0; col < 8; ++col) {
              // exactly the address the kernel forms
              int64_t const got = int64_t(n0) * w.col
                                + int64_t(tid / TPT) * w.thr_major
                                + int64_t(tid % TPT) * w.thr_minor
                                + int64_t(iter) * w.iter
                                + int64_t(col) * w.col;
              // what the format means, written out independently
              int const k = iter * CtaK + tid * StepK;
              int64_t want;
              if constexpr (Lay == WLayout::Native)
                want = (int64_t(n0 + col) * K + k) * Bits / 8;
              else
                want = ((int64_t(k / TS) * N * TS) + int64_t(n0 + col) * TS + (k % TS)) * Bits / 8;
              ++checked;
              if (got != want) {
                if (bad == 0) { first_n = n0; first_tid = tid; first_iter = iter; first_col = col;
                                got0 = got; want0 = want; }
                ++bad;
              }
            }
    }
  (void)stride_bad;
  if (bad == 0) {
    std::printf("  [ok]   layout   %-24s %d points, cute strides == format\n", tag, checked);
    ++g_pass;
  } else {
    std::printf("  [FAIL] layout   %-24s %d/%d bad; first n0=%d tid=%d iter=%d col=%d got %lld want %lld\n",
                tag, bad, checked, first_n, first_tid, first_iter, first_col,
                (long long)got0, (long long)want0);
    ++g_fail;
  }
}

// ---------------------------------------------------------------------------------------------------
// RECORD GATE, with the negative control that makes it mean something: a record describing a DIFFERENT
// offline format must be refused, not read.
template <typename Details, int CtaN, int Chunk>
static void gate_record(const char* tag) {
  int const N = 64, K = 2048, gs = 32;
  QuantOp const q = QuantOp::FinegrainedScaleOnly;
  auto rec = wfmt_record<Details>(N, K, gs, q, 0);
  char const* why = "";

  bool const self_ok = wfmt_matches<Details>(rec, N, K, gs, q, &why);

  // round-trip through a file, as the offline pass would write it
  char path[256];
  std::snprintf(path, sizeof(path), "/tmp/claude-0/-root/57027199-de80-4d5b-b901-e3ed437519e8/scratchpad/wfmt_%s.bin", tag);
  WeightFormatRecord back{};
  bool const io_ok = wfmt_write(path, rec) && wfmt_read(path, back)
                  && std::memcmp(&rec, &back, sizeof(rec)) == 0;

  // negative controls: each perturbation must be REFUSED
  int refused = 0, controls = 0;
  auto expect_refusal = [&](WeightFormatRecord r, const char* what) {
    ++controls;
    char const* w = "";
    if (!wfmt_matches<Details>(r, N, K, gs, q, &w)) ++refused;
    else std::printf("         !! accepted a record with a wrong %s\n", what);
  };
  { auto r = rec; r.layout = int32_t(WLayout::FoldCube); r.fold = 2; r.tm = 32; expect_refusal(r, "layout (cube-folded)"); }
  { auto r = rec; r.lo_bits += 1;                        expect_refusal(r, "low-plane width"); }
  { auto r = rec; r.group_size = 128;                    expect_refusal(r, "group size"); }
  { auto r = rec; r.k = K / 2;                           expect_refusal(r, "problem shape"); }
  { auto r = rec; r.tile_k = (r.tile_k == 256 ? 128 : 256); r.layout = int32_t(WLayout::TileK);
                                                         expect_refusal(r, "tile K"); }
  { auto r = rec; std::snprintf(r.layout_str_lo, sizeof(r.layout_str_lo), "(1,1):(1,1)");
                                                         expect_refusal(r, "layout string"); }
  { auto r = rec; r.magic[0] = 'X';                      expect_refusal(r, "magic"); }

  // and the launcher must actually honour it
  int const fail0 = gemv_fail_count();
  Params bad_p;
  bad_p.n = N; bad_p.k = K; bad_p.groupsize = gs; bad_p.quant = q; bad_p.m = 1;
  bad_p.format = Details::kFormat; bad_p.layout = Details::kLayout;
  auto bad_rec = rec; bad_rec.group_size = 128;
  bad_p.record = &bad_rec;
  bool const launched = launch_gemv<Details, CtaN, Chunk>(bad_p, 0);
  bool const launcher_refused = !launched && gemv_fail_count() == fail0 + 1;

  bool const ok = self_ok && io_ok && refused == controls && launcher_refused;
  if (ok) {
    std::printf("  [ok]   record   %-24s self+io, %d/%d controls refused, launcher honours it\n",
                tag, refused, controls);
    ++g_pass;
  } else {
    std::printf("  [FAIL] record   %-24s self=%d io=%d controls=%d/%d launcher_refused=%d (%s)\n",
                tag, int(self_ok), int(io_ok), refused, controls, int(launcher_refused), why);
    ++g_fail;
  }
  if (self_ok && io_ok) wfmt_print(rec);
}

// ---------------------------------------------------------------------------------------------------
// Instantiations. StepK is set by the SPARSEST plane (StepK*min(bits) >= 32) and Threads by CtaK <= K.
template <WFormat F, WLayout L> struct Pick;
template <WLayout L> struct Pick<WFormat::Int8,  L> { using D = KernelDetails<FP16DetailsA, WFormat::Int8,  L, 16, 128>; };
template <WLayout L> struct Pick<WFormat::Int4,  L> { using D = KernelDetails<FP16DetailsA, WFormat::Int4,  L, 16, 128>; };
template <WLayout L> struct Pick<WFormat::Int2,  L> { using D = KernelDetails<FP16DetailsA, WFormat::Int2,  L, 16, 128>; };
template <WLayout L> struct Pick<WFormat::Int1,  L> { using D = KernelDetails<FP16DetailsA, WFormat::Int1,  L, 32,  64>; };
template <WLayout L> struct Pick<WFormat::Q6_42, L> { using D = KernelDetails<FP16DetailsA, WFormat::Q6_42, L, 16, 128>; };
template <WLayout L> struct Pick<WFormat::Q5_41, L> { using D = KernelDetails<FP16DetailsA, WFormat::Q5_41, L, 32,  64>; };
template <WLayout L> struct Pick<WFormat::Q3_21, L> { using D = KernelDetails<FP16DetailsA, WFormat::Q3_21, L, 32,  64>; };

int main(int argc, char** argv) {
  bool const quick = (argc > 1 && std::string(argv[1]) == "quick");

  std::printf("== layout identity gate (iterator triple == cute Layout) ==\n");
  gate_layout<WLayout::Native, 4, 16, 128, 256>("native int4 s16 t128");
  gate_layout<WLayout::Native, 2, 16, 128, 256>("native int2 s16 t128");
  gate_layout<WLayout::Native, 1, 32,  64, 256>("native int1 s32 t64");
  gate_layout<WLayout::Native, 8, 16, 128, 256>("native int8 s16 t128");
  gate_layout<WLayout::TileK,  4, 16, 128, 256>("tileK  int4 s16 t128 TS256");
  gate_layout<WLayout::TileK,  2, 16, 128, 256>("tileK  int2 s16 t128 TS256");
  gate_layout<WLayout::TileK,  1, 32,  64, 256>("tileK  int1 s32 t64  TS256");
  gate_layout<WLayout::TileK,  4, 16, 128, 128>("tileK  int4 s16 t128 TS128");

  std::printf("\n== weight format record ==\n");
  gate_record<Pick<WFormat::Int4, WLayout::Native>::D, 8, 2>("int4-native");
  gate_record<Pick<WFormat::Int4, WLayout::TileK>::D,  8, 2>("int4-tileK");
  gate_record<Pick<WFormat::Q3_21, WLayout::Native>::D, 8, 2>("q3-native");

  std::printf("\n== converter gate ==\n");
  gate_converter<8, false>("int8");
  gate_converter<4, false>("int4");
  gate_converter<2, false>("int2");
  gate_converter<1, false>("int1");
  gate_converter<2, true>("int2 hi(-off)");
  gate_converter<1, true>("int1 hi(-off)");

  std::printf("\n== int4, full axis sweep (native) ==\n");
  {
    using D = Pick<WFormat::Int4, WLayout::Native>::D;
    for (int gsi = 0; gsi < 3; ++gsi) {
      int const gs = (gsi == 0 ? 32 : gsi == 1 ? 128 : 16);
      for (int qi = 0; qi < 2; ++qi) {
        QuantOp const q = qi ? QuantOp::FinegrainedScaleZero : QuantOp::FinegrainedScaleOnly;
        run_case<D, 2, 2>({1, 64, 2048, q, gs, 0, false, true, false}, 1);
        run_case<D, 4, 2>({1, 64, 2048, q, gs, 0, false, true, false}, 2);
        run_case<D, 8, 2>({1, 64, 2048, q, gs, 0, false, true, false}, 3);
        run_case<D, 8, 4>({1, 64, 2048, q, gs, 0, false, true, false}, 4);
        run_case<D, 8, 8>({1, 64, 2048, q, gs, 0, false, true, false}, 5);
      }
    }
    // per-column scale
    run_case<D, 8, 2>({1, 64, 2048, QuantOp::PerColScaleOnly, 0, 0, false, true, false}, 6);
    // rows > 1 (CtaM dispatch), bias, random magnitudes
    run_case<D, 8, 2>({2, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 0, false, true, true}, 7);
    run_case<D, 8, 2>({3, 64, 2048, QuantOp::FinegrainedScaleZero, 32, 0, false, true, true}, 8);
    run_case<D, 8, 2>({4, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 0, false, true, false}, 9);
    run_case<D, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleZero, 32, 0, false, false, true}, 10);
    run_case<D, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 0, false, false, false}, 10);
    run_case<D, 8, 2>({4, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 8, false, false, false}, 10);
    run_case<D, 8, 2>({1, 64, 2048, QuantOp::PerColScaleOnly,      0,  0, false, false, false}, 10);
    // MoE: uniform and ragged
    run_case<D, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 8, false, true, false}, 11);
    run_case<D, 8, 2>({4, 64, 2048, QuantOp::FinegrainedScaleZero, 32, 8, true, true, true}, 12);
    // tileK layout
    using DT = Pick<WFormat::Int4, WLayout::TileK>::D;
    run_case<DT, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 0, false, true, false}, 13);
    run_case<DT, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleZero, 32, 8, false, true, false}, 14);
  }

  if (!quick) {
    std::printf("\n== every other format, both layouts ==\n");
#define FMT_CASE(F)                                                                                     \
    {                                                                                                   \
      using DN = Pick<F, WLayout::Native>::D;                                                           \
      using DT = Pick<F, WLayout::TileK>::D;                                                            \
      run_case<DN, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 0, false, true, false}, 21);   \
      run_case<DN, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleZero, 32, 8, false, true, false}, 22);   \
      run_case<DN, 8, 2>({4, 64, 2048, QuantOp::FinegrainedScaleZero, 16, 8, true,  true, true}, 23);    \
      run_case<DN, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleZero, 32, 0, false, false, false}, 24);  \
      run_case<DN, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 0, false, false, false}, 27);  \
      run_case<DT, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleOnly, 32, 0, false, true, false}, 25);   \
      run_case<DT, 8, 2>({1, 64, 2048, QuantOp::FinegrainedScaleZero, 32, 8, false, true, false}, 26);   \
    }
    FMT_CASE(WFormat::Int8)
    FMT_CASE(WFormat::Int2)
    FMT_CASE(WFormat::Int1)
    FMT_CASE(WFormat::Q6_42)
    FMT_CASE(WFormat::Q5_41)
    FMT_CASE(WFormat::Q3_21)
#undef FMT_CASE
  }

#if defined(GEMV_GATE_FAST)
  std::printf("\n  *** GEMV_GATE_FAST: only gs {0,32} instantiated; %d rows SKIPPED (not passed). Build without\n"
              "      it before trusting a clean result. ***\n", g_skip);
#endif
  std::printf("\n== summary ==\n  %d passed, %d failed, %d skipped, %d launches refused\n",
              g_pass, g_fail, g_skip, gemv_fail_count());
  return g_fail == 0 ? 0 : 1;
}
