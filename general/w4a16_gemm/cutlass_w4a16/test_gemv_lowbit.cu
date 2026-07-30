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

#include <cuda_runtime.h>
#include "gemv_lowbit/gemv_launcher.hpp"

using namespace ppu_gemv;

#define CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
  std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); std::exit(1); } } while (0)

static int g_pass = 0, g_fail = 0;

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
  CHECK(cudaMalloc(&d_src, h_src.size() * 4));
  CHECK(cudaMalloc(&d_f, N * 4)); CHECK(cudaMalloc(&d_r, N * 4)); CHECK(cudaMalloc(&d_m, N * 4));

  bool ok = true;
  int bad_first = -1;
  std::vector<float> h_f(N), h_r(N);
  std::vector<int> h_m(N);
  for (int trial = 0; trial < 64 && ok; ++trial) {
    for (auto& w : h_src) w = gen();
    CHECK(cudaMemcpy(d_src, h_src.data(), h_src.size() * 4, cudaMemcpyHostToDevice));
    cvt_probe<FP16DetailsA, Bits, SubOffset, N><<<1, 1>>>(d_src, d_f, d_r, d_m);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_f.data(), d_f, N * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_r.data(), d_r, N * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_m.data(), d_m, N * 4, cudaMemcpyDeviceToHost));
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
      CHECK(cudaMemcpy(d_src, h_src.data(), h_src.size() * 4, cudaMemcpyHostToDevice));
      cvt_probe<FP16DetailsA, Bits, SubOffset, N><<<1, 1>>>(d_src, d_f, d_r, d_m);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(h_f.data(), d_f, N * 4, cudaMemcpyDeviceToHost));
      float base = SubOffset ? 0.f : 1024.f;
      int found = -1;
      for (int s = 0; s < N; ++s) if (h_f[s] == base + 1.f) { found = s; break; }
      std::printf(" %d->%d", l, found);
    }
    std::printf("\n");
    ++g_fail;
  }
  CHECK(cudaFree(d_src)); CHECK(cudaFree(d_f)); CHECK(cudaFree(d_r)); CHECK(cudaFree(d_m));
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
  CHECK(cudaMalloc(&dA, hA.size() * 2));   CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&dS, hS.size() * 2));   CHECK(cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice));
  if (has_zero(c.qop)) { CHECK(cudaMalloc(&dZ, hZ.size() * 2)); CHECK(cudaMemcpy(dZ, hZ.data(), hZ.size() * 2, cudaMemcpyHostToDevice)); }
  if (c.bias)          { CHECK(cudaMalloc(&dB, hB.size() * 2)); CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice)); }
  CHECK(cudaMalloc(&dW, pk_lo.size()));    CHECK(cudaMemcpy(dW, pk_lo.data(), pk_lo.size(), cudaMemcpyHostToDevice));
  if (TwoPlane) { CHECK(cudaMalloc(&dWh, pk_hi.size())); CHECK(cudaMemcpy(dWh, pk_hi.data(), pk_hi.size(), cudaMemcpyHostToDevice)); }
  CHECK(cudaMalloc(&dO, size_t(total_rows) * N * 2));
  CHECK(cudaMemset(dO, 0, size_t(total_rows) * N * 2));
  if (L > 0) { CHECK(cudaMalloc(&dOff, offs.size() * 4)); CHECK(cudaMemcpy(dOff, offs.data(), offs.size() * 4, cudaMemcpyHostToDevice)); }

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
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  bool ok = launched && gemv_fail_count() == fail0;
  double max_rel = 0.0; int bad = 0; int bad_at = -1;
  if (ok) {
    std::vector<__half> hO(size_t(total_rows) * N);
    CHECK(cudaMemcpy(hO.data(), dO, hO.size() * 2, cudaMemcpyDeviceToHost));
    double maxg = 1e-9;
    for (auto v : G) maxg = std::max(maxg, double(std::fabs(v)));
    for (size_t i = 0; i < hO.size(); ++i) {
      double const got = __half2float(hO[i]);
      double const want = double(__half2float(__float2half(G[i])));
      if (c.exact) {
        if (got != want) { if (bad == 0) bad_at = int(i); ++bad; }
      } else {
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

  cudaFree(dA); cudaFree(dS); cudaFree(dW); cudaFree(dO);
  if (dZ) cudaFree(dZ);
  if (dB) cudaFree(dB);
  if (dWh) cudaFree(dWh);
  if (dOff) cudaFree(dOff);
  return ok;
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

  std::printf("== converter gate ==\n");
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

  std::printf("\n== summary ==\n  %d passed, %d failed, %d launches refused\n",
              g_pass, g_fail, gemv_fail_count());
  return g_fail == 0 ? 0 : 1;
}
