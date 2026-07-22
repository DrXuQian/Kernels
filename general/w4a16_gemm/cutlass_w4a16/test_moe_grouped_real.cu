// REAL-WEIGHT correctness gate for the grouped mixed-input W4A16 GEMM. Loads a .bin produced by
// real_weight/dump_real_weights.py (real Qwen3.5 GPTQ-Int4 or Q4_K_M expert weights already mapped into the
// collective's convention: q_signed in [-8,7], fp16 group scale[+zero]), runs the SAME pack +
// preprocess_weights_for_mixed_gemm<false,256> + filter_and_run path the validated verify test uses, and
// compares against the Python-side algorithm-independent fp16 golden (A @ dequant(W)).
//
//   ./test_moe_grouped_real <file.bin>
//
// .bin layout (little-endian) -- see dump_real_weights.py:
//   char[8] "RWMOE\0\0\0"; int32 L,M,N,K,gs,mode(0=scale-only,1=scale+zero)
//   f16 A[L][M][K]; int8 q_signed[L][K][N]; f16 scale[L][sk][N]; (f16 zero[L][sk][N] if mode==1); f16 golden[L][M][N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
using int4_t  = cutlass::int4b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

template <class T> static std::vector<T> rd(std::ifstream& f, size_t n) {
  std::vector<T> v(n); f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T)); return v;
}

// Dispatch filter_and_run by (mode, gs) -> the right QuantMode + tile. Mirrors the validated xcheck: TK=128 for
// gs>=128, TK=64 otherwise (gs=32/64 -> FINE per-mma-atom scale). All 64x64/32x32/s3, il auto by n,k%256.
static void run_grouped(QM mode, const half_t* A, const int4_t* B, const half_t* scale, const half_t* zero,
                        half_t** ptr_D, DStride* sd, int const* gm, int Mmax, int N, int K, int L, int gs,
                        GS* gsd, GS const* gsh, int const* offs, char* ws, size_t wsb) {
  #define CALL(QMODE, TK) moe_grouped_ppu::filter_and_run<QMODE, 64, 64, TK, 32, 32, 3>( \
      A, B, scale, zero, ptr_D, sd, gm, Mmax, N, K, L, gs, gsd, gsh, offs, ws, wsb, nullptr)
  if (mode == QM::FinegrainedScaleOnly) { if (gs >= 128) CALL(QM::FinegrainedScaleOnly, 128); else CALL(QM::FinegrainedScaleOnly, 64); }
  else                                   { if (gs >= 128) CALL(QM::FinegrainedScaleZero, 128); else CALL(QM::FinegrainedScaleZero, 64); }
  #undef CALL
}

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: %s <file.bin>\n", argv[0]); return 1; }
  std::ifstream f(argv[1], std::ios::binary);
  if (!f) { std::printf("cannot open %s\n", argv[1]); return 1; }
  char magic[8]; f.read(magic, 8);
  if (std::memcmp(magic, "RWMOE\0\0\0", 8) != 0) { std::printf("bad magic\n"); return 1; }
  int hdr[6]; f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  const int L = hdr[0], M = hdr[1], N = hdr[2], K = hdr[3], gs = hdr[4], mode = hdr[5];
  const int scale_k = (K + gs - 1) / gs;
  const QM qmode = (mode == 0) ? QM::FinegrainedScaleOnly : QM::FinegrainedScaleZero;
  std::printf("[real] L=%d M=%d N=%d K=%d gs=%d mode=%s scale_k=%d\n", L, M, N, K, gs,
              mode == 0 ? "scale-only" : "scale+zero", scale_k);
  if (N % 256 || K % 256) std::printf("  NOTE: N or K not %%256 -> non-interleaved path\n");

  // read all sections
  std::vector<std::vector<half_t>> A(L), scl(L), zro(L), gold(L);
  std::vector<std::vector<int8_t>> qs(L);
  for (int e = 0; e < L; ++e) A[e]  = rd<half_t>(f, (size_t)M * K);
  for (int e = 0; e < L; ++e) qs[e] = rd<int8_t>(f, (size_t)K * N);
  for (int e = 0; e < L; ++e) scl[e] = rd<half_t>(f, (size_t)scale_k * N);
  if (mode == 1) for (int e = 0; e < L; ++e) zro[e] = rd<half_t>(f, (size_t)scale_k * N);
  for (int e = 0; e < L; ++e) gold[e] = rd<half_t>(f, (size_t)M * N);

  // pack q_signed -> 2 nibbles/byte, then preprocess (SAME as test_moe_grouped_verify.cu), shape {K,N}
  std::vector<int8_t> Bbuf((size_t)L * K * N / 2);
  for (int e = 0; e < L; ++e) {
    std::vector<int8_t> packed((size_t)K * N / 2);
    for (size_t i = 0; i < (size_t)K * N / 2; ++i) {
      int8_t lo = qs[e][2 * i] & 0xF, hi = qs[e][2 * i + 1] & 0xF;
      packed[i] = int8_t((hi << 4) | lo);
    }
    preprocess_weights_for_mixed_gemm<false, 256>(
        (int8_t*)&Bbuf[(size_t)e * K * N / 2], packed.data(), {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);
  }

  // contiguous device buffers (uniform M per expert). total = L*M rows; D[total][N] contiguous, ptr_D[e]=D+e*M*N.
  const int total = L * M, Mmax = M;
  // assemble contiguous host arrays (L-strided single bases), then one copy each.
  std::vector<half_t> hAll((size_t)total * K), hScAll((size_t)L * scale_k * N), hZrAll(mode == 1 ? (size_t)L * scale_k * N : 0);
  for (int e = 0; e < L; ++e) {
    std::copy(A[e].begin(),   A[e].end(),   hAll.begin()   + (size_t)e * M * K);
    std::copy(scl[e].begin(), scl[e].end(), hScAll.begin() + (size_t)e * scale_k * N);
    if (mode == 1) std::copy(zro[e].begin(), zro[e].end(), hZrAll.begin() + (size_t)e * scale_k * N);
  }
  cutlass::DeviceAllocation<half_t> dA((size_t)total * K), dScale((size_t)L * scale_k * N),
      dZero(mode == 1 ? (size_t)L * scale_k * N : 1), dD((size_t)total * N);
  cutlass::DeviceAllocation<int4_t> dB((size_t)L * K * N);
  dA.copy_from_host(hAll.data());
  dScale.copy_from_host(hScAll.data());
  if (mode == 1) dZero.copy_from_host(hZrAll.data());
  dB.copy_from_host(reinterpret_cast<int4_t const*>(Bbuf.data()));

  std::vector<GS> shp(L); for (int e = 0; e < L; ++e) shp[e] = cute::make_shape(M, N, K);
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  auto out_stride = [&](int m) { return cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(m, N, 1)); };
  std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L), offs(L);
  for (int e = 0; e < L; ++e) { pdh[e] = dD.get() + (size_t)e * M * N; sdh[e] = out_stride(M); gmh[e] = M; offs[e] = e * M; }
  cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int>     gm(L); gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int>     offdev(L); offdev.copy_from_host(offs.data());
  const size_t wsb = (size_t)cutlass::ceil_div(Mmax, 16) * cutlass::ceil_div(N, 64) * (size_t)L * 64;
  cutlass::DeviceAllocation<char> ws(wsb);

  run_grouped(qmode, dA.get(), dB.get(), dScale.get(), mode == 1 ? dZero.get() : nullptr,
              pd.get(), sd.get(), gm.get(), Mmax, N, K, L, gs, shpd.get(), shp.data(),
              /*offsets=*/offdev.get(), ws.get(), wsb);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<half_t> hD((size_t)total * N); dD.copy_to_host(hD.data());

  auto cmp = [&](const char* name, std::vector<std::vector<double>> const& G) {
    double mr = 0; int bad = 0, we = -1;
    for (int e = 0; e < L; ++e)
      for (size_t i = 0; i < (size_t)M * N; ++i) {
        double g = G[e][i], got = float(hD[(size_t)e * M * N + i]);
        double ad = std::abs(got - g), rel = ad / (std::abs(g) + 1e-3);
        if (rel > mr) { mr = rel; we = e; }
        if (ad > 1e-2 + 5e-2 * std::abs(g)) ++bad;   // abs floor tolerates near-zero fp16 noise
      }
    std::printf("  grp vs %-28s max_rel=%.3e (worst e=%d) bad=%d/%zu -> %s\n",
                name, mr, we, bad, (size_t)L * M * N, bad == 0 ? "MATCH" : "MISMATCH");
    return bad == 0;
  };

  // Candidate host goldens to localize the orientation. q buf is [K*N], scale buf is [scale_k*N] (G1's layout).
  //  G1 = Python golden (scale[sk][n]=scale_buf[sk*N+n], q[k][n]=q_buf[k*N+n]).
  //  G2 = scale read n-major: scale_buf[n*scale_k+sk].
  //  G3 = q read transposed (as [N][K]): q_buf[n*K+k].
  //  G4 = both transposed (q [N][K] + scale n-major).
  std::vector<std::vector<double>> Gpy(L), G2(L), G3(L), G4(L);
  for (int e = 0; e < L; ++e) {
    Gpy[e].assign((size_t)M * N, 0); G2[e].assign((size_t)M * N, 0);
    G3[e].assign((size_t)M * N, 0);  G4[e].assign((size_t)M * N, 0);
    for (size_t i = 0; i < (size_t)M * N; ++i) Gpy[e][i] = float(gold[e][i]);
  }
  for (int e = 0; e < L; ++e)
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double a2 = 0, a3 = 0, a4 = 0;
        for (int k = 0; k < K; ++k) {
          int sk = k / gs;
          double sc_sk_n = float(scl[e][(size_t)sk * N + n]);   // scale[sk][n]
          double sc_n_sk = float(scl[e][(size_t)n * scale_k + sk]); // scale[n][sk]
          double q_k_n = (double)qs[e][(size_t)k * N + n];       // q[k][n]
          double q_n_k = (double)qs[e][(size_t)n * K + k];       // q as [N][K]
          double z_sk_n = mode == 1 ? (double)float(zro[e][(size_t)sk * N + n]) : 0.0;
          double z_n_sk = mode == 1 ? (double)float(zro[e][(size_t)n * scale_k + sk]) : 0.0;
          double a = (double)float(A[e][(size_t)m * K + k]);
          a2 += a * (sc_n_sk * q_k_n + z_n_sk);
          a3 += a * (sc_sk_n * q_n_k + z_sk_n);
          a4 += a * (sc_n_sk * q_n_k + z_n_sk);
        }
        G2[e][(size_t)m * N + n] = a2; G3[e][(size_t)m * N + n] = a3; G4[e][(size_t)m * N + n] = a4;
      }

  bool ok = cmp("G1 python (scale[sk][n],q[k][n])", Gpy);
  cmp("G2 scale[n][sk]                ", G2);
  cmp("G3 q[n][k] transposed          ", G3);
  cmp("G4 both transposed             ", G4);
  std::printf("  gold[e0][0..5] ="); for (int i = 0; i < 6; ++i) std::printf(" %8.3f", float(gold[0][i]));
  std::printf("\n  grp [e0][0..5] ="); for (int i = 0; i < 6; ++i) std::printf(" %8.3f", float(hD[i]));
  std::printf("\n");
  return ok ? 0 : 1;
}
