// W4A16 DEQUANT-isolation diag [box-only] -- the INT4 REFERENCE counterpart of test_w2a16_diag, SAME params
// (L=1, M=N=K=256, gs=32, identity A, same filter_and_run tile template) so the W2DBG layout print for int4
// (KNOWN-GOOD) can be compared line-by-line against int2's, to see whether the offline->cute-layout mapping
// and the mma-needed layout diverge for int2.  Offline input nibble v (signed -8..7); preprocess adds +8;
// converter outputs v; golden W = scale*v + zero.
//   TARGET=test_w4a16_diag ./build.sh ; ./<bin>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"

using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using GS     = moe_grouped_ppu::GroupShape;
using DStride= moe_grouped_ppu::DStride;
using QM     = moe_grouped_ppu::QuantMode;

int main() {
  const int L = 1, M = 256, N = 256, K = 256, gs = 32;
  const int scale_k = (K + gs - 1) / gs;
  std::srand(4321);
  std::printf("[w4a16-diag DEQUANT ref] identity A, RANDOM q4(signed)/scale/zero; D[m][n] should = W[m][n]\n");

  std::vector<int>   q4((size_t)K * N);          // nibble 0..15 (interpreted signed -8..7)
  std::vector<float> hsc((size_t)scale_k * N), hzr((size_t)scale_k * N), hA((size_t)M * K, 0.f);
  for (auto& q : q4)  q = std::rand() % 16;
  for (auto& s : hsc) s = 0.01f + (std::rand() % 8) * 0.01f;
  for (auto& z : hzr) z = (std::rand() % 9 - 4) * 0.01f;
  for (int m = 0; m < M; ++m) hA[(size_t)m*K + m] = 1.f;                     // identity A

  auto sgn = [](int v){ return v < 8 ? v : v - 16; };
  std::vector<double> gD((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    int g = m / gs;
    gD[(size_t)m*N+n] = (double)hsc[(size_t)g*N+n]*sgn(q4[(size_t)m*N+n]) + hzr[(size_t)g*N+n];  // W[m][n]
  }

  // transpose q [K][N]->[N][K] + pack 2 int4/byte + preprocess PACKED_INT4
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n*K + k] = q4[(size_t)k*N + n];
  std::vector<int8_t> packed((size_t)K * N / 2, 0);
  for (size_t i = 0; i < (size_t)K * N / 2; ++i)
    packed[i] = int8_t((qT[2*i] & 0xF) | ((qT[2*i+1] & 0xF) << 4));
  std::vector<int8_t> Bbuf((size_t)K * N / 2);
  preprocess_weights_for_mixed_gemm<false, 256>(
      Bbuf.data(), packed.data(), {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);

  std::vector<half_t> hA16(hA.size()), hSc16(hsc.size()), hZr16(hzr.size());
  for (size_t i=0;i<hA.size();++i)  hA16[i]  = half_t(hA[i]);
  for (size_t i=0;i<hsc.size();++i) hSc16[i] = half_t(hsc[i]);
  for (size_t i=0;i<hzr.size();++i) hZr16[i] = half_t(hzr[i]);
  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dScale((size_t)scale_k*N), dZero((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<int4_t> dB((size_t)K*N);
  dA.copy_from_host(hA16.data()); dScale.copy_from_host(hSc16.data()); dZero.copy_from_host(hZr16.data());
  dB.copy_from_host(reinterpret_cast<int4_t const*>(Bbuf.data()));

  std::vector<GS> shp(L, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  auto out_stride = [&](int m){ return cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(m, N, 1)); };
  std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L), offs(L);
  for (int e=0;e<L;++e){ pdh[e]=dD.get()+(size_t)e*M*N; sdh[e]=out_stride(M); gmh[e]=M; offs[e]=e*M; }
  cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(L); gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  const size_t wsb = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*(size_t)L*64;
  cutlass::DeviceAllocation<char> ws(wsb);

  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3, cutlass::int4b_t>(
      dA.get(), dB.get(), dScale.get(), dZero.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hD((size_t)M*N); dD.copy_to_host(hD.data());

  int bad = 0, shown = 0;
  for (size_t i=0;i<(size_t)M*N;++i) if (std::abs((double)float(hD[i])-gD[i]) > 1e-2 + 5e-2*std::abs(gD[i])) ++bad;
  std::printf("  bad=%d/%d %s\n", bad, M*N, bad==0?"MATCH":"MISMATCH");
  std::printf("  mismatches (m,n,g | q4 sc zr | got exp):\n");
  for (int m=0; m<M && shown<12; ++m) for (int n=0; n<N && shown<12; ++n) {
    double got=(double)float(hD[(size_t)m*N+n]), exp=gD[(size_t)m*N+n];
    if (std::abs(got-exp) > 1e-2 + 5e-2*std::abs(exp)) {
      int g=m/gs;
      std::printf("    m=%3d n=%3d g=%d | q4=%2d(v%3d) sc=%.2f zr=%.2f | got=%.3f exp=%.3f\n",
                  m,n,g, q4[(size_t)m*N+n], sgn(q4[(size_t)m*N+n]), hsc[(size_t)g*N+n], hzr[(size_t)g*N+n], got, exp);
      ++shown;
    }
  }
  return 0;
}
