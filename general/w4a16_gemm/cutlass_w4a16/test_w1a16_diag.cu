// W1A16 DEQUANT-isolation diag [box-only]. A=identity (M=K) so D[m][n]=W[m][n] -> isolates dequant/layout from
// the K-contraction. Random q1/scale/zero (sensitive to the N/within-reg permutation). Reports bad + the first
// mismatches with context so the residual permutation (if any) is visible. Mirror of test_w2a16_diag (uint1b_t,
// 8 uint1/byte, PACKED_INT1, TileShapeK=256 which satisfies int1's Block_K%256==0).
//   TARGET=test_w1a16_diag ./build.sh ; ./<bin>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

int main() {
  // FULL dequant test: RANDOM q1/scale/zero. A=identity so D[m][n] = W[m][n] = scale*q1 + zero, q1 in {0,1}.
  const int L = 1, M = 256, N = 256, K = 256, gs = 32;
  const int scale_k = (K + gs - 1) / gs;
  std::srand(4321);
  std::printf("[w1a16-diag DEQUANT] identity A, RANDOM q1/scale/zero; D[m][n] should = W[m][n]\n");

  std::vector<int>   q1((size_t)K * N);
  std::vector<float> hsc((size_t)scale_k * N), hzr((size_t)scale_k * N), hA((size_t)M * K, 0.f);
  for (auto& q : q1)  q = std::rand() % 2;                                  // 1-bit
  for (auto& s : hsc) s = 0.01f + (std::rand() % 8) * 0.01f;
  for (auto& z : hzr) z = (std::rand() % 9 - 4) * 0.01f;
  for (int m = 0; m < M; ++m) hA[(size_t)m*K + m] = 1.f;                    // identity A

  std::vector<double> gD((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    int g = m / gs;
    gD[(size_t)m*N+n] = (double)hsc[(size_t)g*N+n]*q1[(size_t)m*N+n] + hzr[(size_t)g*N+n];   // W[m][n]
  }

  // transpose q [K][N]->[N][K] + pack 8 uint1/byte + preprocess PACKED_INT1
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n*K + k] = q1[(size_t)k*N + n];
  std::vector<int8_t> packed((size_t)K * N / 8, 0);
  for (size_t i = 0; i < (size_t)K * N / 8; ++i) {
    int8_t b = 0;
    for (int t = 0; t < 8; ++t) b |= int8_t((qT[8*i+t]&1) << t);
    packed[i] = b;
  }
  std::vector<int8_t> Bbuf((size_t)K * N / 8);
  preprocess_weights_for_mixed_gemm<false, 256>(
      Bbuf.data(), packed.data(), {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);

  std::vector<half_t> hA16(hA.size()), hSc16(hsc.size()), hZr16(hzr.size());
  for (size_t i=0;i<hA.size();++i)  hA16[i]  = half_t(hA[i]);
  for (size_t i=0;i<hsc.size();++i) hSc16[i] = half_t(hsc[i]);
  for (size_t i=0;i<hzr.size();++i) hZr16[i] = half_t(hzr[i]);
  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dScale((size_t)scale_k*N), dZero((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint1_t> dB((size_t)K*N);
  dA.copy_from_host(hA16.data()); dScale.copy_from_host(hSc16.data()); dZero.copy_from_host(hZr16.data());
  dB.copy_from_host(reinterpret_cast<uint1_t const*>(Bbuf.data()));

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

  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3, cutlass::uint1b_t>(
      dA.get(), dB.get(), dScale.get(), dZero.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hD((size_t)M*N); dD.copy_to_host(hD.data());

  int bad = 0, shown = 0;
  for (size_t i=0;i<(size_t)M*N;++i) if (std::abs((double)float(hD[i])-gD[i]) > 1e-2 + 5e-2*std::abs(gD[i])) ++bad;
  std::printf("  bad=%d/%d %s\n", bad, M*N, bad==0?"MATCH":"MISMATCH");
  for (int m=0; m<M && shown<12; ++m) for (int n=0; n<N && shown<12; ++n) {
    double got=(double)float(hD[(size_t)m*N+n]), exp=gD[(size_t)m*N+n];
    if (std::abs(got-exp) > 1e-2 + 5e-2*std::abs(exp)) {
      int g=m/gs;
      std::printf("    m=%3d n=%3d g=%d | q1=%d sc=%.2f zr=%.2f | got=%.3f exp=%.3f\n",
                  m,n,g, q1[(size_t)m*N+n], hsc[(size_t)g*N+n], hzr[(size_t)g*N+n], got, exp);
      ++shown;
    }
  }
  return 0;
}
