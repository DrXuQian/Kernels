// W2A16 DEQUANT-isolation diag [box-only]. A=identity (M=K) so D[m][n]=W[m][n] -> isolates dequant/layout from
// the K-contraction. Random q2/scale/zero (sensitive to the N/within-reg permutation). Reports bad + the first
// mismatches with context so the residual permutation (if any) is visible.
//   TARGET=test_w2a16_diag ./build.sh ; ./<bin>
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
using uint2_t = cutlass::uint2b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

int main() {
  // CONTROLLED-INPUT probe (user's half-fill idea): scale=1, zero=0 so D[m][n] == q2[m][n] directly.
  // q2 tagged by the N-half WITHIN each 16-N mma atom: n%16<8 -> 1, n%16>=8 -> 2 (k-independent). If correct,
  // D[0][n%16<8]=1 and D[0][n%16>=8]=2. If the upper N-half reads the lower half, D[0][n%16>=8] shows 1 ->
  // exposes the aliasing directly (no AIU modeling needed).
  const int L = 1, M = 256, N = 256, K = 256, gs = 32;
  const int scale_k = (K + gs - 1) / gs;
  std::printf("[w2a16-probe] scale=1 zero=0, q2[n]=((n%%16)<8?1:2); D[0][n] expect 1 for n%%16<8, 2 for n%%16>=8\n");

  std::vector<int>   q2((size_t)K * N);
  std::vector<float> hsc((size_t)scale_k * N, 1.f), hzr((size_t)scale_k * N, 0.f), hA((size_t)M * K, 0.f);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) q2[(size_t)k*N+n] = ((n % 16) < 8) ? 1 : 2;  // half-tag
  for (int m = 0; m < M; ++m) hA[(size_t)m*K + m] = 1.f;                    // identity A

  std::vector<double> gD((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n)
    gD[(size_t)m*N+n] = (double)q2[(size_t)m*N+n];   // scale=1, zero=0 -> W == q2

  // transpose q [K][N]->[N][K] + pack 4 uint2/byte + preprocess PACKED_INT2
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n*K + k] = q2[(size_t)k*N + n];
  std::vector<int8_t> packed((size_t)K * N / 4, 0);
  for (size_t i = 0; i < (size_t)K * N / 4; ++i)
    packed[i] = int8_t((qT[4*i]&3) | ((qT[4*i+1]&3)<<2) | ((qT[4*i+2]&3)<<4) | ((qT[4*i+3]&3)<<6));
  std::vector<int8_t> Bbuf((size_t)K * N / 4);
  preprocess_weights_for_mixed_gemm<false, 256>(
      Bbuf.data(), packed.data(), {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY);

  std::vector<half_t> hA16(hA.size()), hSc16(hsc.size()), hZr16(hzr.size());
  for (size_t i=0;i<hA.size();++i)  hA16[i]  = half_t(hA[i]);
  for (size_t i=0;i<hsc.size();++i) hSc16[i] = half_t(hsc[i]);
  for (size_t i=0;i<hzr.size();++i) hZr16[i] = half_t(hzr[i]);
  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dScale((size_t)scale_k*N), dZero((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dB((size_t)K*N);
  dA.copy_from_host(hA16.data()); dScale.copy_from_host(hSc16.data()); dZero.copy_from_host(hZr16.data());
  dB.copy_from_host(reinterpret_cast<uint2_t const*>(Bbuf.data()));

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

  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3, cutlass::uint2b_t>(
      dA.get(), dB.get(), dScale.get(), dZero.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hD((size_t)M*N); dD.copy_to_host(hD.data());

  int bad = 0;
  for (size_t i=0;i<(size_t)M*N;++i) if (std::abs((double)float(hD[i])-gD[i]) > 0.3) ++bad;
  std::printf("  bad=%d/%d %s\n", bad, M*N, bad==0?"MATCH":"MISMATCH");
  // Direct look: D[0][n] for n=0..31. Each value = the q2 (tag) that output col n actually read.
  //   1 = read the LOW  N-half (n%16<8)    2 = read the HIGH N-half (n%16>=8)
  // Expect: 1 1 1 1 1 1 1 1  2 2 2 2 2 2 2 2  (repeat). Any n%16>=8 showing 1 = aliased to the low half.
  std::printf("  D[0][0..31] (round) = ");
  for (int n = 0; n < 32; ++n) { if (n%16==0) std::printf("| "); std::printf("%d ", (int)((double)float(hD[(size_t)0*N+n])+0.5)); }
  std::printf("\n  D[1][0..31] (round) = ");
  for (int n = 0; n < 32; ++n) { if (n%16==0) std::printf("| "); std::printf("%d ", (int)((double)float(hD[(size_t)1*N+n])+0.5)); }
  std::printf("\n");
  return 0;
}
