// W2A16 q2 N-PERMUTATION READOUT [box-only]. zero-N is correct (prior probe) but q2's N-map is untested (q2 was
// n-independent). q2 is 2-bit so we bit-slice n into q2 selected by k%4: q2[k][n] = (n >> (2*(k%4))) & 3.
// identity A, scale=1, zero=0 -> D[m][n] = q2[m][n] = (n' >> (2*(m%4))) & 3 where n' = sigma_n(n) is the column
// the kernel actually read. Reconstruct n' = D[0][n] | D[1][n]<<2 | D[2][n]<<4 | D[3][n]<<6 (m=0..3 give k%4=0..3;
// robust to any within-%4 K permutation since the layout diag proved k%4 is preserved). n' vs n reveals sigma_n.
//
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
  const int L = 1, M = 256, N = 256, K = 256, gs = 32;
  const int scale_k = (K + gs - 1) / gs;
  std::printf("[w2a16-diag q2-N] q2[k][n]=(n>>(2*(k%%4)))&3; reconstruct n' from D[0..3][n] -> sigma_n(n)\n");

  std::vector<int>   q2((size_t)K * N);
  std::vector<float> hsc((size_t)scale_k * N, 1.f), hzr((size_t)scale_k * N, 0.f), hA((size_t)M * K, 0.f);
  for (int k=0;k<K;++k) for (int n=0;n<N;++n) q2[(size_t)k*N+n] = (n >> (2*(k%4))) & 3;
  for (int m=0;m<M;++m) hA[(size_t)m*K + m] = 1.f;                          // identity A

  std::vector<int> qT((size_t)K * N);
  for (int k=0;k<K;++k) for (int n=0;n<N;++n) qT[(size_t)n*K + k] = q2[(size_t)k*N + n];
  std::vector<int8_t> packed((size_t)K * N / 4, 0);
  for (size_t i=0;i<(size_t)K*N/4;++i)
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

  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 128, 32, 32, 3, cutlass::uint2b_t>(
      dA.get(), dB.get(), dScale.get(), dZero.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hD((size_t)M*N); dD.copy_to_host(hD.data());

  auto q = [&](int m,int n){ return ((int)std::lround((double)float(hD[(size_t)m*N+n]))) & 3; };
  std::printf("  n : sigma_n(n)   [* if != n]\n   ");
  int bad = 0;
  for (int n=0;n<64;++n){
    int np = q(0,n) | (q(1,n)<<2) | (q(2,n)<<4) | (q(3,n)<<6);
    if (np!=n) ++bad;
    std::printf(" %d:%d%s", n, np, np!=n?"*":"");
  }
  std::printf("\n  (%d of first 64 permuted)\n", bad);
  return 0;
}
