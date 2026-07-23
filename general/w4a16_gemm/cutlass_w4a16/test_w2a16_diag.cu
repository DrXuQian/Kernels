// W2A16 N-PERMUTATION READOUT [box-only]. The dequant probe (random per-(g,n)) MISMATCHed with a structured
// N-side permutation that PRESERVES n%4 (so the earlier n%4-only probes missed it). This reads the exact column
// map: A=identity (M=K), q2[k][n]=k%4 (encodes k%4), scale=1, zero[g][n]=1000+4*n (encodes n). Then
//   D[m][n] = W[m][n] = (m%4) + 1000 + 4*n   ->   n_read = (round(D)-1000)/4 ,  k4_read = (round(D)-1000)%4.
// n_read vs n reveals the N permutation sigma(n); k4_read vs m%4 reveals any K(mod4) permutation.
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
  std::printf("[w2a16-diag N-READOUT] W[k][n]=(k%%4)+1000+4*n; decode n_read=(D-1000)/4, k4_read=(D-1000)%%4\n");

  std::vector<int>   q2((size_t)K * N);
  std::vector<float> hsc((size_t)scale_k * N, 1.f), hzr((size_t)scale_k * N), hA((size_t)M * K, 0.f);
  for (int k=0;k<K;++k) for (int n=0;n<N;++n) q2[(size_t)k*N+n] = k % 4;
  for (int g=0;g<scale_k;++g) for (int n=0;n<N;++n) hzr[(size_t)g*N+n] = 1000.f + 4.f*n;
  for (int m=0;m<M;++m) hA[(size_t)m*K + m] = 1.f;                          // identity A

  std::vector<double> gD((size_t)M * N, 0.0);
  for (int m=0;m<M;++m) for (int n=0;n<N;++n) gD[(size_t)m*N+n] = (m%4) + 1000.0 + 4.0*n;

  std::vector<int> qT((size_t)K * N);
  for (int k=0;k<K;++k) for (int n=0;n<N;++n) qT[(size_t)n*K + k] = q2[(size_t)k*N + n];
  std::vector<int8_t> packed((size_t)K * N / 4, 0);
  for (size_t i=0;i<(size_t)K*N/4;++i)
    packed[i] = int8_t((qT[4*i]&3) | ((qT[4*i+1]&3)<<2) | ((qT[4*i+2]&3)<<4) | ((qT[4*i+3]&3)<<6));
  std::vector<int8_t> Bbuf((size_t)K * N / 4);
  preprocess_weights_for_mixed_gemm<false, 256>(
      Bbuf.data(), packed.data(), {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY);

  auto to_h = [](std::vector<float> const& f){ std::vector<half_t> h(f.size()); for (size_t i=0;i<f.size();++i) h[i]=half_t(f[i]); return h; };
  auto hA16 = to_h(hA), hSc16 = to_h(hsc), hZr16 = to_h(hzr);
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

  auto dec = [&](int m,int n,int& nr,int& kr){ int v=(int)std::lround((double)float(hD[(size_t)m*N+n]))-1000; nr=v/4; kr=((v%4)+4)%4; };
  // row m=0: read sigma(n) for n=0..31  (n_read should == n)
  std::printf("  m=0  n : n_read (exp n)  [* if n_read!=n]\n   ");
  for (int n=0;n<32;++n){ int nr,kr; dec(0,n,nr,kr); std::printf(" %d:%d%s", n, nr, nr!=n?"*":""); }
  std::printf("\n");
  // k4 check across m=0..7 at n=0 (k4_read should == m%4)
  std::printf("  n=0  m : k4_read (exp m%%4):");
  for (int m=0;m<8;++m){ int nr,kr; dec(m,0,nr,kr); std::printf(" m%d:%d(%d)", m, kr, m%4); }
  std::printf("\n");
  return 0;
}
