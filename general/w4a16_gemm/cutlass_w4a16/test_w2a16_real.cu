// W2A16 REAL-WEIGHT gate [box-only]: reads a Q2_K slice .bin extracted from a real model (3B ffn_gate) by
// real_weight/dump_real_weights.py, runs the W2A16 kernel at gs=16 (Q2_K native group size), and compares
// D = A @ dequant(W) vs the Python native golden. This validates the int2 N-half fix on REAL quant values +
// the gs=16 path (the diag only did gs=32/synthetic).
//   .bin layout: <4 int32: M,N,K,gs> A[M*K] f16 | q2[K*N] u8(0..3) | scale[(K/gs)*N] f16 | zero f16 | gold[M*N] f16
//   Build: TARGET=test_w2a16_real ./build.sh ; run: ./<bin> [real_weight/real_q2k_ffn_gate_L0.bin]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
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

template <class T> static std::vector<T> rd(FILE* f, size_t n) { std::vector<T> v(n); if (fread(v.data(), sizeof(T), n, f) != n) { std::printf("short read\n"); exit(1);} return v; }

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "real_weight/real_q2k_ffn_gate_L0.bin";
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::printf("cannot open %s\n", path); return 1; }
  int32_t hdr[4]; if (fread(hdr, 4, 4, f) != 4) { std::printf("bad header\n"); return 1; }
  const int M = hdr[0], N = hdr[1], K = hdr[2], gs = hdr[3], L = 1;
  const int scale_k = K / gs;
  std::printf("[w2a16-real] %s  M=%d N=%d K=%d gs=%d\n", path, M, N, K, gs);
  auto A_h    = rd<uint16_t>(f, (size_t)M * K);          // f16 bits
  auto q2u8   = rd<uint8_t> (f, (size_t)K * N);          // [K][N] 0..3
  auto sc_h   = rd<uint16_t>(f, (size_t)scale_k * N);
  auto zr_h   = rd<uint16_t>(f, (size_t)scale_k * N);
  auto gold_h = rd<uint16_t>(f, (size_t)M * N);
  std::fclose(f);

  // transpose q [K][N]->[N][K] + pack 4 uint2/byte + preprocess PACKED_INT2 (identical to test_w2a16_diag)
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n*K + k] = q2u8[(size_t)k*N + n];
  std::vector<int8_t> packed((size_t)K * N / 4, 0);
  for (size_t i = 0; i < (size_t)K * N / 4; ++i)
    packed[i] = int8_t((qT[4*i]&3) | ((qT[4*i+1]&3)<<2) | ((qT[4*i+2]&3)<<4) | ((qT[4*i+3]&3)<<6));
  std::vector<int8_t> Bbuf((size_t)K * N / 4);
  preprocess_weights_for_mixed_gemm<false, 256>(
      Bbuf.data(), packed.data(), {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY);

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dScale((size_t)scale_k*N), dZero((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dB((size_t)K*N);
  dA.copy_from_host(reinterpret_cast<half_t const*>(A_h.data()));
  dScale.copy_from_host(reinterpret_cast<half_t const*>(sc_h.data()));
  dZero.copy_from_host(reinterpret_cast<half_t const*>(zr_h.data()));
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
  const half_t* gold = reinterpret_cast<const half_t*>(gold_h.data());

  int bad = 0, shown = 0; double maxrel = 0;
  for (size_t i=0;i<(size_t)M*N;++i) {
    double got=(double)float(hD[i]), exp=(double)float(gold[i]);
    double rel = std::abs(got-exp) / (std::abs(exp) + 1e-3);
    if (rel > maxrel) maxrel = rel;
    if (std::abs(got-exp) > 2e-2 + 6e-2*std::abs(exp)) ++bad;
  }
  std::printf("  bad=%d/%d max_rel=%.3e %s\n", bad, M*N, maxrel, bad==0?"MATCH":"MISMATCH");
  for (int m=0; m<M && shown<10; ++m) for (int n=0; n<N && shown<10; ++n) {
    double got=(double)float(hD[(size_t)m*N+n]), exp=(double)float(gold[(size_t)m*N+n]);
    if (std::abs(got-exp) > 2e-2 + 6e-2*std::abs(exp)) { std::printf("    m=%d n=%d got=%.4f exp=%.4f\n", m,n,got,exp); ++shown; }
  }
  return 0;
}
