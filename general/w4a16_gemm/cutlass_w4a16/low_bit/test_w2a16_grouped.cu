// Single-tile W2A16 (uint2b_t) grouped-GEMM correctness driver [box-only]. Self-contained (synthetic Q2 data,
// host dequant golden). Exercises the FULL W2A16 scaffold: pack 4 uint2/byte -> preprocess PACKED_INT2 ->
// filter_and_run<..., cutlass::uint2b_t> -> compare D vs A@dequant(W). On MISMATCH it dumps D vs golden for the
// first rows/cols so a STRUCTURED permutation (within-b16 / permute-map error) is directly visible.
//
//   nvcc -std=c++17 -O3 -I<cutlass/include> -I.. test_w2a16_grouped.cu -o test_w2a16_grouped && ./test_w2a16_grouped
//
// FIRST CUT choices (box-tune from here):
//   gs=32 (launcher dispatches 128/64/32; gs=16 for real Q2_K is a separate re-add). N=K=64 -> il=false
//   (non-interleaved ColumnMajor) to sidestep cutlass LayoutDetailsB<uint2b_t>. mode=scale+zero (affine, Q2_K-like).
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  const int L = 1, M = 64, N = 64, K = 64, gs = 32;      // gs=32 (dispatched); N=K=64 -> il=false
  const int scale_k = (K + gs - 1) / gs;                 // = 2
  const QM qmode = QM::FinegrainedScaleZero;             // affine (scale + zero), like Q2_K
  std::srand(4321);
  std::printf("[w2a16] L=%d M=%d N=%d K=%d gs=%d (scale+zero) scale_k=%d  il=%s\n",
              L, M, N, K, gs, scale_k, (N % 256 || K % 256) ? "false (non-interleaved)" : "true");

  // synth data: q2 in [0,3] laid out [K][N] (q2[k*N+n]); scale/zero [scale_k][N] fp16; A [M][K].
  std::vector<int>   q2((size_t)K * N);
  std::vector<float> hsc((size_t)scale_k * N), hzr((size_t)scale_k * N), hA((size_t)M * K);
  for (auto& q : q2)  q = std::rand() % 4;
  for (auto& s : hsc) s = 0.01f + (std::rand() % 8) * 0.01f;
  for (auto& z : hzr) z = (std::rand() % 9 - 4) * 0.01f;
  for (auto& a : hA)  a = (std::rand() % 15 - 7) * 0.5f;

  // host golden: W[k][n] = scale[k/gs][n]*q2[k][n] + zero[k/gs][n];  D[m][n] = sum_k A[m][k]*W[k][n]
  std::vector<double> gD((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        int g = k / gs;
        double w = (double)hsc[(size_t)g * N + n] * q2[(size_t)k * N + n] + hzr[(size_t)g * N + n];
        acc += (double)hA[(size_t)m * K + k] * w;
      }
      gD[(size_t)m * N + n] = acc;
    }

  // pack 4 uint2/byte over the [K][N] row-major flattening (matches int4's "2 nibbles/byte" but crumbs), then
  // preprocess PACKED_INT2, shape {K,N}. B buffer = K*N/4 bytes.
  std::vector<int8_t> packed((size_t)K * N / 4, 0);
  for (size_t i = 0; i < (size_t)K * N / 4; ++i) {
    packed[i] = int8_t((q2[4*i] & 3) | ((q2[4*i+1] & 3) << 2) | ((q2[4*i+2] & 3) << 4) | ((q2[4*i+3] & 3) << 6));
  }
  std::vector<int8_t> Bbuf((size_t)K * N / 4);
  preprocess_weights_for_mixed_gemm<false, 256>(
      Bbuf.data(), packed.data(), {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY);

  // to fp16 host buffers
  auto to_h = [](std::vector<float> const& f){ std::vector<half_t> h(f.size()); for (size_t i=0;i<f.size();++i) h[i]=half_t(f[i]); return h; };
  auto hA16 = to_h(hA), hSc16 = to_h(hsc), hZr16 = to_h(hzr);

  cutlass::DeviceAllocation<half_t> dA((size_t)M * K), dScale((size_t)scale_k * N), dZero((size_t)scale_k * N), dD((size_t)M * N);
  cutlass::DeviceAllocation<uint2_t> dB((size_t)K * N);          // uint2b_t elems; storage = K*N/4 bytes
  dA.copy_from_host(hA16.data());
  dScale.copy_from_host(hSc16.data());
  dZero.copy_from_host(hZr16.data());
  dB.copy_from_host(reinterpret_cast<uint2_t const*>(Bbuf.data()));

  // grouped args (L=1, single tile). Mirrors test_moe_grouped_real.cu's setup.
  std::vector<GS> shp(L); for (int e = 0; e < L; ++e) shp[e] = cute::make_shape(M, N, K);
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  auto out_stride = [&](int m){ return cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(m, N, 1)); };
  std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L), offs(L);
  for (int e = 0; e < L; ++e) { pdh[e] = dD.get() + (size_t)e * M * N; sdh[e] = out_stride(M); gmh[e] = M; offs[e] = e * M; }
  cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int>     gm(L); gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int>     offdev(L); offdev.copy_from_host(offs.data());
  const size_t wsb = (size_t)cutlass::ceil_div(M, 16) * cutlass::ceil_div(N, 64) * (size_t)L * 64;
  cutlass::DeviceAllocation<char> ws(wsb);

  // TK=64 (gs=32 -> gs<128 path). ElementB = uint2b_t (the W2A16 template arg).
  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 64, 32, 32, 3, cutlass::uint2b_t>(
      dA.get(), dB.get(), dScale.get(), dZero.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<half_t> hD((size_t)M * N); dD.copy_to_host(hD.data());

  // compare
  double mr = 0; int bad = 0;
  for (size_t i = 0; i < (size_t)M * N; ++i) {
    double g = gD[i], got = (double)float(hD[i]);
    double ad = std::abs(got - g), rel = ad / (std::abs(g) + 1e-3);
    if (rel > mr) mr = rel;
    if (ad > 1e-2 + 5e-2 * std::abs(g)) ++bad;
  }
  std::printf("  grp vs host golden  max_rel=%.3e  bad=%d/%d -> %s\n", mr, bad, M * N, bad == 0 ? "MATCH" : "MISMATCH");

  if (bad) {
    // dump first 8x8 of D vs golden so a structured permutation (within-b16 / permute-map) is visible
    std::printf("  D[m][n] (got | gold), first 4x8:\n");
    for (int m = 0; m < 4; ++m) {
      std::printf("   m=%d:", m);
      for (int n = 0; n < 8; ++n) std::printf("  %.2f|%.2f", (double)float(hD[(size_t)m*N+n]), gD[(size_t)m*N+n]);
      std::printf("\n");
    }
  }
  return bad ? 1 : 0;
}
