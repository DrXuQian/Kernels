// W1A16 MULTI-EXPERT + RAGGED grouped correctness gate. Mirror of test_moe_grouped_verify but for uint1b_t
// (8 uint1/byte, PACKED_INT1, TK=256 since int1 needs Block_K % 256 == 0), using int1's perf-winner tile
// 32x128:32x32:s3. Output is contiguous [total][N] via the ptr-array epilogue: ptr_D[e] = D + offs[e]*N.
// The grouped-L=1 path is already proven bit-exact for int1 (test_w1a16_diag, bad=0), so running each expert
// ALONE and requiring the multi-expert run's rows to match validates the GROUPING + contiguous layout only --
// no host golden needed.
//   run: ./test_w1a16_grouped [L] [m_base] [ragged?] [gs]    (any 4th arg => ragged: m_e = m_base*(e%4+1))
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"   // preprocess_weights_for_mixed_gemm
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;

// One grouped launch. int1 requires TK % 256 == 0 (AIU 32B min: TK*1/8 % 32 == 0), so TK=256 is fixed; the
// M/N tile is int1's measured winner (32x128, 4 warps as 1x4) -- A-smem = TileM*TK is int1's limiter.
static void run_grouped(const half_t* A, const uint1_t* B, const half_t* scales,
                        half_t** ptr_D, DStride* stride_D, int const* group_M,
                        int Mmax, int N, int K, int L, int gs,
                        GS* gsd, GS const* gsh, int const* offsets, char* ws, size_t wsb) {
  moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly,
                                  32, 128, 256, 32, 32, 3, cutlass::uint1b_t>(
      A, B, scales, nullptr, ptr_D, stride_D, group_M, Mmax, N, K, L, gs, gsd, gsh, offsets, ws, wsb, nullptr);
}

int main(int argc, char** argv) {
  const int  L      = argc > 1 ? atoi(argv[1]) : 4;
  const int  Mb     = argc > 2 ? atoi(argv[2]) : 128;
  const bool ragged = argc > 3;
  const int  gs     = argc > 4 ? atoi(argv[4]) : 128;
  const int  N = 1024, K = 2048;               // %256==0 -> interleaved-256; K%TK(256)==0
  const int  scale_k = K / gs;

  std::vector<int> me(L), offs(L);
  int total = 0, Mmax = 0;
  for (int e = 0; e < L; ++e) { me[e] = ragged ? Mb * ((e % 4) + 1) : Mb; offs[e] = total; total += me[e]; Mmax = std::max(Mmax, me[e]); }

  srand(1234);
  std::vector<float>  hA((size_t)total * K), hSc((size_t)L * N * scale_k);
  std::vector<int8_t> rawB((size_t)L * N * K);
  for (auto& a : hA)  a = (rand() % 7 - 3) * 0.25f;
  for (auto& s : hSc) s = 0.02f + (rand() % 8) * 0.01f;
  for (auto& b : rawB) b = int8_t(rand() % 2);              // 1-bit

  // pack 8 uint1/byte + preprocess PACKED_INT1 (per expert)
  std::vector<int8_t> packed((size_t)L * N * K / 8), Bbuf((size_t)L * N * K / 8);
  for (int e = 0; e < L; ++e) {
    for (int i = 0; i < N * K / 8; ++i) {
      int8_t b = 0;
      for (int t = 0; t < 8; ++t) b |= int8_t((rawB[(size_t)e * N * K + 8 * i + t] & 1) << t);
      packed[(size_t)e * N * K / 8 + i] = b;
    }
    preprocess_weights_for_mixed_gemm<false, 256>(
        (int8_t*)&Bbuf[(size_t)e * N * K / 8], (int8_t*)&packed[(size_t)e * N * K / 8],
        {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);
  }
  auto to_half = [](std::vector<float> const& f) { std::vector<half_t> h(f.size()); for (size_t i = 0; i < f.size(); ++i) h[i] = half_t(f[i]); return h; };
  auto out_stride = [&](int m) { return cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(m, N, 1)); };

  // ---- FULL run: all L experts, CONTIGUOUS output D[total][N] ----
  cutlass::DeviceAllocation<half_t> A((size_t)total * K), scales((size_t)L * N * scale_k), D((size_t)total * N);
  cutlass::DeviceAllocation<uint1_t> B((size_t)L * N * K);
  { auto h = to_half(hA);  A.copy_from_host(h.data()); }
  { auto h = to_half(hSc); scales.copy_from_host(h.data()); }
  B.copy_from_host(reinterpret_cast<uint1_t const*>(Bbuf.data()));

  std::vector<GS> rshapes(L); for (int e = 0; e < L; ++e) rshapes[e] = cute::make_shape(me[e], N, K);
  cutlass::DeviceAllocation<GS>  rdev(L);   rdev.copy_from_host(rshapes.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);
  for (int e = 0; e < L; ++e) { pdh[e] = D.get() + (size_t)offs[e] * N; sdh[e] = out_stride(me[e]); gmh[e] = me[e]; }
  cutlass::DeviceAllocation<half_t*> pd(L);  pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L);  sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int>     gm(L);  gm.copy_from_host(gmh.data());
  const size_t ws = (size_t)cutlass::ceil_div(Mmax, 16) * cutlass::ceil_div(N, 64) * (size_t)L * 64;
  cutlass::DeviceAllocation<char> wsr(ws);

  run_grouped(A.get(), B.get(), scales.get(), pd.get(), sd.get(), gm.get(),
              Mmax, N, K, L, gs, rdev.get(), rshapes.data(), ragged ? offdev.get() : nullptr, wsr.get(), ws);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hD((size_t)total * N); D.copy_to_host(hD.data());

  // ---- ORACLE: each expert alone (L=1), compare vs the contiguous D rows ----
  double max_rel = 0; int bad = 0, worst_e = -1;
  for (int e = 0; e < L; ++e) {
    const int Me = me[e];
    cutlass::DeviceAllocation<half_t> Ae((size_t)Me * K), Se((size_t)N * scale_k), De((size_t)Me * N);
    cutlass::DeviceAllocation<uint1_t> Be((size_t)N * K);
    { std::vector<half_t> h((size_t)Me * K); for (size_t i = 0; i < (size_t)Me * K; ++i) h[i] = half_t(hA[(size_t)offs[e] * K + i]); Ae.copy_from_host(h.data()); }
    { std::vector<half_t> h((size_t)N * scale_k); for (size_t i = 0; i < (size_t)N * scale_k; ++i) h[i] = half_t(hSc[(size_t)e * N * scale_k + i]); Se.copy_from_host(h.data()); }
    Be.copy_from_host(reinterpret_cast<uint1_t const*>(&Bbuf[(size_t)e * N * K / 8]));

    std::vector<GS> gs1(1, cute::make_shape(Me, N, K));
    cutlass::DeviceAllocation<GS> gs1d(1); gs1d.copy_from_host(gs1.data());
    std::vector<half_t*> pdo{De.get()}; std::vector<DStride> sdo{out_stride(Me)}; std::vector<int> gmo{Me};
    cutlass::DeviceAllocation<half_t*> pd1(1); pd1.copy_from_host(pdo.data());
    cutlass::DeviceAllocation<DStride> sd1(1); sd1.copy_from_host(sdo.data());
    cutlass::DeviceAllocation<int>     gm1(1); gm1.copy_from_host(gmo.data());
    const size_t ws1 = (size_t)cutlass::ceil_div(Me, 16) * cutlass::ceil_div(N, 64) * 64;
    cutlass::DeviceAllocation<char> ws1b(ws1);

    run_grouped(Ae.get(), Be.get(), Se.get(), pd1.get(), sd1.get(), gm1.get(),
                Me, N, K, /*L=*/1, gs, gs1d.get(), gs1.data(), /*offsets=*/nullptr, ws1b.get(), ws1);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::vector<half_t> hDe((size_t)Me * N); De.copy_to_host(hDe.data());

    for (int i = 0; i < Me; ++i)
      for (int j = 0; j < N; ++j) {
        double gold = (double)float(hDe[(size_t)i * N + j]);
        double got  = (double)float(hD[((size_t)offs[e] + i) * N + j]);   // CONTIGUOUS: row offs[e]+i
        double rel  = std::abs(got - gold) / (std::abs(gold) + 1e-3);
        if (rel > max_rel) { max_rel = rel; worst_e = e; }
        if (rel > 5e-2) ++bad;
      }
  }
  std::printf("[w1a16-grouped] L=%d %s Mb=%d N=%d K=%d gs=%d total=%d Mmax=%d (tile 32x128:32x32:s3, TK=256)\n",
              L, ragged ? "ragged" : "uniform", Mb, N, K, gs, total, Mmax);
  std::printf("  grouped-L=%d (contiguous D[total][N]) vs grouped-L=1 oracle: max_rel=%.3e (worst e=%d) bad=%d -> %s\n",
              L, max_rel, worst_e, bad, bad == 0 ? "MATCH" : "MISMATCH");
  return bad == 0 ? 0 : 1;
}
