// MULTI-EXPERT + RAGGED correctness gate for the grouped mixed-input GEMM (step 2a/2b grouping logic).
//
// WHY THIS SHAPE OF TEST. The SINGLE-expert (L=1) grouped kernel is PROVEN bit-exact vs the verified mixed
// kernel AND the dequant golden -- see `bench_cutlass_w4a16 --xcheck` (both comparisons max_rel=0). So the
// GEMM math / layout / scale / interleave are already trusted. What is NOT yet tested is the GROUPING itself:
// the GroupScheduler's tile assignment, the per-expert l_coord slice of B/scale, and the ragged A offset.
//
// So we use grouped-L=1 as a TRUSTED per-expert ORACLE: run each expert ALONE (L=1) to get golden_e, then run
// all L experts TOGETHER (uniform or ragged) and require D[e] == golden_e for every expert. No hand-rolled
// reference, no dequant/packing/orientation guessing (that is exactly what broke the previous version of this
// file) -- it is kernel-vs-itself, so any delta is purely a grouping bug.
//
// Realistic MoE FC shapes: N,K multiples of 256 -> the interleaved-256 path (the one the xcheck proved), gs=128.
// run: ./test_moe_grouped_verify [L] [m_base] [ragged?]   (any 4th arg => ragged: m_e = m_base*(e%4+1))
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"   // preprocess_weights_for_mixed_gemm
#include "moe_grouped_ppu.cuh"

using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using GS     = moe_grouped_ppu::GroupShape;

// One grouped launch, fixed tile 64x64x128/s3 (the xcheck-proven config), gs=128 interleaved.
static void run_grouped(const half_t* A, const int4_t* B, const half_t* scales, half_t* D,
                        int Mmax, int N, int K, int L, int gs,
                        GS* gsd, GS const* gsh, int const* offsets, char* ws, size_t ws_bytes) {
  moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64, 64, 128, 32, 32, 3>(
      A, B, scales, /*zeros=*/nullptr, D, Mmax, N, K, L, gs, gsd, gsh, offsets, ws, ws_bytes, /*stream=*/nullptr);
}

int main(int argc, char** argv) {
  const int  L      = argc > 1 ? atoi(argv[1]) : 4;    // experts
  const int  Mb     = argc > 2 ? atoi(argv[2]) : 128;  // base tokens/expert
  const bool ragged = argc > 3;                        // any 4th arg => ragged
  const int  N = 1024, K = 2048, gs = 128;             // both %256==0 -> interleaved-256 (xcheck-proven path)
  const int  scale_k = K / gs;

  std::vector<int> me(L), offs(L);
  int total = 0, Mmax = 0;
  for (int e = 0; e < L; ++e) { me[e] = ragged ? Mb * ((e % 4) + 1) : Mb; offs[e] = total; total += me[e]; Mmax = std::max(Mmax, me[e]); }

  srand(1234);
  // A is one contiguous [total][K] (ragged rows concatenated). raw signed int4 W_e is [N][K] per expert.
  std::vector<float>  hA((size_t)total * K), hSc((size_t)L * N * scale_k);
  std::vector<int8_t> rawB((size_t)L * N * K);
  for (auto& a : hA)  a = (rand() % 7 - 3) * 0.25f;
  for (auto& s : hSc) s = 0.02f + (rand() % 8) * 0.01f;
  for (auto& b : rawB) b = int8_t(rand() % 15 - 7);

  // preprocess each expert's weight (interleaved-256) into the kernel B_buff layout, exactly like bench.
  std::vector<int8_t> packed((size_t)L * N * K / 2), Bbuf((size_t)L * N * K / 2);
  for (int e = 0; e < L; ++e) {
    for (int i = 0; i < N * K / 2; ++i) {
      int8_t lo = rawB[(size_t)e * N * K + 2 * i] & 0xF, hi = rawB[(size_t)e * N * K + 2 * i + 1] & 0xF;
      packed[(size_t)e * N * K / 2 + i] = int8_t((hi << 4) | lo);
    }
    preprocess_weights_for_mixed_gemm<false, 256>(
        (int8_t*)&Bbuf[(size_t)e * N * K / 2], (int8_t*)&packed[(size_t)e * N * K / 2],
        {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);
  }

  auto to_half = [](std::vector<float> const& f) { std::vector<half_t> h(f.size()); for (size_t i = 0; i < f.size(); ++i) h[i] = half_t(f[i]); return h; };

  // ---- FULL run: all L experts together ----
  cutlass::DeviceAllocation<half_t> A((size_t)total * K), scales((size_t)L * N * scale_k), D((size_t)L * Mmax * N);
  cutlass::DeviceAllocation<int4_t> B((size_t)L * N * K);
  { auto h = to_half(hA);  A.copy_from_host(h.data()); }
  { auto h = to_half(hSc); scales.copy_from_host(h.data()); }
  B.copy_from_host(reinterpret_cast<int4_t const*>(Bbuf.data()));

  std::vector<GS> rshapes(L); for (int e = 0; e < L; ++e) rshapes[e] = cute::make_shape(me[e], N, K);
  cutlass::DeviceAllocation<GS>  rdev(L);   rdev.copy_from_host(rshapes.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  const size_t ws = (size_t)cutlass::ceil_div(Mmax, 16) * cutlass::ceil_div(N, 64) * L * 64;
  cutlass::DeviceAllocation<char> wsr(ws);

  run_grouped(A.get(), B.get(), scales.get(), D.get(), Mmax, N, K, L, gs,
              rdev.get(), rshapes.data(), ragged ? offdev.get() : nullptr, wsr.get(), ws);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hD((size_t)L * Mmax * N); D.copy_to_host(hD.data());

  // ---- ORACLE: each expert ALONE through the SAME kernel at L=1 (trusted). Compare D[e] vs golden_e. ----
  double max_rel = 0; int bad = 0; int worst_e = -1;
  for (int e = 0; e < L; ++e) {
    const int Me = me[e];
    // per-expert device buffers (avoid sub-byte int4 pointer arithmetic on the shared B)
    cutlass::DeviceAllocation<half_t> Ae((size_t)Me * K), Se((size_t)N * scale_k), De((size_t)Me * N);
    cutlass::DeviceAllocation<int4_t> Be((size_t)N * K);
    { std::vector<half_t> h((size_t)Me * K); for (size_t i = 0; i < (size_t)Me * K; ++i) h[i] = half_t(hA[(size_t)offs[e] * K + i]); Ae.copy_from_host(h.data()); }
    { std::vector<half_t> h((size_t)N * scale_k); for (size_t i = 0; i < (size_t)N * scale_k; ++i) h[i] = half_t(hSc[(size_t)e * N * scale_k + i]); Se.copy_from_host(h.data()); }
    Be.copy_from_host(reinterpret_cast<int4_t const*>(&Bbuf[(size_t)e * N * K / 2]));

    std::vector<GS> gs1(1, cute::make_shape(Me, N, K));
    cutlass::DeviceAllocation<GS> gs1d(1); gs1d.copy_from_host(gs1.data());
    const size_t ws1 = (size_t)cutlass::ceil_div(Me, 16) * cutlass::ceil_div(N, 64) * 64;
    cutlass::DeviceAllocation<char> ws1b(ws1);

    run_grouped(Ae.get(), Be.get(), Se.get(), De.get(), Me, N, K, /*L=*/1, gs,
                gs1d.get(), gs1.data(), /*offsets=*/nullptr, ws1b.get(), ws1);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::vector<half_t> hDe((size_t)Me * N); De.copy_to_host(hDe.data());

    for (int i = 0; i < Me; ++i)
      for (int j = 0; j < N; ++j) {
        double gold = (double)float(hDe[(size_t)i * N + j]);                     // golden_e[i][j]
        double got  = (double)float(hD[(size_t)e * Mmax * N + (size_t)i * N + j]); // D[e][i][j] (padded)
        double rel  = std::abs(got - gold) / (std::abs(gold) + 1e-3);
        if (rel > max_rel) { max_rel = rel; worst_e = e; }
        if (rel > 5e-2) ++bad;
      }
  }

  std::printf("verify(grouping): L=%d %s Mb=%d N=%d K=%d gs=%d total=%d Mmax=%d\n",
              L, ragged ? "ragged" : "uniform", Mb, N, K, gs, total, Mmax);
  std::printf("  grouped-L=%d vs grouped-L=1 oracle: max_rel=%.3e (worst expert=%d) bad=%d -> %s\n",
              L, max_rel, worst_e, bad, bad == 0 ? "MATCH" : "MISMATCH");
  return bad == 0 ? 0 : 1;
}
