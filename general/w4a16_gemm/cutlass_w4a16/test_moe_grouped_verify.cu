// CORRECTNESS gate for the ragged grouped mixed-input GEMM (step 2b). Small shapes, scale-only.
//
// The kernel reads B in the PREPROCESSED (preprocess_weights_for_mixed_gemm) layout; the reference
// dequantizes the RAW int4 B. For scale-only the preprocess +8 bias and the kernel's internal -8 cancel, so
// the net weight is scale * raw_int4 (this is exactly why example 16's verify passes). CPU reference thus
// computes D_ref[e][i][j] = sum_k A[e][i][k] * (scale[e][j][k/gs] * raw_B[e][j][k]) per expert on its M_e
// rows, and we compare against the kernel's D[e] (padded [L][M_max][N]). K=128 < 256 -> the non-interleaved
// (ColumnMajor) path, so preprocess with RowsPerTile=-1 to match filter_and_run's il=false branch.
//
// UNTESTED on box; likely needs iteration on the int4 packing / preprocess convention.
#define MOEG_DEBUG   // print the kernel's actual A/B/C/D/scale strides (get the layout, don't guess)
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

int main(int argc, char** argv) {
  // Debug decomposition: run experts=1 first (isolates the single GEMM + swap/dequant); then multi-expert
  // uniform (isolates grouping); then ragged (isolates the per-expert offset).
  const int L  = argc > 1 ? atoi(argv[1]) : 1;       // experts (default 1 = single GEMM)
  const int Mu = argc > 2 ? atoi(argv[2]) : 32;      // base m per expert
  const bool ragged = argc > 3;                      // any 4th arg => ragged (Mu*(e%4+1)), else uniform Mu
  const int N = 64, K = 128, gs = 64;
  const int scale_k = K / gs;                        // 2
  std::vector<int> me(L);
  for (int e=0;e<L;e++) me[e] = ragged ? Mu*((e%4)+1) : Mu;
  int total = 0, Mmax = 0; std::vector<int> offs(L);
  for (int e=0;e<L;e++){ offs[e]=total; total+=me[e]; Mmax=std::max(Mmax,me[e]); }

  srand(1234);
  // Host data: A [sum_tokens][K], raw signed int4 B [L][N][K], scales [L][N][scale_k]. zeros=0.
  std::vector<float> hA((size_t)total*K), hSc((size_t)L*N*scale_k);
  std::vector<int8_t> rawB((size_t)L*N*K);           // one signed int4 per int8 slot (host, pre-pack)
  for (auto& a : hA)  a = (rand()%7 - 3) * 0.25f;
  for (auto& s : hSc) s = 0.02f + (rand()%8)*0.01f;
  for (auto& b : rawB) b = int8_t(rand()%15 - 7);    // [-7,7]

  // Pack raw int4 into 2-per-byte and preprocess per expert into the kernel's B_buff layout.
  std::vector<int8_t> packed((size_t)L*N*K/2), Bbuf((size_t)L*N*K/2);
  auto pack_expert = [&](int e){
    for (int i=0;i<N*K/2;i++){
      int8_t lo = rawB[(size_t)e*N*K + 2*i] & 0xF, hi = rawB[(size_t)e*N*K + 2*i+1] & 0xF;
      packed[(size_t)e*N*K/2 + i] = int8_t((hi<<4) | lo);
    }
    // non-interleaved (K=128<256): RowsPerTile=-1, is_rowmajor=false (B is [N][K] col-major logical)
    preprocess_weights_for_mixed_gemm<false, -1>(
        (int8_t*)&Bbuf[(size_t)e*N*K/2], (int8_t*)&packed[(size_t)e*N*K/2],
        {(size_t)K,(size_t)N}, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);
  };
  for (int e=0;e<L;e++) pack_expert(e);

  // Device buffers.
  cutlass::DeviceAllocation<half_t> A((size_t)total*K), scales((size_t)L*N*scale_k), D((size_t)L*Mmax*N);
  cutlass::DeviceAllocation<int4_t> B((size_t)L*N*K);
  { std::vector<half_t> tmp(hA.size()); for (size_t i=0;i<hA.size();i++) tmp[i]=half_t(hA[i]); A.copy_from_host(tmp.data()); }
  { std::vector<half_t> tmp(hSc.size()); for (size_t i=0;i<hSc.size();i++) tmp[i]=half_t(hSc[i]); scales.copy_from_host(tmp.data()); }
  B.copy_from_host(reinterpret_cast<int4_t const*>(Bbuf.data()));

  // Reference weight = OFFICIAL dequantize_weight(raw B) -- the ground truth the kernel matches (example 16's
  // verify relies on this pairing). Do NOT hand-roll the dequant convention. raw int4 (packed) on device;
  // dequant per expert -> Wdq[e] fp16 [N][K]; zeros = 0.
  cutlass::DeviceAllocation<int4_t> Braw((size_t)L*N*K);
  Braw.copy_from_host(reinterpret_cast<int4_t const*>(packed.data()));
  cutlass::DeviceAllocation<half_t> Wdq((size_t)L*N*K), zeros((size_t)L*N*scale_k);
  { std::vector<half_t> z((size_t)L*N*scale_k, half_t(0.f)); zeros.copy_from_host(z.data()); }
  using SB = cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
  for (int e=0;e<L;e++) {
    auto lB  = cute::make_layout(cute::make_shape(N,K,1), cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N,K,1)));
    auto lsz = cute::make_layout(cute::make_shape(N,scale_k,1));
    dequantize_weight(Wdq.get()+(size_t)e*N*K, Braw.get()+(size_t)e*N*K, lB,
                      scales.get()+(size_t)e*N*scale_k, zeros.get()+(size_t)e*N*scale_k, lsz, gs);
  }
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hWdq((size_t)L*N*K); Wdq.copy_to_host(hWdq.data());

  using GS = moe_grouped_ppu::GroupShape;
  std::vector<GS> rshapes(L); for (int e=0;e<L;e++) rshapes[e]=cute::make_shape(me[e],N,K);
  cutlass::DeviceAllocation<GS> rdev(L); rdev.copy_from_host(rshapes.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  const size_t ws = (size_t)cutlass::ceil_div(Mmax,16)*cutlass::ceil_div(N,64)*L*64;
  cutlass::DeviceAllocation<char> wsr(ws);

  // gs=64 -> FinegrainedGs64; tile must have TK>=gs=64. Use 64x64x64.
  moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64,64,64, 32,32, 3>(
      A.get(), B.get(), scales.get(), nullptr, D.get(), Mmax, N, K, L, gs,
      rdev.get(), rshapes.data(), offdev.get(), wsr.get(), ws, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<half_t> hD((size_t)L*Mmax*N); D.copy_to_host(hD.data());

  // CPU reference per expert = A @ Wdq (Wdq from the official dequant). Compare its M_e rows.
  double max_rel = 0; int bad = 0;
  for (int e=0;e<L;e++)
    for (int i=0;i<me[e];i++)
      for (int j=0;j<N;j++){
        double ref = 0;
        for (int kk=0;kk<K;kk++)
          // Wdq is ROW-major [N][K] (measured: StrideB=(K,1,0)): element (n,k) at n*K + k.
          ref += hA[(size_t)(offs[e]+i)*K + kk] * (double)float(hWdq[(size_t)e*N*K + (size_t)j*K + kk]);
        // D is TRANSPOSED [N][M] per expert (mixed-input swap+transpose): element (m,n) at e*N*Mmax + n*Mmax + m.
        double got = (double)float(hD[(size_t)e*N*Mmax + (size_t)j*Mmax + i]);
        double rel = std::abs(got-ref)/(std::abs(ref)+1e-3);
        if (rel > max_rel) max_rel = rel;
        if (rel > 5e-2) bad++;
      }
  std::printf("verify: L=%d %s Mu=%d N=%d K=%d gs=%d total=%d  max_rel=%.3e  bad=%d -> %s\n",
              L, ragged?"ragged":"uniform", Mu, N, K, gs, total, max_rel, bad, bad==0 ? "MATCH" : "MISMATCH");
  return 0;
}
