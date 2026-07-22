// DATA-SLICE probe for the grouped mixed-input kernel. The routing probe (test_moe_grouped_probe, MOEG_PROBE=1)
// already proved scheduler+epilogue routing is correct (every D-plane e got tag e+1). So the multi-expert bug
// is in the per-expert DATA slice: the collective loads the wrong B/scale/A for e>0.
//
// This isolates WHICH input is stale with controlled, decodable data. Three sub-tests, each varies exactly ONE
// input by expert (the others constant), runs the REAL grouped GEMM (L experts together), and checks each
// D-plane. With A=1, rawB and scale constant within an expert, and scale-only dequant (Wdq = scale*rawB):
//     D[e][i][j] = sum_k A * (scale[e] * rawB[e]) = K * scale[e] * rawB[e]
//   - vary B:     rawB[e]=e+1, scale=1, A=1  -> expect K*(e+1); if every plane == K*1 -> B is stale (reads expert 0)
//   - vary scale: rawB=1, scale[e]=e+1, A=1  -> expect K*(e+1); if every plane == K*1 -> scale is stale
//   - vary A:     A[e]=e+1, rawB=1, scale=1  -> expect K*(e+1); if every plane == K*1 -> A is stale
// A stale value on plane e (e>0) equal to the plane-0 value is the signature of "no l_coord offset applied".
// run: ./test_moe_grouped_dataslice [L] [m_base]     (uniform; MOEG_PROBE must be UNSET so the real GEMM runs)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"

using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;

int main(int argc, char** argv) {
  const int L  = argc > 1 ? atoi(argv[1]) : 4;
  const int M  = argc > 2 ? atoi(argv[2]) : 128;   // uniform tokens/expert
  const int N = 1024, K = 2048, gs = 128, scale_k = K / gs;
  if (std::getenv("MOEG_PROBE")) { std::printf("ERROR: unset MOEG_PROBE (this test needs the real GEMM).\n"); return 2; }

  std::vector<GS> shp(L, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  const size_t ws = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*L*64;
  cutlass::DeviceAllocation<char> wsr(ws);

  // one sub-test: bval(e)/sval(e)/aval(e) give the per-expert constant for B/scale/A. Expected D[e]=K*sval*bval*aval.
  auto sub = [&](const char* name, auto bval, auto sval, auto aval) {
    std::vector<int8_t> rawB((size_t)L*N*K), packed((size_t)L*N*K/2), Bbuf((size_t)L*N*K/2);
    std::vector<float>  hA((size_t)L*M*K), hSc((size_t)L*N*scale_k);
    for (int e=0;e<L;e++){
      for (size_t i=0;i<(size_t)N*K;i++)      rawB[(size_t)e*N*K+i] = int8_t(bval(e));
      for (size_t i=0;i<(size_t)M*K;i++)      hA[(size_t)e*M*K+i]   = float(aval(e));
      for (size_t i=0;i<(size_t)N*scale_k;i++)hSc[(size_t)e*N*scale_k+i] = float(sval(e));
      for (int i=0;i<N*K/2;i++){
        int8_t lo = rawB[(size_t)e*N*K+2*i]&0xF, hi = rawB[(size_t)e*N*K+2*i+1]&0xF;
        packed[(size_t)e*N*K/2+i] = int8_t((hi<<4)|lo);
      }
      preprocess_weights_for_mixed_gemm<false,256>((int8_t*)&Bbuf[(size_t)e*N*K/2], (int8_t*)&packed[(size_t)e*N*K/2],
          {(size_t)K,(size_t)N}, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);
    }
    cutlass::DeviceAllocation<half_t> A((size_t)L*M*K), scales((size_t)L*N*scale_k), D((size_t)L*M*N);
    cutlass::DeviceAllocation<int4_t> B((size_t)L*N*K);
    { std::vector<half_t> t(hA.size());  for(size_t i=0;i<hA.size();i++)  t[i]=half_t(hA[i]);  A.copy_from_host(t.data()); }
    { std::vector<half_t> t(hSc.size()); for(size_t i=0;i<hSc.size();i++) t[i]=half_t(hSc[i]); scales.copy_from_host(t.data()); }
    B.copy_from_host(reinterpret_cast<int4_t const*>(Bbuf.data()));
    // contiguous ptr-array output (uniform M -> plane e at e*M*N)
    std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);
    for (int e=0;e<L;e++){ pdh[e]=D.get()+(size_t)e*M*N; sdh[e]=cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M,N,1)); gmh[e]=M; }
    cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
    cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
    cutlass::DeviceAllocation<int>     gm(L); gm.copy_from_host(gmh.data());

    moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64,64,128, 32,32, 3>(
        A.get(), B.get(), scales.get(), nullptr, pd.get(), sd.get(), gm.get(), M, N, K, L, gs,
        shpd.get(), shp.data(), nullptr, wsr.get(), ws, nullptr);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::vector<half_t> hD((size_t)L*M*N); D.copy_to_host(hD.data());

    std::printf("[%s]\n", name);
    bool sliced_ok = true, all_equal_plane0 = true;
    double p0 = (double)float(hD[0]);
    for (int e=0;e<L;e++){
      double expect = double(K) * double(sval(e)) * double(bval(e)) * double(aval(e));
      double got = (double)float(hD[(size_t)e*M*N + 0]);   // plane e, element (0,0)
      double rel = std::abs(got-expect)/(std::abs(expect)+1e-3);
      if (rel > 5e-2) sliced_ok = false;
      if (std::abs(got - p0) > 1e-3) all_equal_plane0 = false;
      std::printf("   plane e=%d: D[e][0][0]=%.1f  expect=%.1f  %s\n", e, got, expect, rel<5e-2?"ok":"BAD");
    }
    std::printf("   -> %s\n", sliced_ok ? "SLICED OK (per-expert data read correctly)"
                : all_equal_plane0 ? "STALE: every plane == plane 0 -> this input NOT sliced by l_coord"
                : "WRONG but not uniformly plane0 -> partial/other error");
  };

  std::printf("data-slice probe: L=%d M=%d N=%d K=%d gs=%d  (expect D[e]=K*(e+1)=%d*(e+1))\n", L,M,N,K,gs,K);
  sub("vary B (rawB[e]=e+1, scale=1, A=1)",       [](int e){return e+1;}, [](int){return 1;}, [](int){return 1;});
  sub("vary scale (scale[e]=e+1, rawB=1, A=1)",   [](int){return 1;},     [](int e){return e+1;}, [](int){return 1;});
  sub("vary A (A[e]=e+1, rawB=1, scale=1)",       [](int){return 1;},     [](int){return 1;}, [](int e){return e+1;});
  return 0;
}
