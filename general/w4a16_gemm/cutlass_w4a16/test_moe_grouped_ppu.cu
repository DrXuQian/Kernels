// PERF sweep for the grouped mixed-input GEMM (correctness gated by test_moe_grouped_verify).
//
// Runs the trtllm-MoE-style tactic sweep on BOTH a uniform and a ragged token distribution, so we can separate:
//   - grouped-scheduler intrinsic overhead   (visible in UNIFORM, vs the batched kernel's ceiling ~49% MFU)
//   - ragged load-imbalance / padding cost    (the extra gap UNIFORM -> RAGGED at equal total work)
// Compute-bound here (FLOP/byte = 4*M_e >> machine balance ~181), so %MFU is the metric; %HBM shown for context.
// run: ./test_moe_grouped_ppu [experts] [m_base] [n] [k] [gs]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "moe_grouped_ppu.cuh"

using half_t = cutlass::half_t; using int4_t = cutlass::int4b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
static const double TF_PEAK = 500.0, HBM_PEAK = 2766.0;   // ppu001

// Sweep the candidate-tile set on one workload (per-expert token counts `me`), rank by MFU.
static void sweep(const char* label, const std::vector<int>& me, int n, int k, int g) {
  const int L = (int)me.size();
  const int scale_k = (k + g - 1) / g;
  std::vector<int> offs(L); int total = 0, Mmax = 0;
  for (int e = 0; e < L; ++e) { offs[e] = total; total += me[e]; Mmax = std::max(Mmax, me[e]); }

  cutlass::DeviceAllocation<half_t> A((size_t)total*k), scales((size_t)L*n*scale_k), D((size_t)total*n);  // contiguous
  cutlass::DeviceAllocation<int4_t> B((size_t)L*n*k);
  std::vector<GS> shp(L); for (int e=0;e<L;e++) shp[e]=cute::make_shape(me[e],n,k);
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  const size_t ws_bytes = (size_t)cutlass::ceil_div(Mmax,16)*cutlass::ceil_div(n,64)*(size_t)L*64;
  cutlass::DeviceAllocation<char> ws(ws_bytes);
  // ptr-array outputs (TM-independent -> built once, reused across the config sweep): contiguous ptr_D[e]=D+offs[e]*n
  std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);
  for (int e=0;e<L;e++){ pdh[e]=D.get()+(size_t)offs[e]*n; sdh[e]=cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(me[e],n,1)); gmh[e]=me[e]; }
  cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int>     gm(L); gm.copy_from_host(gmh.data());

  std::printf("\n=== %s: experts=%d total_tokens=%d n=%d k=%d gs=%d Mmax=%d ===\n", label, L, total, n, k, g, Mmax);
  // COMPULSORY HBM traffic (each read/written once): A activations + B int4 weights + scales + D output.
  // (Earlier this counted ONLY B -> 4-5x undercount, since A and D dominate at these shapes. This is the ideal
  // lower bound with perfect reuse; actual is higher when M_e/TM>1 re-reads B per m-tile.)
  const double abytes = (double)total * k * 2.0;                                  // A fp16 [total][K]
  const double bbytes = (double)L * n * k / 2.0 + (double)L * n * scale_k * 2.0;  // B int4 + fp16 scales
  const double dbytes = (double)total * n * 2.0;                                  // D fp16 [total][N]
  const double wbytes = abytes + bbytes + dbytes;
  std::printf("%-26s %-9s %-6s %-9s %s\n", "TILE(MxNxK)/WARP/ST", "TFLOP/s", "MFU", "GB/s", "%HBM");

  const int warmup = 10, iters = 50;
  char best_name[64] = ""; double best_mfu = 0.0;
#define TIME(TM,TN,TK,WM,WN,ST) do {                                                                    \
    auto launch = [&]{ moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly,\
        TM, TN, TK, WM, WN, ST>(A.get(), B.get(), scales.get(), nullptr, pd.get(), sd.get(), gm.get(), \
        Mmax, n, k, L, g, shpd.get(), shp.data(), offdev.get(), ws.get(), ws_bytes, nullptr); };        \
    launch(); for (int i=0;i<warmup;i++) launch();                                                      \
    PpuTimer t; t.start(); for (int i=0;i<iters;i++) launch(); t.stop();                                \
    double us = double(t.elapsed_millis())*1e3/iters;                                                   \
    double tf = 2.0*total*n*k/(us*1e-6)/1e12, gbps = wbytes/(us*1e-6)/1e9, mfu = 100.0*tf/TF_PEAK;       \
    const char* nm = #TM "x" #TN "x" #TK "/" #WM "x" #WN "/s" #ST;                                       \
    bool ran = (tf <= TF_PEAK) && (gbps <= 1.5*HBM_PEAK);                                                \
    if (ran) std::printf("%-26s %-9.1f %-6.1f %-9.0f %.1f%%\n", nm, tf, mfu, gbps, 100.0*gbps/HBM_PEAK); \
    else     std::printf("%-26s %-9s %-6s %-9s %s\n", nm, "-","-","-","FAIL (no-op)");                   \
    if (ran && mfu > best_mfu) { best_mfu = mfu; std::snprintf(best_name,sizeof(best_name),"%s",nm); }   \
  } while (0)
  // OCCUPANCY axis: smaller BM / fewer stages -> less shared -> more blocks/CU (acu said theoretical occ was
  // shared-limited to 21.9%; this axis IS the occupancy/shared search -- no separate "tuning" step).
  TIME(32,  64,  128, 32, 32, 2);
  TIME(32,  64,  128, 32, 32, 3);
  TIME(32,  64,  128, 32, 32, 4);
  TIME(32,  128, 128, 32, 64, 3);
  TIME(64,  64,  128, 32, 32, 2);
  TIME(64,  64,  128, 32, 32, 3);   // current baseline winner
  TIME(64,  64,  128, 32, 32, 4);
  // REUSE axis: bigger BM/BN/BK -> fewer B re-reads (helps larger M_e / prefill).
  TIME(64,  128, 128, 32, 64, 3);   // ragged winner (s3)
  TIME(64,  128, 128, 32, 64, 2);   // NEW: winner's s2 -> less shared (more blocks/CU) + fewer regs (kill spill?)
  TIME(128, 64,  128, 64, 32, 3);
  TIME(128, 64,  128, 64, 32, 2);   // NEW: bigger TM (halve B re-reads) + s2 (occupancy)
  TIME(128, 64,  128, 64, 32, 4);
  TIME(128, 128, 128, 64, 64, 3);   // if this fails to compile (shared budget), comment it out & rebuild
  TIME(128, 128, 128, 64, 64, 2);   // NEW: bigger TM + s2
  TIME(64,  64,  256, 32, 32, 2);
  TIME(32,  64,  256, 32, 16, 2);
  TIME(64,  128, 256, 32, 64, 2);   // bigger TK on 64x128; comment out if compile fails
#undef TIME
  std::printf("  WINNER %s: %s at %.1f%% MFU\n", label, best_name, best_mfu);
}

// Single clean launch of the winner config (64x64x128/32x32/s3) on a UNIFORM workload, for a 1-kernel acu
// capture (acu -c 1). Enable with env MOEG_ONE=1.
static void one_launch(int L, int Mb, int n, int k, int g) {
  const int scale_k = (k + g - 1) / g;
  // RAGGED single launch (same 1:2:3:4 skew, total=L*Mb) with the RAGGED winner config, for a clean acu -c 1
  // capture of the production (dropless) path. flat grid + O(log L) binary decode.
  std::vector<int> me(L), offs(L);
  { long wsum=0; for (int e=0;e<L;e++) wsum += (e%4+1);
    long target=(long)L*Mb, acc=0;
    for (int e=0;e<L;e++){ me[e]=(int)(target*(e%4+1)/wsum); acc+=me[e]; } me[L-1]+=(int)(target-acc); }
  int total=0, Mmax=0; for (int e=0;e<L;e++){ offs[e]=total; total+=me[e]; Mmax=std::max(Mmax,me[e]); }
  cutlass::DeviceAllocation<half_t> A((size_t)total*k), scales((size_t)L*n*scale_k), D((size_t)total*n);
  cutlass::DeviceAllocation<int4_t> B((size_t)L*n*k);
  std::vector<GS> shp(L); for (int e=0;e<L;e++) shp[e]=cute::make_shape(me[e],n,k);
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);
  for (int e=0;e<L;e++){ pdh[e]=D.get()+(size_t)offs[e]*n; sdh[e]=cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(me[e],n,1)); gmh[e]=me[e]; }
  cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int>     gm(L); gm.copy_from_host(gmh.data());
  const size_t ws = (size_t)cutlass::ceil_div(Mmax,16)*cutlass::ceil_div(n,64)*(size_t)L*64;
  cutlass::DeviceAllocation<char> wsr(ws);
  std::printf("[MOEG_ONE] single RAGGED launch 64x128x128/32x64/s3 L=%d Mb=%d total=%d n=%d k=%d gs=%d Mmax=%d\n", L,Mb,total,n,k,g,Mmax);
  moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64,128,128, 32,64, 3>(
      A.get(), B.get(), scales.get(), nullptr, pd.get(), sd.get(), gm.get(), Mmax, n, k, L, g,
      shpd.get(), shp.data(), offdev.get(), wsr.get(), ws, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
}

int main(int argc, char** argv) {
  int L  = argc > 1 ? atoi(argv[1]) : 8;
  int Mb = argc > 2 ? atoi(argv[2]) : 512;
  int n  = argc > 3 ? atoi(argv[3]) : 1024;
  int k  = argc > 4 ? atoi(argv[4]) : 2048;
  int g  = argc > 5 ? atoi(argv[5]) : 128;
  if (std::getenv("MOEG_ONE")) { one_launch(L, Mb, n, k, g); return 0; }

  // UNIFORM: every expert has Mb tokens. total = L*Mb.
  std::vector<int> uni(L, Mb);
  sweep("UNIFORM (m=Mb)", uni, n, k, g);

  // RAGGED at the SAME total (L*Mb) as uniform, so the comparison isolates LOAD-IMBALANCE from work amount
  // (the old ragged used Mb*(e%4+1) -> 2.5x more tokens, which alone raised MFU -- an unfair comparison).
  // Keep the 1:2:3:4 skew shape but normalize the sum to L*Mb.
  std::vector<int> rag(L);
  { long wsum = 0; for (int e = 0; e < L; ++e) wsum += (e % 4 + 1);
    long target = (long)L * Mb, acc = 0;
    for (int e = 0; e < L; ++e) { rag[e] = (int)(target * (e % 4 + 1) / wsum); acc += rag[e]; }
    rag[L - 1] += (int)(target - acc); }                          // fix rounding: exact sum == L*Mb
  sweep("RAGGED (1:2:3:4 skew, SAME total=L*Mb)", rag, n, k, g);
  return 0;
}
