// PERF sweep for the grouped mixed-input GEMM (correctness gated by test_moe_grouped_verify).
//
// Modeled on trtllm's MoE GemmProfiler flow: time a candidate-tile SET on the real (ragged) problem and pick the
// best. The candidate set mirrors trtllm's MoE tiles -- crucially it includes NARROW CTA-M (32) tiles, because
// per-expert M_e is small in MoE (unlike dense at M=2048, where 64x64 won). We re-measure winners on PPU; only
// the SEARCH MECHANISM + the shape of the config set are borrowed, not NVIDIA's winning numbers.
//
// MoE is WEIGHT-BANDWIDTH bound (each expert's B read ~once, M_e small), so rank by %HBM, not %MFU.
// run: ./test_moe_grouped_ppu [experts] [m_base] [n] [k] [gs]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "moe_grouped_ppu.cuh"

int main(int argc, char** argv) {
  int L  = argc > 1 ? atoi(argv[1]) : 8;
  int Mb = argc > 2 ? atoi(argv[2]) : 64;      // base tokens/expert; ragged uses Mb*(e%4+1)
  int n  = argc > 3 ? atoi(argv[3]) : 1024;
  int k  = argc > 4 ? atoi(argv[4]) : 2048;
  int g  = argc > 5 ? atoi(argv[5]) : 128;
  using half_t = cutlass::half_t; using int4_t = cutlass::int4b_t;
  const int scale_k = (k + g - 1) / g;
  const double TF_PEAK = 500.0, HBM_PEAK = 2766.0;   // ppu001 HBM peak ~2766 GB/s

  // Ragged workload = the realistic MoE case (variable tokens/expert).
  std::vector<int> me(L), offs(L);
  int total = 0, Mmax = 0;
  for (int e = 0; e < L; ++e) { me[e] = Mb * (e % 4 + 1); offs[e] = total; total += me[e]; Mmax = std::max(Mmax, me[e]); }

  cutlass::DeviceAllocation<half_t> A((size_t)total*k), scales((size_t)L*n*scale_k), D((size_t)L*Mmax*n);
  cutlass::DeviceAllocation<int4_t> B((size_t)L*n*k);
  using GS = moe_grouped_ppu::GroupShape;
  std::vector<GS> shp(L); for (int e=0;e<L;e++) shp[e]=cute::make_shape(me[e],n,k);
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  const size_t ws_bytes = (size_t)cutlass::ceil_div(Mmax,16)*cutlass::ceil_div(n,64)*L*64;
  cutlass::DeviceAllocation<char> ws(ws_bytes);

  std::printf("MoE grouped RAGGED sweep: experts=%d tokens=[", L);
  for (int e=0;e<L;e++) std::printf("%d%s", me[e], e+1<L?",":"]");
  std::printf(" total=%d n=%d k=%d gs=%d Mmax=%d\n", total, n, k, g, Mmax);
  const double wbytes = (double)L*n*k/2.0 + (double)L*n*scale_k*2.0;   // all experts' B + scales, read ~once
  std::printf("%-26s %-9s %-6s %-9s %s\n", "TILE(MxNxK)/WARP/ST", "TFLOP/s", "MFU", "GB/s", "%HBM");

  const int warmup = 10, iters = 50;
  char best_name[64] = ""; double best_hbm = 0.0;
#define TIME(TM,TN,TK,WM,WN,ST) do {                                                                    \
    auto launch = [&]{ moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly,\
        TM, TN, TK, WM, WN, ST>(A.get(), B.get(), scales.get(), nullptr, D.get(), Mmax, n, k, L, g,     \
        shpd.get(), shp.data(), offdev.get(), ws.get(), ws_bytes, nullptr); };                          \
    launch(); for (int i=0;i<warmup;i++) launch();                                                      \
    PpuTimer t; t.start(); for (int i=0;i<iters;i++) launch(); t.stop();                                \
    double us = double(t.elapsed_millis())*1e3/iters;                                                   \
    double tf = 2.0*total*n*k/(us*1e-6)/1e12, gbps = wbytes/(us*1e-6)/1e9;                               \
    const char* nm = #TM "x" #TN "x" #TK "/" #WM "x" #WN "/s" #ST;                                       \
    bool ran = (tf <= TF_PEAK) && (gbps <= 1.5*HBM_PEAK);                                                \
    if (ran) std::printf("%-26s %-9.1f %-6.1f %-9.0f %.1f%%\n", nm, tf, 100.0*tf/TF_PEAK, gbps, 100.0*gbps/HBM_PEAK); \
    else     std::printf("%-26s %-9s %-6s %-9s %s\n", nm, "-","-","-","FAIL (no-op)");                   \
    if (ran && gbps > best_hbm) { best_hbm = gbps; std::snprintf(best_name,sizeof(best_name),"%s",nm); } \
  } while (0)

  // trtllm-MoE-style candidate set: narrow CTA-M (32) tiles for small per-expert M_e, plus the dense
  // neighbourhood as baseline. FinegrainedGs128 requires TK>=128; WM|TM and WN|TN.
  TIME(32,  64,  128, 32, 32, 3);   // narrow-M (trtllm MoE favours these when M_e is small)
  TIME(32,  128, 128, 32, 64, 3);
  TIME(32,  64,  256, 32, 16, 2);
  TIME(64,  64,  128, 32, 32, 3);   // dense winner's neighbourhood (baseline)
  TIME(64,  64,  128, 32, 32, 4);
  TIME(64,  128, 128, 32, 64, 3);
  TIME(128, 64,  128, 64, 32, 3);
  TIME(64,  64,  256, 32, 32, 2);
#undef TIME
  std::printf("  WINNER (by %%HBM, MoE is weight-bound): %s at %.0f GB/s (%.1f%% HBM)\n",
              best_name, best_hbm, 100.0*best_hbm/HBM_PEAK);
  return 0;
}
