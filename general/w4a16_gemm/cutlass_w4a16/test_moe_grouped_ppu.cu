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
#include "helper.h"
#include "moe_grouped_ppu.cuh"

using half_t = cutlass::half_t; using int4_t = cutlass::int4b_t;
using GS = moe_grouped_ppu::GroupShape;
static const double TF_PEAK = 500.0, HBM_PEAK = 2766.0;   // ppu001

// Sweep the candidate-tile set on one workload (per-expert token counts `me`), rank by MFU.
static void sweep(const char* label, const std::vector<int>& me, int n, int k, int g) {
  const int L = (int)me.size();
  const int scale_k = (k + g - 1) / g;
  std::vector<int> offs(L); int total = 0, Mmax = 0;
  for (int e = 0; e < L; ++e) { offs[e] = total; total += me[e]; Mmax = std::max(Mmax, me[e]); }

  cutlass::DeviceAllocation<half_t> A((size_t)total*k), scales((size_t)L*n*scale_k), D((size_t)L*Mmax*n);
  cutlass::DeviceAllocation<int4_t> B((size_t)L*n*k);
  std::vector<GS> shp(L); for (int e=0;e<L;e++) shp[e]=cute::make_shape(me[e],n,k);
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  const size_t ws_bytes = (size_t)cutlass::ceil_div(Mmax,16)*cutlass::ceil_div(n,64)*L*64;
  cutlass::DeviceAllocation<char> ws(ws_bytes);

  std::printf("\n=== %s: experts=%d total_tokens=%d n=%d k=%d gs=%d Mmax=%d ===\n", label, L, total, n, k, g, Mmax);
  const double wbytes = (double)L*n*k/2.0 + (double)L*n*scale_k*2.0;
  std::printf("%-26s %-9s %-6s %-9s %s\n", "TILE(MxNxK)/WARP/ST", "TFLOP/s", "MFU", "GB/s", "%HBM");

  const int warmup = 10, iters = 50;
  char best_name[64] = ""; double best_mfu = 0.0;
#define TIME(TM,TN,TK,WM,WN,ST) do {                                                                    \
    auto launch = [&]{ moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly,\
        TM, TN, TK, WM, WN, ST>(A.get(), B.get(), scales.get(), nullptr, D.get(), Mmax, n, k, L, g,     \
        shpd.get(), shp.data(), offdev.get(), ws.get(), ws_bytes, nullptr); };                          \
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
  TIME(32,  64,  128, 32, 32, 3);
  TIME(32,  128, 128, 32, 64, 3);
  TIME(32,  64,  256, 32, 16, 2);
  TIME(64,  64,  128, 32, 32, 3);
  TIME(64,  64,  128, 32, 32, 4);
  TIME(64,  128, 128, 32, 64, 3);
  TIME(128, 64,  128, 64, 32, 3);
  TIME(64,  64,  256, 32, 32, 2);
#undef TIME
  std::printf("  WINNER %s: %s at %.1f%% MFU\n", label, best_name, best_mfu);
}

int main(int argc, char** argv) {
  int L  = argc > 1 ? atoi(argv[1]) : 8;
  int Mb = argc > 2 ? atoi(argv[2]) : 512;
  int n  = argc > 3 ? atoi(argv[3]) : 1024;
  int k  = argc > 4 ? atoi(argv[4]) : 2048;
  int g  = argc > 5 ? atoi(argv[5]) : 128;

  // UNIFORM: every expert has Mb tokens (Mmax==Mb, no padding, no imbalance) -> isolates grouped-scheduler cost
  // vs the batched kernel (test_moe_gemm_ppu, ~49% MFU) at the same per-expert shape.
  std::vector<int> uni(L, Mb);
  sweep("UNIFORM (m=Mb, no padding/imbalance)", uni, n, k, g);

  // RAGGED: me = Mb*(e%4+1) -> adds load-imbalance + padded-D over the uniform case at higher total work.
  std::vector<int> rag(L); for (int e=0;e<L;e++) rag[e] = Mb * (e % 4 + 1);
  sweep("RAGGED (m=Mb*(e%4+1))", rag, n, k, g);
  return 0;
}
