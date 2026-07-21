// STEP 1 (stepping stone): batched mixed-input GEMM = MoE W4A16 with UNIFORM m_per_expert.
// Validates that actlize's rank-4 mixed-input kernel does the per-expert GEMM (fp16 x int4, group scale) and
// carries the tuning we found for dense. NOT the deliverable -- the ragged (variable tokens/expert) grouped
// GEMM is step 2 (see moe_ffn/w4a16/MOE_W4A16_PPU_PLAN.md). Timing only, uninitialized buffers (data-indep).
//
// Layout: A=[L][m][k] fp16, B=[L][n][k] int4, scales=[L][n][scale_k] fp16, D=[L][m][n] fp16 (L=num_experts).
// run: ./test_moe_gemm_ppu [experts] [m_per_expert] [n] [k] [gs]
#include <cstdio>
#include <cstdlib>
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "moe_gemm_ppu.cuh"

int main(int argc, char** argv) {
  int L  = argc > 1 ? atoi(argv[1]) : 8;      // experts
  int m  = argc > 2 ? atoi(argv[2]) : 512;    // m_per_expert (uniform)
  int n  = argc > 3 ? atoi(argv[3]) : 1024;   // qwen35moe FC1 N
  int k  = argc > 4 ? atoi(argv[4]) : 2048;   // qwen35moe FC1 K
  int g  = argc > 5 ? atoi(argv[5]) : 128;

  using half_t = cutlass::half_t;
  using int4_t = cutlass::int4b_t;
  const int scale_k = (k + g - 1) / g;
  const double TF_PEAK = 500.0, HBM_PEAK = 2200.0;

  const size_t ws_bytes = (size_t)cutlass::ceil_div(m,16)*cutlass::ceil_div(n,64)*L*4;
  cutlass::DeviceAllocation<half_t> A((size_t)L*m*k), scales((size_t)L*n*scale_k), D((size_t)L*m*n);
  cutlass::DeviceAllocation<int4_t> B((size_t)L*n*k);
  cutlass::DeviceAllocation<char>   ws(ws_bytes);

  std::printf("MoE W4A16 batched (uniform m_per_expert): experts=%d m=%d n=%d k=%d gs=%d\n", L,m,n,k,g);
  std::printf("%-28s %-9s %-6s %-9s %s\n", "TILE(MxNxK)/WARP/ST", "TFLOP/s", "MFU", "GB/s", "%HBM");

  const int warmup = 10, iters = 50;
  const double wbytes = (double)L*n*k/2.0 + (double)L*n*scale_k*2.0;  // per-layer weight traffic (all experts)
  char best_name[64] = ""; double best_tf = 0.0;
#define TIME(TM,TN,TK,WM,WN,ST) do {                                                                 \
    auto launch = [&]{ moe_gemm_ppu::filter_and_run<moe_gemm_ppu::QuantMode::FinegrainedScaleOnly,   \
        TM, TN, TK, WM, WN, ST>(A.get(), B.get(), scales.get(), nullptr, D.get(), m, n, k, L, g,     \
        ws.get(), ws_bytes, nullptr); }; \
    launch(); for (int i=0;i<warmup;i++) launch();                                                    \
    PpuTimer t; t.start(); for (int i=0;i<iters;i++) launch(); t.stop();                              \
    double us = double(t.elapsed_millis())*1e3/iters;                                                 \
    double tf = 2.0*L*m*n*k/(us*1e-6)/1e12, gbps = wbytes/(us*1e-6)/1e9;                              \
    const char* nm = #TM "x" #TN "x" #TK "/" #WM "x" #WN "/s" #ST;                                    \
    bool ran = (tf <= TF_PEAK) && (gbps <= 1.5*HBM_PEAK);                                             \
    if (ran) std::printf("%-28s %-9.1f %-6.1f %-9.0f %.1f%%\n", nm, tf, 100.0*tf/TF_PEAK, gbps, 100.0*gbps/HBM_PEAK); \
    else     std::printf("%-28s %-9s %-6s %-9s %s\n", nm, "-","-","-","FAIL (no-op)");                \
    if (ran && tf > best_tf) { best_tf = tf; std::snprintf(best_name,sizeof(best_name),"%s",nm); }     \
  } while (0)

  // Reuse the dense winners' neighbourhood (64x64 was best for gs=128; TK>=gs for finegrained).
  TIME(64,  64,  128, 32, 32, 3);
  TIME(64,  64,  128, 32, 32, 4);
  TIME(128, 64,  128, 64, 32, 3);
  TIME(64,  128, 128, 32, 64, 3);
  TIME(64,  64,  256, 32, 32, 2);
  TIME(32,  64,  256, 32, 16, 2);
#undef TIME
  std::printf("  WINNER: %s at %.1f TFLOP/s (%.1f%% MFU)\n", best_name, best_tf, 100.0*best_tf/TF_PEAK);
  return 0;
}
