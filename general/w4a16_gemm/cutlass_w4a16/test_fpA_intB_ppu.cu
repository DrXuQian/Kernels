// Minimal compile+run gate for fpA_intB_ppu.cuh (the official-structured finegrained launcher on actlize
// v1.0.0). It does NOT verify correctness or report a real number yet -- B is not run through
// preprocess_weights_for_mixed_gemm here, so results are garbage-by-design. The point is to surface the
// [F1]-[F4] compile fixes flagged in fpA_intB_ppu.cuh and confirm the finegrained Gs128 path builds/launches
// on the box. Once green, route this through bench_cutlass_w4a16.cu's data+verify harness for a real number.
//
// Official finegrained path needs block_k >= group_size, so gs=128 uses TK=128 (NOT the generic path's 64).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "fpA_intB_ppu.cuh"

int main(int argc, char** argv) {
  int m = argc > 1 ? atoi(argv[1]) : 2048;
  int n = argc > 2 ? atoi(argv[2]) : 4096;
  int k = argc > 3 ? atoi(argv[3]) : 4096;
  int g = argc > 4 ? atoi(argv[4]) : 128;

  using half_t = cutlass::half_t;
  using int4_t = cutlass::int4b_t;
  const int scale_k = (k + g - 1) / g;
  // Split-k workspace = grid_m*grid_n*sizeof(int) per the kernel, with grid largest for the SMALLEST tile
  // (Bm=16, Bn=64) and largest split_k in the sweep. Size for the worst case so split_k configs actually run
  // -- an undersized workspace made initialize() fail silently and the kernel no-op (garbage >peak TFLOP/s).
  const size_t ws_bytes = (size_t)cutlass::ceil_div(m,16) * cutlass::ceil_div(n,64) * 8 /*max split_k*/ * 4;

  cutlass::DeviceAllocation<half_t> A((size_t)m*k), scales((size_t)n*scale_k), D((size_t)m*n);
  cutlass::DeviceAllocation<int4_t> B((size_t)n*k);       // int4b_t elements (packed by the allocator)
  cutlass::DeviceAllocation<char>   ws(ws_bytes);

  std::printf("fpA_intB official path sweep: m=%d n=%d k=%d gs=%d (FinegrainedGs128, scale-only)\n", m,n,k,g);
  std::printf("%-24s %-10s %s\n", "TILE(MxNxK)/WARP/ST", "TFLOP/s", "MFU");

  // GEMM timing is data-independent, so uninitialized buffers give a valid perf number (correctness checked
  // separately via the bench harness). Each TIME(...) is a distinct compiled instantiation. block_k (TK) is
  // now swept too: TK=128 satisfies the official block_k>=gs gate; TK=64 probes the relaxed gate (does the
  // FinegrainedGs128 kernel accept a group spanning 2 k-tiles?).
  const int warmup = 20, iters = 100;
  char best_name[64] = ""; double best_tf = 0.0;
#define TIME(TM,TN,TK,WM,WN,ST,SPLITK) do {                                                          \
    auto launch = [&]{ fpa_intb_ppu::filter_and_run<fpa_intb_ppu::QuantMode::FinegrainedScaleOnly,   \
        TM, TN, TK, WM, WN, ST>(A.get(), B.get(), scales.get(), nullptr, D.get(), m, n, k, g,        \
        SPLITK, ws.get(), ws_bytes, nullptr); };                                                     \
    launch();                                                                                        \
    for (int i = 0; i < warmup; i++) launch();                                                       \
    PpuTimer t; t.start(); for (int i = 0; i < iters; i++) launch(); t.stop();                       \
    double us = double(t.elapsed_millis()) * 1e3 / iters;                                            \
    double tf = 2.0 * m * n * k / (us * 1e-6) / 1e12;                                                \
    const char* nm = #TM "x" #TN "x" #TK "/" #WM "x" #WN "/s" #ST "/spk" #SPLITK;                     \
    bool ran = tf <= 500.0;  /* faster-than-peak == kernel no-op (init/can_implement bailed) */       \
    if (ran) std::printf("%-32s %-10.1f %.1f%%\n", nm, tf, 100.0*tf/500.0);                            \
    else     std::printf("%-32s %-10s %s\n", nm, "-", "FAIL (no-op: init/can_implement)");            \
    if (ran && tf > best_tf) { best_tf = tf; std::snprintf(best_name, sizeof(best_name), "%s", nm); }  \
  } while (0)

  // EXPANDED SEARCH SPACE. tactic+sweep (not the official LUT): we measure every config and keep the best.
  // Axes: Bm{16,32,64,128} x Bn{64,128} x Bk{64,128,256} x stage{2,3,4} x split_k{1,2,4}. Curated to the
  // promising region so compile time (one cutlass kernel per row) stays bounded. Bk=64/128 with gs=128 uses
  // the relaxed block_k>=gs gate. Run at small m (decode) and m=2048 (prefill); the winner differs by m.

  // --- small Bm (decode regime): Bn=64, vary Bk / stage / split_k ---
  TIME(16, 64, 64,  16, 16, 2, 1);  TIME(16, 64, 64,  16, 16, 2, 2);  TIME(16, 64, 64,  16, 16, 2, 4);
  TIME(16, 64, 128, 16, 16, 2, 1);  TIME(16, 64, 128, 16, 16, 2, 2);
  TIME(16, 64, 256, 16, 16, 2, 1);  TIME(16, 64, 256, 16, 16, 2, 2);
  TIME(32, 64, 64,  32, 16, 2, 1);  TIME(32, 64, 64,  32, 16, 2, 2);  TIME(32, 64, 64,  32, 16, 3, 1);
  TIME(32, 64, 128, 32, 16, 2, 1);  TIME(32, 64, 128, 32, 16, 2, 2);
  TIME(32, 64, 256, 32, 16, 2, 1);  TIME(32, 64, 256, 32, 16, 2, 2);

  // --- mid/large Bm (prefill regime): split_k=1 ---
  TIME(64,  64,  64,  32, 32, 3, 1);  TIME(64,  64,  64,  32, 32, 4, 1);  TIME(64,  64,  64,  32, 32, 2, 2);
  TIME(64,  64,  128, 32, 32, 3, 1);  TIME(64,  64,  256, 32, 32, 2, 1);
  TIME(128, 64,  64,  64, 32, 3, 1);  TIME(128, 64,  128, 64, 32, 3, 1);
  TIME(64,  128, 64,  32, 64, 3, 1);  TIME(128, 128, 64,  64, 64, 3, 1);
#undef TIME

  std::printf("  WINNER m=%d: %s at %.1f TFLOP/s (%.1f%%)\n", m, best_name, best_tf, 100.0*best_tf/500.0);
  // Append the winner to a shape-keyed tactic cache (m,n,k,g|config=,tflops=), our sweep-built LUT analogue.
  {
    std::ofstream f("tactics_fpA_intB_ppu.cache", std::ios::app);
    if (f) f << m << "," << n << "," << k << "," << g << "|config=" << best_name
             << ",tflops=" << best_tf << "\n";
  }
  return 0;
}
