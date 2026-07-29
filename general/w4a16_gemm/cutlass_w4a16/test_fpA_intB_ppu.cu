// Minimal compile+run gate for fpA_intB_ppu.cuh (the official-structured finegrained launcher on actlize
// v1.0.0). It does NOT verify correctness or report a real number yet -- B is not run through
// preprocess_weights_for_mixed_gemm here, so results are garbage-by-design. The point is to surface the
// [F1]-[F4] compile fixes flagged in fpA_intB_ppu.cuh and confirm the finegrained Gs128 path builds/launches
// on the box. Once green, route this through bench_cutlass_w4a16.cu's data+verify harness for a real number.
//
// Official finegrained path needs block_k >= group_size, so gs=128 uses TK=128 (NOT the generic path's 64).
#include <cstdio>
#include <cstring>
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
  // max split_k raised 8 -> 32 for the occupancy ladder below. The comment above is load-bearing: an undersized
  // workspace made initialize() fail SILENTLY and the kernel no-op, and the only symptom was a >peak TFLOP/s number.
  const size_t ws_bytes = (size_t)cutlass::ceil_div(m,16) * cutlass::ceil_div(n,64) * 32 /*max split_k*/ * 4;

  cutlass::DeviceAllocation<half_t> A((size_t)m*k), scales((size_t)n*scale_k), D((size_t)m*n);
  cutlass::DeviceAllocation<int4_t> B((size_t)n*k);       // int4b_t elements (packed by the allocator)
  cutlass::DeviceAllocation<char>   ws(ws_bytes);

  // weight traffic (the binding cost at small m / decode): B is int4 = N*K/2 bytes, plus N*ceil(K/gs) fp16
  // scales. split_k partitions K so total B bytes are unchanged. Achievable HBM read ~2200 GB/s (bw_probe).
  const double wbytes = (double)n*k/2.0 + (double)n*scale_k*2.0;
  const double HBM_PEAK = 2200.0, TF_PEAK = 500.0;

  std::printf("fpA_intB official path sweep: m=%d n=%d k=%d gs=%d (FinegrainedGs128, scale-only)\n", m,n,k,g);
  std::printf("  compute-bound metric = MFU vs 500 TFLOP/s ; memory-bound (small m) = %% of 2200 GB/s\n");
  std::printf("%-32s %-9s %-6s %-9s %-7s %5s\n", "TILE(MxNxK)/WARP/ST/spk", "TFLOP/s", "MFU", "GB/s", "%HBM", "gw/CU");

  // GEMM timing is data-independent, so uninitialized buffers give a valid perf number (correctness checked
  // separately via the bench harness). Each TIME(...) is a distinct compiled instantiation. block_k (TK) is
  // now swept too: TK=128 satisfies the official block_k>=gs gate; TK=64 probes the relaxed gate (does the
  // FinegrainedGs128 kernel accept a group spanning 2 k-tiles?).
  // ONE ROW, ONE LAUNCH -- so acu can answer instead of me decomposing GB/s ratios into a traffic term and a
  // serialisation residual. The split-K attribution was argued from arithmetic twice and corrected once; acu reports DRAM
  // throughput, achieved occupancy and the stall breakdown directly, which is what the argument was standing in for.
  //   FPA_ONLY=<substring of the row name>   run only matching rows, e.g. "16x32x64/16x16/s2/spk8"
  //   FPA_ACU=1                              ONE launch, no warmup (these are NOT timings)
  const char* only = std::getenv("FPA_ONLY");
  const bool  acu  = std::getenv("FPA_ACU") != nullptr;
  if (only) std::printf("  FPA_ONLY='%s' -- every other row is SKIPPED\n", only);
  if (acu)  std::printf("  *** FPA_ACU=1: ONE COLD LAUNCH per row, no warmup. NOT timings. ***\n");
  const int warmup = 20, iters = 100;
  char best_name[64] = ""; double best_tf = 0.0, best_gbps = 0.0, best_score = 0.0;
#define TIME(TM,TN,TK,WM,WN,ST,SPLITK) do {                                                          \
    auto launch = [&]{ fpa_intb_ppu::filter_and_run<fpa_intb_ppu::QuantMode::FinegrainedScaleOnly,   \
        TM, TN, TK, WM, WN, ST>(A.get(), B.get(), scales.get(), nullptr, D.get(), m, n, k, g,        \
        SPLITK, ws.get(), ws_bytes, nullptr); };                                                     \
    const char* nm0 = #TM "x" #TN "x" #TK "/" #WM "x" #WN "/s" #ST "/spk" #SPLITK;                    \
    if (only && !std::strstr(nm0, only)) break;                                                      \
    double us;                                                                                        \
    if (acu) {                                                                                        \
      PpuTimer t; t.start(); launch(); t.stop();                                                       \
      us = double(t.elapsed_millis()) * 1e3;                                                          \
      std::printf("  [acu] ONE COLD launch (not a timing): %s\n", nm0);                               \
    } else {                                                                                          \
      launch();                                                                                       \
      for (int i = 0; i < warmup; i++) launch();                                                      \
      PpuTimer t; t.start(); for (int i = 0; i < iters; i++) launch(); t.stop();                      \
      us = double(t.elapsed_millis()) * 1e3 / iters;                                                  \
    }                                                                                                 \
    double tf = 2.0 * m * n * k / (us * 1e-6) / 1e12;                                                \
    double gbps = wbytes / (us * 1e-6) / 1e9;                                                        \
    const char* nm = #TM "x" #TN "x" #TK "/" #WM "x" #WN "/s" #ST "/spk" #SPLITK;                     \
    /* grid warps per CU = mt * n * TM * split_k / (WM*WN) / CU. Derived and then confirmed twice by acu on the  */ \
    /* grouped kernel (predicted 14.2, measured 13.65). TileN and TileK CANCEL, so split_k is the only factor    */ \
    /* here that is not already at its floor -- which is exactly what this ladder measures.                      */ \
    const double gwcu = double(cutlass::ceil_div(m,(TM))) * double(n) * double(TM) * double(SPLITK)               \
                      / (double(WM) * double(WN)) / 72.0;                                                        \
    /* no-op guard both regimes: faster than compute peak OR faster than HBM peak == kernel never ran */ \
    bool ran = (tf <= TF_PEAK) && (gbps <= 1.5*HBM_PEAK);                                             \
    if (ran) std::printf("%-32s %-9.1f %-6.1f %-9.0f %-6.1f%% %5.1f\n", nm, tf, 100.0*tf/TF_PEAK, gbps, 100.0*gbps/HBM_PEAK, gwcu); \
    else     std::printf("%-32s %-9s %-6s %-9s %-7s %5.1f  FAIL (no-op)\n", nm, "-", "-", "-", "-", gwcu); \
    /* rank by the binding metric: MFU for large m (compute-bound), GB/s for small m (memory-bound) */ \
    double score = (m >= 256) ? tf : gbps;                                                           \
    if (ran && score > best_score) { best_score = score; best_tf = tf; best_gbps = gbps;             \
                                     std::snprintf(best_name, sizeof(best_name), "%s", nm); }         \
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

  // --- SPLIT-K OCCUPANCY LADDER, the one experiment the grouped path cannot run ---
  // The grouped MoE kernel is stuck at grid warps/CU = mt*N*TM/(WM*WN)/72 = 14.2, because WarpN=16 is the MMA atom floor,
  // TileM/WarpM > 1 was measured and lost, and TileN/TileK cancel out of the identity. split_k is the ONLY remaining factor,
  // and it does not exist for the grouped ProblemShape -- but it does for the dense one, with the same collective (every
  // finegrained mainloop policy, fold included, exposes `Schedule = KernelAiuMultistageMixedInput`, which is what the
  // SplitKSerialScheduler specialization enable_ifs on) and the same non-EVT epilogue the split-K kernel needs.
  //
  // So: measure whether raising grid warps/CU past 14.2 keeps buying time, on dense, before building grouped split-K.
  // split_k <= K/TileK, so TileK=64 reaches spk=32 (56.9 warps/CU) while TileK=256 stops at spk=8 (14.2).
  // Run it at the decode shape: <bin> 8 2048 2048 32
  TIME(16, 32, 64,  16, 16, 2, 1);  TIME(16, 32, 64,  16, 16, 2, 2);  TIME(16, 32, 64,  16, 16, 2, 4);
  TIME(16, 32, 64,  16, 16, 2, 8);  TIME(16, 32, 64,  16, 16, 2, 16); TIME(16, 32, 64,  16, 16, 2, 32);
  TIME(16, 32, 256, 16, 16, 2, 1);  TIME(16, 32, 256, 16, 16, 2, 2);  TIME(16, 32, 256, 16, 16, 2, 4);
  TIME(16, 32, 256, 16, 16, 2, 8);
  TIME(16, 64, 64,  16, 16, 2, 8);  TIME(16, 64, 64,  16, 16, 2, 16); TIME(16, 64, 64,  16, 16, 2, 32);

  // --- mid/large Bm (prefill regime): split_k=1 ---
  TIME(64,  64,  64,  32, 32, 3, 1);  TIME(64,  64,  64,  32, 32, 4, 1);  TIME(64,  64,  64,  32, 32, 2, 2);
  TIME(64,  64,  128, 32, 32, 3, 1);  TIME(64,  64,  256, 32, 32, 2, 1);
  TIME(128, 64,  64,  64, 32, 3, 1);  TIME(128, 64,  128, 64, 32, 3, 1);
  TIME(64,  128, 64,  32, 64, 3, 1);  TIME(128, 128, 64,  64, 64, 3, 1);
#undef TIME

  // NOTE: dequant+cublas is NOT a candidate here. It needs OUR fast dequant (dequant_w4_ppu.cuh, cuda_fp16 +
  // custom device kernel) + cublas -- both CUDA-toolchain, and cuda_fp16's __half2 clashes with the PPU SDK's
  // hggc headers that hgcc auto-includes here. Device .o from nvcc cannot link with hgcc device .o either, so
  // it cannot ride this actlize binary. It lives on the nvcc side: ../marlin_ppu dequant_cublas (per-call,
  // our fast dequant). Compare by running both. The official CuTe dequant is accuracy-reference only (slow).

  std::printf("  WINNER m=%d: %s -> %.1f TFLOP/s (%.1f%% MFU) | %.0f GB/s (%.1f%% HBM)\n",
              m, best_name, best_tf, 100.0*best_tf/TF_PEAK, best_gbps, 100.0*best_gbps/HBM_PEAK);
  // Append the winner to a shape-keyed tactic cache (m,n,k,g|config=,tflops=), our sweep-built LUT analogue.
  {
    std::ofstream f("tactics_fpA_intB_ppu.cache", std::ios::app);
    if (f) f << m << "," << n << "," << k << "," << g << "|config=" << best_name
             << ",tflops=" << best_tf << "\n";
  }
  return 0;
}
