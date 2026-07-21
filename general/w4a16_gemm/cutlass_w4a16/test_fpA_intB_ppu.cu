// Minimal compile+run gate for fpA_intB_ppu.cuh (the official-structured finegrained launcher on actlize
// v1.0.0). It does NOT verify correctness or report a real number yet -- B is not run through
// preprocess_weights_for_mixed_gemm here, so results are garbage-by-design. The point is to surface the
// [F1]-[F4] compile fixes flagged in fpA_intB_ppu.cuh and confirm the finegrained Gs128 path builds/launches
// on the box. Once green, route this through bench_cutlass_w4a16.cu's data+verify harness for a real number.
//
// Official finegrained path needs block_k >= group_size, so gs=128 uses TK=128 (NOT the generic path's 64).
#include <cstdio>
#include <cstdlib>
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
  const size_t ws_bytes = (size_t)cutlass::ceil_div(m,128) * cutlass::ceil_div(n,64) * 7 * 4;

  cutlass::DeviceAllocation<half_t> A((size_t)m*k), scales((size_t)n*scale_k), D((size_t)m*n);
  cutlass::DeviceAllocation<int4_t> B((size_t)n*k);       // int4b_t elements (packed by the allocator)
  cutlass::DeviceAllocation<char>   ws(ws_bytes);

  std::printf("fpA_intB official path: m=%d n=%d k=%d gs=%d (TK=128, FinegrainedGs128, scale-only)\n", m,n,k,g);
  // block 128x128x128, warp 64x64 -- block_k=128 >= gs=128 (the official constraint), split_k=1.
  auto launch = [&]{
    fpa_intb_ppu::filter_and_run<fpa_intb_ppu::QuantMode::FinegrainedScaleOnly,
        128, 128, 128, 64, 64, /*Stages=*/3>(
        A.get(), B.get(), scales.get(), /*zeros=*/nullptr, D.get(), m, n, k, g,
        /*split_k=*/1, ws.get(), ws_bytes, /*stream=*/nullptr);
  };
  launch();
  std::printf("launch issued (compile + can_implement + run OK). timing next.\n");

  // GEMM timing is data-independent (no data-dependent branches), so this is a valid perf number even on
  // uninitialized buffers. Correctness is checked separately by routing through the bench data+verify harness.
  const int warmup = 20, iters = 100;
  for (int i = 0; i < warmup; i++) launch();
  PpuTimer timer; timer.start();
  for (int i = 0; i < iters; i++) launch();
  timer.stop();
  double us = double(timer.elapsed_millis()) * 1e3 / iters;
  double tflops = 2.0 * m * n * k / (us * 1e-6) / 1e12;
  std::printf("  [CUTLASS-official gs=%d TK=128] %7.2f us | %6.1f TFLOP/s | %.1f%% of 500\n",
              g, us, tflops, 100.0 * tflops / 500.0);
  return 0;
}
