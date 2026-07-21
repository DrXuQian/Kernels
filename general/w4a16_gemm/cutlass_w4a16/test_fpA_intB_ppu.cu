// Minimal compile+run gate for fpA_intB_ppu.cuh (the official-structured finegrained launcher on actlize
// v1.0.0). It does NOT verify correctness or report a real number yet -- B is not run through
// preprocess_weights_for_mixed_gemm here, so results are garbage-by-design. The point is to surface the
// [F1]-[F4] compile fixes flagged in fpA_intB_ppu.cuh and confirm the finegrained Gs128 path builds/launches
// on the box. Once green, route this through bench_cutlass_w4a16.cu's data+verify harness for a real number.
//
// Official finegrained path needs block_k >= group_size, so gs=128 uses TK=128 (NOT the generic path's 64).
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
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

  half_t *A, *scales, *D; int4_t* B; char* ws;
  size_t ws_bytes = (size_t)cutlass::ceil_div(m,128) * cutlass::ceil_div(n,64) * 7 * 4;
  cudaMalloc(&A, (size_t)m*k*sizeof(half_t));
  cudaMalloc(&B, (size_t)n*k/2);                 // int4 packed
  cudaMalloc(&scales, (size_t)n*scale_k*sizeof(half_t));
  cudaMalloc(&D, (size_t)m*n*sizeof(half_t));
  cudaMalloc(&ws, ws_bytes);
  cudaMemset(A, 0, (size_t)m*k*sizeof(half_t));
  cudaMemset(B, 0, (size_t)n*k/2);
  cudaMemset(scales, 0, (size_t)n*scale_k*sizeof(half_t));

  std::printf("fpA_intB official path: m=%d n=%d k=%d gs=%d (TK=128, FinegrainedGs128, scale-only)\n", m,n,k,g);
  // block 128x128x128, warp 64x64 -- block_k=128 >= gs=128 (the official constraint), split_k=1.
  fpa_intb_ppu::filter_and_run<cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
      128, 128, 128, 64, 64, /*Stages=*/3>(
      A, B, scales, /*zeros=*/nullptr, D, m, n, k, g, /*split_k=*/1, ws, ws_bytes, 0);
  cudaError_t e = cudaDeviceSynchronize();
  std::printf("launch status: %s\n", cudaGetErrorString(e));
  return 0;
}
