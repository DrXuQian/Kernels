// Dequant-weight + cublas fp16 GEMM -- the NVIDIA fpA_intB REFERENCE/fallback scheme, as a comparison option
// against the actlize mixed-input paths (../cutlass_w4a16) and the fused marlin kernel.
//
// The idea: expand the INT4 weight to fp16 ONCE, then run a fully-optimized dense fp16 GEMM (cublasHgemm).
// Two numbers, because the trade-off is entirely about WHEN you pay the dequant:
//   - gemm-only  : weight pre-dequantized offline. Runtime = just the fp16 GEMM. BUT the fp16 weight is 2x
//                  the int4 in HBM (breaks the W4A16 memory saving / "不能两份在显存"). Prefill-only play.
//   - per-call   : dequant charged every forward pass (weight expanded, used by one GEMM, discarded). Honest
//                  when you refuse to keep the fp16 copy resident.
//
// Metrics match ../cutlass_w4a16's sweep: TFLOP/s + MFU (vs 500) and GB/s + %HBM (vs ~2200 achievable). At a
// prefill M this path is compute-bound (the dense GEMM is the point); at decode M~1 it is memory-bound AND
// pays 2x weight bytes, so it is expected to lose there -- that contrast is exactly what we want to show.
//
// build (nvcc + cublas, PPU = no -arch):  see Makefile target `dequant_cublas`
// run:  ./dequant_cublas M N K [gs]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "dequant_w4_ppu.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int N = argc > 2 ? atoi(argv[2]) : 4096;
  int K = argc > 3 ? atoi(argv[3]) : 4096;
  int gs = argc > 4 ? atoi(argv[4]) : 128;
  const int G = K / gs;
  const double TF_PEAK = 500.0, HBM_PEAK = 2200.0;

  half *dA, *dW, *dC, *dS, *dM; int* dB;
  CK(cudaMalloc(&dA, (size_t)M*K*2));
  CK(cudaMalloc(&dW, (size_t)N*K*2));          // dequantized fp16 weight [N][K] -- 2x the int4
  CK(cudaMalloc(&dC, (size_t)M*N*2));
  CK(cudaMalloc(&dB, (size_t)N*K/2));          // int4 packed (n*k/2 bytes)
  CK(cudaMalloc(&dS, (size_t)G*N*2));
  CK(cudaMalloc(&dM, (size_t)G*N*2));
  CK(cudaMemset(dA,0,(size_t)M*K*2)); CK(cudaMemset(dB,0,(size_t)N*K/2));
  CK(cudaMemset(dS,0,(size_t)G*N*2)); CK(cudaMemset(dM,0,(size_t)G*N*2));

  cublasHandle_t cub; cublasCreate(&cub);
  cublasSetMathMode(cub, CUBLAS_TENSOR_OP_MATH);
  const half one = __float2half(1.f), zero = __float2half(0.f);

  auto gemm     = [&]{ cublasHgemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, dW, K, dA, K, &zero, dC, N); };
  auto per_call = [&]{ dequant_w4_ppu::dequant_launch(dB, dS, dM, dW, N, K, gs); gemm(); };

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto tm = [&](auto fn){
    for (int i=0;i<10;i++) fn();
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for (int i=0;i<100;i++) fn(); cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms=0; cudaEventElapsedTime(&ms,a,b); return double(ms)*1e3/100.0;   // us
  };

  auto report = [&](const char* tag, double us, double wbytes){
    double tf = 2.0*M*N*K/(us*1e-6)/1e12, gbps = wbytes/(us*1e-6)/1e9;
    printf("  [%-22s] M=%d %8.2f us | %6.1f TFLOP/s (%.1f%%) | %6.0f GB/s (%.1f%% HBM)\n",
           tag, M, us, tf, 100.0*tf/TF_PEAK, gbps, 100.0*gbps/HBM_PEAK);
  };

  printf("dequant+cublas fp16: M=%d N=%d K=%d gs=%d\n", M, N, K, gs);
  // PER-CALL ONLY. The "offline" (pre-dequantized, gemm-only) variant was dropped: it keeps a fp16 weight
  // resident = 2x the int4 in HBM, which defeats W4A16's whole point ("不能两份在显存"). per-call is the
  // honest number -- dequant expanded, used by one GEMM, discarded -- and tm() brackets { dequant; gemm; }
  // together so the dequant->cublas launch overhead between the two kernels IS counted.
  // Bytes: int4 read (N*K/2) + fp16 weight write in dequant (N*K*2) + fp16 weight read in gemm (N*K*2) + A.
  report("dequant+cublas per-call",  tm(per_call), (double)N*K/2 + (double)N*K*2 + (double)N*K*2 + (double)M*K*2);
  cublasDestroy(cub);
  return 0;
}
