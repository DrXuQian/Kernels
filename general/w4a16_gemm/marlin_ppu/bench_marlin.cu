// Perf bench for the classic Marlin PPU port. Perf is data-independent -> random buffers (correctness validated
// separately in test_marlin_classic_num/splitk). Reports us + TFLOP/s.
// Known perf caveats in this port: write_result/global_reduce use DIRECT global stores (not the NVIDIA smem-staged
// coalesced writes); A smem->reg is element-wise (not ldmatrix). The main cp.async+mma loop is the NVIDIA structure.
//
// build (box):  make bench_marlin   (nvcc -O3, no -arch)
// run:          ./bench_marlin                 # the default W4A16 shape sweep (decode + prefill)
//               ./bench_marlin M N K [iters]   # one shape, e.g.  ./bench_marlin 2048 4096 14336
// env:          MARLIN_NOSPLITK=1 / MARLIN_SPLITK=1   force the split-K choice (default auto: no-split if tiles>=SMs)
//
// Shape constraints (else ret=1): N % thread_n == 0 and K % thread_k == 0, where the config is picked by M:
//   M <= 16  -> thread_n=128, thread_k=128      M > 16  -> thread_n=256, thread_k=64

#include "marlin_classic_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

static double bench_one(int M, int N, int K, int iters) {
    const int max_par = 16;
    const size_t A_i4 = (size_t) M * K / 8, B_i4 = (size_t) (K / 16) * (N * 16 / 32), C_i4 = (size_t) M * N / 8, S_h = (size_t) N;
    // locks: one per (n-tile, par). Worst case is thread_n=128 (the M<=16 configs) -> N/128 n-tiles, NOT N/256.
    const size_t WS = (size_t) (N / 128 + 1) * max_par;
    int4 *dA, *dB, *dC, *dS; int * dWS;
    if (cudaMalloc(&dA, A_i4 * 16) || cudaMalloc(&dB, B_i4 * 16) || cudaMalloc(&dC, C_i4 * 16) ||
        cudaMalloc(&dS, (S_h / 8 + 1) * 16) || cudaMalloc(&dWS, WS * 4)) { printf("  alloc fail\n"); return -1; }
    cudaMemset(dA, 1, A_i4 * 16); cudaMemset(dB, 1, B_i4 * 16); cudaMemset(dS, 0x3c, (S_h / 8) * 16);   // s~=1.0 (0x3c00)
    cudaMemset(dWS, 0, WS * 4);

    auto run = [&] { return marlin_classic_ppu::marlin_cuda(dA, dB, dC, dS, M, N, K, dWS, -1); };
    auto cleanup = [&] { cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dS); cudaFree(dWS); };

    int ret = run(); cudaError_t e = cudaDeviceSynchronize();
    if (ret || e) {
        printf("  M=%-5d N=%-5d K=%-5d : ret=%d err=%s  (unsupported shape/config)\n", M, N, K, ret, cudaGetErrorString(e));
        cleanup(); return -1;
    }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) run();
    cudaDeviceSynchronize();
    cudaMemset(dWS, 0, WS * 4);              // locks must be 0 before each launch set
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) { cudaMemset(dWS, 0, WS * 4); run(); }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    double us = ms * 1e3, tflops = 2.0 * M * N * K / (ms * 1e9);
    printf("  M=%-5d N=%-5d K=%-5d : %8.2f us  %7.1f TFLOP/s\n", M, N, K, us, tflops);
    cudaEventDestroy(a); cudaEventDestroy(b);
    cleanup();
    return us;
}

int main(int argc, char** argv) {
    if (argc >= 4) {
        int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
        int iters = (argc >= 5) ? atoi(argv[4]) : 50;
        printf("classic Marlin PPU perf (W4A16, random data), %d iters:\n", iters);
        return bench_one(M, N, K, iters) < 0 ? 1 : 0;
    }
    printf("classic Marlin PPU perf (W4A16, random data):\n");
    int shapes[][3] = {
        {1,    4096, 4096}, {1,    14336, 4096}, {1,    4096, 14336},   // batch=1 (decode) -- latency / weight-BW bound
        {2048, 4096, 4096}, {2048, 14336, 4096}, {2048, 4096, 14336},   // batch=2048 (prefill) -- compute bound
    };
    for (auto & s : shapes) bench_one(s[0], s[1], s[2], 50);
    return 0;
}
