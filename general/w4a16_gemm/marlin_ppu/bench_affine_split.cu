// Does splitting the affine min term into its OWN kernel beat folding it into Marlin's mainloop?
//
// The min never touches B, so it factors out of the quantized GEMM entirely:
//   C[m][n] = sum_g s[g][n] * sum_{k in g} A[m][k]*(q-8)   +   sum_g m'[g][n] * sum_{k in g} A[m][k]
//             \____________ symmetric Marlin ____________/      \______ Asum[M][G] @ m'[G][N] ______/
// with Asum[m][g] = sum over that group's 32 activations. The correction is a plain fp16 GEMM whose inner
// dimension is G = K/32 -- ONE THIRTY-SECOND of the main GEMM's K, so ~3% of the FLOPs.
//
// Why it might win: folding the min in costs 10-11 MFU points at gs=32 (29.9 against 40.5 sym on
// 2048x4096x4096), because every k of the mainloop pays an extra shared int4 load and an hfma2, and
// frag_m holds 8 registers against a budget MARLIN_MIN_BLOCKS=2 already pins. Split out, the mainloop goes
// back to symmetric and the min is paid once per (m, group) instead of once per (m, k).
//
// Why it might NOT: the correction is a SKINNY gemm (G = 128 at K=4096) and skinny gemms run far below
// peak. At 100 TFLOP/s it is ~21us against the 121us saved; at 25 it is ~85 and the whole thing is a wash.
// That efficiency is the entire question and it is why this measures rather than argues.
//
// build (box): make bench_affine_split && ./bench_affine_split
#include "marlin_gguf_ppu.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

// Asum[m][g] = sum of A[m][k] over the group's gs activations. One block per (m, chunk of groups).
__global__ void asum_kernel(const half* __restrict__ A, half* __restrict__ Asum, int M, int K, int gs) {
    const int G = K / gs;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * G) return;
    const int m = idx / G, g = idx % G;
    float acc = 0.f;
    const half* a = A + (size_t) m * K + (size_t) g * gs;
    for (int i = 0; i < gs; i++) acc += __half2float(a[i]);
    Asum[(size_t) m * G + g] = __float2half(acc);
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 2048;
    const int N = argc > 2 ? atoi(argv[2]) : 4096;
    const int K = argc > 3 ? atoi(argv[3]) : 4096;
    const int gs = 32, G = K / gs, max_par = 128, iters = 20;
    printf("affine split: M=%d N=%d K=%d gs=%d (G=%d, correction is %.1f%% of the main GEMM's FLOPs)\n",
           M, N, K, gs, G, 100.0 / gs);

    const size_t A_h = (size_t) M * K, B_i = (size_t)(K / 16) * (N * 16 / 32) * 4, C_h = (size_t) M * N;
    const size_t S_h = (size_t) G * N;
    half *dA, *dC, *dS, *dM, *dAsum; int *dB, *dWS;
    CK(cudaMalloc(&dA, A_h * 2)); CK(cudaMalloc(&dB, B_i * 4)); CK(cudaMalloc(&dC, C_h * 2));
    CK(cudaMalloc(&dS, S_h * 2)); CK(cudaMalloc(&dM, S_h * 2));
    CK(cudaMalloc(&dAsum, (size_t) M * G * 2));
    CK(cudaMalloc(&dWS, (N / 128 + 1) * max_par * 4));
    CK(cudaMemset(dA, 1, A_h * 2)); CK(cudaMemset(dB, 1, B_i * 4));
    CK(cudaMemset(dS, 0x3c, S_h * 2)); CK(cudaMemset(dM, 0x30, S_h * 2)); CK(cudaMemset(dWS, 0, (N / 128 + 1) * max_par * 4));

    cublasHandle_t cub; cublasCreate(&cub);
    cublasSetMathMode(cub, CUBLAS_TENSOR_OP_MATH);
    const half alpha = __float2half(1.f), beta_acc = __float2half(1.f);

    auto marlin = [&](bool affine) {
        return marlin_gguf_ppu::marlin_cuda(dA, dB, dC, dS, affine ? (void*) dM : nullptr,
                                            M, N, K, dWS, gs, 0, 0, -1, -1, -1, max_par);
    };
    // C += Asum[M x G] * m'[G x N]. cublas is column-major, so compute C^T = m'^T * Asum^T by swapping.
    auto correction = [&] {
        asum_kernel<<<(M * G + 255) / 256, 256>>>(dA, dAsum, M, K, gs);
        cublasHgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, N, M, G, &alpha,
                    dM, N, dAsum, G, &beta_acc, dC, N);
    };

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto time_it = [&](const char* name, auto&& fn) {
        for (int i = 0; i < 3; i++) fn();
        CK(cudaDeviceSynchronize());
        cudaEventRecord(a);
        for (int i = 0; i < iters; i++) fn();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
        printf("  %-34s %8.2f us   %7.1f TFLOP/s  %5.1f%% MFU\n", name, ms * 1e3,
               2.0 * M * N * K / (ms * 1e9), 100.0 * (2.0 * M * N * K / (ms * 1e9)) / 500.0);
        return ms * 1e3;
    };

    const double t_aff  = time_it("fused affine (mins in mainloop)", [&] { marlin(true); });
    const double t_sym  = time_it("symmetric only (no min at all)",  [&] { marlin(false); });
    const double t_corr = time_it("correction alone (asum + hgemm)", [&] { correction(); });
    const double t_spl  = time_it("SPLIT = symmetric + correction",  [&] { marlin(false); correction(); });

    printf("\n  correction costs %.1f us; folding the min in costs %.1f us -> split %s by %.1f%%\n",
           t_corr, t_aff - t_sym, t_spl < t_aff ? "WINS" : "LOSES", 100.0 * (t_aff - t_spl) / t_aff);
    printf("  (timing only -- inputs are memset patterns, correctness lives in test_marlin_gguf)\n");
    cublasDestroy(cub);
    return 0;
}
