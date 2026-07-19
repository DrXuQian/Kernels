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
#include "affine_split_ppu.cuh"   // asum_kernel + affine_split_cuda: ONE copy, shared with test_marlin_gguf
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 2048;
    const int N = argc > 2 ? atoi(argv[2]) : 4096;
    const int K = argc > 3 ? atoi(argv[3]) : 4096;
    // gs from argv so the bench is not permanently pinned to the one groupsize that hid an asum bug:
    // the shuffle reduction was short by half at gs=128 and only gs=32 was ever benched here.
    const int gs = argc > 4 ? atoi(argv[4]) : 32;
    const int G = K / gs, max_par = 128, iters = 20;
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
        affine_split_ppu::asum_kernel<<<(int) (((long long) M * (K / 8) + 255) / 256), 256>>>(dA, dAsum, M, K, gs);
        cublasHgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, N, M, G, &alpha,
                    dM, N, dAsum, G, &beta_acc, dC, N);
    };

    {   // asum self-check against a CPU reduction over random A (memset patterns would hide index bugs)
        std::vector<half> hA((size_t) M * K); std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : hA) x = __float2half(0.1f * nd(rng));
        CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
        affine_split_ppu::asum_kernel<<<(int) (((long long) M * (K / 8) + 255) / 256), 256>>>(dA, dAsum, M, K, gs);
        CK(cudaDeviceSynchronize());
        std::vector<half> got((size_t) M * G);
        CK(cudaMemcpy(got.data(), dAsum, got.size() * 2, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int m = 0; m < M; m += 97)
            for (int g = 0; g < G; g++) {
                double r = 0;
                for (int i = 0; i < gs; i++) r += __half2float(hA[(size_t) m * K + (size_t) g * gs + i]);
                worst = fmax(worst, fabs(r - __half2float(got[(size_t) m * G + g])));
            }
        printf("  asum self-check: max abs err %.2e -> %s\n", worst, worst < 2e-2 ? "MATCH" : "MISMATCH");
        CK(cudaMemset(dA, 1, A_h * 2));
    }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    // flops: the correction's inner dimension is G, NOT K. Printing 2*M*N*K over the correction's time
    // reported 1214 TFLOP/s and 242.9% MFU -- six times a peak that cannot be exceeded. A >100% figure is
    // a broken denominator every time, and this is the second one this session I read past instead of
    // stopping at (the first was 114.5% HBM on a 1792 GB/s part).
    auto time_it = [&](const char* name, double flops, auto&& fn) {
        for (int i = 0; i < 3; i++) fn();
        CK(cudaDeviceSynchronize());
        cudaEventRecord(a);
        for (int i = 0; i < iters; i++) fn();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
        const double tf = flops / (ms * 1e9);
        printf("  %-34s %8.2f us   %7.1f TFLOP/s  %5.1f%% MFU\n", name, ms * 1e3, tf, 100.0 * tf / 500.0);
        return ms * 1e3;
    };

    const double F_main = 2.0 * M * N * K, F_corr = 2.0 * (double) M * N * G;
    const double t_aff  = time_it("fused affine (mins in mainloop)", F_main, [&] { marlin(true); });
    const double t_sym  = time_it("symmetric only (no min at all)",  F_main, [&] { marlin(false); });
    // Split, because "the correction is slow" does not say WHICH part. asum is bandwidth work (read all of
    // A); the hgemm is a skinny GEMM plus a read-modify-write of all of C, which is a floor the separate
    // -kernel shape cannot avoid -- only fusing into Marlin's epilogue could.
    const double t_asum = time_it("  asum only (read A, reduce by gs)", F_corr,
        [&] { affine_split_ppu::asum_kernel<<<(int) (((long long) M * (K / 8) + 255) / 256), 256>>>(dA, dAsum, M, K, gs); });
    const double t_hg   = time_it("  hgemm only (skinny, + C rmw)", F_corr,
        [&] { cublasHgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, N, M, G, &alpha, dM, N, dAsum, G, &beta_acc, dC, N); });
    const double t_corr = time_it("correction alone (asum + hgemm)", F_corr, [&] { correction(); });
    const double t_spl  = time_it("SPLIT = symmetric + correction",  F_main, [&] { marlin(false); correction(); });
    const double bytes_floor = (double) M * K * 2 + 2.0 * M * N * 2;   // read A, read+write C
    printf("  correction traffic floor: %.1f MB -> %.1f us at 2766 GB/s (asum %.1f + hgemm %.1f measured)\n",
           bytes_floor / 1e6, bytes_floor / 2766e3, t_asum, t_hg);

    printf("\n  correction costs %.1f us; folding the min in costs %.1f us -> split %s by %.1f%%\n",
           t_corr, t_aff - t_sym, t_spl < t_aff ? "WINS" : "LOSES", 100.0 * (t_aff - t_spl) / t_aff);
    printf("  (timing only -- inputs are memset patterns, correctness lives in test_marlin_gguf)\n");
    cublasDestroy(cub);
    return 0;
}
