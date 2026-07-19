// Which W4A16 path wins at which M? Scans batch size across the three we have, all on the SAME data,
// all verified against one CPU reference before any of them is timed.
//
//   fused   : Marlin with the affine min folded into the mainloop           (what we ship today)
//   split   : symmetric Marlin + Asum[M x G] @ m'                           (min as its own pass, ~20% at M=2048)
//   dequant : expand B to fp16 once, then a dense cublasHgemm               (no fused dequant at all)
//
// The shape of the answer is forced by the arithmetic and only the crossover is unknown. dequant's extra
// cost is FIXED at one round trip of the fp16 weights (write N*K*2, read it back); the MFU it buys grows
// with M. So at M=1 it must lose badly -- it quadruples weight traffic on a bandwidth-bound problem -- and
// at large M it should win if a dense fp16 GEMM clears Marlin's ~40.6% at gs=32. Where they cross is the
// number this exists to produce.
//
// build (box): make bench_prefill_paths && ./bench_prefill_paths
#include "marlin_gguf_ppu.cuh"
#include "affine_split_ppu.cuh"
#include "dequant_w4_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 4096;
    const int K = argc > 2 ? atoi(argv[2]) : 4096;
    const int gs = 32, G = K / gs, max_par = 128;
    printf("W4A16 prefill paths on PPU, N=%d K=%d gs=%d (fp16 peak 500 TFLOP/s)\n", N, K, gs);

    const int Mv[] = {1, 16, 64, 128, 256, 512, 1024, 2048, 4096};
    const int MMAX = 4096;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> qd(-8, 7);

    std::vector<half> hA((size_t) MMAX * K);
    for (auto& x : hA) x = __float2half(0.1f * nd(rng));
    std::vector<int> Bq((size_t) N * K);
    for (auto& x : Bq) x = qd(rng);
    std::vector<half> sPlain((size_t) G * N), mPlain((size_t) G * N), sDev((size_t) G * N);
    for (size_t i = 0; i < sPlain.size(); i++) {
        sPlain[i] = __float2half(0.4f + 0.6f * ((rng() % 100) / 100.0f));
        mPlain[i] = __float2half(((int) (rng() % 100) - 50) / 20.0f);
    }
    marlin_gguf_ppu::marlin_permute_scales(sPlain.data(), sDev.data(), G, N);
    std::vector<half> mDev((size_t) G * N);
    marlin_gguf_ppu::marlin_permute_scales(mPlain.data(), mDev.data(), G, N);

    // Marlin's B packing (ktile-major, per-lane), verbatim from the tests.
    std::vector<int> hB((size_t) (K / 16) * (N * 16 / 32) * 4);
    {
        auto Bu = [&](int n, int k) { return (Bq[(size_t) n * K + k] + 8) & 0xf; };
        for (int kt = 0; kt < K / 16; kt++)
            for (int nb = 0; nb < N / 16; nb++)
                for (int l = 0; l < 32; l++) {
                    int n = nb * 16 + l / 4, kb = kt * 16 + (l % 4) * 2, q = 0;
                    q |= Bu(n, kb) << 0;      q |= Bu(n, kb + 1) << 16;
                    q |= Bu(n, kb + 8) << 4;  q |= Bu(n, kb + 9) << 20;
                    q |= Bu(n + 8, kb) << 8;  q |= Bu(n + 8, kb + 1) << 24;
                    q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
                    size_t idx = (size_t) (N / 2) * kt + (nb / 4) * 32 + l;
                    hB[idx * 4 + (nb % 4)] = q;
                }
    }

    half *dA, *dC, *dS, *dM, *dSp, *dMp, *dAsum, *dW; int *dB, *dWS;
    CK(cudaMalloc(&dA, hA.size() * 2));       CK(cudaMalloc(&dB, hB.size() * 4));
    CK(cudaMalloc(&dC, (size_t) MMAX * N * 2));
    CK(cudaMalloc(&dS, sDev.size() * 2));     CK(cudaMalloc(&dM, mDev.size() * 2));
    CK(cudaMalloc(&dSp, sPlain.size() * 2));  CK(cudaMalloc(&dMp, mPlain.size() * 2));
    CK(cudaMalloc(&dAsum, (size_t) MMAX * G * 2));
    CK(cudaMalloc(&dW, (size_t) N * K * 2));
    CK(cudaMalloc(&dWS, (N / 128 + 1) * max_par * 4));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS, sDev.data(), sDev.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dM, mDev.data(), mDev.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSp, sPlain.data(), sPlain.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dMp, mPlain.data(), mPlain.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dWS, 0, (N / 128 + 1) * max_par * 4));

    cublasHandle_t cub; cublasCreate(&cub);
    cublasSetMathMode(cub, CUBLAS_TENSOR_OP_MATH);
    const half one = __float2half(1.f), zero = __float2half(0.f);

    auto run_fused = [&](int M) {
        return marlin_gguf_ppu::marlin_cuda(dA, dB, dC, dS, dM, M, N, K, dWS, gs, 0, 0, -1, -1, -1, max_par);
    };
    auto run_split = [&](int M) {
        return affine_split_ppu::affine_split_cuda(dA, dB, dC, dS, dMp, M, N, K, dWS, gs, dAsum, cub, max_par);
    };
    // The dequant is charged IN FULL on every call: a weight is expanded once per forward pass and used by
    // exactly one GEMM, so amortizing it across iterations would be measuring a reuse that never happens.
    auto run_dequant = [&](int M) {
        dequant_w4_ppu::dequant_launch(dB, dSp, dMp, dW, N, K, gs);
        cublasHgemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, dW, K, dA, K, &zero, dC, N);
        return 0;
    };

    // ---- correctness first, on a small M, against a CPU reference none of the three shares ----
    {
        const int Mc = 64;
        std::vector<double> ref((size_t) Mc * N);
        for (int m = 0; m < Mc; m++)
            for (int n = 0; n < N; n++) {
                double acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (double) __half2float(hA[(size_t) m * K + k]) *
                           ((double) Bq[(size_t) n * K + k] * __half2float(sPlain[(size_t)(k / gs) * N + n])
                            + __half2float(mPlain[(size_t)(k / gs) * N + n]));
                ref[(size_t) m * N + n] = acc;
            }
        std::vector<half> got((size_t) Mc * N);
        auto check = [&](const char* name, auto&& fn) {
            CK(cudaMemset(dC, 0, (size_t) Mc * N * 2));
            fn(Mc); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(got.data(), dC, got.size() * 2, cudaMemcpyDeviceToHost));
            double ma = 0, rm = 0;
            for (size_t i = 0; i < got.size(); i++) {
                ma = fmax(ma, fabs(__half2float(got[i]) - ref[i])); rm = fmax(rm, fabs(ref[i]));
            }
            const double rel = ma / (rm + 1e-9);
            printf("  correctness %-8s rel %.2e -> %s\n", name, rel, rel < 3e-2 ? "MATCH" : "MISMATCH");
            return rel < 3e-2;
        };
        bool ok = true;
        ok &= check("fused",   run_fused);
        ok &= check("split",   run_split);
        ok &= check("dequant", run_dequant);
        if (!ok) { printf("  a path is WRONG -- not timing it\n"); return 1; }
    }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto tm = [&](int M, auto&& fn) {
        const int it = M >= 1024 ? 10 : 50;
        for (int i = 0; i < 3; i++) fn(M);
        CK(cudaDeviceSynchronize());
        cudaEventRecord(a);
        for (int i = 0; i < it; i++) fn(M);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        return (double) ms * 1e3 / it;
    };

    printf("\n  %-6s %11s %11s %11s   %-8s %s\n", "M", "fused us", "split us", "dequant us", "winner", "MFU of winner");
    for (int M : Mv) {
        const double tf = tm(M, run_fused), ts = tm(M, run_split), td = tm(M, run_dequant);
        const double best = fmin(tf, fmin(ts, td));
        const char* w = (best == td) ? "dequant" : (best == ts) ? "split" : "fused";
        printf("  %-6d %11.2f %11.2f %11.2f   %-8s %5.1f%%\n", M, tf, ts, td, w,
               100.0 * (2.0 * M * N * K / (best * 1e3)) / 500e9);
    }
    printf("\n  dequant charges a full weight expansion (%.1f MB write + read back) on every call.\n",
           2.0 * N * K * 2 / 1e6);
    cublasDestroy(cub);
    return 0;
}
