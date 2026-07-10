// Numerical test for GROUPED scales (groupsize=128, what real quantized models use).
//
// This path was never exercised: every other test in this directory runs groupsize=-1. And it was broken. Grouped
// scales are applied inside matmul() via scale(frag_b, frag_s[k%2][j], i), and frag_s was filled NVIDIA-style -- one
// int4 starting at s_sh_rd = 8*warp_n + lane/4, which puts j at a half-stride of 2 and lane/4 at a stride of 8. PPU's
// dequant hands frag_b0 column (warp_n*4 + j)*16 + lane/4 and frag_b1 that column + 8, so j needs a stride of 16 and
// lane/4 a stride of 1. 22 of the 24 (warp_n, lane, j) combinations disagreed.
//
// K=512 with groupsize=128 gives FOUR groups, so the kernel must actually switch scale groups as it walks K. A K=128
// test would have one group and could not tell a working implementation from a broken one -- the same trap as s=1.0.
//
// B is packed with the config-independent formula derived for gemv_w4a16_ppu.cu:
//     int4 idx = (N/2)*ktile + (nblock/4)*32 + lane,  and the (nblock%4)-th uint32 of it
// which reproduces the classic path's per-config packing exactly (verified: for (1,8,8) it equals tid + 256*k).
//
// build (box):  make marlin_classic_group   (nvcc -O3, no -arch).  Run: ./marlin_classic_group

#include "marlin_classic_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    using namespace marlin_classic_ppu;
    const int M = 16, N = 128, K = 512, groupsize = 128, max_par = 128;
    const int GROUPS = K / groupsize;                 // 4 -- the point of the test

    std::vector<half> hA(M * K), hC(M * N, __float2half(0.f));
    std::vector<half> hS((size_t) GROUPS * N);        // s[g][n], row-major, column order (our format)
    std::vector<int>  Bdeq((size_t) N * K), hB((size_t) (K / 16) * (N * 16 / 32) * 4);
    srand(1234);
    for (auto & x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto & x : Bdeq) x = (rand() & 0xf) - 8;
    // Distinct scale per (group, column): a bug that reads the wrong group, or the wrong column, cannot hide.
    for (int g = 0; g < GROUPS; g++)
        for (int n = 0; n < N; n++) hS[(size_t) g * N + n] = __float2half(0.5f + (rand() % 100) / 100.0f);

    auto Bu = [&](int n, int k) { return (Bdeq[(size_t) n * K + k] + 8) & 0xf; };
    for (int ktile = 0; ktile < K / 16; ktile++)
        for (int nblock = 0; nblock < N / 16; nblock++)
            for (int lane = 0; lane < 32; lane++) {
                int n = nblock * 16 + lane / 4, kb = ktile * 16 + (lane % 4) * 2, q = 0;
                q |= Bu(n, kb)         << 0;  q |= Bu(n, kb + 1)     << 16;
                q |= Bu(n, kb + 8)     << 4;  q |= Bu(n, kb + 9)     << 20;
                q |= Bu(n + 8, kb)     << 8;  q |= Bu(n + 8, kb + 1) << 24;
                q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
                size_t idx = (size_t) (N / 2) * ktile + (nblock / 4) * 32 + lane;
                hB[idx * 4 + (nblock % 4)] = q;
            }

    // C[m][n] = sum_g s[g][n] * sum_{k in group g} A[m][k] * q[n][k]
    std::vector<float> ref((size_t) M * N);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float total = 0;
        for (int g = 0; g < GROUPS; g++) {
            float acc = 0;
            for (int k = g * groupsize; k < (g + 1) * groupsize; ++k)
                acc += __half2float(hA[(size_t) m * K + k]) * (float) Bdeq[(size_t) n * K + k];
            total += acc * __half2float(hS[(size_t) g * N + n]);
        }
        ref[(size_t) m * N + n] = total;
    }

    half * dA, * dC, * dS; int * dB, * dWS;
    cudaMalloc(&dA, hA.size() * 2); cudaMalloc(&dB, hB.size() * 4); cudaMalloc(&dC, hC.size() * 2);
    cudaMalloc(&dS, hS.size() * 2); cudaMalloc(&dWS, (N / 128 + 1) * max_par * 4);
    cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dWS, 0, (N / 128 + 1) * max_par * 4);

    int ret = marlin_cuda(dA, dB, dC, dS, M, N, K, dWS, groupsize);
    cudaError_t e = cudaDeviceSynchronize();
    if (ret || e) { printf("ret=%d err=%s\n", ret, cudaGetErrorString(e)); return 2; }
    cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost);

    double max_abs = 0, ref_max = 0;
    for (size_t i = 0; i < (size_t) M * N; ++i) {
        max_abs = fmax(max_abs, fabs(__half2float(hC[i]) - ref[i]));
        ref_max = fmax(ref_max, fabs(ref[i]));
    }
    double rel = max_abs / (ref_max + 1e-9);
    printf("classic Marlin PPU grouped scales (M%d N%d K%d, groupsize=%d -> %d groups, cfg 1/8/8): "
           "max_abs=%.4e rel=%.2e (|ref|max=%.2f) -> %s\n",
           M, N, K, groupsize, GROUPS, max_abs, rel, ref_max, rel < 3e-2 ? "MATCH" : "MISMATCH");
    printf("  C[0..3]=%.3f %.3f %.3f %.3f  ref=%.3f %.3f %.3f %.3f\n",
           __half2float(hC[0]), __half2float(hC[1]), __half2float(hC[2]), __half2float(hC[3]),
           ref[0], ref[1], ref[2], ref[3]);
    return rel < 3e-2 ? 0 : 1;
}
