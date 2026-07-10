// Test for the simple plain PPU Marlin kernel. B is packed in the format the kernel reads (bidx + the validated
// per-lane dequant packing), so it's a self-consistent end-to-end correctness check of the
// KERNEL (K-loop + mma_n16 + K-warp reduce + acc write). Whether this packing == vLLM's Marlin gmem repack is a
// separate wiring question; here we validate the kernel math.
//
// build (box):  make marlin_ppu_test   (nvcc -O3, no -arch: ppu001).  Run: ./marlin_ppu_test

#include "marlin_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    using namespace marlin_ppu;
    const int M = 16, N = 128, K = 128;            // config MB=1, NB=8
    const int MB = 1, NB = 8, KB = 8, GB = -1;
    const int k_tiles = K / 16, nblocks = N / 16;

    std::vector<half>  hA(M * K), hS(N, __float2half(1.0f)), hC(M * N, __float2half(0.f));
    std::vector<int>   Bdeq(N * K), hBq((size_t) k_tiles * nblocks * 32);
    srand(1234);
    for (auto & x : hA)  x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto & x : Bdeq) x = (rand() & 0xf) - 8;

    auto Bu = [&](int n, int k) { return (Bdeq[n * K + k] + 8) & 0xf; };
    for (int kt = 0; kt < k_tiles; ++kt) for (int nb = 0; nb < nblocks; ++nb) for (int lane = 0; lane < 32; ++lane) {
        int n = nb * 16 + lane / 4, kb = kt * 16 + (lane % 4) * 2, q = 0;
        q |= Bu(n, kb)         << 0;  q |= Bu(n, kb + 1)     << 16;
        q |= Bu(n, kb + 8)     << 4;  q |= Bu(n, kb + 9)     << 20;
        q |= Bu(n + 8, kb)     << 8;  q |= Bu(n + 8, kb + 1) << 24;
        q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
        hBq[(size_t) (kt * nblocks + nb) * 32 + lane] = q;
    }
    std::vector<float> ref(M * N);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float acc = 0; for (int k = 0; k < K; ++k) acc += __half2float(hA[m * K + k]) * (float) Bdeq[n * K + k];
        ref[m * N + n] = acc;                       // scale = 1
    }

    half * dA, * dC, * dS; int * dBq;
    cudaMalloc(&dA, hA.size() * 2); cudaMalloc(&dC, hC.size() * 2); cudaMalloc(&dS, hS.size() * 2);
    cudaMalloc(&dBq, hBq.size() * 4);
    cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dBq, hBq.data(), hBq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice);

    launch_simple<MB, NB, KB, GB>((const int4 *) dA, (const int4 *) dBq, dC, (const int4 *) dS, M, N, K, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("kernel ERR %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost);

    double max_abs = 0, ref_max = 0;
    for (int i = 0; i < M * N; ++i) { double d = fabs(__half2float(hC[i]) - ref[i]); max_abs = fmax(max_abs, d); ref_max = fmax(ref_max, fabs(ref[i])); }
    double rel = max_abs / (ref_max + 1e-9);
    printf("plain PPU Marlin (M%d N%d K%d, MB%d NB%d): max_abs=%.4e rel=%.2e (|ref|max=%.2f) -> %s\n",
           M, N, K, MB, NB, max_abs, rel, ref_max, rel < 3e-2 ? "MATCH" : "MISMATCH");
    printf("  C[0..3]=%.3f %.3f %.3f %.3f  ref=%.3f %.3f %.3f %.3f\n",
           __half2float(hC[0]), __half2float(hC[1]), __half2float(hC[2]), __half2float(hC[3]), ref[0], ref[1], ref[2], ref[3]);
    return rel < 3e-2 ? 0 : 1;
}
