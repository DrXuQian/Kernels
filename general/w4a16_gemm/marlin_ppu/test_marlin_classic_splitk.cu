// Split-K test for the classic Marlin PPU port: exercises global_reduce (cross-CTA K reduction). Shape N=128 K=512
// cfg (1,8,8) -> n_tiles=1, k_tiles=4, so with many SMs the single output tile's K is split across 4 CTAs (slice_row
// 0..3), each computing a 128-wide K slice; global_reduce sums them in L2. B packing = the single-stage trace +
// the slice_row offset (b_gl_rd += 512*r): B_gmem[tid + 256*k + 512*r], global k-tile = r*8 + k*4 + warp_k (0..31).
//
// build (box):  make marlin_classic_splitk   (nvcc -O3, no -arch).  Run: ./marlin_classic_splitk
// (If it doesn't split-K on this GPU -- few SMs -- it still validates the non-split path; check slice_count via ncu.)

#include "marlin_classic_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    using namespace marlin_classic_ppu;
    const int M = 16, N = 128, K = 512, groupsize = -1, max_par = 128;   // must match marlin_cuda's default (workspace contract)
    const int NWN = 2, NWK = 4, k_tiles = K / 16 / 8;   // = 4 CTA K-slices

    std::vector<half> hA(M * K), hS(N, __float2half(1.0f)), hC(M * N, __float2half(0.f));
    std::vector<int>  Bdeq(N * K), hB((size_t) (K / 16) * (N * 16 / 32) * 4);
    srand(1234);
    for (auto & x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto & x : Bdeq) x = (rand() & 0xf) - 8;

    auto Bu = [&](int n, int k) { return (Bdeq[n * K + k] + 8) & 0xf; };
    auto packq = [&](int nblock, int ktile, int lane) {
        int n = nblock * 16 + lane / 4, kb = ktile * 16 + (lane % 4) * 2, q = 0;
        q |= Bu(n, kb)         << 0;  q |= Bu(n, kb + 1)     << 16;
        q |= Bu(n, kb + 8)     << 4;  q |= Bu(n, kb + 9)     << 20;
        q |= Bu(n + 8, kb)     << 8;  q |= Bu(n + 8, kb + 1) << 24;
        q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
        return q;
    };
    for (int r = 0; r < k_tiles; ++r) for (int k = 0; k < 2; ++k) for (int tid = 0; tid < 256; ++tid) {
        int warp = tid / 32, lane = tid % 32, warp_n = warp % NWN, warp_k = warp / NWN;
        int gktile = r * 8 + k * NWK + warp_k;             // global 16-k-tile index (0..31)
        size_t idx = (size_t) (tid + 256 * k + 512 * r) * 4;
        for (int j = 0; j < 4; ++j) hB[idx + j] = packq(warp_n * 4 + j, gktile, lane);
    }
    std::vector<float> ref(M * N);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float acc = 0; for (int k = 0; k < K; ++k) acc += __half2float(hA[m * K + k]) * (float) Bdeq[n * K + k];
        ref[m * N + n] = acc;
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
    for (int i = 0; i < M * N; ++i) { double d = fabs(__half2float(hC[i]) - ref[i]); max_abs = fmax(max_abs, d); ref_max = fmax(ref_max, fabs(ref[i])); }
    double rel = max_abs / (ref_max + 1e-9);
    printf("classic Marlin PPU split-K (M%d N%d K%d, k_tiles=%d): max_abs=%.4e rel=%.2e (|ref|max=%.2f) -> %s\n",
           M, N, K, k_tiles, max_abs, rel, ref_max, rel < 4e-2 ? "MATCH" : "MISMATCH");
    printf("  C[0..3]=%.2f %.2f %.2f %.2f  ref=%.2f %.2f %.2f %.2f\n",
           __half2float(hC[0]), __half2float(hC[1]), __half2float(hC[2]), __half2float(hC[3]), ref[0], ref[1], ref[2], ref[3]);
    return rel < 4e-2 ? 0 : 1;
}
