// Compile/run gate for the ported classic Marlin (marlin_classic_ppu.cuh). Random buffers, small shape, launch,
// check it COMPILES and RUNS without a CUDA error. This surfaces the two classic-port unknowns on the box:
//   (1) does plain ldmatrix.x4 (ldsm4) work on the PPU? (2) does the whole kernel compile?
// NUMERICAL validation needs B in the real Marlin offline-repack format -> next step (test_marlin_classic_ref.cu).
//
// build (box):  make marlin_classic_test   (nvcc -O3, no -arch: ppu001).  Run: ./marlin_classic_test

#include "marlin_classic_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    using namespace marlin_classic_ppu;
    const int M = 16, N = 128, K = 128, groupsize = -1, max_par = 128;   // must match marlin_cuda's default (workspace contract)
    const size_t A_i4 = (size_t) M * K / 8, B_i4 = (size_t) (K / 16) * (N * 16 / 32), C_i4 = (size_t) M * N / 8, S_h = (size_t) N;
    const size_t WS = (size_t) (N / 128 + 1) * max_par;

    std::vector<half> hA(M * K), hS(S_h, __float2half(1.0f)), hC(M * N, __float2half(0.f));
    std::vector<int>  hB(B_i4 * 4);
    srand(1234);
    for (auto & x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto & x : hB) x = rand();

    int4 *dA, *dB, *dC, *dS; int * dWS;
    cudaMalloc(&dA, A_i4 * 16); cudaMalloc(&dB, B_i4 * 16); cudaMalloc(&dC, C_i4 * 16);
    cudaMalloc(&dS, (S_h / 8) * 16); cudaMalloc(&dWS, WS * 4);
    cudaMemcpy(dA, hA.data(), A_i4 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), B_i4 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dS, hS.data(), (S_h / 8) * 16, cudaMemcpyHostToDevice);
    cudaMemset(dWS, 0, WS * 4);

    int ret = marlin_cuda(dA, dB, dC, dS, M, N, K, dWS, groupsize);
    cudaError_t e = cudaDeviceSynchronize();
    printf("classic Marlin PPU: ret=%d  launch/exec=%s\n", ret, cudaGetErrorString(e));
    if (ret) { printf("  (ret nonzero: %s)\n", ret == 1 ? "ERR_PROB_SHAPE" : "ERR_KERN_SHAPE"); return 1; }
    if (e) return 2;
    cudaMemcpy(hC.data(), dC, C_i4 * 16, cudaMemcpyDeviceToHost);
    printf("  ran OK. C[0..7] = ");
    for (int i = 0; i < 8; ++i) printf("%.2f ", __half2float(hC[i]));
    printf("\n  NOTE: numerical validation TODO (needs the real Marlin offline B repack).\n");
    return 0;
}
