// Numerical test for the ported classic Marlin (marlin_classic_ppu.cuh). B is packed in the format the kernel reads,
// derived by tracing the classic B path for config (1,8,8,-1), M16 N128 K128:
//   sh_b[256*k + tid] = B_gmem[tid + 256*k]  (b_gl_rd=tid, b_gl_rd_delta_i=256, one stage). frag_b_quant[k] = that
//   int4 (4 int32, j=0..3). A k-tile = matmul_k*N_WARPS_K + warp_k. n-block(j) = warp_n*4 + j. Per-lane int32
//   packing follows the kernel's dequant path (dequant -> {b0,b1}). scale=1 (write_result scale is FIXME). Validates A(element-wise)
//   + B path + mma_n16 + K-warp reduce + acc write, end-to-end vs CPU.
//
// build (box):  make marlin_classic_num   (nvcc -O3, no -arch).  Run: ./marlin_classic_num

#include "marlin_classic_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    using namespace marlin_classic_ppu;
    const int M = 16, N = 128, K = 128, groupsize = -1, max_par = 128;   // must match marlin_cuda's default (workspace contract)
    const int NWN = 2, NWK = 4;                        // thread_n_blocks/4=2, warps_k=4  (config 1,8,8)

    std::vector<half> hA(M * K), hS(N), hC(M * N, __float2half(0.f));
    std::vector<int>  Bdeq(N * K), hB((size_t) (K / 16) * (N * 16 / 32) * 4);   // int4 buffer as ints
    srand(1234);
    for (auto & x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    // NON-TRIVIAL per-column scales. They used to be all 1.0, which is the multiplicative identity -- so a kernel that
    // never applies them passed anyway. write_result did exactly that (it read frag_s and dropped it), and this test
    // could not see it. Any scale test must use values that are not 1.
    for (auto & x : hS) x = __float2half(0.5f + (rand() % 100) / 100.0f);   // [0.5, 1.49]
    for (auto & x : Bdeq) x = (rand() & 0xf) - 8;

    auto Bu = [&](int n, int k) { return (Bdeq[n * K + k] + 8) & 0xf; };
    auto packq = [&](int nblock, int ktile, int lane) {                        // i4_gemm per-lane packing
        int n = nblock * 16 + lane / 4, kb = ktile * 16 + (lane % 4) * 2, q = 0;
        q |= Bu(n, kb)         << 0;  q |= Bu(n, kb + 1)     << 16;
        q |= Bu(n, kb + 8)     << 4;  q |= Bu(n, kb + 9)     << 20;
        q |= Bu(n + 8, kb)     << 8;  q |= Bu(n + 8, kb + 1) << 24;
        q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
        return q;
    };
    for (int k = 0; k < 2; ++k) for (int tid = 0; tid < 256; ++tid) {          // sh_b[256k+tid] = B_gmem[tid+256k]
        int warp = tid / 32, lane = tid % 32, warp_n = warp % NWN, warp_k = warp / NWN;
        int ktile = k * NWK + warp_k;
        size_t idx = (size_t) (tid + 256 * k) * 4;                             // int4 = 4 ints
        for (int j = 0; j < 4; ++j) hB[idx + j] = packq(warp_n * 4 + j, ktile, lane);
    }
    std::vector<float> ref(M * N);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float acc = 0; for (int k = 0; k < K; ++k) acc += __half2float(hA[m * K + k]) * (float) Bdeq[n * K + k];
        ref[m * N + n] = acc * __half2float(hS[n]);   // C = (A @ dequant(B)) * s   (groupsize=-1: per-column)
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
    printf("classic Marlin PPU numerical (M%d N%d K%d, cfg 1/8/8): max_abs=%.4e rel=%.2e (|ref|max=%.2f) -> %s\n",
           M, N, K, max_abs, rel, ref_max, rel < 3e-2 ? "MATCH" : "MISMATCH");
    printf("  C[0..3]=%.3f %.3f %.3f %.3f  ref=%.3f %.3f %.3f %.3f\n",
           __half2float(hC[0]), __half2float(hC[1]), __half2float(hC[2]), __half2float(hC[3]), ref[0], ref[1], ref[2], ref[3]);
    return rel < 3e-2 ? 0 : 1;
}
