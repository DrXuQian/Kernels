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
// TWO configs are exercised, because the scale index depends on BOTH of the parameters that differ between them:
//     M<=16 -> (1,8,8):   thread_n_blocks/4 = 2 n-warps, group_blocks/thread_k_blocks = 8/8 = 1 (new group each stage)
//     M>16  -> (2,16,4):  4 n-warps,                     8/4 = 2 (group changes every SECOND stage)   <- PREFILL RUNS THIS
// warp_n = (tid/32) % (thread_n_blocks/4) feeds the column index directly, and the stage->group mapping feeds
// sh_s_stage. A test that only runs M=16 never executes the config prefill actually uses.
//
// build (box):  make marlin_classic_group   (nvcc -O3, no -arch).  Run: ./marlin_classic_group

#include "marlin_classic_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static int run_case(int M, int N, int K) {
    using namespace marlin_classic_ppu;
    const int groupsize = 128, max_par = 128;
    const int GROUPS = K / groupsize;                 // >= 2, so the kernel must switch groups while walking K

    std::vector<half> hA(M * K), hC(M * N, __float2half(0.f));
    std::vector<half> hS((size_t) GROUPS * N);        // s[g][n], row-major, column order (our format)
    std::vector<int>  Bdeq((size_t) N * K), hB((size_t) (K / 16) * (N * 16 / 32) * 4);
    srand(1234);
    for (auto & x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto & x : Bdeq) x = (rand() & 0xf) - 8;
    // Distinct scale per (group, column): a bug that reads the wrong group, or the wrong column, cannot hide.
    for (int g = 0; g < GROUPS; g++)
        for (int n = 0; n < N; n++) hS[(size_t) g * N + n] = __float2half(0.5f + (rand() % 100) / 100.0f);

    // hS above is the PLAIN [G][N] layout our kernel reads by column. hSdev is what actually gets uploaded.
    std::vector<half> hSdev = hS;
#ifdef TEST_SCALE_PERM
    // Upstream Marlin/vLLM does NOT ship plain scales -- its repack applies _scale_perm:
    //     for i in range(8): scale_perm += [i + 8*j for j in range(8)]
    //     s = s.reshape((-1, 64))[:, scale_perm]
    // i.e. an 8x8 transpose inside every 64-column chunk. Pair this with -DMARLIN_NVIDIA_SCALE_MAP
    // (the kernel's one-int4-per-lane read) and the question "can we eat a vLLM checkpoint as-is?"
    // becomes a PASS/FAIL, instead of something I derive on a whiteboard and get wrong.
    for (int g = 0; g < GROUPS; g++)
        for (int c0 = 0; c0 < N; c0 += 64)
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++)
                    hSdev[(size_t) g * N + c0 + 8 * i + j] = hS[(size_t) g * N + c0 + i + 8 * j];
#endif

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
    cudaMemcpy(dS, hSdev.data(), hS.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dWS, 0, (N / 128 + 1) * max_par * 4);

    int ret = marlin_cuda(dA, dB, dC, dS, M, N, K, dWS, groupsize);
    cudaError_t e = cudaDeviceSynchronize();
    if (ret || e) { printf("  M%-4d N%-5d K%-4d: ret=%d err=%s\n", M, N, K, ret, cudaGetErrorString(e)); return 2; }
    cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost);

    double max_abs = 0, ref_max = 0;
    for (size_t i = 0; i < (size_t) M * N; ++i) {
        max_abs = fmax(max_abs, fabs(__half2float(hC[i]) - ref[i]));
        ref_max = fmax(ref_max, fabs(ref[i]));
    }
    double rel = max_abs / (ref_max + 1e-9);
    const int mb = (M <= 16) ? 1 : 2, nb = (M <= 16) ? 8 : 16, kb = (M <= 16) ? 8 : 4;
    printf("  M%-4d N%-5d K%-4d gs=%d (%d groups)  cfg (%d,%d,%d): max_abs=%.4e rel=%.2e (|ref|max=%.1f) -> %s\n",
           M, N, K, groupsize, GROUPS, mb, nb, kb, max_abs, rel, ref_max, rel < 3e-2 ? "MATCH" : "MISMATCH");
    if (rel >= 3e-2)
        printf("       C[0..3]=%.3f %.3f %.3f %.3f  ref=%.3f %.3f %.3f %.3f\n",
               __half2float(hC[0]), __half2float(hC[1]), __half2float(hC[2]), __half2float(hC[3]),
               ref[0], ref[1], ref[2], ref[3]);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dS); cudaFree(dWS);
    return rel < 3e-2 ? 0 : 1;
}

int main() {
    printf("classic Marlin PPU grouped scales (groupsize=128):\n");
    int bad = 0;
    bad |= run_case(  16,  128,  512);   // (1,8,8)   -- 2 n-warps, group changes every stage
    bad |= run_case(  32,  256,  512);   // (2,16,4)  -- 4 n-warps, group changes every 2nd stage  <- PREFILL's config
    bad |= run_case(  64,  256, 1024);   // (2,16,4)  -- 8 groups, par branch, more k-tiles
    bad |= run_case( 128,  512, 1024);   // (2,16,4)  -- wider N (2 slice_cols), exercises s_gl_rd's slice_col term
    printf("%s\n", bad ? "SOME CASES FAILED" : "all grouped-scale cases MATCH");
    return bad;
}
