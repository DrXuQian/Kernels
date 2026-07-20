// Perf for the Q4_K MoE on the AIU path. Correctness lives in test_moe_aiu (all cases MATCH).
//
// READ THE WEIGHT TRAFFIC COLUMN, NOT TFLOP/s. The fp16 MoE GEMM established that this kernel class is
// weight/memory bound, which is why it minimizes TOTAL M-TILES (sum_e ceil(rows_e/BMR)) rather than flops:
// each m-tile re-streams its expert's weight block. So the number that moves with BMR is bytes, and a
// TFLOP/s figure can improve while the thing that actually binds gets worse.
//
// weight_min = the weight each expert is READ AT LEAST once for: sum over experts with rows > 0 of
// (N*K/2) bytes, times n-blocks/1 (one pass per (expert, n-block) if L2 holds the block across its
// m-tiles). weight_tiles = what the m-tile count implies if L2 holds nothing. The truth is between them,
// and the ratio to weight_min is the load-balance/L2 efficiency the persistent scheduler exists to fix.
//
// Shapes default to Qwen3.5-A3B's FC1 (N=1024 K=2048); pass N K to change.
// Build: make bench_moe_aiu   Run: ./bench_moe_aiu [N] [K] [tokens] [topk] [experts]
#include "marlin_moe_aiu_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e_=(x); if(e_){printf("cuda %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);} } while(0)

template <int NST, int BMR>
static double bench(int n_experts, int N, int K, int total_rows, const std::vector<int>& rows_expert, int iters) {
    using namespace marlin_moe_aiu_ppu;
    const int gs = 32, G = K / gs, IH = aiu_img_h(N), IW = aiu_img_w(K);
    std::vector<int> bounds(n_experts + 1, 0);
    for (int t = 0; t < total_rows; t++) bounds[rows_expert[t] + 1]++;
    for (int e = 0; e < n_experts; e++) bounds[e + 1] += bounds[e];
    const auto sch = moe_sched(bounds.data(), n_experts, N, BMR);

    half *dA, *dS; unsigned short* dImg; float* dC; int *dB2, *dM2;
    CK(cudaMalloc(&dA, (size_t) total_rows * K * 2));
    CK(cudaMalloc(&dImg, (size_t) n_experts * IH * IW * 2));
    CK(cudaMalloc(&dS, (size_t) n_experts * G * N * 2));
    CK(cudaMalloc(&dC, (size_t) total_rows * N * 4));
    CK(cudaMalloc(&dB2, (n_experts + 1) * 4)); CK(cudaMalloc(&dM2, (n_experts + 1) * 4));
    CK(cudaMemset(dA, 0x11, (size_t) total_rows * K * 2));
    CK(cudaMemset(dImg, 0x22, (size_t) n_experts * IH * IW * 2));
    CK(cudaMemset(dS, 0x3c, (size_t) n_experts * G * N * 2));
    CK(cudaMemcpy(dB2, bounds.data(), (n_experts + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dM2, sch.mblk_prefix.data(), (n_experts + 1) * 4, cudaMemcpyHostToDevice));

    auto run = [&] { launch_moe_q4k<NST, BMR>(dA, dImg, dS, dB2, dM2, dC, total_rows, N, K,
                                              n_experts, sch.total_tiles, gs); };
    run(); CK(cudaDeviceSynchronize());
    // MOE_NCU=1: exactly ONE launch, no warmup, no timing loop -- a clean profiler capture. Same
    // convention as GEMV_NCU / FA_NCU. The printed ms is meaningless then.
    if (getenv("MOE_NCU")) {
        printf("  MOE_NCU: single launch NST=%d BMR=%d (tiles=%d mtiles=%d) -- timing skipped\n",
               NST, BMR, sch.total_tiles, sch.mblk_prefix[n_experts]);
        cudaFree(dA); cudaFree(dImg); cudaFree(dS); cudaFree(dC); cudaFree(dB2); cudaFree(dM2);
        return 0.0;
    }
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) run();
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for (int i = 0; i < iters; i++) run(); cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;

    int active = 0; for (int e = 0; e < n_experts; e++) if (bounds[e+1] > bounds[e]) active++;
    const double wbytes_min = (double) active * N * K / 2.0;                 // each active expert once
    const double wbytes_tiles = (double) sch.mblk_prefix[n_experts] * N * K / 2.0;  // once per m-tile
    const double flop = 2.0 * total_rows * N * K;
    // The OTHER half of the BMR trade, and the half the "minimize m-tiles" rule leaves out: rows actually
    // COMPUTED is mtiles*BMR, not total_rows. Raising BMR cuts weight re-streaming and inflates masked
    // work at the same time, so BMR=256 can have the fewest m-tiles (1.0x weight) and still lose badly --
    // measured here, 0.762ms against 0.416 at BMR=128, with 2.0x the rows computed. Both columns or
    // neither.
    const double rows_done = (double) sch.mblk_prefix[n_experts] * BMR;
    printf("  NST=%d BMR=%-4d tiles=%-5d mtiles=%-5d | %8.3f ms | %6.1f TFLOP/s | "
           "weight %.1fx | rows computed %.2fx (%.0f/%d) | %5.0f GB/s at min\n",
           NST, BMR, sch.total_tiles, sch.mblk_prefix[n_experts], ms, flop / (ms * 1e9),
           wbytes_tiles / wbytes_min, rows_done / total_rows, rows_done, total_rows,
           wbytes_min / (ms * 1e6));
    (void) active;
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dA); cudaFree(dImg); cudaFree(dS); cudaFree(dC); cudaFree(dB2); cudaFree(dM2);
    return ms;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 1024, K = argc > 2 ? atoi(argv[2]) : 2048;
    const int tokens = argc > 3 ? atoi(argv[3]) : 2048, topk = argc > 4 ? atoi(argv[4]) : 8;
    const int n_experts = argc > 5 ? atoi(argv[5]) : 128;
    // Pick ONE config so a profiler capture has one kernel in it. Without this the sweep launches five.
    const int selBMR = argc > 6 ? atoi(argv[6]) : 0, selNST = argc > 7 ? atoi(argv[7]) : 4;
    const int total_rows = tokens * topk;
    printf("Q4_K MoE on AIU: N=%d K=%d tokens=%d topk=%d experts=%d -> rows=%d (avg %.0f/expert)\n",
           N, K, tokens, topk, n_experts, total_rows, (double) total_rows / n_experts);

    std::mt19937 rng(7);
    std::vector<int> re(total_rows);
    for (auto& e : re) e = (int) (rng() % n_experts);
    std::sort(re.begin(), re.end());

    // BMR sweep is the point: bigger BMR -> fewer m-tiles -> the weight is re-streamed less. That is the
    // fp16 kernel's central finding and the reason Marlin's 32-row tile was the wrong shape here.
    if (selBMR) {   // single config: for acu, or for A/B-ing one knob without the other four in the way
        if (selNST == 2) { if (selBMR == 64) bench<2,64>(n_experts,N,K,total_rows,re,20); else if (selBMR == 256) bench<2,256>(n_experts,N,K,total_rows,re,20); else bench<2,128>(n_experts,N,K,total_rows,re,20); }
        else if (selNST == 6) { if (selBMR == 64) bench<6,64>(n_experts,N,K,total_rows,re,20); else if (selBMR == 256) bench<6,256>(n_experts,N,K,total_rows,re,20); else bench<6,128>(n_experts,N,K,total_rows,re,20); }
        else { if (selBMR == 64) bench<4,64>(n_experts,N,K,total_rows,re,20); else if (selBMR == 256) bench<4,256>(n_experts,N,K,total_rows,re,20); else bench<4,128>(n_experts,N,K,total_rows,re,20); }
        return 0;
    }
    printf("\n  BMR sweep (weight re-streaming is what this moves):\n");
    // BMR=32 is not expressible: 4 warps in m x 16 rows each makes 64 the minimum (static_assert in the
    // header). So the sweep starts at 64 -- which is also where Marlin's 32-row tile could never reach.
    bench<4, 64> (n_experts, N, K, total_rows, re, 20);
    bench<4, 128>(n_experts, N, K, total_rows, re, 20);
    bench<4, 256>(n_experts, N, K, total_rows, re, 20);
    printf("\n  NST sweep at BMR=128 (AIU bulk load is one descriptor, so depth is nearly free):\n");
    bench<2, 128>(n_experts, N, K, total_rows, re, 20);
    bench<6, 128>(n_experts, N, K, total_rows, re, 20);
    return 0;
}
