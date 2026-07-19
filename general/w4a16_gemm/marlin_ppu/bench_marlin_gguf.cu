// Perf bench for the GGUF W4A16 Marlin port (marlin_gguf_ppu.cuh). Perf is data-independent -> random buffers;
// correctness lives in test_marlin_gguf.
//
// THE QUESTION THIS EXISTS TO ANSWER. We moved GGUF off the int8 MMQ path (~10% MFU on PPU) onto the fp16 Marlin
// structure because marlin_classic reaches 47% of the 500 TFLOP/s fp16 peak. But that 47% was measured at gs=128,
// and GGUF Q4_K is gs=32 -- FOUR TIMES the scale-fetch rate, four times the group-rows staged per pipeline stage,
// and a group index that now has to be recomputed per (k, warp). So the number that matters is not the absolute
// TFLOP/s, it is the gs=32 vs gs=128 DELTA on identical shapes. Hence every shape is run at both, back to back.
//
// AFFINE (Q4_K's min term) is swept too. The claim in the header is that it is free -- hmul2 -> hfma2, same
// instruction count -- but it also doubles the staged scale bytes (mins ride alongside s in shared) and adds a
// second cp.async stream. "Free in the ALU" and "free end-to-end" are different claims; this measures the second.
//
// build (box):  make bench_marlin_gguf
// run:          ./bench_marlin_gguf                 # the sweep: {gs=128, gs=32} x {symmetric, affine}
//               ./bench_marlin_gguf M N K [iters]
// env:          MARLIN_MAX_PAR / MARLIN_HBM_GBS / MARLIN_WORKSET_MB -- as in bench_marlin
//               MARLIN_PEAK_TFLOPS=<n>   fp16 tensor-core peak for the %MFU column (PPU: 500, measured by mma_peak)
//               MARLIN_NOSPLITK=1 / MARLIN_SPLITK=1 / MARLIN_BLOCKS=<n>  -- read by marlin_cuda itself, and via
//                 getenv into a `static`, so they are latched ONCE per process: sweeping them needs separate runs.
//
// DECODE (M=1) IS AN OCCUPANCY PROBLEM BEFORE IT IS A BANDWIDTH ONE. marlin_cuda picks thread_n=128 for M<=16, so
// there are only N/128 output tiles -- 32 at N=4096 on a 72-CU part. That is printed per line. Marlin is a prefill
// kernel; the decode answer on this part is the dedicated GEMV (82% HBM, reads Marlin's B format in place), not a
// tuned Marlin. Compare against marlin_classic at the same gs before reading any decode number as a regression:
//   MARLIN_GROUPSIZE=128 ./bench_marlin 1 4096 4096
#include "marlin_gguf_ppu.cuh"
#include <cstdio>
#include <cstdlib>

static int    get_max_par()  { const char* e = getenv("MARLIN_MAX_PAR");     int v = e ? atoi(e) : 128; return v > 0 ? v : 128; }
static double get_hbm_gbs()  { const char* e = getenv("MARLIN_HBM_GBS");  double v = e ? atof(e) : 2700.0; return v > 0 ? v : 2700.0; }
static double get_peak()     { const char* e = getenv("MARLIN_PEAK_TFLOPS"); double v = e ? atof(e) : 500.0; return v > 0 ? v : 500.0; }
static int    get_rot(size_t b) { const char* e = getenv("MARLIN_WORKSET_MB"); if (!e) return 1;
                                  size_t w = (size_t) atoi(e) << 20; int r = (int)((w + b - 1) / b); return r < 1 ? 1 : r; }

// Compulsory traffic. The scales are the term that gs moves: one fp16 row per group per column, so gs=32 carries
// 4x the scale bytes of gs=128, and AFFINE doubles that again (s and m' both). At prefill sizes this is small next
// to B; the gs cost, if any, should show up as lost TFLOP/s, not as extra GB/s -- which is itself worth checking.
static double compulsory_bytes(int M, int N, int K, int gs, bool affine) {
    const double s_rows = (gs == -1) ? 1.0 : (double)(K / gs);
    return (double) N * K / 2 + (double) M * K * 2 + (double) M * N * 2
         + s_rows * N * 2 * (affine ? 2 : 1);
}

static double bench_one(int M, int N, int K, int gs, bool affine, int iters) {
    const int max_par = get_max_par();
    const size_t s_rows = (gs == -1) ? 1 : (size_t)(K / gs);
    const size_t A_i4 = (size_t) M * K / 8, B_i4 = (size_t)(K / 16) * (N * 16 / 32), C_i4 = (size_t) M * N / 8;
    const size_t S_h = s_rows * (size_t) N, WS = (size_t)(N / 128 + 1) * max_par, b_bytes = B_i4 * 16;
    const int rot = get_rot(b_bytes);
    int4 *dA, *dB, *dC, *dS, *dMn; int* dWS;
    if (cudaMalloc(&dA, A_i4 * 16) || cudaMalloc(&dB, b_bytes * rot) || cudaMalloc(&dC, C_i4 * 16) ||
        cudaMalloc(&dS, (S_h / 8 + 1) * 16) || cudaMalloc(&dMn, (S_h / 8 + 1) * 16) || cudaMalloc(&dWS, WS * 4)) {
        printf("  alloc fail\n"); return -1;
    }
    cudaMemset(dA, 1, A_i4 * 16); cudaMemset(dB, 1, b_bytes * rot);
    cudaMemset(dS, 0x3c, (S_h / 8) * 16); cudaMemset(dMn, 0, (S_h / 8) * 16);   // s ~= 1.0 (0x3c00), m' = 0
    cudaMemset(dWS, 0, WS * 4);

    auto Brot  = [&](int i) { return (const int4*)((char*) dB + (size_t)(i % rot) * b_bytes); };
    auto run_i = [&](int i) { return marlin_gguf_ppu::marlin_cuda(dA, Brot(i), dC, dS, affine ? dMn : nullptr,
                                        M, N, K, dWS, gs, 0, 0, -1, -1, -1, max_par); };
    auto cleanup = [&] { cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dS); cudaFree(dMn); cudaFree(dWS); };

    int ret = run_i(0); cudaError_t e = cudaDeviceSynchronize();
    if (ret || e) { printf("  gs=%-3d %-4s M=%-5d N=%-5d K=%-5d : ret=%d err=%s\n",
                           gs, affine ? "aff" : "sym", M, N, K, ret, cudaGetErrorString(e)); cleanup(); return -1; }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) run_i(i);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) { cudaMemset(dWS, 0, WS * 4); run_i(i); }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    double gbs = compulsory_bytes(M, N, K, gs, affine) / (ms * 1e6);
    // n_tiles vs CUs is the number to read at M=1. marlin_cuda picks thread_n=128 for M<=16, so decode has only
    // N/128 output tiles -- 32 at N=4096, against 72 CUs. More than half the machine is idle before a single byte
    // moves, which caps %HBM no matter how good the inner loop is. Printed so a decode line cannot be misread as
    // a bandwidth result when it is really an occupancy one.
    int cus = 0; cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, 0);
    const int n_tiles = N / (M <= 16 ? 128 : 256);
    printf("  gs=%-3d %-4s M=%-5d N=%-5d K=%-5d : %8.2f us  %7.1f TFLOP/s  %5.1f%% MFU  %6.0f GB/s %5.1f%% HBM"
           "   [%d tiles / %d CU]\n",
           gs, affine ? "aff" : "sym", M, N, K, ms * 1e3, tflops, 100.0 * tflops / get_peak(),
           gbs, 100.0 * gbs / get_hbm_gbs(), n_tiles, cus);
    cudaEventDestroy(a); cudaEventDestroy(b); cleanup();
    return tflops;
}

int main(int argc, char** argv) {
    printf("GGUF W4A16 Marlin on PPU (fp16 peak %.0f TFLOP/s)\n", get_peak());
    if (argc >= 4) {
        int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]), it = (argc >= 5) ? atoi(argv[4]) : 50;
        for (int gs : {128, 32}) for (bool aff : {false, true}) bench_one(M, N, K, gs, aff, it);
        return 0;
    }
    int shapes[][3] = {
        // PREFILL ONLY. The M=1 rows that used to be here measured Marlin at batch 1, which is not the decode
        // path -- decode goes through gemv_w4a16_ppu.cu (GGUF Q4_K: build gemv_w4a16_t32). Reading Marlin's
        // 16.8% HBM at M=1 against the GEMV's ~52% compares two different kernels, so the rows are gone.
        {2048, 4096, 4096}, {2048, 14336, 4096}, {2048, 4096, 14336},   // compute bound -- read %MFU
    };
    // Paired so the gs=32 cost is a difference between adjacent lines on one shape, not a comparison against a
    // number from another binary on another day.
    for (auto& s : shapes) {
        for (int gs : {128, 32}) for (bool aff : {false, true}) bench_one(s[0], s[1], s[2], gs, aff, 50);
        printf("\n");
    }
    return 0;
}
