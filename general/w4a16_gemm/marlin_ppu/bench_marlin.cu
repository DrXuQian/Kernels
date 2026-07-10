// Perf bench for the classic Marlin PPU port. Perf is data-independent -> random buffers (correctness validated
// separately in test_marlin_classic_num/splitk). Reports us + TFLOP/s.
// Known perf caveats in this port: global_reduce still uses DIRECT global stores (split-K only; the no-split-K
// heuristic keeps it off the prefill path). write_result is smem-staged/coalesced; A smem->reg is ldmatrix.x4.
// The main cp.async+mma loop is the NVIDIA structure.
//
// Reported time is END-TO-END per marlin_cuda() call, which issues ceil(tot_m_blocks/MB/par) kernels back-to-back --
// a profiler shows ONE of them (4096^3 with max_par=16: 8 launches, so asys ~67us vs the 517us reported here).
//
// build (box):  make bench_marlin   (nvcc -O3, no -arch)
// run:          ./bench_marlin                 # the default W4A16 shape sweep (decode + prefill)
//               ./bench_marlin M N K [iters]   # one shape, e.g.  ./bench_marlin 2048 4096 14336
// env:          MARLIN_NOSPLITK=1 / MARLIN_SPLITK=1   force the split-K choice (default auto: no-split if tiles>=SMs)
//               MARLIN_MAX_PAR=<n>                    m-tiles per launch (default 128); scales the locks workspace
//               MARLIN_BLOCKS=<n>                     override gridDim outright (decode: only N/128 output tiles exist)
//               MARLIN_HBM_GBS=<n>                    HBM peak for the %HBM column (default 2700)
//
// Shape constraints (else ret=1): N % thread_n == 0 and K % thread_k == 0, where the config is picked by M:
//   M <= 16  -> thread_n=128, thread_k=128      M > 16  -> thread_n=256, thread_k=64

#include "marlin_classic_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

// max_par caps how many m-tiles ONE launch covers: a launch does 16*MARLIN_MAX_MB*par rows, so marlin_cuda issues
// ceil(tot_m_blocks/MB/par) kernels back-to-back. NVIDIA's default of 16 was tuned for a 108-SM A100; on a 72-CU PPU it
// pinned the grid at cols*16 = 256 CTA -> 4 waves whose last fills 40/72 CUs, a tail paid once per launch.
//
// Raising it wins TWICE, and the two effects are separable (measured, 4096^3):
//   tail amortization : 8 launches @ 88.9% tail eff -> 1 @ 98.1%   ~= 48.5 us
//   launch latency    : ~2.1 us x 7 launches saved                  ~= 15   us
//   16 -> 266.3   32 -> 272.4   64 -> 293.9   128 -> 303.6 TFLOP/s (total -63 us, predicted -63.5)
// The 32 case isolates the second effect: identical tail eff (88.9%), 4 fewer launches, +2.3%.
// 2048x4096x14336 saturates at par = tot_m_blocks/MB = 64 (294.9 -> 312.5); 128 is then a no-op (noise floor ~0.6%).
//
// The locks workspace scales with max_par -- see marlin_cuda's WORKSPACE CONTRACT. Keep them in sync or it writes OOB.
static int get_max_par() { const char* e = getenv("MARLIN_MAX_PAR"); int v = e ? atoi(e) : 128; return v > 0 ? v : 128; }

// PPU HBM peak, GB/s. Override with MARLIN_HBM_GBS.
static double get_hbm_gbs() { const char* e = getenv("MARLIN_HBM_GBS"); double v = e ? atof(e) : 2700.0; return v > 0 ? v : 2700.0; }

// COMPULSORY traffic: every byte the problem must touch at least once. Excludes split-K's global_reduce, which
// re-reads and re-writes C once per extra slice -- so on a split-K path the reported %HBM UNDERSTATES real traffic.
//
// Which number to read: prefill (M>=2048) is compute bound, look at TFLOP/s. Decode (M=1) is weight-bandwidth bound --
// 2*M*N*K FLOPs over a stream of N*K/2 INT4 weight bytes gives an arithmetic intensity of ~4*M flop/byte, so at M=1 the
// tensor cores are idle by construction and TFLOP/s says nothing. Read %HBM. A tuned bf16 GEMV on this part hits ~82%.
static double compulsory_bytes(int M, int N, int K) {
    return (double) N * K / 2      // B: INT4 weights -- dominates at M=1
         + (double) M * K * 2      // A: fp16 activations
         + (double) M * N * 2      // C: fp16 output
         + (double) N * 2;         // scales: fp16, one per column
}

// -DMARLIN_BENCH_NOREDUCE makes each split-K slice write its OWN C plane (slot = slice_idx). Results are wrong by
// design -- no reduce runs -- but the timing models a splitk_parallel mainloop without cache-line contention. Slots
// needed = max slice_count = ceildiv(k_tiles, iters); worst case in our sweeps is 112 (N=4096 K=14336, blocks=3584).
// 128 slots costs 1 MB and cannot be overrun by any grid we probe. Too few slots = out-of-bounds writes, not a bad number.
#ifdef MARLIN_BENCH_NOREDUCE
static const int C_SLOTS = 128;
#else
static const int C_SLOTS = 1;
#endif

static double bench_one(int M, int N, int K, int iters) {
    const int max_par = get_max_par();
    const size_t A_i4 = (size_t) M * K / 8, B_i4 = (size_t) (K / 16) * (N * 16 / 32), C_i4 = (size_t) M * N / 8, S_h = (size_t) N;
    // locks: one per (n-tile, par). Worst case is thread_n=128 (the M<=16 configs) -> N/128 n-tiles, NOT N/256.
    const size_t WS = (size_t) (N / 128 + 1) * max_par;
    int4 *dA, *dB, *dC, *dS; int * dWS;
    if (cudaMalloc(&dA, A_i4 * 16) || cudaMalloc(&dB, B_i4 * 16) || cudaMalloc(&dC, C_i4 * 16 * C_SLOTS) ||
        cudaMalloc(&dS, (S_h / 8 + 1) * 16) || cudaMalloc(&dWS, WS * 4)) { printf("  alloc fail\n"); return -1; }
    cudaMemset(dA, 1, A_i4 * 16); cudaMemset(dB, 1, B_i4 * 16); cudaMemset(dS, 0x3c, (S_h / 8) * 16);   // s~=1.0 (0x3c00)
    cudaMemset(dWS, 0, WS * 4);

    auto run = [&] { return marlin_classic_ppu::marlin_cuda(dA, dB, dC, dS, M, N, K, dWS, -1,
                                                            /*dev=*/0, /*stream=*/0, /*thread_k=*/-1, /*thread_n=*/-1,
                                                            /*sms=*/-1, max_par); };
    auto cleanup = [&] { cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dS); cudaFree(dWS); };

    int ret = run(); cudaError_t e = cudaDeviceSynchronize();
    if (ret || e) {
        printf("  M=%-5d N=%-5d K=%-5d : ret=%d err=%s  (unsupported shape/config)\n", M, N, K, ret, cudaGetErrorString(e));
        cleanup(); return -1;
    }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) run();
    cudaDeviceSynchronize();
    cudaMemset(dWS, 0, WS * 4);              // locks must be 0 before each launch set
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) { cudaMemset(dWS, 0, WS * 4); run(); }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    double us = ms * 1e3, tflops = 2.0 * M * N * K / (ms * 1e9);
    double gbs = compulsory_bytes(M, N, K) / (ms * 1e6);          // bytes / (ms*1e-3) / 1e9
    printf("  M=%-5d N=%-5d K=%-5d : %8.2f us  %7.1f TFLOP/s  %7.0f GB/s  %5.1f%% HBM\n",
           M, N, K, us, tflops, gbs, 100.0 * gbs / get_hbm_gbs());
    cudaEventDestroy(a); cudaEventDestroy(b);
    cleanup();
    return us;
}

int main(int argc, char** argv) {
    if (argc >= 4) {
        int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
        int iters = (argc >= 5) ? atoi(argv[4]) : 50;
        printf("classic Marlin PPU perf (W4A16, random data), %d iters:\n", iters);
        return bench_one(M, N, K, iters) < 0 ? 1 : 0;
    }
    printf("classic Marlin PPU perf (W4A16, random data):\n");
    int shapes[][3] = {
        {1,    4096, 4096}, {1,    14336, 4096}, {1,    4096, 14336},   // batch=1 (decode) -- latency / weight-BW bound
        {2048, 4096, 4096}, {2048, 14336, 4096}, {2048, 4096, 14336},   // batch=2048 (prefill) -- compute bound
    };
    for (auto & s : shapes) bench_one(s[0], s[1], s[2], 50);
    return 0;
}
