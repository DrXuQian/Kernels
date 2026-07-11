// W4A16 decode GEMV on an N-MAJOR INT4 layout -- the v8 structure from the bf16 GEMV sweep (ppu_tests/gemv_ppu.cu),
// which reached ~82% of HBM by copying cuBLAS: lean register LDG, one output row per block, NO shared / cp.async /
// split-K. In that sweep AIU (swzl and bulk-linear), cp.async pipelines, split-K, multi-row and prefetch rings ALL LOST
// to the dumbest kernel. Do not re-run those.
//
// WHY A SECOND WEIGHT LAYOUT. gemv_w4a16_ppu.cu reads Marlin's B in place, and that layout is the root cause of decode
// stalling at 35-54% of HBM. Every constraint traces back to one line of the packing:
//
//   ONE int4 packs 64 COLUMNS x 4 k  ->  a block must consume all 64 columns or waste 12 of every 16 bytes
//                                    ->  grid.x = N/64 = 64 blocks -- cannot even fill 72 CUs
//                                    ->  split-K is FORCED to get parallelism -- and split-K is a known loser
//   walking K for fixed columns jumps (N/2) int4 = 8N bytes = 32 KB at N=4096
//                                    ->  DRAM row buffer is 1-2 KB, so nearly every 512 B read opens a new row
//
// N-major fixes both. Column n's K nibbles are contiguous: 2048 B at K=4096. A block owns COLS columns, and with
// T=128 threads x 16 B the block sweeps one column's entire 2048 B in a single step, then the next column, which sits
// immediately after it -> COLS*2048 B of fully sequential DRAM. grid = N/COLS.
//
// A-REUSE IS WHY COLS > 1. INT4 packs 4x denser than the fp16 activation, so per byte of B a GEMV needs 4 bytes of A.
// With one column per block that would make A 4x the weight traffic; the bf16 sweep found the activation, not the
// weight unpack, was what capped it. COLS columns share one A load, cutting A to 1/COLS.
//
// PACKING. One uint32 holds 8 CONSECUTIVE k of one column, in the nibble order Marlin's dequant emits, so we reuse its
// lop3 verbatim -- 4 instructions for 4 halves:
//     dequant(q)      -> {k+0, k+1}, {k+2, k+3}          nibbles at shifts 0,16 and 4,20
//     dequant(q >> 8) -> {k+4, k+5}, {k+6, k+7}          nibbles at shifts 8,24 and 12,28
// which is exactly A[k..k+7] read as one int4 (4 half2). No shuffles, no repack.
//
// Portable (only lop3, which is standard PTX), so correctness is checkable off-box. Build: nvcc -O3 -std=c++17.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_){printf("CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__); exit(1);} } while(0)

#ifndef NM_COLS
#define NM_COLS 8        // output columns per block. grid = N/COLS. Sweeps against A-reuse vs block count.
#endif
#ifndef NM_THREADS
#define NM_THREADS 128   // v8's width
#endif
// Chunks issued before any is consumed. DEFAULT 1, and read this before raising it:
//
// The staging arrays are q4[COLS][U] + av[4][U] int4, i.e. 4*(COLS+4)*U registers JUST to hold loads in flight.
// COLS=8 U=2 measured 167 registers (COLS=16 U=2: 254, at the cap). On the PPU that caps occupancy at
// 131072/(167*32) = 24.5 warps/CU -- BELOW the 28.4 the grid would otherwise give. The unroll pays for itself only
// if it is actually used.
//
// And it often is NOT: a thread walks nchunk = K/32 chunks with stride T, so it gets ceil(nchunk/T) of them.
// At K=4096, T=128 that is exactly ONE -- every u > 0 is skipped and the registers are pure waste. That is the
// configuration this kernel was first benchmarked in, and it lost to the ktile-major kernel because of it.
// Raise U only when K/32 > NM_THREADS.
#ifndef NM_UNROLL
#define NM_UNROLL 1
#endif

struct FragB { half2 v[2]; };
template <int lut> __device__ inline int lop3(int a, int b, int c) {
    int r; asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(lut)); return r;
}
// Marlin's dequant, verbatim. DO NOT fold a scale into these constants: lo = 1024+q and SUB = 1032 are both exact in
// fp16 so `lo - SUB` is exact, and folding s in rounds two ~1000-magnitude products before subtracting them to get a
// ~10-magnitude result. Measured median relative error 4.3e-2, max 5.0e-1. (See marlin_classic_ppu.cuh.)
__device__ __forceinline__ FragB dequant(int q) {
    const int LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400;
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX), hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    const int SUB = 0x64086408, MUL = 0x2c002c00, ADD = 0xd480d480;
    FragB b;
    b.v[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
    b.v[1] = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));
    return b;
}

__device__ __forceinline__ float warp_reduce(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Bn: [N][K/32] int4  -- column n's K nibbles, k contiguous. One int4 = 4 uint32 = 32 consecutive k.
// s:  [N][K/gs] half  -- N-MAJOR too, so a lane walking groups reads contiguous halves.
// One block owns COLS columns; T threads stride the K/32 int4 of each. No split-K, no shared staging.
template <int COLS, int T, int U>
__global__ __launch_bounds__(T) void gemv_nmajor(
        const int4* __restrict__ Bn, const half* __restrict__ A, const half* __restrict__ s,
        half* __restrict__ C, int N, int K, int chunks_per_group) {
    const int col0 = blockIdx.x * COLS;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int nchunk = K / 32;                       // int4 per column
    const int ngroup = K / (32 * chunks_per_group);  // scale rows per column

    const int4* wp[COLS];
    #pragma unroll
    for (int c = 0; c < COLS; c++) wp[c] = Bn + (long long)(col0 + c) * nchunk;

    float facc[COLS];
    #pragma unroll
    for (int c = 0; c < COLS; c++) facc[c] = 0.f;

    for (int i0 = tid; i0 < nchunk; i0 += T * U) {
        int4 q4[COLS][U];
        int4 av[4][U];                                // A for the 32 k of each chunk: 4 int4 = 32 halves
        half  sg[COLS][U];                            // this chunk's scale, per column
        #pragma unroll
        for (int u = 0; u < U; u++) {
            const int i = i0 + u * T;
            if (i >= nchunk) continue;
            #pragma unroll
            for (int c = 0; c < COLS; c++) q4[c][u] = wp[c][i];                     // 512 B contiguous per warp
            #pragma unroll
            for (int e = 0; e < 4; e++)
                av[e][u] = *reinterpret_cast<const int4*>(&A[i * 32 + e * 8]);      // shared by all COLS
            const int g = i / chunks_per_group;                                     // whole chunk is in one group
            #pragma unroll
            for (int c = 0; c < COLS; c++) sg[c][u] = s[(long long)(col0 + c) * ngroup + g];
        }
        #pragma unroll
        for (int u = 0; u < U; u++) {
            const int i = i0 + u * T;
            if (i >= nchunk) continue;
            #pragma unroll
            for (int c = 0; c < COLS; c++) {
                const int* q = reinterpret_cast<const int*>(&q4[c][u]);
                half2 acc = __float2half2_rn(0.f);
                #pragma unroll
                for (int e = 0; e < 4; e++) {                                       // 4 uint32 = 4 x 8 k
                    const FragB b0 = dequant(q[e]);        // {k+0,k+1}, {k+2,k+3}
                    const FragB b1 = dequant(q[e] >> 8);   // {k+4,k+5}, {k+6,k+7}
                    const half2* a = reinterpret_cast<const half2*>(&av[e][u]);
                    acc = __hfma2(b0.v[0], a[0], acc);
                    acc = __hfma2(b0.v[1], a[1], acc);
                    acc = __hfma2(b1.v[0], a[2], acc);
                    acc = __hfma2(b1.v[1], a[3], acc);
                }
                const float2 f = __half22float2(acc);      // one chunk = one group -> scale and fold to fp32 here
                facc[c] += (f.x + f.y) * __half2float(sg[c][u]);
            }
        }
    }

    __shared__ float sm[COLS][T / 32];
    #pragma unroll
    for (int c = 0; c < COLS; c++) { const float v = warp_reduce(facc[c]); if (lane == 0) sm[c][wid] = v; }
    __syncthreads();
    if (wid == 0) {
        #pragma unroll
        for (int c = 0; c < COLS; c++) {
            float v = (lane < (T >> 5)) ? sm[c][lane] : 0.f;
            v = warp_reduce(v);
            if (lane == 0) C[col0 + c] = __float2half(v);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------

// Host packing. uint32 (n, chunk c, sub e) holds k = c*32 + e*8 .. +7, in the nibble order dequant emits.
static void pack_nmajor(const std::vector<int>& Bdeq, std::vector<int>& hB, int N, int K) {
    auto q4 = [&](int n, int k) { return (Bdeq[(size_t) n * K + k] + 8) & 0xf; };   // to unsigned [0,15]
    for (int n = 0; n < N; n++)
        for (int k0 = 0; k0 < K; k0 += 8) {
            int q = 0;
            q |= q4(n, k0 + 0) <<  0;  q |= q4(n, k0 + 1) << 16;    // dequant(q).v[0]
            q |= q4(n, k0 + 2) <<  4;  q |= q4(n, k0 + 3) << 20;    // dequant(q).v[1]
            q |= q4(n, k0 + 4) <<  8;  q |= q4(n, k0 + 5) << 24;    // dequant(q>>8).v[0]
            q |= q4(n, k0 + 6) << 12;  q |= q4(n, k0 + 7) << 28;    // dequant(q>>8).v[1]
            hB[(size_t) n * (K / 8) + k0 / 8] = q;
        }
}

static double get_peak() { const char* e = getenv("NM_HBM_GBS"); return e ? atof(e) : 2700.0; }
// The weights are 8-29 MB and the PPU's LLC is 64-128 MB, so a loop over ONE buffer measures cache, not HBM. Rotate.
static int get_rot(size_t b) { const char* e = getenv("NM_WORKSET_MB"); if (!e) return 1;
    size_t w = (size_t) atoi(e) << 20; int r = (int)((w + b - 1) / b); return r < 1 ? 1 : r; }

static int bench(int N, int K, int iters) {
    const int gs = getenv("NM_GROUPSIZE") ? atoi(getenv("NM_GROUPSIZE")) : 128;
    if (N % NM_COLS || K % 256 || (gs != -1 && (gs % 32 || K % gs))) {
        printf("  N=%d K=%d gs=%d unsupported (N%%%d, K%%256, gs%%32)\n", N, K, gs, NM_COLS); return -1; }
    const int gsz = (gs == -1) ? K : gs;
    const int chunks_per_group = gsz / 32;            // int4 chunks per scale group
    const int GROUPS = K / gsz;

    std::vector<half> hA(K), hC(N);
    std::vector<half> hS((size_t) N * GROUPS);        // N-MAJOR: s[n][g]
    std::vector<int>  Bdeq((size_t) N * K), hB((size_t) N * (K / 8));
    srand(1234);
    for (auto& x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto& x : Bdeq) x = (rand() & 0xf) - 8;
    for (auto& x : hS) x = __float2half(0.5f + (rand() % 100) / 100.0f);   // non-trivial: s=1 would hide a dropped scale
    pack_nmajor(Bdeq, hB, N, K);

    std::vector<double> ref(N);
    for (int n = 0; n < N; n++) {
        double t = 0;
        for (int g = 0; g < GROUPS; g++) {
            double a = 0;
            for (int k = g * gsz; k < (g + 1) * gsz; k++) a += (double) __half2float(hA[k]) * Bdeq[(size_t) n * K + k];
            t += a * __half2float(hS[(size_t) n * GROUPS + g]);
        }
        ref[n] = t;
    }

    const size_t b_bytes = hB.size() * 4;
    const int rot = get_rot(b_bytes);
    int4 *dB; half *dA, *dS, *dC;
    CUDA_CHECK(cudaMalloc(&dB, b_bytes * rot));
    CUDA_CHECK(cudaMalloc(&dA, K * 2)); CUDA_CHECK(cudaMalloc(&dS, hS.size() * 2)); CUDA_CHECK(cudaMalloc(&dC, N * 2));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_bytes, cudaMemcpyHostToDevice));
    for (int r = 1; r < rot; r++) CUDA_CHECK(cudaMemcpy((char*) dB + (size_t) r * b_bytes, dB, b_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice));

    auto Brot = [&](int i) { return (const int4*) ((char*) dB + (size_t)(i % rot) * b_bytes); };
    auto run = [&](int i) { gemv_nmajor<NM_COLS, NM_THREADS, NM_UNROLL><<<N / NM_COLS, NM_THREADS>>>(
                                Brot(i), dA, dS, dC, N, K, chunks_per_group); };

    run(0); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * 2, cudaMemcpyDeviceToHost));
    double ma = 0, rm = 0;
    for (int n = 0; n < N; n++) { ma = fmax(ma, fabs(__half2float(hC[n]) - ref[n])); rm = fmax(rm, fabs(ref[n])); }
    const double rel = ma / (rm + 1e-9);
    printf("  correctness N=%d K=%d gs=%d: max_abs=%.4e rel=%.2e -> %s\n", N, K, gs, ma, rel, rel < 3e-2 ? "MATCH" : "MISMATCH");
    if (rel >= 3e-2) return -1;

    for (int i = 0; i < 10; i++) run(i);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) run(i);
    cudaEventRecord(b); CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;

    const double bytes = (double) N * K / 2 + (double) K * 2 + (double) N * 2 + (double) N * GROUPS * 2;
    const double gbs = bytes / (ms * 1e6);
    printf("  M=1     N=%-5d K=%-5d : %8.2f us  %7.0f GB/s  %5.1f%% HBM   (COLS=%d T=%d U=%d, %d blocks, wset=%zu MB)\n",
           N, K, ms * 1e3, gbs, 100.0 * gbs / get_peak(), NM_COLS, NM_THREADS, NM_UNROLL, N / NM_COLS,
           (b_bytes * rot) >> 20);
    cudaFree(dB); cudaFree(dA); cudaFree(dS); cudaFree(dC);
    return 0;
}

int main(int argc, char** argv) {
    printf("W4A16 GEMV, N-MAJOR layout (v8 structure: lean LDG, no shared/cp.async/split-K):\n");
    if (!getenv("NM_WORKSET_MB"))
        printf("  NOTE: weights are LLC-resident. Set NM_WORKSET_MB=256 for cold-weight, HBM-bound timings.\n");
    if (argc >= 3) return bench(atoi(argv[1]), atoi(argv[2]), argc >= 4 ? atoi(argv[3]) : 200) < 0;
    int sh[][2] = { {4096, 4096}, {14336, 4096}, {4096, 14336} };
    for (auto& s : sh) if (bench(s[0], s[1], 200) < 0) return 1;
    return 0;
}
