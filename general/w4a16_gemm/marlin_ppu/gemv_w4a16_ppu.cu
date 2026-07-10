// W4A16 GEMV (M=1 decode) reading Marlin's INT4 B format in place.
//
// WHY THIS EXISTS. Marlin is a GEMM: its tile is 16x128 and decode uses 1 of those 16 rows. Worse, its (tile, k)
// decomposition obeys blocks * iters ~= n_tiles * k_tiles, so grid size and pipeline depth trade against each other and
// decode tops out around 37% of HBM no matter how the knobs are set (measured; see marlin_classic_ppu.cuh). A GEMV has
// no such conservation law: each block owns a slice of N columns and walks all of K, so `iters` is long (pipeline is
// fine) and `blocks` is large (warps are plentiful), independently.
//
// B LAYOUT (decoded from Marlin's b_gl_rd + the packq in test_marlin_classic_num.cu). Crucially it is INDEPENDENT of
// thread_n_blocks -- folding slice_col*b_sh_stride + warp_n*32 into (nblock/4)*32 makes the config drop out -- so
// prefill and decode share ONE packed weight tensor.
//
//   int4 index   idx = (N/2) * ktile  +  (nblock/4) * 32  +  lane          ktile = k/16, nblock = n/16
//   the j-th uint32 of that int4 (j = nblock % 4) holds B[n][k] at
//       lane  = 4*(n % 8) + (k % 8)/2
//       shift = 4*(k%16 >= 8) + 8*(n%16 >= 8) + 16*(k % 2)
//       value = ((q >> shift) & 0xf) - 8
//
// So ONE int4 covers 4 consecutive nblocks = 64 n-columns x 4 k-values. A block must consume all 64 columns or it
// wastes 12 of every 16 bytes -- that pins grid.x = N/64.
//
// Portable: no ppu.* asm, so this builds and runs on stock CUDA too (that is how correctness was checked off-box).
//
// build:  nvcc -O3 -std=c++17 -o gemv_w4a16 gemv_w4a16_ppu.cu     (on the box: no -arch)
// run:    ./gemv_w4a16                # correctness + the decode shape sweep
//         ./gemv_w4a16 N K [iters]

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while(0)

static const int N_PER_BLOCK = 64;   // one int4 spans exactly this many columns
static const int GEMV_THREADS = 128; // 4 warps; each warp takes a different ktile

// How many independent int4 loads a thread issues before consuming any. Costs GEMV_UNROLL*4 registers.
#ifndef GEMV_UNROLL
#define GEMV_UNROLL 4
#endif

// Marlin's dequant, verbatim. 4 instructions (2 lop3 + hsub2 + hfma2) yield 4 halves; the naive
// shift/and/int-sub/int-to-float path costs ~5 instructions PER WEIGHT.
//
// Its output order is exactly what a GEMV over this packing wants -- because both come from the same packq:
//   dequant(q)      -> [0] = {B[n][kb],   B[n][kb+1]}    [1] = {B[n][kb+8],   B[n][kb+9]}
//   dequant(q >> 8) -> [0] = {B[n+8][kb], B[n+8][kb+1]}  [1] = {B[n+8][kb+8], B[n+8][kb+9]}
// (the >> 8 is arithmetic, but every mask below reads bits at or under position 23, and sign fills only 31:24)
struct FragB { half2 v[2]; };
template <int lut> __device__ inline int lop3(int a, int b, int c) {
  int res; asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)); return res;
}
__device__ __forceinline__ FragB dequant(int q) {
  const int LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX), hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64086408, MUL = 0x2c002c00, ADD = 0xd480d480;
  FragB b;
  b.v[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  b.v[1] = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));
  return b;
}

// Each thread consumes one int4 per step: 4 nblocks (j) x {lo,hi} n-halves = 8 columns, x 4 k-values.
// acc[j][hi] accumulates column n = (nb_group*4 + j)*16 + lane/4 + 8*hi.
template <int SPLIT_K>
__global__ void __launch_bounds__(GEMV_THREADS)
gemv_w4a16(const int4* __restrict__ B, const half* __restrict__ A, const half* __restrict__ s,
           float* __restrict__ partial, int N, int K) {
    const int nb_group = blockIdx.x;                 // 4 nblocks = 64 columns
    const int slice    = blockIdx.y;
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;

    const int k_tiles = K / 16;
    const int kt_per_slice = k_tiles / SPLIT_K;
    const int kt_begin = slice * kt_per_slice;
    const int kt_end   = kt_begin + kt_per_slice;

    const int b_stride = N / 2;                      // int4 per ktile
    const int lane_k = (lane % 4) * 2;               // this lane's k phase within the 16-wide ktile

    // half2 accumulators: each slot sums (kt_per_slice/4)*2 products. At SPLIT_K=4, K=4096 that is 32 terms, so the
    // fp16 rounding is ~sqrt(32)*2^-11 = 2.8e-3 relative -- an order worse than an fp32 accumulator, well inside the
    // 3e-2 check. SPLIT_K bounds the chain, which is a second reason not to run this at SPLIT_K=1 on long K.
    half2 hacc[4][2];
    #pragma unroll
    for (int j = 0; j < 4; j++) { hacc[j][0] = __float2half2_rn(0.f); hacc[j][1] = __float2half2_rn(0.f); }

    // ILP. acu on N=4096 K=4096 says Memory Dependency 5.333 dominates (Not Selected 1.275, achieved occupancy 80%):
    // warps are plentiful and idle, all waiting on loads. With one int4 load per iteration, consumed immediately by
    // dequant, the load-to-use distance is zero and a thread never has a second request in flight.
    //
    // So issue GEMV_UNROLL independent loads before touching any of them. This is the documented EXCEPTION to "U-unroll
    // hurts bandwidth kernels": that rule is about grid-stride loops, where unrolling cuts the block count. Here the
    // block count is pinned by N and SPLIT_K, so ILP is the only lever left for hiding latency.
    const int step = GEMV_THREADS / 32;
    const int cnt = (kt_end - kt_begin) / step;        // ktiles this thread owns
    const int main_cnt = (cnt / GEMV_UNROLL) * GEMV_UNROLL;

#ifdef GEMV_BW_ONLY
    // SPEED-ONLY, RESULT IS WRONG. Same addresses, same loop, same reduce -- only the dequant is replaced, by the
    // cheapest arithmetic that still consumes every loaded byte (so the loads cannot be optimized away). It told us the
    // kernel was ALU bound, not bandwidth bound: BW_ONLY ran 9.66 us against the naive kernel's 18.97 us on N=14336
    // K=4096, and the 9.31 us gap scaled with the WEIGHT COUNT (2.21 us on the 3.5x smaller N=4096 shape).
    #define GEMV_BODY(q4_, kt_)                                                                      \
        do {                                                                                         \
            const half2 a01 = *reinterpret_cast<const half2*>(&A[(kt_) * 16 + lane_k]);              \
            const int q[4] = { (q4_).x, (q4_).y, (q4_).z, (q4_).w };                                 \
            _Pragma("unroll")                                                                        \
            for (int j = 0; j < 4; j++)                                                              \
                hacc[j][0] = __hfma2(__halves2half2(__int2half_rn(q[j] & 1),                         \
                                                    __int2half_rn(q[j] >> 31)), a01, hacc[j][0]);    \
        } while (0)
#else
    #define GEMV_BODY(q4_, kt_)                                                                      \
        do {                                                                                         \
            const int kb = (kt_) * 16 + lane_k;        /* even -> both half2 loads are 4B aligned */ \
            const half2 a01 = *reinterpret_cast<const half2*>(&A[kb]);      /* {A[kb],   A[kb+1]} */ \
            const half2 a89 = *reinterpret_cast<const half2*>(&A[kb + 8]);  /* {A[kb+8], A[kb+9]} */ \
            const int q[4] = { (q4_).x, (q4_).y, (q4_).z, (q4_).w };                                 \
            _Pragma("unroll")                                                                        \
            for (int j = 0; j < 4; j++) {                                                            \
                const FragB b0 = dequant(q[j]);        /* column n   */                              \
                const FragB b1 = dequant(q[j] >> 8);   /* column n+8 */                              \
                hacc[j][0] = __hfma2(b0.v[0], a01, hacc[j][0]);   /* k = kb, kb+1   */               \
                hacc[j][0] = __hfma2(b0.v[1], a89, hacc[j][0]);   /* k = kb+8, kb+9 */               \
                hacc[j][1] = __hfma2(b1.v[0], a01, hacc[j][1]);                                      \
                hacc[j][1] = __hfma2(b1.v[1], a89, hacc[j][1]);                                      \
            }                                                                                        \
        } while (0)
#endif

    int i = 0;
    for (; i < main_cnt; i += GEMV_UNROLL) {           // GEMV_UNROLL loads issued back-to-back, none consumed yet
        int4 q4[GEMV_UNROLL];
        #pragma unroll
        for (int u = 0; u < GEMV_UNROLL; u++) {
            const int kt = kt_begin + wid + (i + u) * step;
            q4[u] = B[(long long) b_stride * kt + nb_group * 32 + lane];
        }
        #pragma unroll
        for (int u = 0; u < GEMV_UNROLL; u++) GEMV_BODY(q4[u], kt_begin + wid + (i + u) * step);
    }
    for (; i < cnt; i++) {                             // tail, when cnt is not a multiple of GEMV_UNROLL
        const int kt = kt_begin + wid + i * step;
        GEMV_BODY(B[(long long) b_stride * kt + nb_group * 32 + lane], kt);
    }
    #undef GEMV_BODY

    float acc[4][2];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const float2 f = __half22float2(hacc[j][h]);
            acc[j][h] = f.x + f.y;                     // the two half2 lanes hold different k, so fold them
        }

    // A column n is held by the 4 lanes sharing lane/4 (they differ only in k phase). Fold them.
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            acc[j][h] += __shfl_xor_sync(0xffffffff, acc[j][h], 1);
            acc[j][h] += __shfl_xor_sync(0xffffffff, acc[j][h], 2);
        }
    // now lane%4 == 0 holds the warp's partial for column (nb_group*4+j)*16 + lane/4 + 8*h

    __shared__ float sm[GEMV_THREADS / 32][N_PER_BLOCK];
    if (lane % 4 == 0) {
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++) sm[wid][j * 16 + lane / 4 + 8 * h] = acc[j][h];
    }
    __syncthreads();

    // 64 columns, 128 threads -> first 64 threads each reduce one column across the 4 warps.
    if (threadIdx.x < N_PER_BLOCK) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < GEMV_THREADS / 32; w++) v += sm[w][threadIdx.x];
        const int n = nb_group * N_PER_BLOCK + threadIdx.x;
        partial[(long long) slice * N + n] = v;
    }
}

// out[n] = half(scale[n] * sum_slice partial[slice][n])
__global__ void gemv_reduce(const float* __restrict__ partial, const half* __restrict__ s,
                            half* __restrict__ C, int N, int split_k) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float v = 0.f;
    for (int i = 0; i < split_k; i++) v += partial[(long long) i * N + n];
    C[n] = __float2half(v * __half2float(s[n]));
}

// ---------------------------------------------------------------------------------------------------------------

// SPLIT_K trades block count against per-thread loop length -- the same tension that caps Marlin, except here the two
// are NOT tied to a tile decomposition, so we can sit anywhere on the curve. grid = (N/64) x SPLIT_K.
//
// Measured optima on the box land at the SAME block count, not the same SPLIT_K:
//     N=4096  best sk=16 ->  64 * 16 = 1024 blocks   (6.96 us, == the BW_ONLY bound of 6.93)
//     N=14336 best sk=4  -> 224 *  4 =  896 blocks   (11.40 us)
// i.e. ~1000 blocks, ~13 per CU. So pick sk to hit that, rather than tuning per shape. Longer K also needs a shorter
// fp16 accumulator chain, and a larger sk shortens it -- the two wants agree.
//
// Clamped to a power of two in [1,32]; K/16/sk must stay divisible by the 4 warps.
static const int GEMV_TARGET_BLOCKS = 1024;
static int auto_split_k(int N, int K) {
    int want = GEMV_TARGET_BLOCKS / (N / N_PER_BLOCK);
    int sk = 1;
    while (sk * 2 <= want && sk < 32) sk *= 2;
    while (sk > 1 && (K / 16) % (sk * (GEMV_THREADS / 32))) sk /= 2;   // keep ktiles/slice a multiple of 4
    return sk;
}
static int get_split_k(int N, int K) { const char* e = getenv("GEMV_SPLIT_K"); int v = e ? atoi(e) : 0; return v > 0 ? v : auto_split_k(N, K); }
static int SPLIT_K = 8;

static void launch(const int4* B, const half* A, const half* s, float* partial, half* C, int N, int K) {
    dim3 grid(N / N_PER_BLOCK, SPLIT_K);
    switch (SPLIT_K) {
        case  1: gemv_w4a16< 1><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case  2: gemv_w4a16< 2><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case  4: gemv_w4a16< 4><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case  8: gemv_w4a16< 8><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case 16: gemv_w4a16<16><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case 32: gemv_w4a16<32><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        default: printf("  GEMV_SPLIT_K=%d unsupported (1/2/4/8/16/32)\n", SPLIT_K); exit(1);
    }
    if (SPLIT_K > 1) gemv_reduce<<<(N + 255) / 256, 256>>>(partial, s, C, N, SPLIT_K);
    else             gemv_reduce<<<(N + 255) / 256, 256>>>(partial, s, C, N, 1);
}

// Marlin's packing, verbatim from test_marlin_classic_num.cu's packq, generalized to any N.
static void pack_B(const std::vector<int>& Bdeq, std::vector<int>& hB, int N, int K) {
    auto Bu = [&](int n, int k) { return (Bdeq[(size_t) n * K + k] + 8) & 0xf; };
    for (int ktile = 0; ktile < K / 16; ktile++)
        for (int nblock = 0; nblock < N / 16; nblock++)
            for (int lane = 0; lane < 32; lane++) {
                int n = nblock * 16 + lane / 4, kb = ktile * 16 + (lane % 4) * 2, q = 0;
                q |= Bu(n, kb)         << 0;  q |= Bu(n, kb + 1)     << 16;
                q |= Bu(n, kb + 8)     << 4;  q |= Bu(n, kb + 9)     << 20;
                q |= Bu(n + 8, kb)     << 8;  q |= Bu(n + 8, kb + 1) << 24;
                q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
                size_t idx = (size_t) (N / 2) * ktile + (nblock / 4) * 32 + lane;   // int4
                hB[idx * 4 + (nblock % 4)] = q;
            }
}

static double bench_one(int N, int K, int iters, bool check) {
    SPLIT_K = get_split_k(N, K);
    // each slice needs a whole number of ktiles, and the 4 warps must divide them evenly
    if (N % 64 || K % (16 * SPLIT_K) || (K / 16 / SPLIT_K) % (GEMV_THREADS / 32)) {
        printf("  N=%d K=%d SPLIT_K=%d unsupported (N%%64, K%%%d, ktiles/slice %% 4)\n", N, K, SPLIT_K, 16 * SPLIT_K); return -1; }

    std::vector<half> hA(K), hS(N, __float2half(1.0f)), hC(N);
    std::vector<int> Bdeq((size_t) N * K), hB((size_t) (K / 16) * (N / 2) * 4);
    srand(1234);
    for (auto& x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto& x : Bdeq) x = (rand() & 0xf) - 8;
    pack_B(Bdeq, hB, N, K);

    int4 *dB; half *dA, *dS, *dC; float* dP;
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * 4));
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
    CUDA_CHECK(cudaMalloc(&dS, hS.size() * 2));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * 2));
    CUDA_CHECK(cudaMalloc(&dP, (size_t) 32 * N * 4));   // max SPLIT_K
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice));

    if (!getenv("GEMV_NCU")) { launch(dB, dA, dS, dP, dC, N, K); CUDA_CHECK(cudaDeviceSynchronize()); }

    if (getenv("GEMV_NCU")) check = false;
#ifdef GEMV_BW_ONLY
    if (check) printf("  (GEMV_BW_ONLY: dequant replaced by a stub -- results are WRONG on purpose, timing only)\n");
    check = false;
#endif
    if (check) {
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));
        double max_abs = 0, ref_max = 0;
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double) __half2float(hA[k]) * Bdeq[(size_t) n * K + k];
            max_abs = fmax(max_abs, fabs(__half2float(hC[n]) - acc));
            ref_max = fmax(ref_max, fabs(acc));
        }
        double rel = max_abs / (ref_max + 1e-9);
        printf("  correctness N=%d K=%d: max_abs=%.4e rel=%.2e (|ref|max=%.2f) -> %s\n",
               N, K, max_abs, rel, ref_max, rel < 3e-2 ? "MATCH" : "MISMATCH");
        if (rel >= 3e-2) { printf("  MISMATCH -- aborting bench\n"); return -1; }
    }

    // GEMV_NCU=1: exactly one launch of each kernel, no warmup, no timing loop -- a clean profiler capture.
    // (Same convention as GEMV_NCU / FA_NCU / CONVERT_NCU in ppu_tests.) The printed us is then meaningless.
    if (getenv("GEMV_NCU")) {
        launch(dB, dA, dS, dP, dC, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  GEMV_NCU: single launch done (N=%d K=%d sk=%d, %d blocks) -- timing skipped\n",
               N, K, SPLIT_K, (N / N_PER_BLOCK) * SPLIT_K);
        cudaFree(dB); cudaFree(dA); cudaFree(dS); cudaFree(dC); cudaFree(dP);
        return 0;
    }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 10; i++) launch(dB, dA, dS, dP, dC, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) launch(dB, dA, dS, dP, dC, N, K);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;

    const char* e = getenv("MARLIN_HBM_GBS");
    double peak = e ? atof(e) : 2700.0;
    double bytes = (double) N * K / 2 + (double) K * 2 + (double) N * 2 + (double) N * 2;  // B + A + C + scales
    double gbs = bytes / (ms * 1e6);
    printf("  M=1     N=%-5d K=%-5d : %8.2f us  %7.1f TFLOP/s  %7.0f GB/s  %5.1f%% HBM   (sk=%d, %d blocks)\n",
           N, K, ms * 1e3, 2.0 * N * K / (ms * 1e9), gbs, 100.0 * gbs / peak, SPLIT_K, (N / N_PER_BLOCK) * SPLIT_K);

    cudaFree(dB); cudaFree(dA); cudaFree(dS); cudaFree(dC); cudaFree(dP);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1e3;
}

int main(int argc, char** argv) {
    if (argc >= 3) {
        int N = atoi(argv[1]), K = atoi(argv[2]);
        int iters = (argc >= 4) ? atoi(argv[3]) : 200;
        printf("W4A16 GEMV (Marlin B format), SPLIT_K=auto:\n");
        return bench_one(N, K, iters, true) < 0 ? 1 : 0;
    }
    printf("W4A16 GEMV (Marlin B format), SPLIT_K=auto, 200 iters:\n");
    int shapes[][2] = { {4096, 4096}, {14336, 4096}, {4096, 14336} };
    for (auto& s : shapes) if (bench_one(s[0], s[1], 200, true) < 0) return 1;
    return 0;
}
