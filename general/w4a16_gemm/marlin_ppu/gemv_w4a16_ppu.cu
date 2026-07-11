// W4A16 GEMV (M=1 decode) reading Marlin's INT4 B format in place.
//
// WHY THIS EXISTS. Marlin is a GEMM: its tile is 16x128 and decode uses 1 of those 16 rows, and it wastes most of a
// dequant on rows nobody reads. A purpose-built GEMV is 1.0-1.9x faster on cold weights (see README).
//
// IT DOES NOT ESCAPE MARLIN'S CONSERVATION LAW. This file used to claim it did. It does not:
//
//     loads_per_thread x warps_per_CU  =  (K/16) * (N/64) / CUs      -- independent of SPLIT_K and of GEMV_THREADS
//
// (= 228 at N=K=4096 on 72 CUs). SPLIT_K and GEMV_THREADS only slide along that hyperbola, which is exactly why every
// SPLIT_K sweep looked like it "cancelled out" -- it did. Measured on that curve, cold, gs=128, N=4096 K=4096:
//
//     T=32   16 loads/thread, 14.2 warps/CU -> 10.64 us    (too few warps to hide DRAM latency)
//     T=64    8 loads/thread, 28.4 warps/CU ->  9.13 us    <- the optimum
//     T=128   4 loads/thread, 56.9 warps/CU -> 10.57 us
//     T=256   2 loads/thread, 113  warps/CU -> 12.51 us    (fixed cost per block no longer amortized)
//
// Getting past the hyperbola means changing the CONSTANT: cut the per-block fixed cost (block launch, the shared
// reduce, the partial write, and the separate reduce kernel's ~1.1 us launch), or make each DRAM access carry more.
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

// Warps per block. Together with SPLIT_K this picks the point on the conservation hyperbola (see the file header):
// more loads per thread costs warps per CU, one for one, and the product is fixed by the problem and the CU count.
//
// MEASURED: 64 is the optimum on BOTH shapes (cold, gs=128). N=4096 K=4096: 10.57 -> 9.13 us (+13.6%).
// N=14336 K=4096: 21.62 -> 21.58 us (flat). Both extremes lose -- see the conservation law at the top of the file.
#ifndef GEMV_THREADS
#define GEMV_THREADS 64
#endif

// How many independent int4 loads a thread issues before consuming any.
//
// MEASURED on the box, and it barely matters -- 2 is a small win, 4 and 8 are losses:
//   N=4096  sk=8   U=1 7.20   U=2 6.94   U=4 7.59   U=8 8.85 us
//   N=14336 sk=4   U=1 11.90  U=2 11.72  U=4 12.93  U=8 14.95 us
//
// So Memory Dependency (5.333, the top stall) is NOT a shortage of in-flight loads. The likely cause is the stride:
// B is ktile-major (Marlin packed it for the mma), so walking K jumps a whole N-row each step. A thread's consecutive
// ktiles are step*8N bytes apart -- 128 KB at N=4096, 448 KB at N=14336 -- and U concurrent loads scatter across
// U times that, thrashing TLB and LLC. That predicts exactly what we see: U hurts, and it hurts more as N grows.
// This is the price of sharing one packed weight tensor with prefill.
#ifndef GEMV_UNROLL
#define GEMV_UNROLL 2
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
           float* __restrict__ partial, half* __restrict__ C, int* __restrict__ counter,
           int N, int K, int kt_per_group) {
    const int nb_group = blockIdx.x;                 // 4 nblocks = 64 columns
    const int slice    = blockIdx.y;
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;

    const int k_tiles = K / 16;
    const int kt_per_slice = k_tiles / SPLIT_K;
    const int kt_begin = slice * kt_per_slice;
    const int kt_end   = kt_begin + kt_per_slice;

    const int b_stride = N / 2;                      // int4 per ktile
    const int lane_k = (lane % 4) * 2;               // this lane's k phase within the 16-wide ktile

    // GROUPED SCALES. Real quantized models use groupsize=128, so s varies along K and CANNOT be applied once at the
    // end -- which is exactly what this kernel used to do (and its test set every scale to 1.0, the multiplicative
    // identity, so neither bug was visible). Same pair of bugs we just fixed in marlin_classic_ppu.cuh.
    //
    // C[n] = sum_g s[g][n] * (sum_{k in group g} A[k] * q[n][k])
    //
    // So walk K one group at a time: accumulate the group's partial in half2, then FLUSH it into an fp32 accumulator
    // scaled by that group's s. kt_per_group = groupsize/16; the caller passes k_tiles for per-column scales (gs == -1),
    // which collapses to a single group and reproduces the old behaviour exactly.
    //
    // This also IMPROVES precision: the fp16 chain is now bounded by one group (8 ktiles / 4 warps = 2 per thread at
    // groupsize=128) instead of the whole slice, and everything across groups is fp32.
    float facc[4][2] = {{0.f}};

    for (int g0 = kt_begin; g0 < kt_end; g0 += kt_per_group) {
        const int g  = g0 / kt_per_group;                     // scale row
        const int g1 = min(g0 + kt_per_group, kt_end);

        half2 hacc[4][2];
        #pragma unroll
        for (int j = 0; j < 4; j++) { hacc[j][0] = __float2half2_rn(0.f); hacc[j][1] = __float2half2_rn(0.f); }

        // Prefetch this group's 8 scales HERE, not in the flush. acu (cold, T=64, 4096^2) says Memory Dependency 2.630
        // dominates everything -- 4.3x the next stall, and 30x the ALU stalls. But GEMV_UNROLL only hoists the B loads:
        // the 16 A loads and the 16 scale loads per warp were issued and consumed in the same breath, load-to-use
        // distance ZERO, and they OUTNUMBER the 8 B loads 4 to 1. That is why more MLP (U=4) and more warps (sk=32)
        // both failed to help a memory-dependency-bound kernel: neither touched the loads that were actually stalling.
        // -DGEMV_NO_HOIST restores the old placement.
#ifndef GEMV_NO_HOIST
        half sg[4][2];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++)
                sg[j][h] = s[(long long) g * N + (nb_group * 4 + j) * 16 + lane / 4 + 8 * h];
#endif

        // ILP. acu (cache-resident) put Memory Dependency at 5.333 with warps idle, so issue GEMV_UNROLL independent
        // int4 loads before consuming any. Measured: U=2 is the optimum and it is worth MORE on cold weights (+10.5%)
        // than warm (+3.6%); U>=4 loses, and loses harder as N grows -- B is ktile-major, so consecutive ktiles are
        // 8N bytes apart and U concurrent loads scatter across U times that.
        const int step = GEMV_THREADS / 32;
        const int n_kt = (g1 - g0 > wid) ? ((g1 - g0 - wid + step - 1) / step) : 0;   // ktiles THIS thread owns here
        const int main_n = (n_kt / GEMV_UNROLL) * GEMV_UNROLL;
        int kt = g0 + wid;

#ifdef GEMV_BW_ONLY
        // SPEED-ONLY, RESULT IS WRONG. Same addresses, same loop, same reduce -- only the dequant is replaced, by the
        // cheapest arithmetic that still consumes every loaded byte. It told us the kernel was ALU bound, not bandwidth
        // bound: 9.66 us vs the naive kernel's 18.97 us on N=14336 K=4096, and the gap scaled with the WEIGHT COUNT.
        #define GEMV_BODY(q4_, a01_, a89_)                                                               \
            do {                                                                                         \
                const half2 aa = (a01_);                                                                 \
                const int q[4] = { (q4_).x, (q4_).y, (q4_).z, (q4_).w };                                 \
                _Pragma("unroll")                                                                        \
                for (int j = 0; j < 4; j++)                                                              \
                    hacc[j][0] = __hfma2(__halves2half2(__int2half_rn(q[j] & 1),                         \
                                                        __int2half_rn(q[j] >> 31)), aa, hacc[j][0]);     \
            } while (0)
#else
        // A now arrives as a PARAMETER -- it is loaded up front with the B int4, not fetched here and used one
        // instruction later. That is the whole point; see the Memory Dependency note above.
        #define GEMV_BODY(q4_, a01_, a89_)                                                               \
            do {                                                                                         \
                const half2 aa = (a01_), ab = (a89_);                                                    \
                const int q[4] = { (q4_).x, (q4_).y, (q4_).z, (q4_).w };                                 \
                _Pragma("unroll")                                                                        \
                for (int j = 0; j < 4; j++) {                                                            \
                    const FragB b0 = dequant(q[j]);        /* column n   */                              \
                    const FragB b1 = dequant(q[j] >> 8);   /* column n+8 */                              \
                    hacc[j][0] = __hfma2(b0.v[0], aa, hacc[j][0]);    /* k = kb, kb+1   */               \
                    hacc[j][0] = __hfma2(b0.v[1], ab, hacc[j][0]);    /* k = kb+8, kb+9 */               \
                    hacc[j][1] = __hfma2(b1.v[0], aa, hacc[j][1]);                                       \
                    hacc[j][1] = __hfma2(b1.v[1], ab, hacc[j][1]);                                       \
                }                                                                                        \
            } while (0)
#endif
        int i = 0;
        for (; i < main_n; i += GEMV_UNROLL) {          // ALL of B and A issued back-to-back, none consumed yet
            int4 q4[GEMV_UNROLL];
            half2 av0[GEMV_UNROLL], av1[GEMV_UNROLL];
            #pragma unroll
            for (int u = 0; u < GEMV_UNROLL; u++) {
                const int ktu = kt + u * step, kb = ktu * 16 + lane_k;   // kb even -> half2 loads are 4B aligned
                q4[u]  = B[(long long) b_stride * ktu + nb_group * 32 + lane];
                av0[u] = *reinterpret_cast<const half2*>(&A[kb]);        // {A[kb],   A[kb+1]}
                av1[u] = *reinterpret_cast<const half2*>(&A[kb + 8]);    // {A[kb+8], A[kb+9]}
            }
            #pragma unroll
            for (int u = 0; u < GEMV_UNROLL; u++) GEMV_BODY(q4[u], av0[u], av1[u]);
            kt += GEMV_UNROLL * step;
        }
        for (; i < n_kt; i++) {                        // tail, when n_kt is not a multiple of GEMV_UNROLL
            const int kb = kt * 16 + lane_k;
            GEMV_BODY(B[(long long) b_stride * kt + nb_group * 32 + lane],
                      *reinterpret_cast<const half2*>(&A[kb]),
                      *reinterpret_cast<const half2*>(&A[kb + 8]));
            kt += step;
        }
        #undef GEMV_BODY

        // FLUSH: fold this group's half2 partial into fp32, scaled by this group's s. The two half2 lanes hold
        // different k, so they add. Column of slot (j,h) is (nb_group*4 + j)*16 + lane/4 + 8*h.
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const float2 f = __half22float2(hacc[j][h]);
#ifdef GEMV_NO_HOIST
                const int col = (nb_group * 4 + j) * 16 + lane / 4 + 8 * h;
                facc[j][h] += (f.x + f.y) * __half2float(s[(long long) g * N + col]);
#else
                facc[j][h] += (f.x + f.y) * __half2float(sg[j][h]);   // prefetched at the top of the group
#endif
            }
    }

    float acc[4][2];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int h = 0; h < 2; h++) acc[j][h] = facc[j][h];

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

    // Reduce the 64 columns across the warps. GEMV_THREADS may be fewer than 64, so each thread may own >1 column.
    #define GEMV_COL_LOOP(body) \
        for (int c = threadIdx.x; c < N_PER_BLOCK; c += GEMV_THREADS) { body }

    if (SPLIT_K == 1) {                                // nothing to reduce across
        GEMV_COL_LOOP({
            float v = 0.f;
            #pragma unroll
            for (int w = 0; w < GEMV_THREADS / 32; w++) v += sm[w][c];
            C[nb_group * N_PER_BLOCK + c] = __float2half(v);   // scales already applied per group
        })
        return;
    }

#ifndef GEMV_FUSED_REDUCE
    GEMV_COL_LOOP({
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < GEMV_THREADS / 32; w++) v += sm[w][c];
        partial[(long long) slice * N + nb_group * N_PER_BLOCK + c] = v;
    })
#else
    // Last-CTA reduce: the final slice to finish this nb_group folds the partials in place, so the second kernel (and
    // its ~2.1 us launch, 30% of a 7 us GEMV) disappears. Ordering is the whole game here, and follows CUDA's
    // threadFenceReduction: __syncthreads() retires this block's partial stores, then thread 0 issues __threadfence()
    // to push them device-visible, and only then takes a ticket. The block that draws the last ticket is guaranteed to
    // observe every other block's partials.
    //
    // `partial` is read through volatile so the winner cannot serve the other slices' values out of its own L1, which
    // is not coherent across CTAs.
    GEMV_COL_LOOP({
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < GEMV_THREADS / 32; w++) v += sm[w][c];
        partial[(long long) slice * N + nb_group * N_PER_BLOCK + c] = v;
    })
    __syncthreads();

    __shared__ bool s_last;
    if (threadIdx.x == 0) {
        __threadfence();
        s_last = (atomicAdd(&counter[nb_group], 1) == SPLIT_K - 1);
    }
    __syncthreads();
    if (!s_last) return;

    GEMV_COL_LOOP({
        volatile const float* p = partial + nb_group * N_PER_BLOCK + c;
        float v = 0.f;
        for (int i = 0; i < SPLIT_K; i++) v += p[(long long) i * N];
        C[nb_group * N_PER_BLOCK + c] = __float2half(v);   // scales already applied per group
    })
    if (threadIdx.x == 0) counter[nb_group] = 0;       // rearm for the next launch; the stream orders it
#endif
    #undef GEMV_COL_LOOP
}

// out[n] = half(sum_slice partial[slice][n]). The scales are applied per group inside the mainloop, so they must NOT
// be applied again here -- with groupsize=128 there is no single s[n] to apply.
__global__ void gemv_reduce(const float* __restrict__ partial, half* __restrict__ C, int N, int split_k) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float v = 0.f;
    for (int i = 0; i < split_k; i++) v += partial[(long long) i * N + n];
    C[n] = __float2half(v);
}

// ---------------------------------------------------------------------------------------------------------------

// SPLIT_K trades block count against per-thread loop length -- the same tension that caps Marlin, except here the two
// are NOT tied to a tile decomposition, so we can sit anywhere on the curve. grid = (N/64) x SPLIT_K.
//
// SPLIT_K sets both the block count and (on the fused path) the contention on counter[nb_group].
//
// THE OPTIMUM DEPENDS ON WHETHER THE WEIGHTS ARE IN CACHE, and the two answers disagree on every knob. The PPU LLC is
// 64-128 MB, so all decode shapes (8.4-29.4 MB of weights) are LLC-resident in a naive benchmark loop. A served model
// reads cold weights, so the HBM-resident column is the one that matters. Measured (U=2, GEMV_WORKSET_MB=256):
//
//                      cache-resident            HBM-resident
//   N=4096   sk=8   fused 5.83 / sep 6.85     fused 9.92  / sep 10.61
//            sk=16  fused 8.80 / sep 7.44     fused 10.13 / sep  9.29   <- HBM optimum
//            sk=32                            fused 14.63 / sep 10.64
//   N=14336  sk=4   fused 11.41 / sep 11.78   fused 18.98 / sep 18.40   <- HBM optimum
//            sk=8   fused 14.88 / sep 13.87   fused 23.01 / sep 19.20
//
// Cold weights need MORE blocks to cover DRAM latency (N=4096 moves sk 8 -> 16), and at that sk the fused reduce's
// contention already costs more than the launch it saves. So the crossover moves left and SEPARATE wins outright.
// Defaults follow the HBM column: separate reduce, ~1024 blocks, no sk cap. -DGEMV_FUSED_REDUCE restores the fused
// path, which is the better choice only when the weights genuinely stay resident.
//
// Clamped to a power of two; K/16/sk must stay divisible by the 4 warps.
static const int GEMV_TARGET_BLOCKS = 1024;
#ifdef GEMV_FUSED_REDUCE
static const int GEMV_MAX_SPLIT_K = 8;    // fused contention grows with sk
#else
static const int GEMV_MAX_SPLIT_K = 32;   // separate reduce has no contention
#endif
static int auto_split_k(int N, int K) {
    int want = GEMV_TARGET_BLOCKS / (N / N_PER_BLOCK);
    int sk = 1;
    while (sk * 2 <= want && sk < GEMV_MAX_SPLIT_K) sk *= 2;
    while (sk > 1 && (K / 16) % (sk * (GEMV_THREADS / 32))) sk /= 2;   // keep ktiles/slice a multiple of 4
    return sk;
}
// The weight stream fits in LLC (BW_ONLY reported 112.9% of HBM peak on N=14336's 29.4 MB), so a 200-iteration loop
// over ONE B buffer measures cache, not HBM: only iteration 0 touches DRAM. Every %HBM number in this file is therefore
// optimistic -- real decode re-reads weights that nothing kept warm.
//
// GEMV_WORKSET_MB=<n> allocates ceil(n MB / sizeof(B)) identical copies of B and rotates through them, so consecutive
// iterations cannot hit each other's lines. Shape and kernel are untouched. Set it past the LLC (256 is a safe start).
static int get_rot(size_t b_bytes) {
    const char* e = getenv("GEMV_WORKSET_MB");
    if (!e) return 1;
    size_t want = (size_t) atoi(e) << 20;
    int r = (int) ((want + b_bytes - 1) / b_bytes);
    return r < 1 ? 1 : r;
}

static int get_split_k(int N, int K) { const char* e = getenv("GEMV_SPLIT_K"); int v = e ? atoi(e) : 0; return v > 0 ? v : auto_split_k(N, K); }

// -1 = per-column scales; 128 = grouped, WHAT REAL QUANTIZED MODELS USE. Grouped scales vary along K, so they cannot be
// applied once at the end -- the kernel folds each group's partial into an fp32 accumulator, scaled by that group's s.
static int get_groupsize() { const char* e = getenv("GEMV_GROUPSIZE"); return e ? atoi(e) : -1; }
static int SPLIT_K = 8;

// kt_per_group = groupsize/16, or k_tiles when groupsize == -1 (one group spanning all of K -> per-column scales).
static void launch(const int4* B, const half* A, const half* s, float* partial, half* C, int* counter,
                   int N, int K, int kt_per_group) {
    dim3 grid(N / N_PER_BLOCK, SPLIT_K);
    #define GEMV_CASE(SK) case SK: gemv_w4a16<SK><<<grid, GEMV_THREADS>>>(B, A, s, partial, C, counter, N, K, kt_per_group); break;
    switch (SPLIT_K) {
        GEMV_CASE(1) GEMV_CASE(2) GEMV_CASE(4) GEMV_CASE(8) GEMV_CASE(16) GEMV_CASE(32)
        default: printf("  GEMV_SPLIT_K=%d unsupported (1/2/4/8/16/32)\n", SPLIT_K); exit(1);
    }
    #undef GEMV_CASE
#ifndef GEMV_FUSED_REDUCE
    if (SPLIT_K > 1) gemv_reduce<<<(N + 255) / 256, 256>>>(partial, C, N, SPLIT_K);
#endif
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

    const int gs = get_groupsize();
    if (gs != -1 && (gs % 16 || K % gs)) { printf("  GEMV_GROUPSIZE=%d unsupported (need %%16==0, K%%gs==0)\n", gs); return -1; }
    // GEMV_SCALE_BUG=1 reproduces the ORIGINAL bug: pass kt_per_group = k_tiles even when gs=128, so the kernel sees a
    // single group and applies only s[0][n] -- a per-column scale on a grouped problem. It must MISMATCH, which is how
    // we know the test is not blind. (The kernel really did this, and its test set every scale to 1.0 so nothing showed.)
    const int kt_per_group = (gs == -1 || getenv("GEMV_SCALE_BUG")) ? (K / 16) : (gs / 16);
    const int GROUPS = (gs == -1) ? 1 : (K / gs);

    std::vector<half> hA(K), hS((size_t) GROUPS * N), hC(N);
    std::vector<int> Bdeq((size_t) N * K), hB((size_t) (K / 16) * (N / 2) * 4);
    srand(1234);
    for (auto& x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto& x : Bdeq) x = (rand() & 0xf) - 8;
    // NON-TRIVIAL scales, distinct per (group, column). They used to be all 1.0 -- the multiplicative identity -- so a
    // kernel that dropped the scale entirely passed anyway, and a kernel that applied a per-COLUMN scale to a GROUPED
    // problem also passed. Both bugs were live in this file.
    for (auto& x : hS) x = __float2half(0.5f + (rand() % 100) / 100.0f);
    pack_B(Bdeq, hB, N, K);

    const size_t b_bytes = hB.size() * 4;
    const int rot = get_rot(b_bytes);
    int4 *dB; half *dA, *dS, *dC; float* dP; int* dCnt;
    CUDA_CHECK(cudaMalloc(&dB, b_bytes * rot));
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
    CUDA_CHECK(cudaMalloc(&dS, hS.size() * 2));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * 2));
    CUDA_CHECK(cudaMalloc(&dP, (size_t) 32 * N * 4));   // max SPLIT_K
    CUDA_CHECK(cudaMalloc(&dCnt, (size_t) (N / N_PER_BLOCK) * 4));
    CUDA_CHECK(cudaMemset(dCnt, 0, (size_t) (N / N_PER_BLOCK) * 4));   // the winning CTA rearms it each launch
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_bytes, cudaMemcpyHostToDevice));
    for (int r = 1; r < rot; r++)   // identical copies; correctness is unaffected, cache residency is not
        CUDA_CHECK(cudaMemcpy((char*) dB + (size_t) r * b_bytes, dB, b_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice));

    if (!getenv("GEMV_NCU")) { launch(dB, dA, dS, dP, dC, dCnt, N, K, kt_per_group); CUDA_CHECK(cudaDeviceSynchronize()); }

    if (getenv("GEMV_NCU")) check = false;
#ifdef GEMV_BW_ONLY
    if (check) printf("  (GEMV_BW_ONLY: dequant replaced by a stub -- results are WRONG on purpose, timing only)\n");
    check = false;
#endif
    // Verified BEFORE and AFTER the timing loop. The fused reduce rearms `counter` from inside the winning CTA, so a
    // leaked count would corrupt every launch after the first -- which a pre-bench check alone cannot see.
    // C[n] = sum_g s[g][n] * sum_{k in group g} A[k] * q[n][k]
    const int gsz = (gs == -1) ? K : gs;
    std::vector<double> ref(N);
    for (int n = 0; n < N; n++) {
        double total = 0;
        for (int g = 0; g < GROUPS; g++) {
            double acc = 0;
            for (int k = g * gsz; k < (g + 1) * gsz; k++) acc += (double) __half2float(hA[k]) * Bdeq[(size_t) n * K + k];
            total += acc * __half2float(hS[(size_t) g * N + n]);
        }
        ref[n] = total;
    }
    auto verify = [&](const char* when) -> bool {
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));
        double max_abs = 0, ref_max = 0;
        for (int n = 0; n < N; n++) {
            max_abs = fmax(max_abs, fabs(__half2float(hC[n]) - ref[n]));
            ref_max = fmax(ref_max, fabs(ref[n]));
        }
        double rel = max_abs / (ref_max + 1e-9);
        printf("  correctness N=%d K=%d %s: max_abs=%.4e rel=%.2e (|ref|max=%.2f) -> %s\n",
               N, K, when, max_abs, rel, ref_max, rel < 3e-2 ? "MATCH" : "MISMATCH");
        return rel < 3e-2;
    };
    if (check && !verify("pre-bench")) { printf("  aborting bench\n"); return -1; }

    // GEMV_NCU=1: exactly one launch of each kernel, no warmup, no timing loop -- a clean profiler capture.
    // (Same convention as GEMV_NCU / FA_NCU / CONVERT_NCU in ppu_tests.) The printed us is then meaningless.
    if (getenv("GEMV_NCU")) {
        launch(dB, dA, dS, dP, dC, dCnt, N, K, kt_per_group);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  GEMV_NCU: single launch done (N=%d K=%d sk=%d, %d blocks) -- timing skipped\n",
               N, K, SPLIT_K, (N / N_PER_BLOCK) * SPLIT_K);
        cudaFree(dB); cudaFree(dA); cudaFree(dS); cudaFree(dC); cudaFree(dP); cudaFree(dCnt);
        return 0;
    }

    auto Brot = [&](int i) { return (const int4*) ((char*) dB + (size_t) (i % rot) * b_bytes); };
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 10; i++) launch(Brot(i), dA, dS, dP, dC, dCnt, N, K, kt_per_group);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) launch(Brot(i), dA, dS, dP, dC, dCnt, N, K, kt_per_group);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    if (check && !verify("post-bench")) { printf("  counter leaked across launches\n"); return -1; }

    const char* e = getenv("MARLIN_HBM_GBS");
    double peak = e ? atof(e) : 2700.0;
    double bytes = (double) N * K / 2 + (double) K * 2 + (double) N * 2 + (double) GROUPS * N * 2;  // B + A + C + scales
    double gbs = bytes / (ms * 1e6);
    printf("  M=1     N=%-5d K=%-5d : %8.2f us  %7.1f TFLOP/s  %7.0f GB/s  %5.1f%% HBM   (gs=%d, sk=%d, %d blocks, wset=%zu MB)\n",
           N, K, ms * 1e3, 2.0 * N * K / (ms * 1e9), gbs, 100.0 * gbs / peak,
           gs, SPLIT_K, (N / N_PER_BLOCK) * SPLIT_K, (b_bytes * rot) >> 20);

    cudaFree(dB); cudaFree(dA); cudaFree(dS); cudaFree(dC); cudaFree(dP); cudaFree(dCnt);
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
    if (!getenv("GEMV_WORKSET_MB"))
        printf("  NOTE: weights are LLC-resident (PPU LLC is 64-128 MB). Set GEMV_WORKSET_MB=256 for cold-weight,\n"
               "        HBM-bound timings -- the regime a served model actually runs in.\n");
    int shapes[][2] = { {4096, 4096}, {14336, 4096}, {4096, 14336} };
    for (auto& s : shapes) if (bench_one(s[0], s[1], 200, true) < 0) return 1;
    return 0;
}
