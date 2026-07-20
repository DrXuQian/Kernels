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
#include <set>
#include <algorithm>

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

// How many ktiles' worth of loads (B int4 + both A half2) a thread issues before consuming any.
//
// This only became meaningful once the A loads were hoisted alongside B (see the Memory Dependency note in the group
// loop). Before that, U hoisted 8 of the 40 loads per warp and the other 32 were fetched-and-used one instruction
// later, which is why raising U used to make things WORSE.
//
// MEASURED cold, gs=128, T=64 (hoisted):
//                    N=4096 K=4096      N=14336 K=4096
//   U=1                 10.33 us            26.02 us
//   U=2                  9.51               21.36
//   U=4                  9.42               20.68     <- default
//   U=8                 10.18               24.94
//
// SHAPE-DEPENDENT, and worth being honest about: on N=14336 the hoist + U=4 is a real 4.3% (21.58 -> 20.68), but on
// N=4096 it is a ~2% REGRESSION against the old no-hoist U=2 (9.14 -> 9.42). That shape gives each warp only 8 ktiles,
// so U=4 is two iterations and the extra registers and setup never amortize; N=14336 gives 32 and they do.
//
// So the "80% of loads have zero load-to-use distance" diagnosis was accurate as a DESCRIPTION and not load-bearing as
// a CAUSE. Memory Dependency (2.630, dominant) is mostly the 8 cold B loads from DRAM; A and the scales hit L1/L2 and
// are cheap. Hoisting them only buys room for a deeper B pipeline, which needs a long loop to pay off.
// -DGEMV_NO_HOIST restores the old placement.
#ifndef GEMV_UNROLL
// Overridable so the main_n cliff can be swept. n_kt = kt_per_group/step ktiles per warp per group, and
// main_n = (n_kt/GEMV_UNROLL)*GEMV_UNROLL -- so once kt_per_group/step < GEMV_UNROLL the hoisted path is
// SILENTLY skipped entirely and every B load is consumed at load-to-use distance zero. At step=2, U=4 that
// is every gs <= 64. Suspected cause of gs=32 costing +67% on PPU; sweep -DGEMV_UNROLL to test.
#ifndef GEMV_UNROLL
#define GEMV_UNROLL 4
#endif
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
// AFFINE (GGUF Q4_1/Q4_K): the weight is w = (q-8)*s + m', with m' = m + 8s folded on the host, so the
// symmetric dequant above is reused untouched. Distributing over the dot product,
//     C[n] = sum_g [ s[g][n] * sum_{k in g} A[k]*(q-8)  +  m'[g][n] * sum_{k in g} A[k] ]
// the min term is RANK-1: the A-sum is a scalar shared by all 64 columns of the block, and A is already in
// registers, so it costs one half2 add per body and one fma per column at the flush. It touches no B load,
// which is what the %HBM number is actually made of.
//
// The per-thread A-sum must cover exactly the k this thread consumed, so that the existing lane/warp
// reduction totals the whole group -- it does, since every (j,h) slot of a thread shares the same k.
// OCCUPANCY. acu: theoretical 62.5%, achieved 41.3%, "limited by the number of required registers" -- 88
// per thread gives 10 warps per scheduler against a hardware max of 16. On the PPU that is 131072 regs and
// 64 warps per CU, so saturating needs <= 131072/(64*32) = 64 regs/thread.
//
// This matters for the same reason the loop structure does. gs=32 leaves only ONE B load in flight per warp
// (main_n = 0 for every gs <= 64), and the GEMV_KTPG probe showed the whole +67% is structural, not the
// scale data -- 19.89us for gs=32's data through gs=128's structure against 19.87 for gs=128 itself. More
// resident warps attacks that from the other side: instead of raising per-warp memory-level parallelism by
// restructuring the loop, raise the number of warps with a load outstanding.
//
// GEMV_MIN_BLOCKS is the blocks-per-CU ptxas must fit, which caps registers accordingly. At 64 threads a
// block is 2 warps, so 32 blocks = 64 warps = full occupancy and a 64-register budget. 0 leaves it alone.
#ifndef GEMV_MIN_BLOCKS
#define GEMV_MIN_BLOCKS 0
#endif
#if GEMV_MIN_BLOCKS > 0
#define GEMV_LB __launch_bounds__(GEMV_THREADS, GEMV_MIN_BLOCKS)
#else
#define GEMV_LB __launch_bounds__(GEMV_THREADS)
#endif

template <int SPLIT_K, bool AFFINE>
__global__ void GEMV_LB
gemv_w4a16(const int4* __restrict__ B, const half* __restrict__ A, const half* __restrict__ s,
           const half* __restrict__ mn,
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

    // THE gs=32 CLIFF. This loop is group-major and GEMV_UNROLL hoists INSIDE one group, so the ktiles a
    // warp owns per group is kt_per_group/step -- at gs=32 that is 2/step, i.e. ONE at GEMV_THREADS=64.
    // main_n = (1/4)*4 = 0, so the hoisted path is skipped entirely and every B load is consumed
    // immediately. gs=-1 makes the group the whole slice, so n_kt is large and the hoist works.
    //
    // Measured on ppu001, same kernel, N=4096 K=14336: gs=-1 87.9% HBM against gs=32 47.2%. The scale
    // reads are not the difference -- GEMV_SCALE_PERM was tried and lost on PPU. Losing the unroll is.
    //
    // -DGEMV_XGROUP walks KTILES instead of groups and flushes when the group changes: nothing about the
    // B or A loads depends on the group, only the accumulate and flush do, so U independent loads can be
    // issued across a group boundary. This file's own note predicted the cliff; the fix belongs here too.
#ifdef GEMV_XGROUP
    {
        const int my_first = kt_begin + wid;
        const int my_n = (kt_end - kt_begin - wid + (GEMV_THREADS/32) - 1) / (GEMV_THREADS/32);
        int cur_g = -1;
        half2 hacc[4][2]; half sg[4][2], mg[4][2]; half2 asum2 = __float2half2_rn(0.f);
        #define GX_OPEN(gg) do { cur_g = (gg);                                                     \
            _Pragma("unroll") for (int j = 0; j < 4; j++) { hacc[j][0] = __float2half2_rn(0.f);     \
                                                           hacc[j][1] = __float2half2_rn(0.f); }   \
            asum2 = __float2half2_rn(0.f);                                                          \
            _Pragma("unroll") for (int j = 0; j < 4; j++)                                           \
                _Pragma("unroll") for (int h = 0; h < 2; h++) {                                     \
                    const int c_ = (nb_group * 4 + j) * 16 + lane / 4 + 8 * h;                      \
                    sg[j][h] = s[(long long) cur_g * N + c_];                                       \
                    if (AFFINE) mg[j][h] = mn[(long long) cur_g * N + c_]; }                        \
        } while (0)
        #define GX_FLUSH() do { if (cur_g >= 0) {                                                   \
            float as_ = 0.f;                                                                        \
            if (AFFINE) { const float2 fa = __half22float2(asum2); as_ = fa.x + fa.y; }              \
            _Pragma("unroll") for (int j = 0; j < 4; j++)                                            \
                _Pragma("unroll") for (int h = 0; h < 2; h++) {                                      \
                    const float2 f = __half22float2(hacc[j][h]);                                     \
                    facc[j][h] += (f.x + f.y) * __half2float(sg[j][h]);                              \
                    if (AFFINE) facc[j][h] += as_ * __half2float(mg[j][h]); } } } while (0)
        #define GX_BODY(kt_, q4_, aa_, ab_) do {                                                     \
            const int g_ = (kt_) / kt_per_group;                                                     \
            if (g_ != cur_g) { GX_FLUSH(); GX_OPEN(g_); }                                            \
            const half2 aa = (aa_), ab = (ab_);                                                      \
            if (AFFINE) asum2 = __hadd2(asum2, __hadd2(aa, ab));                                     \
            const int qs[4] = { (q4_).x, (q4_).y, (q4_).z, (q4_).w };                                \
            _Pragma("unroll") for (int j = 0; j < 4; j++) {                                          \
                const FragB b0 = dequant(qs[j]);                                                     \
                const FragB b1 = dequant(qs[j] >> 8);                                                \
                hacc[j][0] = __hfma2(b0.v[0], aa, hacc[j][0]);                                       \
                hacc[j][0] = __hfma2(b0.v[1], ab, hacc[j][0]);                                       \
                hacc[j][1] = __hfma2(b1.v[0], aa, hacc[j][1]);                                       \
                hacc[j][1] = __hfma2(b1.v[1], ab, hacc[j][1]); } } while (0)

        const int main_n2 = (my_n / GEMV_UNROLL) * GEMV_UNROLL;
        int kt2 = my_first, i2 = 0;
        for (; i2 < main_n2; i2 += GEMV_UNROLL) {
            int4 q4[GEMV_UNROLL]; half2 av0[GEMV_UNROLL], av1[GEMV_UNROLL];
            #pragma unroll
            for (int u = 0; u < GEMV_UNROLL; u++) {
                const int ktu = kt2 + u * (GEMV_THREADS / 32), kb = ktu * 16 + lane_k;
                q4[u]  = B[(long long) b_stride * ktu + nb_group * 32 + lane];
                av0[u] = *reinterpret_cast<const half2*>(&A[kb]);
                av1[u] = *reinterpret_cast<const half2*>(&A[kb + 8]);
            }
            #pragma unroll
            for (int u = 0; u < GEMV_UNROLL; u++) GX_BODY(kt2 + u * (GEMV_THREADS / 32), q4[u], av0[u], av1[u]);
            kt2 += GEMV_UNROLL * (GEMV_THREADS / 32);
        }
        for (; i2 < my_n; i2++) {
            const int kb = kt2 * 16 + lane_k;
            GX_BODY(kt2, B[(long long) b_stride * kt2 + nb_group * 32 + lane],
                    *reinterpret_cast<const half2*>(&A[kb]),
                    *reinterpret_cast<const half2*>(&A[kb + 8]));
            kt2 += (GEMV_THREADS / 32);
        }
        GX_FLUSH();
        #undef GX_BODY
        #undef GX_FLUSH
        #undef GX_OPEN
    }
    goto gemv_acc_done;
#endif
    for (int g0 = kt_begin; g0 < kt_end; g0 += kt_per_group) {
        const int g  = g0 / kt_per_group;                     // scale row
        const int g1 = min(g0 + kt_per_group, kt_end);

        half2 hacc[4][2];
        #pragma unroll
        for (int j = 0; j < 4; j++) { hacc[j][0] = __float2half2_rn(0.f); hacc[j][1] = __float2half2_rn(0.f); }
        half2 asum2 = __float2half2_rn(0.f);    // AFFINE: this thread's sum of A over the group

        // Prefetch this group's 8 scales HERE, not in the flush. acu (cold, T=64, 4096^2) says Memory Dependency 2.630
        // dominates everything -- 4.3x the next stall, and 30x the ALU stalls. But GEMV_UNROLL only hoists the B loads:
        // the 16 A loads and the 16 scale loads per warp were issued and consumed in the same breath, load-to-use
        // distance ZERO, and they OUTNUMBER the 8 B loads 4 to 1. That is why more MLP (U=4) and more warps (sk=32)
        // both failed to help a memory-dependency-bound kernel: neither touched the loads that were actually stalling.
        // -DGEMV_NO_HOIST restores the old placement.
#ifndef GEMV_NO_HOIST
        half sg[4][2];
#ifdef GEMV_SCALE_PERM
        // ONE int4 instead of EIGHT scalar half loads (and 2 instead of 16 with mins). A lane's 8 columns
        // are j*16 + lane/4 + 8h, i.e. {lane/4 + 8m : m=0..7} -- scattered over 128 B in the plain layout.
        // Marlin's _scale_perm (out[8i+j] = plain[i+8j]) makes exactly that set contiguous, so a lane's
        // whole group of scales is a single 16 B load and slot (j,h) is element 2j+h. This is what
        // _scale_perm is FOR, and the GEMV's per-lane column set is the same formula as Marlin's B
        // fragment, so it carries over untouched.
        //
        // Aimed at gs=32. Per thread per group the kernel issues 1 B load against 8 scale loads (16 with
        // mins); gs=128 issues 4 B loads against the same 8. That 4x worse ratio is why gs=32 costs +67%
        // on PPU against +9.8% on a 5090 -- this file's own notes already had Memory Dependency as the
        // dominant stall with the scale loads outnumbering B 4 to 1.
        {
            const int4 sv = reinterpret_cast<const int4*>(s)[(long long) g * (N / 8) + nb_group * 8 + lane / 4];
            const half* sp = reinterpret_cast<const half*>(&sv);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                #pragma unroll
                for (int h = 0; h < 2; h++) sg[j][h] = sp[2 * j + h];
        }
#else
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++)
                sg[j][h] = s[(long long) g * N + (nb_group * 4 + j) * 16 + lane / 4 + 8 * h];
#endif
#endif
        half mg[4][2];
        if (AFFINE) {
#ifdef GEMV_SCALE_PERM
            const int4 mv = reinterpret_cast<const int4*>(mn)[(long long) g * (N / 8) + nb_group * 8 + lane / 4];
            const half* mp = reinterpret_cast<const half*>(&mv);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                #pragma unroll
                for (int h = 0; h < 2; h++) mg[j][h] = mp[2 * j + h];
#else
            #pragma unroll
            for (int j = 0; j < 4; j++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    mg[j][h] = mn[(long long) g * N + (nb_group * 4 + j) * 16 + lane / 4 + 8 * h];
#endif
        }

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
                if (AFFINE) asum2 = __hadd2(asum2, __hadd2(aa, ab));   /* rank-1 min term */             \
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
        float asum = 0.f;
        half asum_h = __float2half(0.f);
        if (AFFINE) {
#ifdef GEMV_FLUSH_H16
            asum_h = __hadd(__low2half(asum2), __high2half(asum2));
#else
            const float2 fa = __half22float2(asum2); asum = fa.x + fa.y;
#endif
        }
        (void) asum; (void) asum_h;
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
#if defined(GEMV_FLUSH_H16) && !defined(GEMV_NO_HOIST)
                // acu (gs=32 vs gs=128) put v.cnvt at 1,382,400 against 350,208 -- +295%, and the single
                // most-executed opcode in the kernel, ahead of v.fma.f16 (1,376,256, UNCHANGED). The real
                // math does not grow with gs at all; the per-group FLUSH does, 4x with 4x the groups. Each
                // (j,h) used to cost 3 conversions (2 for hacc, 1 for the scale) plus an fp32 add and fma,
                // and 4 conversions when affine.
                //
                // Reducing and scaling in half instead costs ONE conversion: the group partial is already a
                // half2 accumulator, so folding its two lanes and applying the scale there loses nothing
                // structural, and the cross-group accumulator stays fp32 where the range actually matters.
                half v = __hmul(__hadd(__low2half(hacc[j][h]), __high2half(hacc[j][h])), sg[j][h]);
                if (AFFINE) v = __hfma(asum_h, mg[j][h], v);
                facc[j][h] += __half2float(v);
#else
                const float2 f = __half22float2(hacc[j][h]);
#ifdef GEMV_NO_HOIST
                const int col = (nb_group * 4 + j) * 16 + lane / 4 + 8 * h;
                facc[j][h] += (f.x + f.y) * __half2float(s[(long long) g * N + col]);
#else
                facc[j][h] += (f.x + f.y) * __half2float(sg[j][h]);   // prefetched at the top of the group
#endif
                if (AFFINE) facc[j][h] += asum * __half2float(mg[j][h]);
#endif
            }
    }

    gemv_acc_done: ;
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


#ifdef GEMV_AIU
// ============================================================================================================
// PROBE -- RESULT: AIU LOSES. MEASURED, CORRECT, AND THE DOOR IS NOW CLOSED FOR GOOD.
//
//                          LDG (default)      AIU
//   N=4096  K=4096         9.38 us / 34.2%   10.23 us / 31.4%   -9.1%
//   N=14336 K=4096        20.69    / 54.3%   21.79    / 51.5%   -5.3%
//
// Both MATCH at rel 5.0e-04, identical to LDG -- so this is a correct kernel losing, not a broken one.
//
// WHAT MAKES THIS CONCLUSIVE: it tests exactly the case the bf16 GEMV sweep never did. That sweep ran AIU against a
// ROW-MAJOR weight, where reads were already contiguous and the AIU had nothing to offer. Ours is ktile-major, so
// walking K jumps 8N = 32 KB, and the AIU's 2D descriptor gathers those strided rows IN HARDWARE -- one instruction
// pulls 8 ktiles across the stride. That advantage is real and it still is not enough. The four mechanical reasons
// below are simply larger than it:
//
// Everything we know says this should LOSE, and the bf16 GEMV sweep measured it losing (v2 swzl, v2b double-buffered,
// v7 bulk-linear -- all beaten by the dumbest register-LDG kernel; AIU ran 38% of HBM vs 70-76%). The reasons:
//   - GEMV has ZERO weight reuse. AIU does global->TSM; you still need TSM->register. Plain LDG is one hop. Shared
//     only pays when data is read more than once, and PPU shared loads are expensive (1 transaction per 4-byte word,
//     no half-merging -- measured in marlin_classic_ppu.cuh).
//   - AIU is a SINGLE-THREAD bulk DMA (ThrID = Layout<_1>): one thread issues it, everyone else waits on
//     commit/wait_group. Saturating HBM needs many outstanding requests; LDG gets them from many warps x many
//     independent loads, AIU gets far fewer, each behind a barrier.
//   - Its two selling points are both worth ZERO here: the swzl layout buys conflict-free ldmatrix (there is no mma),
//     and a big cube amortizes over many mma reads (there is no reuse).
//
// BUT that sweep tested AIU on a ROW-MAJOR weight, where the reads were already contiguous and the AIU had nothing to
// offer. Our B is ktile-major: walking K jumps 8N = 32 KB. The AIU's 2D descriptor gathers strided rows IN HARDWARE --
// one instruction pulls cube_h ktiles across that stride. That is the one thing it could uniquely do here, and it was
// never measured. "AIU loses on a contiguous layout" does not automatically transfer to a strided one.
//
// Geometry (verified by enumeration): B as b16 is [k_tiles][4N]. A block's 64 columns are 256 b16 = 512 B of each
// ktile row. cube_w maxes at 64 b16 (128 B), so 4 cubes cover them. cube_h = kt_per_group = 8 -> one AIU batch is
// exactly one scale group. Shared = 4*8*64*2 = 4 KB per block (57 KB/CU at 14.2 blocks/CU -- no occupancy cost).
// Lane l's int4 (the l-th of the block's 32) lands in cube l/8 at slot l%8:
//     shared int4 index = (l/8)*cube_h*8 + h*8 + (l%8)
// ============================================================================================================
__device__ __forceinline__ unsigned cvta_s(const void* p) { return (unsigned) __cvta_generic_to_shared((void*) p); }
__device__ __forceinline__ void aiu_linear_b16(unsigned smem, const void* gmem,
        int dim_h, int dim_w, int cube_h, int cube_w, int coord_h, int coord_w) {
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b16 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem), "l"(gmem), "r"(1), "r"(dim_h), "r"(dim_w),
           "r"(0), "r"(coord_h), "r"(coord_w), "r"(1), "r"(cube_h), "r"(cube_w), "r"(1));
}
__device__ __forceinline__ void aiu_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void aiu_wait0()  { asm volatile("cp.async.wait_group 0;\n"); }

template <int SPLIT_K>
__global__ void __launch_bounds__(GEMV_THREADS)
gemv_w4a16_aiu(const int4* __restrict__ B, const half* __restrict__ A, const half* __restrict__ s,
               float* __restrict__ partial, half* __restrict__ C, int* __restrict__ counter,
               int N, int K, int kt_per_group) {
    constexpr int AW = 64;                       // AIU cube width in b16 (128 B -- the hardware max contiguous)
    const int nb_group = blockIdx.x, slice = blockIdx.y;
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    constexpr int NW = GEMV_THREADS / 32;

    const int k_tiles = K / 16, kt_per_slice = k_tiles / SPLIT_K;
    const int kt_begin = slice * kt_per_slice, kt_end = kt_begin + kt_per_slice;
    const int lane_k = (lane % 4) * 2;
    const int H = kt_per_group;                  // ktiles per AIU batch == one scale group

    extern __shared__ half sh[];                 // 4 cubes x [H][AW] b16
    const int4* sh4 = reinterpret_cast<const int4*>(sh);

    float facc[4][2] = {{0.f}};

    for (int g0 = kt_begin; g0 < kt_end; g0 += H) {
        const int g = g0 / H;
        // 4 AIU strips, one per 64-b16 cube.
        //
        // THREAD 0 ISSUES ALL FOUR, IN A LOOP. The AIU is a SINGLE-THREAD bulk DMA (ThrID = Layout<_1>), so the
        // descriptor operands must be uniform. My first version used `if (threadIdx.x < 4)`, i.e. four lanes of the
        // SAME warp issuing with DIFFERENT operands -- lane-divergent issue of a single-thread instruction. It
        // produced garbage (rel = 1.5). fa_ppu.cu's v2 does `if (tid == 0) for (w...) aiu_load(...)`; v7 does
        // `if (lane == 0)`, one issuer per warp -- issuers in DIFFERENT warps are fine, divergent lanes are not.
        // Every thread still commits, to keep the per-thread cp.async group stack consistent.
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int j = 0; j < 4; j++)
                aiu_linear_b16(cvta_s(sh + j * H * AW), B, k_tiles, 4 * N, H, AW,
                               g0, nb_group * 256 + j * AW);
        }
        aiu_commit();
        aiu_wait0();
        __syncthreads();

        half2 hacc[4][2];
        #pragma unroll
        for (int j = 0; j < 4; j++) { hacc[j][0] = __float2half2_rn(0.f); hacc[j][1] = __float2half2_rn(0.f); }
        half2 asum2 = __float2half2_rn(0.f);    // AFFINE: this thread's sum of A over the group
        half sg[4][2];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h2 = 0; h2 < 2; h2++)
                sg[j][h2] = s[(long long) g * N + (nb_group * 4 + j) * 16 + lane / 4 + 8 * h2];

        for (int h = wid; h < H; h += NW) {
            const int4 q4 = sh4[(lane / 8) * H * 8 + h * 8 + (lane % 8)];   // == the LDG kernel's B[...+lane]
            const int kb = (g0 + h) * 16 + lane_k;
            const half2 aa = *reinterpret_cast<const half2*>(&A[kb]);
            const half2 ab = *reinterpret_cast<const half2*>(&A[kb + 8]);
            const int q[4] = { q4.x, q4.y, q4.z, q4.w };
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const FragB b0 = dequant(q[j]);
                const FragB b1 = dequant(q[j] >> 8);
                hacc[j][0] = __hfma2(b0.v[0], aa, hacc[j][0]);
                hacc[j][0] = __hfma2(b0.v[1], ab, hacc[j][0]);
                hacc[j][1] = __hfma2(b1.v[0], aa, hacc[j][1]);
                hacc[j][1] = __hfma2(b1.v[1], ab, hacc[j][1]);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                const float2 f = __half22float2(hacc[j][h2]);
                facc[j][h2] += (f.x + f.y) * __half2float(sg[j][h2]);
            }
        __syncthreads();                          // single-buffer TSM: done reading before the next AIU overwrites
    }

    float acc[4][2];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int h2 = 0; h2 < 2; h2++) {
            float v = facc[j][h2];
            v += __shfl_xor_sync(0xffffffff, v, 1);
            v += __shfl_xor_sync(0xffffffff, v, 2);
            acc[j][h2] = v;
        }
    __shared__ float sm[NW][N_PER_BLOCK];
    if (lane % 4 == 0) {
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h2 = 0; h2 < 2; h2++) sm[wid][j * 16 + lane / 4 + 8 * h2] = acc[j][h2];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < N_PER_BLOCK; c += GEMV_THREADS) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; w++) v += sm[w][c];
        const int n = nb_group * N_PER_BLOCK + c;
        if (SPLIT_K == 1) C[n] = __float2half(v);
        else              partial[(long long) slice * N + n] = v;
    }
}
#endif  // GEMV_AIU

// out[n] = half(sum_slice partial[slice][n]). The scales are applied per group inside the mainloop, so they must NOT
// be applied again here -- with groupsize=128 there is no single s[n] to apply.
// n_rows folds the MoE case in: partial is [rows][split_k][N] and C is [rows][N]. rows=1 is the dense
// layout unchanged, so the dense call sites need no edit.
__global__ void gemv_reduce(const float* __restrict__ partial, half* __restrict__ C, int N, int split_k,
                            int n_rows = 1) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long) n_rows * N) return;
    const int r = (int) (i / N), n = (int) (i % N);
    const float* pr = partial + (long long) r * split_k * N;
    float v = 0.f;
    for (int k = 0; k < split_k; k++) v += pr[(long long) k * N + n];
    C[i] = __float2half(v);
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
// Blocks to aim for. The occupancy arithmetic said to raise this: acu had theoretical 62.5% against an
// ACHIEVED 41.3%, and 1024 blocks over 72 CUs is 14.2 blocks = 28.4 of 64 warps per CU = 44%, matching the
// achieved figure exactly. So the grid, not the register count, was what the kernel actually sat on.
//
// It was still WRONG for the PPU. Doubling the grid (CU-aware, 72*32 = 2304 -> SPLIT_K=32) made gs=32
// SLOWER there: 33.26 -> 36.76 us sym, 35.26 -> 39.60 aff, while a 5090 improved to parity with gs=128.
// More slices buy occupancy and pay for it in partial writes and reduce work, and on 72 CUs the second
// term wins. Reverted to the measured-good 1024; GEMV_TARGET_BLOCKS=<n> sweeps it.
//
// The occupancy number was real and the arithmetic was right -- filling the CUs simply is not what this
// kernel is short of. Fourth explanation for the gs=32 cost to die, and the third to look right on a 5090
// and wrong on the PPU.
// Expressed as a target number of WARPS, because that is the quantity that has to be held. Box, gs=32
// affine, cold, the 2x2 of threads-per-block x SPLIT_K:
//
//   shape             (64,16) base   (32,16)   (32,32)
//   N4096  K4096         13.43        12.78     11.15   -17%
//   N14336 K4096         38.70        37.43     26.29   -32%   (base auto-picked sk=4, only 896 blocks)
//   N4096  K14336        35.73        34.63     26.54   -26%
//
// Two quantities move opposite ways and both have to be held:
//   flushes   ~ groups x WARPS PER GROUP    T=32 halves it (one warp owns the group's 2 ktiles at gs=32)
//   occupancy ~ blocks x WARPS PER BLOCK    T=32 halves that too
// The (32,16) column is the control and shows why it must be both: alone, T=32 buys 2-3%, because halving
// the flushes and halving residency nearly cancel. Restoring the warp count with sk=32 is what cashes it in.
// Symmetrically (64,32) LOST on the box, 33.26 -> 36.76: it paid the extra reduce for occupancy it did not
// need and bought no reduction in flushes.
//
// So: target warps. 2048 is the configuration already verified at T=64/sk=16 (1024 blocks x 2 warps), and
// it now yields sk=32 at T=32 on its own. GEMV_TARGET_BLOCKS overrides, in blocks, for sweeping.
static int gemv_target_blocks() {
    if (const char* e = getenv("GEMV_TARGET_BLOCKS")) { int v = atoi(e); if (v > 0) return v; }
    return 2048 / (GEMV_THREADS / 32);
}
#ifdef GEMV_FUSED_REDUCE
static const int GEMV_MAX_SPLIT_K = 8;    // fused contention grows with sk
#else
static const int GEMV_MAX_SPLIT_K = 32;   // separate reduce has no contention
#endif
// SPLIT_K sets the grid: blocks = (N/64) * SPLIT_K. That is what ACHIEVED occupancy is made of, and it was
// the real limit -- acu reported theoretical 62.5% (88 registers) but achieved 41.3%, and 1024 blocks over
// 72 CUs is 14.2 blocks = 28.4 of 64 warps per CU = 44%, which is the achieved figure. So the registers set
// a ceiling the kernel never reached; too few blocks set the floor it sat on.
//
// kt_per_group belongs in this decision and was missing. A slice that does not hold a whole number of groups
// makes kt_begin land mid-group, and the kernel's g = g0/kt_per_group is then simply the wrong scale row --
// silently wrong results, not a rejected shape. GEMV_SPLIT_K=32 at gs=128 hits exactly that (28 ktiles per
// slice against a group of 8); the auto heuristic only ever avoided it by accident.
//
// It also cuts the other way: gs=32 has the SMALLEST group and so the loosest divisibility, meaning it can
// take the most slices -- which is what it needs, since it was the case starved for blocks.
// n_rows is part of the grid and was missing from this decision. grid = (N/64) * SPLIT_K * n_rows, so at
// MoE decode the row dimension already supplies 8x the blocks and choosing SPLIT_K as if it did not cuts
// the work per block by the same factor. Measured: the dense case reads 29.4 MB with 1024 blocks (28.7 KB
// each) and hits 88% HBM; MoE read 8.4 MB with 4096 blocks (2 KB each) and got 24.9%, on the SAME kernel.
// A block that issues four warp-loads cannot amortise its own epilogue.
static int auto_split_k(int N, int K, int kt_per_group, int n_rows = 1) {
    const int cols = (N / N_PER_BLOCK) * (n_rows > 0 ? n_rows : 1), k_tiles = K / 16, tgt = gemv_target_blocks();
    // gs == -1 sets kt_per_group = k_tiles, i.e. ONE group spanning all of K. Splitting is still fine there:
    // g = g0/kt_per_group is 0 in every slice, so nothing straddles. Only a real per-group scale constrains.
    const int kpg = (kt_per_group >= k_tiles) ? 1 : kt_per_group;
    int best = 1;
    for (int sk = 1; sk <= GEMV_MAX_SPLIT_K; sk *= 2) {
        if (k_tiles % sk) continue;
        const int kts = k_tiles / sk;                    // ktiles per slice
        if (kts % (GEMV_THREADS / 32)) continue;         // warps must divide the slice
        if (kts % kpg) continue;                         // groups must not straddle a slice boundary
        if ((long long) cols * sk > tgt) break;
        best = sk;
    }
    return best;
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

static int get_split_k(int N, int K, int kt_per_group, int n_rows = 1) {
    const char* e = getenv("GEMV_SPLIT_K"); int v = e ? atoi(e) : 0;
    return v > 0 ? v : auto_split_k(N, K, kt_per_group, n_rows);
}

// -1 = per-column scales; 128 = grouped, WHAT REAL QUANTIZED MODELS USE. Grouped scales vary along K, so they cannot be
// applied once at the end -- the kernel folds each group's partial into an fp32 accumulator, scaled by that group's s.
static int get_groupsize() { const char* e = getenv("GEMV_GROUPSIZE"); return e ? atoi(e) : -1; }
static int SPLIT_K = 8;

// kt_per_group = groupsize/16, or k_tiles when groupsize == -1 (one group spanning all of K -> per-column scales).
static void launch(const int4* B, const half* A, const half* s, const half* mn, float* partial, half* C,
                   int* counter, int N, int K, int kt_per_group) {
    dim3 grid(N / N_PER_BLOCK, SPLIT_K);
#ifdef GEMV_AIU
    const int smem = 4 * kt_per_group * 64 * (int) sizeof(half);   // 4 cubes x [kt_per_group][64] b16
    // The AIU variant stays symmetric-only: it stages B through a different path and the min term would need
    // its own staging. Fail loudly rather than silently dropping the term.
    if (mn) { printf("  GEMV_AIU does not support affine (Q4_K min) yet\n"); exit(1); }
    #define GEMV_CASE(SK) case SK: gemv_w4a16_aiu<SK><<<grid, GEMV_THREADS, smem>>>(B, A, s, partial, C, counter, N, K, kt_per_group); break;
#else
    #define GEMV_ARGS partial, C, counter, N, K, kt_per_group
    #define GEMV_CASE(SK) case SK: if (mn) gemv_w4a16<SK, true ><<<grid, GEMV_THREADS>>>(B, A, s, mn, GEMV_ARGS); \
                                   else    gemv_w4a16<SK, false><<<grid, GEMV_THREADS>>>(B, A, s, nullptr, GEMV_ARGS); break;
#endif
    switch (SPLIT_K) {
        GEMV_CASE(1) GEMV_CASE(2) GEMV_CASE(4) GEMV_CASE(8) GEMV_CASE(16) GEMV_CASE(32)
        default: printf("  GEMV_SPLIT_K=%d unsupported (1/2/4/8/16/32)\n", SPLIT_K); exit(1);
    }
    #undef GEMV_CASE
#ifndef GEMV_FUSED_REDUCE
    // one reduce over ALL rows: partial is [rows][SPLIT_K][N], C is [rows][N]
    if (SPLIT_K > 1) gemv_reduce<<<(N + 255) / 256, 256>>>(partial, C, N, SPLIT_K);
#endif
    #undef GEMV_ARGS
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
    const int gs = get_groupsize();
    if (gs != -1 && (gs % 16 || K % gs)) { printf("  GEMV_GROUPSIZE=%d unsupported (need %%16==0, K%%gs==0)\n", gs); return -1; }
    // GEMV_SCALE_BUG=1 reproduces the ORIGINAL bug: pass kt_per_group = k_tiles even when gs=128, so the kernel sees a
    // single group and applies only s[0][n] -- a per-column scale on a grouped problem. It must MISMATCH, which is how
    // we know the test is not blind. (The kernel really did this, and its test set every scale to 1.0 so nothing showed.)
    int kt_per_group = (gs == -1 || getenv("GEMV_SCALE_BUG")) ? (K / 16) : (gs / 16);
    // GEMV_KTPG=<n> overrides the GROUP STRUCTURE while leaving the scale DATA at whatever gs says. The
    // result is then numerically WRONG (the kernel applies one group's scale across n ktiles of data), so
    // this is a timing probe only, in the same spirit as GEMV_BW_ONLY -- correctness checking is disabled.
    //
    // It exists because three explanations for the +67% gs=32 costs on PPU have now been refuted: the
    // scale-load ratio (_scale_perm made it WORSE), the main_n unroll bypass (the gs=64 control did not
    // show the predicted cliff), and the flush's arithmetic (cutting 2/3 of v.cnvt moved nothing, even
    // though acu had it as the most-executed opcode at +295%). What all three share is that they assume
    // the cost is in the WORK. This isolates the other possibility: the group STRUCTURE itself -- one
    // serialization point per group, and only main_n>0 keeps more than one B load in flight.
    //
    //   GEMV_GROUPSIZE=32 GEMV_KTPG=8  -> gs=32's scale traffic and layout, gs=128's loop structure
    // If that lands near gs=128's 19.9 us, the cost is entirely structural and no amount of trimming the
    // per-group work will touch it. If it stays near 33 us, the structure is innocent and the scale data
    // itself is what costs -- and all three refuted hypotheses were looking at the wrong half of it.
    bool ktpg_probe = false;
    if (const char* e = getenv("GEMV_KTPG")) { kt_per_group = atoi(e); ktpg_probe = true; }

    // SPLIT_K is chosen AFTER kt_per_group, because it depends on it -- see auto_split_k.
    SPLIT_K = get_split_k(N, K, kt_per_group);
    if (N % 64 || K % (16 * SPLIT_K) || (K / 16 / SPLIT_K) % (GEMV_THREADS / 32)) {
        printf("  N=%d K=%d SPLIT_K=%d unsupported (N%%64, K%%%d, ktiles/slice %% 4)\n", N, K, SPLIT_K, 16 * SPLIT_K); return -1; }
    // A slice must hold a whole number of GROUPS. Otherwise kt_begin lands mid-group and the kernel's
    // g = g0/kt_per_group reads the wrong scale row -- wrong numbers, not a rejected shape. Previously
    // unchecked, so an explicit GEMV_SPLIT_K=32 at gs=128 computed garbage that only the correctness test
    // caught. Reject it here instead.
    if (kt_per_group < K / 16 && (K / 16 / SPLIT_K) % kt_per_group) {
        printf("  SPLIT_K=%d incompatible with gs=%d: %d ktiles/slice is not a multiple of the group's %d "
               "-- groups would straddle slice boundaries\n", SPLIT_K, gs, K / 16 / SPLIT_K, kt_per_group);
        return -1; }
    const int GROUPS = (gs == -1) ? 1 : (K / gs);

    // AFFINE: GGUF Q4_1/Q4_K carry a per-group min. The host folds m' = m + 8s so the symmetric (q-8) dequant is
    // reused, exactly as marlin_repack_q4k does -- see marlin_gguf_ppu.cuh. Mins are drawn with BOTH signs and a
    // magnitude comparable to s*|q|, so a kernel that dropped the term cannot pass by the term being small.
    const bool affine = getenv("GEMV_AFFINE") != nullptr;
    // GEMV_AFFINE_DROP=1 keeps the affine REFERENCE but runs the SYMMETRIC kernel, i.e. drops the min term. It must
    // MISMATCH. Same guard as GEMV_SCALE_BUG, and for the same reason: this file's scales were once all 1.0, so a
    // kernel that ignored them passed. A test that has never been shown to fail proves nothing.
    const bool affine_drop = getenv("GEMV_AFFINE_DROP") != nullptr;
    std::vector<half> hA(K), hS((size_t) GROUPS * N), hM((size_t) GROUPS * N), hC(N);
    std::vector<int> Bdeq((size_t) N * K), hB((size_t) (K / 16) * (N / 2) * 4);
    srand(1234);
    for (auto& x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto& x : Bdeq) x = (rand() & 0xf) - 8;
    // NON-TRIVIAL scales, distinct per (group, column). They used to be all 1.0 -- the multiplicative identity -- so a
    // kernel that dropped the scale entirely passed anyway, and a kernel that applied a per-COLUMN scale to a GROUPED
    // problem also passed. Both bugs were live in this file.
    for (auto& x : hS) x = __float2half(0.5f + (rand() % 100) / 100.0f);
    for (auto& x : hM) x = __float2half(affine ? (rand() % 100 - 50) / 25.0f : 0.0f);
    pack_B(Bdeq, hB, N, K);

    const size_t b_bytes = hB.size() * 4;
    const int rot = get_rot(b_bytes);
    int4 *dB; half *dA, *dS, *dC; float* dP; int* dCnt;
    CUDA_CHECK(cudaMalloc(&dB, b_bytes * rot));
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
    CUDA_CHECK(cudaMalloc(&dS, hS.size() * 2));
    half* dM; CUDA_CHECK(cudaMalloc(&dM, hM.size() * 2));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * 2));
    CUDA_CHECK(cudaMalloc(&dP, (size_t) 32 * N * 4));   // max SPLIT_K
    CUDA_CHECK(cudaMalloc(&dCnt, (size_t) (N / N_PER_BLOCK) * 4));
    CUDA_CHECK(cudaMemset(dCnt, 0, (size_t) (N / N_PER_BLOCK) * 4));   // the winning CTA rearms it each launch
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_bytes, cudaMemcpyHostToDevice));
    for (int r = 1; r < rot; r++)   // identical copies; correctness is unaffected, cache residency is not
        CUDA_CHECK(cudaMemcpy((char*) dB + (size_t) r * b_bytes, dB, b_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dM, hM.data(), hM.size() * 2, cudaMemcpyHostToDevice));
#ifdef GEMV_SCALE_PERM
    // Marlin's _scale_perm, host side, under the SAME #ifdef as the kernel's int4 read so the two cannot
    // drift apart. The CPU reference below keeps using the UNPERMUTED hS/hM, so a wrong permutation shows
    // up as a mismatch instead of cancelling against itself.
    {
        auto perm = [&](const std::vector<half>& v) {
            std::vector<half> t(v.size());
            for (int g = 0; g < GROUPS; g++)
                for (int c0 = 0; c0 < N; c0 += 64)
                    for (int i = 0; i < 8; i++)
                        for (int j = 0; j < 8; j++)
                            t[(size_t) g * N + c0 + 8 * i + j] = v[(size_t) g * N + c0 + i + 8 * j];
            return t;
        };
        const std::vector<half> sp = perm(hS), mp = perm(hM);
        CUDA_CHECK(cudaMemcpy(dS, sp.data(), sp.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dM, mp.data(), mp.size() * 2, cudaMemcpyHostToDevice));
    }
#endif

    if (!getenv("GEMV_NCU")) { launch(dB, dA, dS, (affine && !affine_drop) ? dM : nullptr, dP, dC, dCnt, N, K, kt_per_group); CUDA_CHECK(cudaDeviceSynchronize()); }

    if (getenv("GEMV_NCU")) check = false;
    if (ktpg_probe) { printf("  (GEMV_KTPG=%d: group STRUCTURE decoupled from the scale data -- results are WRONG on purpose, timing only)\n", kt_per_group); check = false; }
#ifdef GEMV_BW_ONLY
    if (check) printf("  (GEMV_BW_ONLY: dequant replaced by a stub -- results are WRONG on purpose, timing only)\n");
    check = false;
#endif
    // Verified BEFORE and AFTER the timing loop. The fused reduce rearms `counter` from inside the winning CTA, so a
    // leaked count would corrupt every launch after the first -- which a pre-bench check alone cannot see.
    // C[n] = sum_g [ s[g][n] * sum_{k in g} A[k]*q[n][k]  +  m'[g][n] * sum_{k in g} A[k] ]
    // The min term is written out as its own A-sum rather than folded into a dequantized weight, so the reference
    // does not reproduce the kernel's factorization -- a shared misreading of the rank-1 form cannot cancel.
    const int gsz = (gs == -1) ? K : gs;
    std::vector<double> ref(N);
    for (int n = 0; n < N; n++) {
        double total = 0;
        for (int g = 0; g < GROUPS; g++) {
            double acc = 0;
            double asum = 0;
            for (int k = g * gsz; k < (g + 1) * gsz; k++) {
                acc  += (double) __half2float(hA[k]) * Bdeq[(size_t) n * K + k];
                asum += (double) __half2float(hA[k]);
            }
            total += acc * __half2float(hS[(size_t) g * N + n]);
            if (affine) total += asum * __half2float(hM[(size_t) g * N + n]);
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
        launch(dB, dA, dS, (affine && !affine_drop) ? dM : nullptr, dP, dC, dCnt, N, K, kt_per_group);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  GEMV_NCU: single launch done (N=%d K=%d sk=%d, %d blocks) -- timing skipped\n",
               N, K, SPLIT_K, (N / N_PER_BLOCK) * SPLIT_K);
        cudaFree(dB); cudaFree(dA); cudaFree(dS); cudaFree(dM); cudaFree(dC); cudaFree(dP); cudaFree(dCnt);
        return 0;
    }

    auto Brot = [&](int i) { return (const int4*) ((char*) dB + (size_t) (i % rot) * b_bytes); };
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 10; i++) launch(Brot(i), dA, dS, (affine && !affine_drop) ? dM : nullptr, dP, dC, dCnt, N, K, kt_per_group);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) launch(Brot(i), dA, dS, (affine && !affine_drop) ? dM : nullptr, dP, dC, dCnt, N, K, kt_per_group);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    if (check && !verify("post-bench")) { printf("  counter leaked across launches\n"); return -1; }

    const char* e = getenv("MARLIN_HBM_GBS");
    double peak = e ? atof(e) : 2700.0;
    // scales: one fp16 row per group per column, DOUBLED when affine (s and m'). At gs=32 this is 4x the gs=128
    // scale traffic, which is the term to watch -- B is unchanged, so any %HBM move has to come from here.
    double bytes = (double) N * K / 2 + (double) K * 2 + (double) N * 2
                 + (double) GROUPS * N * 2 * (affine ? 2 : 1);            // B + A + C + scales(+mins)
    double gbs = bytes / (ms * 1e6);
    printf("  M=1 %-3s N=%-5d K=%-5d : %8.2f us  %7.1f TFLOP/s  %7.0f GB/s  %5.1f%% HBM   (gs=%d, sk=%d, %d blocks, wset=%zu MB)\n",
           affine ? "aff" : "sym",
           N, K, ms * 1e3, 2.0 * N * K / (ms * 1e9), gbs, 100.0 * gbs / peak,
           gs, SPLIT_K, (N / N_PER_BLOCK) * SPLIT_K, (b_bytes * rot) >> 20);
    (void) 0;

    cudaFree(dB); cudaFree(dA); cudaFree(dS); cudaFree(dM); cudaFree(dC); cudaFree(dP); cudaFree(dCnt);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1e3;
}

// ---------------------------------------------------------------------------------------------------------------
// MoE DECODE -- ITS OWN KERNEL, because bolting a blockIdx.z onto the dense one is just running the GEMV
// n_rows times and exploits nothing about the MoE structure.
//
// What that version wasted, at batch 1 x top-8: all 8 rows belong to the SAME token, so they share one
// activation vector, yet each row re-read it; and each row paid a full epilogue (shuffle -> shared ->
// partial write) for the same 64 output columns, eight times over. A block did 8 KB of weight work and
// then a complete reduce.
//
// Here ONE block owns 64 columns x ALL rows: A is read once, the K loop runs per row over that row's
// expert, and the epilogue is paid once for the whole column block. Work per block goes from 8 KB to
// n_rows * 8 KB (64 KB at 8 rows, above the dense case's 28.7 KB), and the grid drops from
// (N/64)*sk*rows to (N/64)*sk.
//
// Rows sharing an expert also stop re-reading its weights: the loop can group them, though at batch 1 the
// draw is usually 8 distinct experts so that pays off only at larger batch.
#ifndef MOE_ROWS_PER_BLOCK
#define MOE_ROWS_PER_BLOCK 1
#endif

template <int SPLIT_K, bool AFFINE>
__global__ void __launch_bounds__(GEMV_THREADS)
moe_gemv_rows(const int4* __restrict__ B, const half* __restrict__ A, const half* __restrict__ s,
              const half* __restrict__ mn, const int* __restrict__ row_expert,
              const int* __restrict__ row_token, float* __restrict__ partial, half* __restrict__ C,
              int N, int K, int kt_per_group, int n_rows,
              long long b_expert_stride, long long s_expert_stride) {
    const int nb_group = blockIdx.x, slice = blockIdx.y;
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32, step = GEMV_THREADS / 32;
    const int k_tiles = K / 16, kt_per_slice = k_tiles / SPLIT_K;
    const int kt_begin = slice * kt_per_slice, kt_end = kt_begin + kt_per_slice;
    const int b_stride = N / 2, lane_k = (lane % 4) * 2;

    __shared__ float sm[GEMV_THREADS / 32][N_PER_BLOCK];

    // ROWS PER BLOCK is a knob, not a choice between two kernels. RPB=1 puts every row in its own block
    // (maximum parallelism, A re-read and epilogue paid per row); RPB=n_rows gives one block all of them
    // (A read once, one epilogue, but the grid loses the row dimension entirely). Measured on ppu001,
    // N=1024 K=2048, 8 rows: RPB=1 gave 10.70 us at 1024 blocks, RPB=8 gave 17.94 at 512 -- so the A and
    // epilogue savings are worth LESS than the parallelism they cost, and the right value is in between.
    // GEMV_MAX_SPLIT_K=32 caps this grid at 16*32 = 512 blocks, which is why RPB=8 cannot buy them back.
    const int r_lo = (int) blockIdx.z * MOE_ROWS_PER_BLOCK;
    const int r_hi = min(r_lo + MOE_ROWS_PER_BLOCK, n_rows);
    for (int r = r_lo; r < r_hi; r++) {
        const int e = row_expert[r];
        const int4* Be = B + e * b_expert_stride;
        const half* se = s + e * s_expert_stride;
        const half* me = AFFINE ? mn + e * s_expert_stride : nullptr;
        const half* At = A + (long long) row_token[r] * K;

        // KTILE-MAJOR, not group-major. With gs=32 a group is 2 ktiles, so a group-major loop gives each
        // warp kt_per_group/step of them: one at 64 threads, and at 256 threads six of eight warps get
        // NOTHING while still joining the cross-warp reduce. Measured at a fixed 1024 blocks: 2048 warps
        // 10.86 us, 4096 warps 18.08, 8192 warps 24.95 -- more warps made it monotonically worse, because
        // the group cannot feed them, not because warps are unhelpful.
        //
        // Walking ktiles across group boundaries fixes both: warps divide the whole SLICE (16 ktiles at
        // sk=8) so none starve, and the unroll is no longer capped by what one group holds. Only the
        // accumulate and the flush care about the group, so the flush moves to the boundary crossing.
        float facc[4][2] = {{0.f}};
        {
            const int my_first = kt_begin + wid;
            const int my_n = (kt_end - kt_begin - wid + step - 1) / step;
            int cur_g = -1;
            half2 hacc[4][2]; half sg[4][2], mg[4][2]; half2 asum2 = __float2half2_rn(0.f);
            #define MR_OPEN(gg) do { cur_g = (gg);                                                  \
                _Pragma("unroll") for (int j = 0; j < 4; j++) { hacc[j][0] = __float2half2_rn(0.f);  \
                                                               hacc[j][1] = __float2half2_rn(0.f); }\
                asum2 = __float2half2_rn(0.f);                                                       \
                _Pragma("unroll") for (int j = 0; j < 4; j++)                                        \
                    _Pragma("unroll") for (int h = 0; h < 2; h++) {                                  \
                        const int c_ = (nb_group * 4 + j) * 16 + lane / 4 + 8 * h;                   \
                        sg[j][h] = se[(long long) cur_g * N + c_];                                   \
                        if (AFFINE) mg[j][h] = me[(long long) cur_g * N + c_]; } } while (0)
            #define MR_FLUSH() do { if (cur_g >= 0) {                                                \
                float as_ = 0.f;                                                                     \
                if (AFFINE) { const float2 fa = __half22float2(asum2); as_ = fa.x + fa.y; }           \
                _Pragma("unroll") for (int j = 0; j < 4; j++)                                         \
                    _Pragma("unroll") for (int h = 0; h < 2; h++) {                                   \
                        const float2 f = __half22float2(hacc[j][h]);                                  \
                        facc[j][h] += (f.x + f.y) * __half2float(sg[j][h]);                           \
                        if (AFFINE) facc[j][h] += as_ * __half2float(mg[j][h]); } } } while (0)
            #define MR_BODY(kt_, q4_, aa_, ab_) do {                                                 \
                const int g_ = (kt_) / kt_per_group;                                                 \
                if (g_ != cur_g) { MR_FLUSH(); MR_OPEN(g_); }                                        \
                const half2 aa = (aa_), ab = (ab_);                                                  \
                if (AFFINE) asum2 = __hadd2(asum2, __hadd2(aa, ab));                                 \
                const int qs[4] = { (q4_).x, (q4_).y, (q4_).z, (q4_).w };                            \
                _Pragma("unroll") for (int j = 0; j < 4; j++) {                                      \
                    const FragB b0 = dequant(qs[j]), b1 = dequant(qs[j] >> 8);                       \
                    hacc[j][0] = __hfma2(b0.v[0], aa, hacc[j][0]);                                   \
                    hacc[j][0] = __hfma2(b0.v[1], ab, hacc[j][0]);                                   \
                    hacc[j][1] = __hfma2(b1.v[0], aa, hacc[j][1]);                                   \
                    hacc[j][1] = __hfma2(b1.v[1], ab, hacc[j][1]); } } while (0)

            int kt = my_first, done = 0;
            for (; done + GEMV_UNROLL <= my_n; done += GEMV_UNROLL) {
                int4 q4[GEMV_UNROLL]; half2 av0[GEMV_UNROLL], av1[GEMV_UNROLL];
                #pragma unroll
                for (int u = 0; u < GEMV_UNROLL; u++) {
                    const int ktu = kt + u * step, kb = ktu * 16 + lane_k;
                    q4[u]  = Be[(long long) b_stride * ktu + nb_group * 32 + lane];
                    av0[u] = *reinterpret_cast<const half2*>(&At[kb]);
                    av1[u] = *reinterpret_cast<const half2*>(&At[kb + 8]);
                }
                #pragma unroll
                for (int u = 0; u < GEMV_UNROLL; u++) MR_BODY(kt + u * step, q4[u], av0[u], av1[u]);
                kt += GEMV_UNROLL * step;
            }
            for (; done < my_n; done++) {
                const int kb = kt * 16 + lane_k;
                MR_BODY(kt, Be[(long long) b_stride * kt + nb_group * 32 + lane],
                        *reinterpret_cast<const half2*>(&At[kb]),
                        *reinterpret_cast<const half2*>(&At[kb + 8]));
                kt += step;
            }
            MR_FLUSH();
            #undef MR_BODY
            #undef MR_FLUSH
            #undef MR_OPEN
        }
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                facc[j][h] += __shfl_xor_sync(0xffffffff, facc[j][h], 1);
                facc[j][h] += __shfl_xor_sync(0xffffffff, facc[j][h], 2);
            }
        if (lane % 4 == 0)
            #pragma unroll
            for (int j = 0; j < 4; j++)
                #pragma unroll
                for (int h = 0; h < 2; h++) sm[wid][j * 16 + lane / 4 + 8 * h] = facc[j][h];
        __syncthreads();
        for (int c = threadIdx.x; c < N_PER_BLOCK; c += GEMV_THREADS) {
            float v = 0.f;
            #pragma unroll
            for (int w = 0; w < GEMV_THREADS / 32; w++) v += sm[w][c];
            if (SPLIT_K == 1) C[(long long) r * N + nb_group * N_PER_BLOCK + c] = __float2half(v);
            else partial[((long long) r * SPLIT_K + slice) * N + nb_group * N_PER_BLOCK + c] = v;
        }
        __syncthreads();
    }
}

static void launch_moe(const int4* B, const half* A, const half* s, const half* mn,
                       const int* row_expert, const int* row_token, float* partial, half* C,
                       int N, int K, int kt_per_group, int n_rows,
                       long long b_str, long long s_str) {
    const int rz = (n_rows + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK;
    dim3 grid(N / N_PER_BLOCK, SPLIT_K, rz);
#define MOE_CASE(SK) case SK: if (mn) moe_gemv_rows<SK, true ><<<grid, GEMV_THREADS>>>(B,A,s,mn,row_expert,row_token,partial,C,N,K,kt_per_group,n_rows,b_str,s_str); \
                              else    moe_gemv_rows<SK, false><<<grid, GEMV_THREADS>>>(B,A,s,nullptr,row_expert,row_token,partial,C,N,K,kt_per_group,n_rows,b_str,s_str); break;
    switch (SPLIT_K) {
        MOE_CASE(1) MOE_CASE(2) MOE_CASE(4) MOE_CASE(8) MOE_CASE(16) MOE_CASE(32)
        default: printf("  MoE SPLIT_K=%d unsupported\n", SPLIT_K); exit(1);
    }
#undef MOE_CASE
    if (SPLIT_K > 1)
        gemv_reduce<<<(int) (((long long) n_rows * N + 255) / 256), 256>>>(partial, C, N, SPLIT_K, n_rows);
}

// ---------------------------------------------------------------------------------------------------------------
// MoE DECODE. Same kernel, same B packing, same scales -- only a row dimension and an expert lookup. This
// lives here rather than in a separate file because a separate file is what cost the last stretch: every
// problem re-derived there (the main_n unroll cliff, a bench measuring cache, split-K slices needing whole
// scale groups, the fused-vs-separate reduce) was already solved and commented in THIS one.
//   ./gemv_w4a16 moe [N] [K] [tokens] [topk] [experts]
static int moe_case(int N, int K, int tokens, int topk, int n_experts, bool check) {
    const int gs = get_groupsize() < 0 ? 32 : get_groupsize();
    const int G = K / gs, n_rows = tokens * topk;
    // n_rows is NOT in this grid any more. moe_gemv_rows gives one block all the rows, so the grid is
    // (N/64) * SPLIT_K -- passing n_rows here (which I had just added for the z-dimension version) made
    // auto_split_k believe there were 8x the blocks and pick sk=8, giving 128 blocks on 72 CUs and 44 us.
    // Trading the row dimension for epilogue amortisation means SPLIT_K has to buy those blocks back.
    // the grid's row dimension is ceil(n_rows / MOE_ROWS_PER_BLOCK), so that is what auto_split_k must see
    SPLIT_K = get_split_k(N, K, gs / 16, (n_rows + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK);
    if (N % 64 || K % (16 * SPLIT_K) || (K / 16 / SPLIT_K) % (GEMV_THREADS / 32)) {
        printf("  MoE N=%d K=%d SPLIT_K=%d unsupported\n", N, K, SPLIT_K); return 2; }

    srand(4321);
    std::vector<int> rexp(n_rows), rtok(n_rows);
    for (int t = 0; t < tokens; t++) for (int k = 0; k < topk; k++) {
        rexp[t * topk + k] = rand() % n_experts; rtok[t * topk + k] = t; }
    std::vector<half> hA((size_t) tokens * K), hS((size_t) n_experts * G * N), hM((size_t) n_experts * G * N);
    const bool affine = getenv("GEMV_AFFINE") != nullptr;
    for (auto& x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (size_t i = 0; i < hS.size(); i++) {
        hS[i] = __float2half(0.5f + (rand() % 100) / 100.0f);
        hM[i] = __float2half(affine ? ((rand() % 100) - 50) / 25.0f : 0.0f);
    }
    std::vector<int> Bdeq((size_t) n_experts * N * K), hB((size_t) n_experts * (K / 16) * (N / 2) * 4);
    for (auto& x : Bdeq) x = (rand() & 0xf) - 8;
    const size_t bw = (size_t) (K / 16) * (N / 2) * 4;
    for (int e = 0; e < n_experts; e++) {
        std::vector<int> one(Bdeq.begin() + (size_t) e * N * K, Bdeq.begin() + (size_t) (e + 1) * N * K);
        std::vector<int> outv(bw);
        pack_B(one, outv, N, K);
        std::copy(outv.begin(), outv.end(), hB.begin() + (size_t) e * bw);
    }

    int4* dB; half *dA, *dS, *dM, *dC; float* dP; int *dCnt, *dRe, *dRt;
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * 4));        CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
    CUDA_CHECK(cudaMalloc(&dS, hS.size() * 2));        CUDA_CHECK(cudaMalloc(&dM, hM.size() * 2));
    CUDA_CHECK(cudaMalloc(&dC, (size_t) n_rows * N * 2));
    CUDA_CHECK(cudaMalloc(&dP, (size_t) n_rows * 32 * N * 4));
    CUDA_CHECK(cudaMalloc(&dCnt, (size_t) n_rows * (N / N_PER_BLOCK) * 4));
    CUDA_CHECK(cudaMalloc(&dRe, n_rows * 4));          CUDA_CHECK(cudaMalloc(&dRt, n_rows * 4));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dM, hM.data(), hM.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRe, rexp.data(), n_rows * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRt, rtok.data(), n_rows * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dCnt, 0, (size_t) n_rows * (N / N_PER_BLOCK) * 4));

    // UNITS DIFFER because the pointer types do: B is const int4*, s is const half*. Passing the scale
    // stride in int4 units made it 8x too small, so every expert past the first read the wrong scale rows
    // -- rel 0.33, i.e. mostly right, which is what a wrong-but-plausible scale looks like.
    const long long b_str = (long long) (K / 16) * (N / 2);        // int4  per expert
    const long long s_str = (long long) G * N;                     // halves per expert
    auto run = [&] { launch_moe(dB, dA, dS, affine ? dM : nullptr, dRe, dRt, dP, dC,
                                N, K, gs / 16, n_rows, b_str, s_str); };
    run(); CUDA_CHECK(cudaDeviceSynchronize());
    // GEMV_NCU was honoured in bench_one but not here, so `moe` put its two correctness cases plus warmup
    // plus a 50-iteration loop into the capture -- six or more kernels where the hook exists to leave one.
    if (getenv("GEMV_NCU")) {
        printf("  GEMV_NCU: single launch rows=%d E=%d N=%d K=%d sk=%d rpb=%d -- timing skipped\n",
               n_rows, n_experts, N, K, SPLIT_K, MOE_ROWS_PER_BLOCK);
        cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dM);cudaFree(dC);cudaFree(dP);cudaFree(dCnt);cudaFree(dRe);cudaFree(dRt);
        return 0;
    }

    if (check) {
        std::vector<half> got((size_t) n_rows * N);
        CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * 2, cudaMemcpyDeviceToHost));
        double ma = 0, rm = 0;
        for (int r = 0; r < n_rows; r++) {
            const int e = rexp[r], t = rtok[r];
            for (int n = 0; n < N; n++) {
                double acc = 0;
                for (int k = 0; k < K; k++) {
                    const size_t si = (size_t) e * G * N + (size_t) (k / gs) * N + n;
                    acc += (double) __half2float(hA[(size_t) t * K + k]) *
                           ((double) Bdeq[((size_t) e * N + n) * K + k] * __half2float(hS[si])
                            + (affine ? __half2float(hM[si]) : 0.0));
                }
                ma = fmax(ma, fabs(__half2float(got[(size_t) r * N + n]) - acc)); rm = fmax(rm, fabs(acc));
            }
        }
        const double rel = ma / (rm + 1e-9);
        printf("  MoE %-3s rows=%-4d E=%-4d N=%-5d K=%-5d gs=%-3d sk=%-2d  rel %.2e -> %s\n",
               affine ? "aff" : "sym", n_rows, n_experts, N, K, gs, SPLIT_K, rel, rel < 3e-2 ? "MATCH" : "MISMATCH");
        cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dM);cudaFree(dC);cudaFree(dP);cudaFree(dCnt);cudaFree(dRe);cudaFree(dRt);
        return rel < 3e-2 ? 0 : 1;
    }
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; i++) run(); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(a); for (int i = 0; i < 50; i++) run(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 50;
    std::set<int> uniq(rexp.begin(), rexp.end());
    const double wb = (double) uniq.size() * N * K / 2.0;
    const int blocks_ = (N / N_PER_BLOCK) * SPLIT_K * ((n_rows + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK);
    printf("  MoE %-3s rows=%-4d E=%-4d N=%-5d K=%-5d sk=%-2d rpb=%-2d blk=%-5d %5.1f KB/blk | %7.2f us | %6.0f GB/s | %5.1f%% HBM\n",
           affine ? "aff" : "sym", n_rows, n_experts, N, K, SPLIT_K, MOE_ROWS_PER_BLOCK, blocks_, wb / blocks_ / 1024.0, ms * 1e3,
           wb / (ms * 1e6), 100.0 * wb / (ms * 1e6) / (getenv("MARLIN_HBM_GBS") ? atof(getenv("MARLIN_HBM_GBS")) : 2766.0));
    cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dM);cudaFree(dC);cudaFree(dP);cudaFree(dCnt);cudaFree(dRe);cudaFree(dRt);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && argv[1][0] == 'm') {    // moe [N] [K] [tokens] [topk] [experts]
        const int N = argc > 2 ? atoi(argv[2]) : 1024, K = argc > 3 ? atoi(argv[3]) : 2048;
        const int tok = argc > 4 ? atoi(argv[4]) : 1, tk = argc > 5 ? atoi(argv[5]) : 8;
        const int E = argc > 6 ? atoi(argv[6]) : 256;
        printf("MoE decode GEMV -- dedicated kernel (one block owns 64 cols x ALL rows)\n");
        if (getenv("GEMV_NCU")) {   // profiling: ONE kernel in the capture, nothing else
            printf("  GEMV_NCU set: correctness cases skipped so the capture holds one kernel.\n"
                   "  Verify first with the same args and no GEMV_NCU.\n");
            return moe_case(N, K, tok, tk, E, false);
        }
        int bad = 0;
        bad |= moe_case(256, 512, 1, 8, 8, true);      // correctness first, small E for the CPU reference
        bad |= moe_case(256, 512, 2, 8, 16, true);
        printf("%s\n", bad ? "SOME CASES FAILED" : "all MoE cases MATCH");
        if (bad) return 1;
        return moe_case(N, K, tok, tk, E, false);
    }
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
