#ifndef MARLIN_MOE_GEMV_PPU_CUH
#define MARLIN_MOE_GEMV_PPU_CUH
// GGUF Q4_K MoE DECODE: a grouped GEMV, not the grouped GEMM.
//
// WHY A SEPARATE KERNEL. At decode a token routes to top-k experts, so each expert sees ONE row --
// total rows = tokens*topk, which is 8 at batch 1. The grouped GEMM (marlin_moe_aiu_ppu.cuh) is
// weight/compute structured and lands at 52% of the fp16 peak on prefill shapes; at one row per expert it
// would compute a whole BMR-row tile and mask all but one line away. The dense case already measured this
// exactly: at M=1 Marlin gets 16.8% HBM where the purpose-built GEMV gets ~52%. Decode is BANDWIDTH bound
// and the target is %HBM, not TFLOP/s -- the weights are 8 * N*K/2 bytes and the tensor cores are idle by
// construction (arithmetic intensity ~4 flop/byte at one row).
//
// Structure follows gemv_w4a16_ppu.cu, which is the box-verified decode shape: 64 columns per block, one
// int4 per lane covering exactly those 64, GEMV_THREADS=32 or 64, per-32 scales folded in per group. The
// only additions are a row dimension (blockIdx.z over the expanded (token, expert) pairs) and an expert
// lookup that offsets B and s. B stays in Marlin's packed layout, unchanged and unrepacked.
//
// AFFINE: split out, as everywhere else. The min term is rank-1 -- C[r][n] += m'[e][g][n] * Asum[t][g] --
// and Asum is per TOKEN, so at decode it is a handful of values computed once. Not folded into this loop.
#include <cuda_fp16.h>

namespace marlin_moe_gemv_ppu {

#define MOEV_NPB 64            // columns per block; one int4 per lane spans exactly this many

// Marlin's dequant, verbatim: 4 instructions yield 4 halves.
__device__ __forceinline__ void dq4(int q, half2* out) {
    const int LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400;
    const int SUB = 0x64086408, MUL = 0x2c002c00, ADD = 0xd480d480;
    int lo = (q & LO) | EX, hi = (q & HI) | EX;
    out[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
    out[1] = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL),
                     *reinterpret_cast<const half2*>(&ADD));
}

// One block = (64-column chunk, k-slice, expanded row). row_expert[r] picks the weight matrix; row_token[r]
// picks the activation row, so several rows of one token share its A without duplicating it.
// U = independent int4 loads issued before ANY is consumed. acu on this kernel: Memory Dependency is the
// dominant stall while DRAM sits at 39.5%, i.e. LATENCY bound, not bandwidth bound -- too few loads in
// flight. Occupancy cannot fix it either: grid is 4096 warps against 4608 slots, so the kernel is under
// ONE wave, has no steady state, and averages 63% achieved against 100% theoretical. More blocks was
// already tried and turned over (the reduce takes the win back), so the parallelism has to come from
// INSIDE the warp.
//
// U and SPLIT_K pull against each other: sk raises block count but cuts ktiles per warp (sk=32 leaves 4,
// sk=64 leaves 2, and there is nothing left to unroll). So sweep them together rather than maximising
// either. The dense bf16 GEMV found U=2 best, but it has no dequant and an order of magnitude fewer
// instructions per byte, so its optimum is not assumed to carry.
template <int THREADS, int SPLIT_K, int U = 1>
__global__ void __launch_bounds__(THREADS)
moe_gemv_q4k(const int4* __restrict__ B, const half* __restrict__ A, const half* __restrict__ s,
             const int* __restrict__ row_expert, const int* __restrict__ row_token,
             float* __restrict__ partial, half* __restrict__ C, int* __restrict__ counter,
             int N, int K, int gs, int n_rows) {
    const int nbg = blockIdx.x, slice = blockIdx.y, r = blockIdx.z;
    if (r >= n_rows) return;
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32, step = THREADS / 32;

    const int e = row_expert[r], t = row_token[r];
    const long long b_expert = (long long) (K / 16) * (N / 2);          // int4 per expert
    const int4* Be = B + e * b_expert;
    const half* se = s + (long long) e * (K / gs) * N;
    const half* At = A + (long long) t * K;

    const int k_tiles = K / 16, kt_per_slice = k_tiles / SPLIT_K;
    const int kt_begin = slice * kt_per_slice, kt_end = kt_begin + kt_per_slice;
    const int b_stride = N / 2, lane_k = (lane % 4) * 2, kt_per_group = gs / 16;

    float facc[4][2] = {{0.f}};

    // LOADS ARE HOISTED ACROSS GROUP BOUNDARIES. The previous shape looped over scale groups and unrolled
    // INSIDE one, but a group is gs/16 = 2 ktiles and step is 1 at T=32, so in-flight loads were capped at
    // 2 no matter what U said -- and U=4 made main_n = (2/4)*4 = 0, silently skipping the hoisted path
    // entirely and running one load at a time. That is exactly the main_n cliff documented in
    // gemv_w4a16_ppu.cu, hit again here.
    //
    // Nothing about the B or A loads depends on the group; only the accumulate and the flush do. So walk
    // ktiles directly, issue U loads regardless of where the group boundary falls, and flush when the
    // group changes. In-flight loads are then U, not min(U, ktiles-per-group).
    const int my_first = kt_begin + wid;
    const int my_n = (kt_end - kt_begin - wid + step - 1) / step;
    int cur_g = -1;
    half2 hacc[4][2];
    half sg[4][2];
    #define MOEV_FLUSH() do {                                                            \
        if (cur_g >= 0) {                                                                \
            _Pragma("unroll")                                                             \
            for (int j = 0; j < 4; j++)                                                  \
                _Pragma("unroll")                                                         \
                for (int h = 0; h < 2; h++) {                                            \
                    const float2 f = __half22float2(hacc[j][h]);                         \
                    facc[j][h] += (f.x + f.y) * __half2float(sg[j][h]);                  \
                }                                                                         \
        } } while (0)
    #define MOEV_OPEN(g_) do {                                                           \
        cur_g = (g_);                                                                     \
        _Pragma("unroll")                                                                 \
        for (int j = 0; j < 4; j++) { hacc[j][0] = __float2half2_rn(0.f);                \
                                      hacc[j][1] = __float2half2_rn(0.f); }              \
        _Pragma("unroll")                                                                 \
        for (int j = 0; j < 4; j++)                                                      \
            _Pragma("unroll")                                                             \
            for (int h = 0; h < 2; h++)                                                  \
                sg[j][h] = se[(long long) cur_g * N + (nbg * 4 + j) * 16 + lane / 4 + 8 * h]; \
        } while (0)
    #define MOEV_BODY(kt_, q4_, aa_, ab_) do {                                           \
        const int g_ = (kt_) / kt_per_group;                                             \
        if (g_ != cur_g) { MOEV_FLUSH(); MOEV_OPEN(g_); }                                \
        const half2 aa = (aa_), ab = (ab_);                                              \
        const int qs[4] = { (q4_).x, (q4_).y, (q4_).z, (q4_).w };                        \
        _Pragma("unroll")                                                                 \
        for (int j = 0; j < 4; j++) {                                                    \
            half2 b0[2], b1[2];                                                          \
            dq4(qs[j], b0); dq4(qs[j] >> 8, b1);                                         \
            hacc[j][0] = __hfma2(b0[0], aa, hacc[j][0]);                                 \
            hacc[j][0] = __hfma2(b0[1], ab, hacc[j][0]);                                 \
            hacc[j][1] = __hfma2(b1[0], aa, hacc[j][1]);                                 \
            hacc[j][1] = __hfma2(b1[1], ab, hacc[j][1]);                                 \
        } } while (0)

    const int main_n = (my_n / U) * U;
    int kt = my_first, i = 0;
    for (; i < main_n; i += U) {
        int4 q4[U]; half2 av0[U], av1[U];
        #pragma unroll
        for (int u = 0; u < U; u++) {                 // ALL U issued before any is consumed
            const int ktu = kt + u * step, kb = ktu * 16 + lane_k;
            q4[u]  = Be[(long long) b_stride * ktu + nbg * 32 + lane];
            av0[u] = *reinterpret_cast<const half2*>(&At[kb]);
            av1[u] = *reinterpret_cast<const half2*>(&At[kb + 8]);
        }
        #pragma unroll
        for (int u = 0; u < U; u++) MOEV_BODY(kt + u * step, q4[u], av0[u], av1[u]);
        kt += U * step;
    }
    for (; i < my_n; i++) {
        const int kb = kt * 16 + lane_k;
        MOEV_BODY(kt, Be[(long long) b_stride * kt + nbg * 32 + lane],
                  *reinterpret_cast<const half2*>(&At[kb]),
                  *reinterpret_cast<const half2*>(&At[kb + 8]));
        kt += step;
    }
    MOEV_FLUSH();
    #undef MOEV_BODY
    #undef MOEV_OPEN
    #undef MOEV_FLUSH

    // fold the 4 lanes that share a column (they differ only in k phase), then across warps
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            facc[j][h] += __shfl_xor_sync(0xffffffff, facc[j][h], 1);
            facc[j][h] += __shfl_xor_sync(0xffffffff, facc[j][h], 2);
        }
    __shared__ float sm[THREADS / 32][MOEV_NPB];
    if (lane % 4 == 0)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++) sm[wid][j * 16 + lane / 4 + 8 * h] = facc[j][h];
    __syncthreads();

    if (SPLIT_K == 1) {
        for (int c = threadIdx.x; c < MOEV_NPB; c += THREADS) {
            float v = 0.f;
            #pragma unroll
            for (int w = 0; w < THREADS / 32; w++) v += sm[w][c];
            C[(long long) r * N + nbg * MOEV_NPB + c] = __float2half(v);
        }
        return;
    }
#ifndef MOEV_FUSED_REDUCE
    // SEPARATE reduce (default): each block writes its partial and a second kernel folds them IN PARALLEL.
    // Measured better than the fused path everywhere here, and the gap grows with SPLIT_K -- 9.11 vs 12.12
    // us at sk=16, 13.35 vs 39.75 at sk=64.
    for (int c = threadIdx.x; c < MOEV_NPB; c += THREADS) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < THREADS / 32; w++) v += sm[w][c];
        partial[((long long) r * SPLIT_K + slice) * N + nbg * MOEV_NPB + c] = v;
    }
    (void) counter;
    return;
#else
    // FUSED LAST-CTA REDUCE -- A REGRESSION HERE, kept only for comparison. It removes the second launch,
    // but the winning CTA then reads all SPLIT_K partials SERIALLY while every other CTA waits at the
    // fence and contends on the counter, where a separate kernel does that fold in parallel. It wins in
    // gemv_w4a16_ppu.cu because that kernel is already saturated and runs at low SPLIT_K; neither holds
    // here, and I carried the conclusion over without checking either premise. The separate reduce kernel is what made SPLIT_K turn over: acu shows the main
    // kernel still improving past sk=16 (8.65 us at 32 against 10.50 at 16) while the TOTAL regresses. And
    // more blocks is the only lever left -- the grid is 4096 warps against 72*64 = 4608 slots, i.e. under
    // one wave, with no steady state, so occupancy averages 63%. Removing the second launch lets sk rise
    // to 32/64 (8192/16384 blocks = several waves) without paying for it.
    //
    // Ordering follows CUDA's threadFenceReduction: __syncthreads() retires this block's partial stores,
    // thread 0 issues __threadfence() to make them device-visible, and only then takes a ticket. The block
    // that draws the last ticket is guaranteed to observe every other slice. `partial` is read through
    // volatile so the winner cannot serve another slice's value out of its own non-coherent L1.
    for (int c = threadIdx.x; c < MOEV_NPB; c += THREADS) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < THREADS / 32; w++) v += sm[w][c];
        partial[((long long) r * SPLIT_K + slice) * N + nbg * MOEV_NPB + c] = v;
    }
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) {
        __threadfence();
        const int ticket = atomicAdd(&counter[(long long) r * (N / MOEV_NPB) + nbg], 1);
        last = (ticket == SPLIT_K - 1);
        if (last) counter[(long long) r * (N / MOEV_NPB) + nbg] = 0;   // rearm for the next launch
    }
    __syncthreads();
    if (!last) return;
    const volatile float* pv = partial;
    for (int c = threadIdx.x; c < MOEV_NPB; c += THREADS) {
        float v = 0.f;
        for (int sl = 0; sl < SPLIT_K; sl++)
            v += pv[((long long) r * SPLIT_K + sl) * N + nbg * MOEV_NPB + c];
        C[(long long) r * N + nbg * MOEV_NPB + c] = __float2half(v);
    }
#endif
}

__global__ void moe_gemv_reduce(const float* __restrict__ partial, half* __restrict__ C,
                                int N, int SPLIT_K, int n_rows) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long) n_rows * N) return;
    const int r = (int) (i / N), n = (int) (i % N);
    float v = 0.f;
    for (int s = 0; s < SPLIT_K; s++) v += partial[((long long) r * SPLIT_K + s) * N + n];
    C[i] = __float2half(v);
}

}  // namespace marlin_moe_gemv_ppu
#endif
