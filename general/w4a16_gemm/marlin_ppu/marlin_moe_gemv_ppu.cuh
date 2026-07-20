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
             float* __restrict__ partial, half* __restrict__ C,
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
    for (int g0 = kt_begin; g0 < kt_end; g0 += kt_per_group) {
        const int g = g0 / kt_per_group, g1 = min(g0 + kt_per_group, kt_end);
        half2 hacc[4][2];
        #pragma unroll
        for (int j = 0; j < 4; j++) { hacc[j][0] = __float2half2_rn(0.f); hacc[j][1] = __float2half2_rn(0.f); }
        // this group's 8 scales, hoisted (same reason as the GEMM: the read otherwise sits on the flush)
        half sg[4][2];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++)
                sg[j][h] = se[(long long) g * N + (nbg * 4 + j) * 16 + lane / 4 + 8 * h];

        #define MOEV_BODY(q4_, aa_, ab_) do {                                        \
            const half2 aa = (aa_), ab = (ab_);                                       \
            const int qs[4] = { (q4_).x, (q4_).y, (q4_).z, (q4_).w };                 \
            _Pragma("unroll")                                                          \
            for (int j = 0; j < 4; j++) {                                             \
                half2 b0[2], b1[2];                                                   \
                dq4(qs[j], b0); dq4(qs[j] >> 8, b1);                                  \
                hacc[j][0] = __hfma2(b0[0], aa, hacc[j][0]);                          \
                hacc[j][0] = __hfma2(b0[1], ab, hacc[j][0]);                          \
                hacc[j][1] = __hfma2(b1[0], aa, hacc[j][1]);                          \
                hacc[j][1] = __hfma2(b1[1], ab, hacc[j][1]);                          \
            }                                                                          \
        } while (0)

        // ktiles THIS thread owns in this group, then U at a time with ALL loads issued before any use.
        const int n_kt = (g1 - g0 > wid) ? ((g1 - g0 - wid + step - 1) / step) : 0;
        const int main_n = (n_kt / U) * U;
        int kt = g0 + wid, i = 0;
        for (; i < main_n; i += U) {
            int4 q4[U]; half2 av0[U], av1[U];
            #pragma unroll
            for (int u = 0; u < U; u++) {
                const int ktu = kt + u * step, kb = ktu * 16 + lane_k;
                q4[u]  = Be[(long long) b_stride * ktu + nbg * 32 + lane];
                av0[u] = *reinterpret_cast<const half2*>(&At[kb]);
                av1[u] = *reinterpret_cast<const half2*>(&At[kb + 8]);
            }
            #pragma unroll
            for (int u = 0; u < U; u++) MOEV_BODY(q4[u], av0[u], av1[u]);
            kt += U * step;
        }
        for (; i < n_kt; i++) {                    // tail when n_kt is not a multiple of U
            const int kb = kt * 16 + lane_k;
            MOEV_BODY(Be[(long long) b_stride * kt + nbg * 32 + lane],
                      *reinterpret_cast<const half2*>(&At[kb]),
                      *reinterpret_cast<const half2*>(&At[kb + 8]));
            kt += step;
        }
        #undef MOEV_BODY
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const float2 f = __half22float2(hacc[j][h]);
                facc[j][h] += (f.x + f.y) * __half2float(sg[j][h]);
            }
    }

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

    for (int c = threadIdx.x; c < MOEV_NPB; c += THREADS) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < THREADS / 32; w++) v += sm[w][c];
        if (SPLIT_K == 1) C[(long long) r * N + nbg * MOEV_NPB + c] = __float2half(v);
        else partial[((long long) r * SPLIT_K + slice) * N + nbg * MOEV_NPB + c] = v;
    }
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
