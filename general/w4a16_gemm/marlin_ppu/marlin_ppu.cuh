// Plain-CUDA Marlin FP16xINT4 GEMM ported to the PPU (ppu001). The PPU fragment layouts are known, so we
// hand-write the fragment ops with the validated PPU building blocks from
// ppu_tests/test_marlin_n8x2_to_n16.cu. Structure follows the original Marlin (obsidian/marlin.md).
//
// PPU vs NVIDIA (established by the microbenches):
//   - A/B operand fragment layouts == NVIDIA -> A/B gmem->smem, cp.async pipeline, dispatcher, dequant, B packing
//     port UNCHANGED. A smem->reg uses element-wise frag_i/frag_j reads (plain As, no swizzle) instead of ldsm4.
//   - MMA: two NVIDIA m16n8 -> one ppu.mma.m16n16 (B = {b0,b1}, 8-float acc; test_marlin verified).
//   - C fragment layout DIFFERS -> reductions/write use acc_i/acc_j = C[lane/4 + 8*(l>=4)][lane%4 + 4*(l%4)].
//
// This header is standalone. Build the test with: make marlin_ppu_test.

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace marlin_ppu {

enum { ERR_PROB_SHAPE = 1, ERR_KERN_SHAPE = 2 };
constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

template <typename T, int n> struct Vec { T elems[n]; __device__ T & operator[](int i) { return elems[i]; } };
using FragA = Vec<half2, 4>;   // A operand: 4 half2 (m16k16)
using FragB = Vec<half2, 2>;   // one n8 B tile: 2 half2
using FragC = Vec<float, 8>;   // PPU n16 acc: 8 floats (was 2x FragC<4> on NVIDIA)
using FragS = Vec<half2, 1>;   // scale
using I4    = Vec<int, 4>;

// ---- PPU fragment (lane,l) -> (row,col), verified in test_marlin_n8x2_to_n16.cu ----
__device__ __forceinline__ int frag_i(int lane, int l) { return (lane / 4) + (l / 2) * 8; }   // A/B outer (m or n)
__device__ __forceinline__ int frag_j(int lane, int l) { return (lane % 4) + (l % 2) * 4; }   // A/B k half2-col
__device__ __forceinline__ int acc_i (int lane, int l) { return (lane / 4) + (((l >> 2) & 1) << 3); }   // C row
__device__ __forceinline__ int acc_j (int lane, int l) { return (lane % 4) + ((l % 4) << 2); }          // C col

// ---- lop3 + INT4 dequant (unchanged from NVIDIA / vLLM dequant.h) ----
template <int lut> __device__ inline int lop3(int a, int b, int c) {
    int r; asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(lut)); return r;
}
__device__ inline FragB dequant(int q) {   // int32 -> {n0-8, n2-8}, {n1-8, n3-8}
    const int LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400, SUB = 0x64086408, MUL = 0x2c002c00, ADD = 0xd480d480;
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX), hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    FragB fb;
    fb[0] = __hsub2(*reinterpret_cast<half2 *>(&lo), *reinterpret_cast<const half2 *>(&SUB));
    fb[1] = __hfma2(*reinterpret_cast<half2 *>(&hi), *reinterpret_cast<const half2 *>(&MUL), *reinterpret_cast<const half2 *>(&ADD));
    return fb;
}
__device__ inline void scale(FragB & fb, FragS & s, int i) {
    half2 ss = __half2half2(reinterpret_cast<__half *>(&s)[i]);
    fb[0] = __hmul2(fb[0], ss); fb[1] = __hmul2(fb[1], ss);
}

// ---- PPU tensor core: 2 NVIDIA n8 MMAs fused into 1 ppu.mma.m16n16k16 (test_marlin verified) ----
// a = A frag (4 half2 = 4 uint32); b0 = n8 cols 0-7, b1 = n8 cols 8-15 (each FragB = 2 half2 = 2 uint32).
// acc[8]: floats {0,1,4,5} = cols 0-7 tile, {2,3,6,7} = cols 8-15 tile.
__device__ inline void mma_n16(const FragA & a, const FragB & b0, const FragB & b1, FragC & acc) {
    const unsigned * A = reinterpret_cast<const unsigned *>(&a);
    unsigned B[4] = { *reinterpret_cast<const unsigned *>(&b0.elems[0]), *reinterpret_cast<const unsigned *>(&b0.elems[1]),
                      *reinterpret_cast<const unsigned *>(&b1.elems[0]), *reinterpret_cast<const unsigned *>(&b1.elems[1]) };
    float * d = acc.elems;
    asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]));
}

// ---- plain cp.async (== SM80, works verbatim on PPU) ----
__device__ inline void cp_async4(void * smem, const void * glob, bool pred = true) {
    unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %0, 0; @p cp.async.cg.shared.global [%1], [%2], 16; }\n"
                 :: "r"((int) pred), "r"(s), "l"(glob));
}
__device__ inline void cp_async_fence()  { asm volatile("cp.async.commit_group;\n"); }
template <int n> __device__ inline void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

// ---- cross-CTA barriers (unchanged) ----
__device__ inline void barrier_acquire(int * lock, int count) {
    if (threadIdx.x == 0) { int st = -1;
        do asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(st) : "l"(lock)); while (st != count); }
    __syncthreads();
}
__device__ inline void barrier_release(int * lock, bool reset = false) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) { lock[0] = 0; return; }
        asm volatile("fence.acq_rel.gpu;\n"); int v = 1;
        asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" :: "l"(lock), "r"(v));
    }
}

} // namespace marlin_ppu


// ============================================================================
// Simple plain Marlin kernel for PPU (correctness-first: one CTA per output tile, non-pipelined).
// Marlin's stripe dispatcher + multistage cp.async pipeline are the perf follow-up. Uses the validated PPU pieces.
// A: (M,K) fp16 int4-packed. B: Marlin int4 layout (K/16, N*16/32). C: (M,N) fp16. s: scales.
// Grid: (n_tiles, m_tiles). Block: 256 (8 warps). N_WARPS_N warps in N, N_WARPS_K in K (K-warps reduced at the end).
// ============================================================================
namespace marlin_ppu {

template <int thread_m_blocks, int thread_n_blocks, int thread_k_blocks, int group_blocks>
__global__ void marlin_simple(const int4 * __restrict__ A, const int4 * __restrict__ B,
        half * __restrict__ C, const int4 * __restrict__ s, int prob_m, int prob_n, int prob_k) {
    constexpr int threads = 256;
    constexpr int N_WARPS_N = thread_n_blocks / 4;          // warps spanning N (each does 4 n-blocks = 64 cols)
    constexpr int N_WARPS_K = (threads / 32) / N_WARPS_N;   // warps spanning K (reduced at the end)
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wn = warp % N_WARPS_N, wk = warp / N_WARPS_N;

    const int n_tile = blockIdx.x, m_tile = blockIdx.y;     // output tile (m_tile*16*m_blocks, n_tile*16*n_blocks)
    const int m0 = m_tile * 16 * thread_m_blocks;
    const int n0 = n_tile * 16 * thread_n_blocks;

    const int k_tiles = prob_k / 16;                        // total 16-wide K tiles
    // per-thread A: fp16 (M,K) row-major, half units. B: int4 layout. See obsidian/marlin.md for the packing.
    const half  * Ah = reinterpret_cast<const half *>(A);
    const int   * Bq = reinterpret_cast<const int  *>(B);   // int32 view of packed B
    const half  * Sh = reinterpret_cast<const half *>(s);

    // accumulators: this warp owns n-blocks [wn*4, wn*4+4) = 4 n16-tiles; all thread_m_blocks m-tiles.
    FragC frag_c[thread_m_blocks][4];
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int l = 0; l < 8; ++l) frag_c[i][j][l] = 0.0f;

    // K loop: each K-warp wk handles K-tiles kt = wk, wk+N_WARPS_K, ...  (partial sums reduced later)
    for (int kt = wk; kt < k_tiles; kt += N_WARPS_K) {
        const int k = kt * 16;                              // K offset of this 16-tile
        // --- A fragment (element-wise from gmem, PPU frag layout; plain, no swizzle) ---
        FragA fa[thread_m_blocks];
        #pragma unroll
        for (int i = 0; i < thread_m_blocks; ++i) {
            const int mrow = m0 + i * 16;
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = mrow + frag_i(lane, l), c = k + 2 * frag_j(lane, l);
                fa[i][l] = make_half2(Ah[(int64_t) r * prob_k + c], Ah[(int64_t) r * prob_k + c + 1]);
            }
        }
        // --- B: this warp's 4 n16-tiles. Marlin packs so a thread reads one int32 per (n16-tile) that dequants to
        //     its {b0,b1} fragment. B gmem index per obsidian/marlin.md: (K/16, N*16/32) int32-of-int4. ---
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int nblk = wn * 4 + j;                    // which 16-col n-block within the CTA tile
            const int n_global = n0 + nblk * 16;
            // packed B int32 for (this k-tile, this n-block, this lane): layout below matches the validated per-lane packing.
            // FIXME(box): confirm this gmem index vs the real Marlin repack (obsidian notes: b_gl_stride etc).
            const int bidx = (kt * (prob_n / 16) + (n_global / 16)) * 32 + lane;   // 32 int32 per (k-tile,n-block)
            int q = Bq[bidx];
            FragB b0 = dequant(q), b1 = dequant(q >> 8);
            if (group_blocks != -1) { /* grouped scale TODO */ }
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; ++i) mma_n16(fa[i], b0, b1, frag_c[i][j]);
        }
    }

    // --- reduce partial sums across the N_WARPS_K K-warps (register-aligned tree) ---
    extern __shared__ float rsh[];
    if (N_WARPS_K > 1) {
        constexpr int FC = thread_m_blocks * 4 * 8;         // floats per thread
        float * fp = &frag_c[0][0][0];
        for (int stride = N_WARPS_K / 2; stride >= 1; stride /= 2) {
            if (wk >= stride && wk < 2 * stride) {
                float * dst = rsh + (((wn * (N_WARPS_K / 2) + (wk - stride)) * 32 + lane) * FC);
                for (int f = 0; f < FC; ++f) dst[f] = fp[f];
            }
            __syncthreads();
            if (wk < stride) {
                float * src = rsh + (((wn * (N_WARPS_K / 2) + wk) * 32 + lane) * FC);
                for (int f = 0; f < FC; ++f) fp[f] += src[f];
            }
            __syncthreads();
        }
    }

    // --- write result (only wk==0 warps hold the sum). Direct global store via PPU acc layout + per-column scale. ---
    if (wk == 0) {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nblk = wn * 4 + j, ncol0 = n0 + nblk * 16;
                #pragma unroll
                for (int l = 0; l < 8; ++l) {
                    int r = m0 + i * 16 + acc_i(lane, l);
                    int cc = ncol0 + acc_j(lane, l);
                    if (r < prob_m && cc < prob_n) {
                        float v = frag_c[i][j][l];
                        if (group_blocks == -1) v *= __half2float(Sh[cc]);   // per-column scale
                        C[(int64_t) r * prob_n + cc] = __float2half(v);
                    }
                }
            }
    }
}

// Host launch for the simple kernel (one CTA per output tile).
template <int MB, int NB, int KB, int GB>
static void launch_simple(const int4 * A, const int4 * B, half * C, const int4 * s,
        int M, int N, int K, cudaStream_t stream) {
    dim3 grid(N / (16 * NB), ceildiv(M, 16 * MB));
    constexpr int N_WARPS_N = NB / 4, N_WARPS_K = 8 / N_WARPS_N;
    const size_t shmem = (size_t) N_WARPS_N * (N_WARPS_K / 2) * 32 * (MB * 4 * 8) * sizeof(float);
    cudaFuncSetAttribute(marlin_simple<MB, NB, KB, GB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem);
    marlin_simple<MB, NB, KB, GB><<<grid, 256, shmem, stream>>>(A, B, C, s, M, N, K);
}

} // namespace marlin_ppu
