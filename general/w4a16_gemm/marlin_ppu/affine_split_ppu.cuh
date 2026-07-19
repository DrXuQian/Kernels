#ifndef AFFINE_SPLIT_PPU_CUH
#define AFFINE_SPLIT_PPU_CUH
// The affine (GGUF min / zero-point) term, split OUT of Marlin's mainloop into its own pass.
//
// The min never touches B, so it factors out of the quantized GEMM entirely:
//   C[m][n] = sum_g s[g][n] * sum_{k in g} A[m][k]*(q-8)   +   sum_g m'[g][n] * sum_{k in g} A[m][k]
//             \____________ symmetric Marlin ____________/      \______ Asum[M x G] @ m'[G x N] ______/
// with G = K/gs, so the correction is 1/gs of the main GEMM's FLOPs and the min is paid once per
// (m, group) instead of once per (m, k). Measured on ppu001, 2048x4096x4096 gs=32: fused affine 459us
// (29.9% MFU) against 366us (37.5%) split, where symmetric alone is 338 (40.6%) -- so it recovers all but
// ~3 points of the 10.7 the min was costing. Why it is worth it: the cost was never the hfma2, it was the
// per-k shared load of frag_m (MARLIN_MINS_CONST measured 40.3% -- hfma2 kept, load dropped).
//
// LAYOUTS DIFFER between the two paths and it matters: fused Marlin reads mins in the _scale_perm layout,
// while this correction reads m' as a PLAIN [G][N] matrix. Passing the permuted buffer here is silently
// wrong, so the two are not interchangeable.
//
// NOT for decode. At M=1 the GEMM is weight-bandwidth bound and this adds a full read of A plus a
// read-modify-write of C for a correction that gemv_w4a16_ppu.cu already gets rank-1 and nearly free.
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "marlin_gguf_ppu.cuh"

namespace affine_split_ppu {

// Asum[m][g] = sum of A[m][k] over that group's gs activations.
//
// One thread per int4 (8 halves) so lanes address contiguously and a warp pulls a solid 512 B per
// instruction. The obvious version -- one thread per group, scalar loop over gs halves -- puts consecutive
// threads gs*2 bytes apart and makes every 2-byte load a 32-way scatter; it measured 5x off the bandwidth
// floor (35.3us against 6.1) and was two thirds of the whole correction.
__global__ void asum_kernel(const half* __restrict__ A, half* __restrict__ Asum, int M, int K, int gs) {
    const int G = K / gs, per_group = gs / 8;              // int4 per group (4 at gs=32)
    const long long nvec = (long long) M * (K / 8);
    const long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvec) return;

    const int4 v = reinterpret_cast<const int4*>(A)[idx];
    const half2* h = reinterpret_cast<const half2*>(&v);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; i++) { const float2 f = __half22float2(h[i]); acc += f.x + f.y; }

    // The per_group lanes sharing a group are adjacent in idx, hence adjacent lanes.
    #pragma unroll
    for (int off = 1; off < 8; off <<= 1)
        if (off < per_group) acc += __shfl_down_sync(0xffffffff, acc, off);

    if (idx % per_group == 0) {
        const long long vg = idx / per_group;
        Asum[(vg / G) * (long long) G + (vg % G)] = __float2half(acc);
    }
}

inline void asum_launch(const half* A, half* Asum, int M, int K, int gs, cudaStream_t stream = 0) {
    const int blocks = (int) (((long long) M * (K / 8) + 255) / 256);
    asum_kernel<<<blocks, 256, 0, stream>>>(A, Asum, M, K, gs);
}

// C = symmetric_marlin(A, B, s) + Asum(A) @ mins_plain.
//   mins_plain : [G][N] fp16, PLAIN layout (NOT _scale_perm -- see the note above)
//   Asum_ws    : caller-owned scratch, M*G halves
// Returns marlin_cuda's status.
inline int affine_split_cuda(const void* A, const void* B, void* C, void* s, const half* mins_plain,
                             int M, int N, int K, void* workspace, int gs,
                             half* Asum_ws, cublasHandle_t cub, int max_par = 128) {
    const int G = K / gs;
    const int ret = marlin_gguf_ppu::marlin_cuda(A, B, C, s, nullptr /*symmetric*/, M, N, K,
                                                 workspace, gs, 0, 0, -1, -1, -1, max_par);
    if (ret) return ret;
    asum_launch(reinterpret_cast<const half*>(A), Asum_ws, M, K, gs);
    // cublas is column-major; C^T = mins^T * Asum^T is the same call with the operands swapped. beta=1
    // accumulates onto what Marlin just wrote.
    const half alpha = __float2half(1.f), beta = __float2half(1.f);
    cublasHgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, N, M, G, &alpha,
                mins_plain, N, Asum_ws, G, &beta, reinterpret_cast<half*>(C), N);
    return 0;
}

}  // namespace affine_split_ppu
#endif
