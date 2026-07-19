#ifndef DEQUANT_W4_PPU_CUH
#define DEQUANT_W4_PPU_CUH
// Dequantize Marlin-format INT4 weights to a plain fp16 matrix, so a large prefill can run as a DENSE
// fp16 GEMM instead of dequantizing inside the mainloop.
//
// The case for it: Marlin pays for the fused dequant in MFU. On ppu001 it reaches 60.7% with per-column
// scales, 47% grouped, and 40.6% at gs=32 -- ~20 points lost to doing the dequant per k. A dense fp16
// GEMM has none of that. The price is one round trip of the fp16 weights through HBM (write N*K*2, read
// it back), which at N=K=4096 is ~24us against a ~338us GEMM, i.e. ~7%. If a dense fp16 GEMM clears about
// 47% MFU the trade pays; that number is what bench_prefill_paths measures rather than assumes.
//
// It is a PREFILL-ONLY trade, and the reason is worth stating precisely: the extra traffic is fixed at
// 2*N*K*2 bytes while the compute it buys grows with M. At M=1 the GEMM is weight-bandwidth bound and
// this quadruples the weight traffic (int4 -> fp16, twice) for no compute win at all -- strictly worse.
// Somewhere in between they cross, which is what the M scan is for.
//
// MEMORY: the fp16 buffer is N*K*2 bytes and must be SCRATCH, reused across layers -- expanding a whole
// Q4_K model would turn ~18 GB into ~72 GB. One layer's largest weight at a time is what fits.
//
// SCALES ARE PLAIN [G][N] here, not _scale_perm. The permuted layout exists for Marlin's fragment reads
// and means nothing to a dense GEMM; feeding this the permuted buffer is silently wrong.
#include <cuda_fp16.h>

namespace dequant_w4_ppu {

// Inverse of Marlin's B packing (see pack_B in the tests). One thread per (ktile, nblock, lane) reads the
// int holding that lane's 8 nibbles and writes them to W[n][k]. Output is row-major [N][K] -- chosen over
// [K][N] because a lane's 8 values land on 4 ADJACENT k-pairs, so the writes are four 4-byte stores
// instead of eight scattered 2-byte ones.
__global__ void dequant_kernel(const int* __restrict__ B, const half* __restrict__ s,
                               const half* __restrict__ mn, half* __restrict__ W,
                               int N, int K, int gs) {
    const int lane = threadIdx.x % 32, tid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int nblocks = N / 16, ktiles = K / 16;
    if (tid >= nblocks * ktiles) return;
    const int ktile = tid / nblocks, nblock = tid % nblocks;

    const long long idx = (long long) (N / 2) * ktile + (nblock / 4) * 32 + lane;
    const int q = B[idx * 4 + (nblock % 4)];

    const int n0 = nblock * 16 + lane / 4;
    const int kb = ktile * 16 + (lane % 4) * 2;
    const int G = K / gs;

    // (shift, n offset, k offset) exactly mirroring the packer's eight nibble slots.
    const int sh[8] = { 0, 16,  4, 20,  8, 24, 12, 28 };
    const int dn[8] = { 0,  0,  0,  0,  8,  8,  8,  8 };
    const int dk[8] = { 0,  1,  8,  9,  0,  1,  8,  9 };
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int n = n0 + dn[i], k = kb + dk[i];
        const int g = k / gs;
        const float qv = (float) ((q >> sh[i]) & 0xf) - 8.0f;      // symmetric dequant, matches Marlin
        float w = qv * __half2float(s[(long long) g * N + n]);
        if (mn) w += __half2float(mn[(long long) g * N + n]);      // m' already folded with +8s on the host
        W[(long long) n * K + k] = __float2half(w);
    }
}

// W_out: row-major [N][K] fp16, caller-owned, N*K halves.
inline void dequant_launch(const int* B, const half* s, const half* mn, half* W_out,
                           int N, int K, int gs, cudaStream_t stream = 0) {
    const int warps_per_block = 8, units = (N / 16) * (K / 16);
    dequant_kernel<<<(units + warps_per_block - 1) / warps_per_block, warps_per_block * 32, 0, stream>>>(
        B, s, mn, W_out, N, K, gs);
}

}  // namespace dequant_w4_ppu
#endif
