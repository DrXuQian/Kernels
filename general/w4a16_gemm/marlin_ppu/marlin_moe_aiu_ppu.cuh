#ifndef MARLIN_MOE_AIU_PPU_CUH
#define MARLIN_MOE_AIU_PPU_CUH
// GGUF Q4_K grouped (MoE) GEMM on the AIU path -- the fp16 MoE GEMM's skeleton with an INT4 B.
//
// WHY NOT THE MARLIN DERIVATION. Two independent problems, one fix. (1) SHAPE: the fp16 MoE kernel is
// weight/memory bound, so the quantity to minimize is the TOTAL NUMBER OF M-TILES, sum_e ceil(rows_e/BMR)
// -- weight traffic scales with it, and its ragged default is BMR=128. Marlin's m-tile is 16*MARLIN_MAX_MB
// = 32 rows, a quarter of that, and MAX_MB is pinned at 2 by register spills, so it is structural rather
// than tunable. (2) LOAD PATH: MoE is bandwidth bound, which is exactly where the AIU bulk load and its
// nearly free deep pipeline pay -- more so for Q4_K than fp16, since the weights are half the bytes.
// (Dense W4A16 is the opposite case and stays on Marlin: there the gap is the lop3->hsub2->mma dependency
// chain, and AIU does not shorten a dependency chain. See memory ppu-w4a16-path-by-m.)
//
// THE B PATH, measured not assumed (aiu_int4_smoke, PASS on ppu001). AIU and ldmatrix.swzl both move b16,
// and swzl hands a lane one 32-bit register from each of four different 8x8 matrices -- NOT the four
// consecutive int32 Marlin's dequant wants. The map is bijective and inverts, so a pre-permutation fixes
// it, and the useful part is that it is LOCAL and TILEABLE: within a 16x16 b16 tile,
//
//     reg j of lane L sits at  row = L/4 + 8*(j/2),  cols = 2*(L%4) + 8*(j%2)  and +1
//
// A 16x16 b16 tile is 256 b16 = 128 int32 = 32 lanes x 4 regs, which is EXACTLY Marlin's B for one ktile
// x 64 columns, and Marlin's four ints are its four n-blocks j=0..3 -- so ld_swzl's four registers land on
// j=0..3 directly. The permutation folds into the load-time repack we already perform for GGUF Q4_K, so it
// costs nothing at runtime.
//
// SCALES: gs=32, per-32, as in the dense path. The affine min is NOT folded into the mainloop -- dense
// measurements put its whole cost in the per-k shared load of frag_m, and splitting it out recovered 7.6
// of 10.7 MFU points. Here it splits even more cleanly: Asum is per TOKEN and INDEPENDENT of the expert,
// so it is computed once for the whole batch and only the second (tiny) GEMM is grouped.
//
// STATUS: NOT YET RUN, and two things in the inner loop are reasoned rather than measured -- the scale
// column mapping (a lane's column within the n16 tile taken as lane/4, +8 for the >>8 half) and the
// b16-level correspondence between the AIU cube's k offset and kabs. Both are exactly the kind of index
// assumption that has been wrong every time this session, so they get checked against the reference
// before any timing. The Marlin-derived MoE (marlin_moe_gguf_ppu.cuh, all cases MATCH including ragged
// routing) is the reference this gets checked against before any timing.
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include "marlin_gguf_ppu.cuh"   // dequant() / scale() / FragB -- the verified INT4 -> fp16 path, reused as is

namespace marlin_moe_aiu_ppu {

#define MOE_BN 128
#define MOE_BK 64            // == AIU cube width in k; 4 mma k-substeps
#define MOE_KSUB (MOE_BK / 16)
#define MOE_WN 4
#define MOE_NWARP 8

static __device__ __forceinline__ unsigned cvta(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }

// AIU asm, verbatim from the fp16 MoE GEMM / fa_ppu.cu. .b16 is dtype-agnostic; INT4 rides it as packed b16.
static __device__ __forceinline__ void aiu_load(unsigned smem, const void* gmem,
        int dim_h, int dim_w, int cube_h, int cube_w, int coord_h, int coord_w) {
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem), "l"(gmem), "r"(1), "r"(dim_h), "r"(dim_w),
           "r"(0), "r"(coord_h), "r"(coord_w), "r"(1), "r"(cube_h), "r"(cube_w), "r"(1));
}
static __device__ __forceinline__ void ld_swzl(int* v, const void* cube_base, int coord_h, int coord_w, int cube_h) {
    const int cbo = coord_w * (int) sizeof(half);
    asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};\n"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
        : "l"(cube_base), "r"(0), "r"(coord_h), "r"(1), "r"(cube_h), "r"(1), "r"(cbo));
}
__device__ __forceinline__ void mma_f16(float* acc, const int* A, const int* B) {
    asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]));
}
__device__ __forceinline__ int acc_i(int lane, int l) { return (lane / 4) + (((l >> 2) & 1) << 3); }
__device__ __forceinline__ int acc_j(int lane, int l) { return (lane % 4) + ((l % 4) << 2); }

// ---------------------------------------------------------------------------------------------------
// HOST: where reg j of lane L lives inside a 16x16 b16 tile. Measured by aiu_int4_smoke, not derived from
// documentation -- the two b16 halves of a register are adjacent, the register index splits into an
// 8-row and an 8-column step, and the lane splits into a row within 8 and a column-pair within 8.
inline void swzl_pos(int lane, int reg, int& row, int& col0) {
    row  = lane / 4 + 8 * (reg / 2);
    col0 = 2 * (lane % 4) + 8 * (reg % 2);
}

// Marlin's packed B (ints, ktile-major per-lane) -> the b16 image the AIU cube loads.
// Image layout: [16 * (N/64)] x [16 * (K/16)] b16. Tile (ktile kt, 64-column group g) occupies
// rows [16g, 16g+16) and cols [16kt, 16kt+16), and within it swzl_pos places each lane's four ints.
// One such tile IS Marlin's 32 lanes x 4 ints for that (kt, g) -- the correspondence is exact, which is
// what makes this a permutation and not a repacking of a different shape.
inline void marlin_b_to_aiu_image(const int* B_marlin, int N, int K, unsigned short* img) {
    const int ngroups = N / 64, ktiles = K / 16, W = 16 * ktiles;
    for (int kt = 0; kt < ktiles; kt++)
        for (int g = 0; g < ngroups; g++)
            for (int lane = 0; lane < 32; lane++)
                for (int j = 0; j < 4; j++) {
                    const long long idx = (long long) (N / 2) * kt + (long long) g * 32 + lane;
                    const int word = B_marlin[idx * 4 + j];
                    int r, c; swzl_pos(lane, j, r, c);
                    const long long base = (long long) (16 * g + r) * W + 16 * kt + c;
                    img[base]     = (unsigned short) (word & 0xffff);
                    img[base + 1] = (unsigned short) ((unsigned) word >> 16);
                }
}
inline int aiu_img_h(int N) { return 16 * (N / 64); }
inline int aiu_img_w(int K) { return 16 * (K / 16); }

// ---------------------------------------------------------------------------------------------------
// Persistent scheduler, mirroring the fp16 kernel: grid-stride over a global (m-block, n-block) queue,
// expert by binary search over a per-expert m-block prefix sum. Grid is #CU * (256KB / shmem) -- fill every
// CU slot, not one block per CU. Rows are MASKED, not padded: expert_bounds are the true ragged bounds,
// padz zeroes the real OOB, and over-reading into the next expert's rows is harmless because those output
// rows are never stored.
template <int NST, int BMR>
__global__ void moe_q4k_aiu(const half* __restrict__ A, const unsigned short* __restrict__ Bimg,
        const half* __restrict__ s, const int* __restrict__ expert_bounds,
        const int* __restrict__ mblk_prefix, float* __restrict__ C,
        int total_rows, int N, int K, int n_experts, int total_tiles, int gs) {
    constexpr int MIR = BMR / 32, WROW = MIR * 16;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / MOE_WN, wn = warp % MOE_WN;
    const int nk = K / MOE_BK, num_nb = N / MOE_BN;
    const int img_w = aiu_img_w(K);
    const int bcube_h = 16 * (MOE_BN / 64), bcube_w = 16 * (MOE_BK / 16);

    extern __shared__ char smem_raw[];
    half* As = reinterpret_cast<half*>(smem_raw);
    unsigned short* Bs = reinterpret_cast<unsigned short*>(smem_raw) + (size_t) NST * BMR * MOE_BK;

    for (int iter = 0; ; ++iter) {
        const int t = iter * (int) gridDim.x + blockIdx.x;
        if (t >= total_tiles) break;
        const int mblk_g = t / num_nb, nb = t % num_nb;
        int lo = 0, hi = n_experts;
        while (lo + 1 < hi) { const int mid = (lo + hi) >> 1; if (mblk_prefix[mid] <= mblk_g) lo = mid; else hi = mid; }
        const int e = lo;
        const int m0 = expert_bounds[e] + (mblk_g - mblk_prefix[e]) * BMR;
        const int m_end = expert_bounds[e + 1];
        // B image of expert e; the n-block selects a row band of the image.
        const unsigned short* Be = Bimg + (long long) e * aiu_img_h(N) * img_w;
        const int b_row0 = 16 * (nb * MOE_BN / 64);
        const half* se = s + (long long) e * (K / 32) * N;

        float acc[MIR][4][8];
        #pragma unroll
        for (int mi = 0; mi < MIR; ++mi) for (int nj = 0; nj < 4; ++nj) for (int l = 0; l < 8; ++l) acc[mi][nj][l] = 0.f;

        #pragma unroll
        for (int st = 0; st < NST - 1; ++st) {
            if (st < nk && tid == 0) {
                aiu_load(cvta(As + (size_t) st * BMR * MOE_BK), A, total_rows, K, BMR, MOE_BK, m0, st * MOE_BK);
                aiu_load(cvta(Bs + (size_t) st * bcube_h * bcube_w), Be, aiu_img_h(N), img_w,
                         bcube_h, bcube_w, b_row0, st * bcube_w);
            }
            asm volatile("cp.async.commit_group;");
        }
        for (int kk = 0; kk < nk; ++kk) {
            const int nx = kk + NST - 1;
            if (nx < nk && tid == 0) {
                aiu_load(cvta(As + (size_t) (nx % NST) * BMR * MOE_BK), A, total_rows, K, BMR, MOE_BK, m0, nx * MOE_BK);
                aiu_load(cvta(Bs + (size_t) (nx % NST) * bcube_h * bcube_w), Be, aiu_img_h(N), img_w,
                         bcube_h, bcube_w, b_row0, nx * bcube_w);
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group %0;" :: "n"(NST - 2));
            __syncthreads();

            const int stg = kk % NST;
            const half* Ac = As + (size_t) stg * BMR * MOE_BK;
            const unsigned short* Bc = Bs + (size_t) stg * bcube_h * bcube_w;
            #pragma unroll
            for (int ks = 0; ks < MOE_KSUB; ++ks) {
                const int ko = ks * 16, kabs = kk * MOE_BK + ko;
                int fa[MIR][4], fb[4];
                #pragma unroll
                for (int mi = 0; mi < MIR; ++mi) ld_swzl(fa[mi], Ac, wm * WROW + mi * 16, ko, BMR);
                // one ld_swzl gives this lane its four packed ints = the four n-blocks j=0..3 of a
                // 64-column group; wn selects which 64-column group of the BN-wide tile.
                ld_swzl(fb, Bc, 16 * wn, ko, bcube_h);
                #pragma unroll
                for (int nj = 0; nj < 4; ++nj) {
                    // Marlin's consumption of one packed int, unchanged: the low byte-lane dequants to
                    // columns n and the >>8 lane to columns n+8, and the ppu m16n16k16 B operand is the
                    // two n8 fragments CONCATENATED -- {b0[0], b0[1], b1[0], b1[1]}.
                    marlin_gguf_ppu::FragB b0 = marlin_gguf_ppu::dequant(fb[nj]);
                    marlin_gguf_ppu::FragB b1 = marlin_gguf_ppu::dequant(fb[nj] >> 8);
                    // gs=32 per-32 scale. A lane's column within the n16 tile is lane/4 for b0 and +8 for
                    // b1, matching dequant's output positions.
                    const int col0 = nb * MOE_BN + wn * 64 + nj * 16 + (lane / 4);
                    const long long srow = (long long) (kabs / gs) * N;
                    marlin_gguf_ppu::FragS s0, s1;
                    reinterpret_cast<half*>(&s0)[0] = se[srow + col0];
                    reinterpret_cast<half*>(&s0)[1] = se[srow + col0];
                    reinterpret_cast<half*>(&s1)[0] = se[srow + col0 + 8];
                    reinterpret_cast<half*>(&s1)[1] = se[srow + col0 + 8];
                    marlin_gguf_ppu::scale(b0, s0, 0);
                    marlin_gguf_ppu::scale(b1, s1, 0);
                    int Bf[4];
                    Bf[0] = *reinterpret_cast<int*>(&b0.v[0]); Bf[1] = *reinterpret_cast<int*>(&b0.v[1]);
                    Bf[2] = *reinterpret_cast<int*>(&b1.v[0]); Bf[3] = *reinterpret_cast<int*>(&b1.v[1]);
                    #pragma unroll
                    for (int mi = 0; mi < MIR; ++mi) mma_f16(acc[mi][nj], fa[mi], Bf);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int mi = 0; mi < MIR; ++mi) for (int nj = 0; nj < 4; ++nj) for (int l = 0; l < 8; ++l) {
            const int row = m0 + wm * WROW + mi * 16 + acc_i(lane, l);
            const int col = nb * MOE_BN + wn * 64 + nj * 16 + acc_j(lane, l);
            if (row < m_end) C[(long long) row * N + col] = acc[mi][nj][l];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------------
// HOST: routing plan -> the scheduler's inputs. bounds are the TRUE ragged per-expert row bounds (no
// padding -- the kernel masks); mblk_prefix is the per-expert m-block prefix sum the tile queue binary
// searches. total_tiles = mblk_prefix[n_experts] * (N/BN).
struct MoeSched {
  std::vector<int> mblk_prefix;   // n_experts+1
  int total_tiles;
};
inline MoeSched moe_sched(const int* expert_bounds, int n_experts, int N, int BMR) {
  MoeSched s; s.mblk_prefix.assign(n_experts + 1, 0);
  for (int e = 0; e < n_experts; e++) {
    const int rows = expert_bounds[e + 1] - expert_bounds[e];
    s.mblk_prefix[e + 1] = s.mblk_prefix[e] + (rows + BMR - 1) / BMR;
  }
  s.total_tiles = s.mblk_prefix[n_experts] * (N / MOE_BN);
  return s;
}

template <int NST, int BMR>
inline void launch_moe_q4k(const half* A, const unsigned short* Bimg, const half* s,
        const int* d_bounds, const int* d_mblk, float* C,
        int total_rows, int N, int K, int n_experts, int total_tiles, int gs, cudaStream_t stream = 0) {
  const int bcube_h = 16 * (MOE_BN / 64), bcube_w = 16 * (MOE_BK / 16);
  const size_t shmem = (size_t) NST * (BMR * MOE_BK * sizeof(half) + bcube_h * bcube_w * sizeof(unsigned short));
  cudaFuncSetAttribute(moe_q4k_aiu<NST, BMR>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem);
  // Grid fills every CU SLOT, not one block per CU -- the fp16 kernel's finding, and the reason its
  // persistent variant beat the static grid (idle blocks on empty experts, wave tail, 14% L2 hit).
  int cus = 0; cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, 0);
  const int blk_cu = (int) ((size_t) 256 * 1024 / shmem);
  int grid = cus * (blk_cu > 0 ? blk_cu : 1);
  if (const char* g = getenv("MOE_GRID")) grid = atoi(g);
  if (grid > total_tiles) grid = total_tiles > 0 ? total_tiles : 1;
  moe_q4k_aiu<NST, BMR><<<grid, MOE_NWARP * 32, shmem, stream>>>(
      A, Bimg, s, d_bounds, d_mblk, C, total_rows, N, K, n_experts, total_tiles, gs);
}

// AFFINE min, split out -- the dense measurement put the fused min's whole cost in the per-k shared load
// of frag_m, and here it factors even more cleanly: Asum[t][g] is per TOKEN and INDEPENDENT of the expert,
// so it is computed ONCE for the whole batch and only the second (K/gs-deep) GEMM is grouped.
//   C[t][n] += sum_g m'[e(t)][g][n] * Asum[t][g]
// One block per (token, n-block chunk); m' is PLAIN [G][N] per expert, not _scale_perm -- that layout is
// for Marlin's fragment reads and means nothing here, and passing the permuted buffer is silently wrong.
__global__ void moe_min_correct(const half* __restrict__ Asum, const half* __restrict__ mins,
        const int* __restrict__ row_expert, float* __restrict__ C, int total_rows, int N, int G) {
  const int row = blockIdx.x;
  if (row >= total_rows) return;
  const int e = row_expert[row];
  const half* me = mins + (long long) e * G * N;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    float acc = 0.f;
    for (int g = 0; g < G; g++) acc += __half2float(Asum[(long long) row * G + g]) * __half2float(me[(long long) g * N + n]);
    C[(long long) row * N + n] += acc;
  }
}

}  // namespace marlin_moe_aiu_ppu
#endif
