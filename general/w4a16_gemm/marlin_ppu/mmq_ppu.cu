/*
 * Standalone llama.cpp MMQ (quantized matmul) -- Q4_0 and Q4_K -- extracted from
 * ggml/src/ggml-cuda/{mmq.cuh, quantize.cu, mma.cuh} and rebuilt with no ggml dependency.
 *
 * WHAT MMQ IS, AND WHY IT IS NOT MARLIN.  Marlin dequantizes INT4 weights to FP16 and runs a FP16
 * tensor core. MMQ instead dequantizes the INT4 weights to INT8, quantizes the ACTIVATIONS to INT8
 * (Q8_1: an int8 quant + a per-32-value scale d + a per-32-value sum s), and runs the entire K-loop
 * on an INTEGER tensor core, rescaling the int32 accumulator once per 32-value block. On NVIDIA the
 * payoff is that int8 tensor cores run at 2x the fp16 rate. Whether that holds on the PPU is exactly
 * what mma_int8_smoke.cu is for -- if the PPU has no int8 MMA, this file has no reason to exist and
 * the right move is to dequantize GGUF's Q4_K to fp16 and reuse the proven ppu.mma.m16n16k16.
 *
 * Build:
 *   NVIDIA reference (H800):   nvcc -O3 -std=c++17 -arch=sm_90 -DMMQ_NV  -o mmq mmq_ppu.cu
 *   PPU (box, ppu001, NO -arch): nvcc -O3 -std=c++17 -DMMQ_PPU -o mmq mmq_ppu.cu
 * Run:  ./mmq            (self-check, both quant types)
 *       MMQ_BENCH=1 ./mmq
 *
 * SELF-CHECK DISCIPLINE (learned the hard way on the Marlin port, where every test set scale = 1.0
 * and so hid TWO separate scale bugs for days): the reference here uses NON-TRIVIAL, PER-BLOCK,
 * VARYING d / dmin / sub-block scales and mins. A test with flat scales would pass on a kernel that
 * ignores them entirely.
 *
 * Two references are computed, and they answer different questions:
 *   ref_exact  -- the kernel's OWN formula (from the quantized X and the quantized Y), in double.
 *                 The kernel must match this to ~1e-5. This isolates KERNEL bugs.
 *   ref_true   -- the honest fp32 A*B. The gap to ref_exact is the QUANTIZATION error (~1e-2 for
 *                 4-bit) and is not a bug. Reporting only this number would let a real kernel bug
 *                 hide inside the quantization noise.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

#if !defined(MMQ_NV) && !defined(MMQ_PPU)
#define MMQ_NV
#endif

// ============================================================================================
// Block formats (verbatim from ggml-common.h)
// ============================================================================================
#define QK4_0 32
#define QI4_0 4          // 32-bit words of quants per q4_0 block (16 bytes / 4)
struct block_q4_0 { half d; uint8_t qs[QK4_0 / 2]; };
static_assert(sizeof(block_q4_0) == 18, "q4_0");

#define QK_K 256
struct block_q4_K { half d; half dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; };
static_assert(sizeof(block_q4_K) == 144, "q4_K");

#define QK8_1 32
#define QI8_1 8
// 128 activation values of ONE column, with the four 32-value (d, s) pairs hoisted to the front.
// 144 B == 36 ints == MMQ_TILE_Y_K, so the shared tile is a bit-identical copy of global memory.
struct block_q8_1_mmq { half2 ds4[4]; int8_t qs[4 * QK8_1]; };
static_assert(sizeof(block_q8_1_mmq) == 144, "q8_1_mmq");

#define MMQ_TILE_Y_K      36   // ints per column per 128-k slab
#define MMQ_TILE_X_K      76   // ints per weight row per 256-k tile: 64 quants + 8 scales + 4 pad
#define MMQ_TILE_NE_K     32   // ints of quants per half-tile (128 k)
#define MMQ_ITER_K       256   // k advanced per mainloop iteration

// The +4 padding is what keeps the ldmatrix accesses bank-conflict-free; it is not slack.
static_assert(MMQ_TILE_X_K % 8 == 4, "tile-x stride must be 4 mod 8 for conflict-free ldmatrix");

// ============================================================================================
// Backend: fragment layouts and the MMA.
//
// NVIDIA  mma.m16n8k32.row.col.s32.s8.s8.s32   -- C is 16x8, 4 s32 regs
// PPU     ppu.mma.m16n16k32.row.col.s32.s8.s8.s32 -- C is 16x16, 8 s32 regs (two n8 fused)
//
// A (both): filled by ldmatrix.x4.b16, which hands back the four 8x8 b16 tiles in the order its 32
// lane addresses named them: v0=(rows 0-7, k lo) v1=(rows 8-15, k lo) v2=(rows 0-7, k hi)
// v3=(rows 8-15, k hi), i.e. row bit = l%2, k bit = l/2. NVIDIA's mma consumes that order directly.
// The PPU's mma wants row bit = l/2, k bit = l%2 -- so v1 and v2 swap, which is FREE (bind the asm's
// 2nd and 3rd outputs to a[2] and a[1]). Same trick as marlin_classic_ppu.cuh's ppu_ldmatrix_x4.
// ============================================================================================
#ifdef MMQ_NV
#define MMQ_TILE_N   8    // columns of Y per C tile
#define MMQ_C_NE     4    // s32 accumulator regs
#define MMQ_B_NE     2    // B operand regs
#define MMQ_NA       2    // distinct weight rows a lane touches -> index with (l / 2)
#define MMQ_NB       2    // distinct Y columns a lane touches  -> index with (l % 2)
#define ACC_A_IDX(l) ((l) / 2)
#define ACC_B_IDX(l) ((l) % 2)
__device__ __forceinline__ int acc_i(int t, int l) { return 8 * (l / 2) + t / 4; }
__device__ __forceinline__ int acc_j(int t, int l) { return 2 * (t % 4) + (l % 2); }
// B: lane t serves column t/4; its two regs hold k-words (t%4) and (t%4)+4.
__device__ __forceinline__ int b_n(int t, int l) { return t / 4; }
__device__ __forceinline__ int b_k(int t, int l) { return 4 * l + (t % 4); }
// A operand by (row, k-word), for building the fragment straight out of RAW shared memory instead of
// via ldmatrix. Same (row, k) set ldmatrix would have handed back -- established ON HARDWARE in
// mma_int8_smoke.cu. (ggml's tile<16,8,int>::get_i/get_j describe the ACCUMULATOR, not A; using them
// here silently zeroes C rows 8-15.)
__device__ __forceinline__ int a_row(int t, int l) { return t / 4 + 8 * (l % 2); }
__device__ __forceinline__ int a_kw (int t, int l) { return (t % 4) + 4 * (l / 2); }
#else
#define MMQ_TILE_N  16
#define MMQ_C_NE     8
#define MMQ_B_NE     4
#define MMQ_NA       2    // -> index with (l >> 2)
#define MMQ_NB       4    // -> index with (l % 4)
#define ACC_A_IDX(l) ((l) >> 2)
#define ACC_B_IDX(l) ((l) % 4)
__device__ __forceinline__ int acc_i(int t, int l) { return t / 4 + (((l >> 2) & 1) << 3); }
__device__ __forceinline__ int acc_j(int t, int l) { return t % 4 + ((l % 4) << 2); }
// B (16 cols): same fragment formula as the PPU A operand -- row.col makes A and B symmetric.
__device__ __forceinline__ int b_n(int t, int l) { return t / 4 + 8 * (l / 2); }
__device__ __forceinline__ int b_k(int t, int l) { return (t % 4) + 4 * (l % 2); }
// PPU: the A operand has the SAME formula as B (row.col makes them symmetric). Note the bits are
// swapped relative to NVIDIA -- this is the v1<->v2 swap that ppu_ldmatrix_x4 does for free.
__device__ __forceinline__ int a_row(int t, int l) { return t / 4 + 8 * (l / 2); }
__device__ __forceinline__ int a_kw (int t, int l) { return (t % 4) + 4 * (l % 2); }
#endif

__device__ __forceinline__ void mmq_ldmatrix_A(uint32_t* a, const int* smem) {
#ifdef MMQ_NV
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "l"(smem));
#else
  // v1 <-> v2: ldmatrix's natural order is NVIDIA's; the PPU mma wants the row bit in l/2.
  asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[2]), "=r"(a[1]), "=r"(a[3]) : "l"(smem));
#endif
}

__device__ __forceinline__ void mmq_mma(int* c, const uint32_t* a, const uint32_t* b) {
#ifdef MMQ_NV
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  asm("ppu.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]), "+r"(c[5]), "+r"(c[6]), "+r"(c[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
#endif
}

// ============================================================================================
// Tiling. mmq_y = 128 weight rows per block, 8 warps of 32 -> each warp owns 16 rows (one C tile
// in the m direction), and walks all MMQ_X columns. This is llama.cpp's ntx == 1 case; it keeps the
// port honest before any of ntx==2 / stream-k / cp.async gets layered on.
// ============================================================================================
#ifndef MMQ_X
#define MMQ_X 64
#endif
#define MMQ_Y      128
#define MMQ_NWARPS   8
#define MMQ_WARP    32
static_assert(MMQ_NWARPS * 16 == MMQ_Y, "one 16-row C tile per warp must cover mmq_y");
static_assert(MMQ_X % MMQ_TILE_N == 0, "MMQ_X must be a whole number of C tiles");
#define MMQ_SUM_NE (MMQ_X / MMQ_TILE_N * MMQ_C_NE)   // == MMQ_X/2 for both backends

// ============================================================================================
// Y quantization (quantize_mmq_q8_1). One thread per 4 floats.
//
// NOTE on `s`: it is the sum of the RAW fp32 activations over the 32-value group, NOT d*sum(q).
// Q4_K's affine correction term is -dmin*m_b * s, so s must be the un-quantized sum or the min
// correction is biased. Recomputing s as d*sum(q) "looks equivalent" and is not.
//
// NOTE on amax == 0: upstream's quantize_mmq_q8_1 does NOT guard this (its non-MMQ sibling does),
// so d_inv = 127/0 = inf and roundf(0*inf) = NaN. In the zero-padded tail of a row that is exactly
// the case. Guarded here.
// ============================================================================================
__global__ void quantize_q8_1_mmq(const float* __restrict__ x, block_q8_1_mmq* __restrict__ y,
                                  const int K, const int K_padded, const int M) {
  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;  // k offset, 4 per thread
  if (i0 >= K_padded) return;
  const int j = blockIdx.x;  // column of Y

  float4 xi = make_float4(0.f, 0.f, 0.f, 0.f);
  if (i0 + 3 < K) xi = *(const float4*)(x + (int64_t)j * K + i0);

  float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
  float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
  for (int off = 4; off > 0; off >>= 1) {  // 32 vals / 4 per thread = 8 lanes
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off, 32));
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, off, 32);
  }

  const float d = amax / 127.0f;
  const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  char4 q;
  q.x = (int8_t)roundf(xi.x * d_inv);
  q.y = (int8_t)roundf(xi.y * d_inv);
  q.z = (int8_t)roundf(xi.z * d_inv);
  q.w = (int8_t)roundf(xi.w * d_inv);

  const int64_t ib = (i0 / (4 * QK8_1)) * M + j;  // k-slab major, column minor
  const int iqs = i0 % (4 * QK8_1);
  ((char4*)y[ib].qs)[iqs / 4] = q;
  if (iqs % 32 == 0) y[ib].ds4[iqs / 32] = make_half2(d, sum);
}

// ============================================================================================
// Weight tile loads.  Both normalize a 256-k slice of 128 weight rows into the SAME shared layout:
//    ints [i*76 +  0 .. +63]  quants, 64 ints = 256 int8, grouped so that ints [8b, 8b+8) are the
//                             32 values of 32-value block b -- which is what the dot product's
//                             k0/QI8_1 indexing assumes.
//    ints [i*76 + 64 .. +71]  8 scales: float d (Q4_0) or half2(d*sc_b, -dmin*m_b) (Q4_K)
//    ints [i*76 + 72 .. +75]  padding
// After this the dot product is IDENTICAL for the two types except for the scale arithmetic. That
// normalization is the whole trick of MMQ and it is why upstream has no vec_dot_q4_0/q4_K at all.
// ============================================================================================

// sizeof(block_q4_0) == 18, so in an array of them qs lands at byte 18*b + 2 -- only 2-byte aligned.
// A plain 32-bit load there is a misaligned access (it faults). ggml calls this get_int_b2.
// block_q4_K is 144 B with qs at offset 16, so it can be read with a plain int load (get_int_b4).
__device__ __forceinline__ int get_int_b2(const void* p, const int i32) {
  const uint16_t* h = (const uint16_t*)p;
  return (int)h[2 * i32 + 0] | ((int)h[2 * i32 + 1] << 16);
}

// Q4_0: quants are shifted to SIGNED [-8,7] here, so the dot product needs no correction term.
__device__ __forceinline__ void load_tiles_q4_0(const char* __restrict__ x, int* __restrict__ tile,
                                                const int kb0, const int i_max, const int stride) {
  int* x_qs = tile;
  float* x_df = (float*)(x_qs + 2 * MMQ_TILE_NE_K);
  const int t = threadIdx.x;

  const int kbx = t / QI4_0;   // 0..7  which q4_0 block inside the 256-k tile
  const int kqsx = t % QI4_0;  // 0..3  which int inside that block's 16 qs bytes
#pragma unroll
  for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS) {
    const int i = min(i0 + (int)threadIdx.y, i_max);
    const block_q4_0* b = (const block_q4_0*)x + kb0 + i * stride + kbx;
    const int qs0 = get_int_b2(b->qs, kqsx);
    x_qs[i * MMQ_TILE_X_K + kbx * (2 * QI4_0) + kqsx + 0]     = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
    x_qs[i * MMQ_TILE_X_K + kbx * (2 * QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
  }

  const int kbxd = t % 8;
#pragma unroll
  for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * 4) {
    const int i = min(i0 + (int)threadIdx.y * 4 + t / 8, i_max);
    const block_q4_0* b = (const block_q4_0*)x + kb0 + i * stride + kbxd;
    x_df[i * MMQ_TILE_X_K + kbxd] = __half2float(b->d);
  }
}

// The 12 packed bytes of a Q4_K super-block hold 8 six-bit scales and 8 six-bit mins.
// ksc: 0 -> sc0..sc3, 1 -> sc4..sc7, 2 -> m0..m3, 3 -> m4..m7. Returns 4 six-bit values, one per byte.
__device__ __forceinline__ int unpack_scales_q4_K(const int* scales, const int ksc) {
  return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F) |  // low 4 bits
         ((scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030);                          // high 2 bits
}

// Q4_K: quants stay UNSIGNED [0,15]. The affine "-min" term never touches the tensor core -- it is a
// rank-1 correction -dmin*m_b * s_y[j][b] applied straight to the fp accumulator.
__device__ __forceinline__ void load_tiles_q4_K(const char* __restrict__ x, int* __restrict__ tile,
                                                const int kb0, const int i_max, const int stride) {
  int* x_qs = tile;
  half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);
  const int t = threadIdx.x;

#pragma unroll
  for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS) {
    const int i = min(i0 + (int)threadIdx.y, i_max);
    const block_q4_K* b = (const block_q4_K*)x + kb0 + i * stride;
    const int qs0 = *(const int*)(b->qs + 4 * t);
    // src int t -> dst 16*(t/8) + t%8 (low nibbles) and +8 (high nibbles), which lands sub-block b
    // exactly on dst ints [8b, 8b+8).
    x_qs[i * MMQ_TILE_X_K + 16 * (t / 8) + t % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
    x_qs[i * MMQ_TILE_X_K + 16 * (t / 8) + t % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
  }

  // 2 threads cooperate per row, 16 rows per warp.
#pragma unroll
  for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * 16) {
    const int i = min(i0 + (int)threadIdx.y * 16 + t / 2, i_max);
    const block_q4_K* b = (const block_q4_K*)x + kb0 + i * stride;
    const int* scales = (const int*)b->scales;
    const int ksc = t % 2;

    const int sc32 = unpack_scales_q4_K(scales, ksc + 0);
    const int m32  = unpack_scales_q4_K(scales, ksc + 2);
    const uint8_t* sc8 = (const uint8_t*)&sc32;
    const uint8_t* m8  = (const uint8_t*)&m32;

    const half2 dm = __hmul2(make_half2(b->d, b->dmin), make_half2(1.0f, -1.0f));  // (d, -dmin)
#pragma unroll
    for (int l = 0; l < 4; l++)
      x_dm[i * MMQ_TILE_X_K + 4 * ksc + l] = __hmul2(dm, make_half2((float)sc8[l], (float)m8[l]));
  }
}

// ============================================================================================
// The dot product. One warp owns rows [i0, i0+16) and walks all MMQ_X columns.
// k00 selects the first or second 128-k half of the 256-k tile (in ints: 0 or 32).
//
// The int32 accumulator is REZEROED every 32 k-values and folded into the fp sum, because the scale
// changes every 32 values. That is not a missed optimization -- it is the arithmetic.
// ============================================================================================
template <bool IS_Q4_K>
__device__ __forceinline__ void vec_dot(const int* __restrict__ tile_x, const int* __restrict__ tile_y,
                                        float* __restrict__ sum, const int k00) {
  const int t = threadIdx.x;
  const int i0 = threadIdx.y * 16;

  const int* x_qs = tile_x;
  const float* x_df = (const float*)(x_qs + 2 * MMQ_TILE_NE_K);
  const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
  const int* y_qs = tile_y + 4;      // the 4 leading ints are the ds4 scales
  const half2* y_ds = (const half2*)tile_y;

  uint32_t A[MMQ_TILE_NE_K / QI8_1][4];
  float dA[MMQ_TILE_NE_K / QI8_1][MMQ_NA];    // Q4_0: d ; Q4_K: d*sc
  float mA[MMQ_TILE_NE_K / QI8_1][MMQ_NA];    // Q4_K: -dmin*m
#pragma unroll
  for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
    const int k0 = k00 + k01;
    mmq_ldmatrix_A(A[k01 / QI8_1],
                   x_qs + (i0 + t % 16) * MMQ_TILE_X_K + k0 + (t / 16) * 4);
#pragma unroll
    for (int a = 0; a < MMQ_NA; a++) {
      const int i = i0 + (t / 4) + 8 * a;      // the two weight rows this lane accumulates
      if (IS_Q4_K) {
        const float2 dm = __half22float2(x_dm[i * MMQ_TILE_X_K + k0 / QI8_1]);
        dA[k01 / QI8_1][a] = dm.x;
        mA[k01 / QI8_1][a] = dm.y;
      } else {
        dA[k01 / QI8_1][a] = x_df[i * MMQ_TILE_X_K + k0 / QI8_1];
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_TILE_N) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
      uint32_t B[MMQ_B_NE];
#pragma unroll
      for (int l = 0; l < MMQ_B_NE; l++)
        B[l] = y_qs[(j0 + b_n(t, l)) * MMQ_TILE_Y_K + k01 + b_k(t, l)];

      float dB[MMQ_NB], sB[MMQ_NB];
#pragma unroll
      for (int l = 0; l < MMQ_NB; l++) {
        const int j = j0 + acc_j(t, l);        // acc_j(t, l) for l < MMQ_NB enumerates this lane's columns
        const float2 ds = __half22float2(y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
        dB[l] = ds.x;
        sB[l] = ds.y;
      }

      int C[MMQ_C_NE] = {0};
      mmq_mma(C, A[k01 / QI8_1], B);

#pragma unroll
      for (int l = 0; l < MMQ_C_NE; l++) {
        const int a = ACC_A_IDX(l), b = ACC_B_IDX(l);
        float* s = &sum[(j0 / MMQ_TILE_N) * MMQ_C_NE + l];
        *s += dA[k01 / QI8_1][a] * dB[b] * (float)C[l];
        if (IS_Q4_K) *s += mA[k01 / QI8_1][a] * sB[b];  // rank-1 min correction, no tensor core
      }
    }
  }
}

// ============================================================================================
// Mainloop. Plain 2D tiling, single-buffered, no cp.async -- deliberately. Get the arithmetic and
// the layouts right first; the pipeline is a perf question and it comes after the PPU tells us
// whether the int8 tensor core is even there.
// ============================================================================================
// MMQ_MIN_BLOCKS: cap registers so this many blocks fit per CU. MEASURED, and it is worth 40% on
// Q4_K. Left to itself the compiler gives Q4_K 254 registers at MMQ_X=64, which is ONE block (8 of
// 64 warps) per SM -- there are no spare warps to hide the memory latency behind, and the tensor
// core idles. Forcing 2 blocks caps it at 128 regs:
//     MIN_BLOCKS   regs   Q4_K TOP/s   Q4_0 TOP/s
//         1        254      70.1         72.8
//         2        128      98.2        111.9     <- peak
//         3         80      59.1        110.2     (Q4_K now SPILLS; Q4_0 has room to spare)
// The peak is at 2 for exactly the reason it was on Marlin: it is the largest block count you can
// buy before the register cap starts spilling. Q4_K spills first because its min-correction path
// carries a second set of per-row scales (mA) that Q4_0 does not have.
#ifndef MMQ_MIN_BLOCKS
#define MMQ_MIN_BLOCKS 2
#endif
template <bool IS_Q4_K>
__global__ __launch_bounds__(MMQ_WARP* MMQ_NWARPS, MMQ_MIN_BLOCKS) void mul_mat_q(
    const char* __restrict__ x, const block_q8_1_mmq* __restrict__ y, float* __restrict__ dst,
    const int N, const int K, const int M, const int stride_row_x) {
  extern __shared__ int smem[];
  int* tile_y = smem;
  int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;

  const int qk = IS_Q4_K ? QK_K : QK4_0;
  const int blocks_per_iter = MMQ_ITER_K / qk;   // Q4_0: 8, Q4_K: 1

  const int i_tile = blockIdx.x;   // 128 weight rows
  const int j_tile = blockIdx.y;   // MMQ_X columns
  const int i_max = N - i_tile * MMQ_Y - 1;

  float sum[MMQ_SUM_NE] = {0.0f};

  const char* xb = x + (size_t)i_tile * MMQ_Y * stride_row_x * (IS_Q4_K ? sizeof(block_q4_K) : sizeof(block_q4_0));
  const int* yb = (const int*)(y + (size_t)j_tile * MMQ_X);

  for (int kb0 = 0; kb0 < K / qk; kb0 += blocks_per_iter) {
    if (IS_Q4_K) load_tiles_q4_K(xb, tile_x, kb0, i_max, stride_row_x);
    else         load_tiles_q4_0(xb, tile_x, kb0, i_max, stride_row_x);

    const int kslab = kb0 * qk / 128;   // which 128-k slab of Y
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int* y0 = yb + (size_t)(kslab + half) * M * MMQ_TILE_Y_K;
      for (int l = threadIdx.y * MMQ_WARP + threadIdx.x; l < MMQ_X * MMQ_TILE_Y_K;
           l += MMQ_NWARPS * MMQ_WARP)
        tile_y[l] = y0[l];
      __syncthreads();
      vec_dot<IS_Q4_K>(tile_x, tile_y, sum, half * MMQ_TILE_NE_K);
      __syncthreads();
    }
  }

  // Write back. dst is column-major (ggml's convention): dst[j*N + i].
  const int t = threadIdx.x;
  const int i0 = threadIdx.y * 16;
#pragma unroll
  for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_TILE_N)
#pragma unroll
    for (int l = 0; l < MMQ_C_NE; l++) {
      const int j = j_tile * MMQ_X + j0 + acc_j(t, l);
      const int i = i_tile * MMQ_Y + i0 + acc_i(t, l);
      if (i < N && j < M) dst[(size_t)j * N + i] = sum[(j0 / MMQ_TILE_N) * MMQ_C_NE + l];
    }
}

// ============================================================================================
// PIPELINED Q4_K MAINLOOP (-DMMQ_PIPE).  This is Marlin's structure grafted onto MMQ.
//
// WHY THE REWRITE.  Upstream MMQ cannot use cp.async AT ALL, and the reason is structural: it
// dequantizes on the global->shared path (global -> registers -> nibble extract -> shared), and
// cp.async is a pure DMA that cannot transform anything in flight. So MMQ is stuck with a
// synchronous load and 4 __syncthreads per 256-k iteration, and it shows -- upstream itself only
// reaches 12-15% of the int8 tensor core.
//
// Marlin does the opposite: cp.async the RAW packed weights into shared, and dequantize on the
// shared->register path. That is what this does.
//
//   shared holds RAW block_q4_K super-blocks. An A fragment register is then just
//       word = sh[row*36 + 4 + (b/2)*8 + kw];   A = (word >> 4*(b%2)) & 0x0F0F0F0F
//   -- one LDS + one shift + one and. No ldmatrix, no normalized tile_x, no shared write pass.
//
// Two facts make Q4_K fit this like a glove, and both are worth stating because they are why the
// design works rather than merely compiles:
//
//   (1) sizeof(block_q4_K) == 144 == 9*16, so cp.async's 16 B granularity divides a super-block
//       EXACTLY, and every block start is 16 B aligned. (block_q4_0 is 18 B and is not even 4 B
//       aligned -- it cannot be cp.async'd at all without repacking. Q4_K keeps the old path.)
//
//   (2) The 36-int row stride is bank-conflict-free FOR FREE, no padding. For an A-fragment read
//       lane t wants row (t/4 + const) and k-word (t%4 + const), so
//           bank = (36*(t/4) + (t%4) + C) % 32 = (4*(t/4) + (t%4) + C) % 32 = (t + C) % 32
//       because 36 == 32 + 4. All 32 lanes land on 32 distinct banks. The B read out of the y tile
//       has the same 36-int stride and so gets the same property.
//
// Registers also drop: the old vec_dot hoisted A for 4 k-tiles at once (16 regs) plus dA/mA (16).
// Here A is rebuilt per sub-block (4 regs) and still reused across every j0 -- identical reuse, ~24
// fewer live registers. That matters: this kernel was proven register-bound (MIN_BLOCKS=2 was +40%).
// ============================================================================================
#ifdef MMQ_PIPE

// STAGES=2 MEASURED BEST -- and 3/4 are not just neutral, they LOSE (137.9 / 136.1 / 116.3 at
// STAGES=2/3/4, MMQ_X=64). Each extra stage costs 36 KB of shared, and shared is now what caps
// blocks per CU (72 KB -> 3 blocks, 108 KB -> 2). Two stages already hide the latency; a third buys
// buffer we don't need at the price of the occupancy that was hiding it. Real trade-off, not noise.
#ifndef MMQ_STAGES
#define MMQ_STAGES 2
#endif

#define SH_X_INTS (MMQ_Y * 36)          // 128 raw q4_K super-blocks, 36 ints each
#define SH_Y_INTS (MMQ_X * 2 * 36)      // MMQ_X columns x two 128-k slabs

__device__ __forceinline__ void cp_async16(int* smem, const void* g) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(s), "l"(g), "n"(16));
}
__device__ __forceinline__ void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// Stage the raw weights + the raw quantized activations for ONE 256-k iteration. Both are bulk 16 B
// copies with no transform, which is exactly what cp.async can do and what upstream's design forbids.
__device__ __forceinline__ void fetch_stage(int* sh_x, int* sh_y, const char* __restrict__ x,
                                            const int* __restrict__ y, const int kb0, const int i_max,
                                            const int stride_row, const int M) {
  const int tid = threadIdx.y * MMQ_WARP + threadIdx.x;

  // weights: 128 rows x 9 chunks of 16 B
  for (int c = tid; c < MMQ_Y * 9; c += MMQ_NWARPS * MMQ_WARP) {
    const int row = min(c / 9, i_max);
    const int ch = c % 9;
    const char* g = x + ((size_t)row * stride_row + kb0) * sizeof(block_q4_K) + ch * 16;
    cp_async16(sh_x + (c / 9) * 36 + ch * 4, g);
  }
  // activations: two 128-k slabs, each MMQ_X contiguous block_q8_1_mmq (144 B) -> MMQ_X*9 chunks
  for (int c = tid; c < MMQ_X * 2 * 9; c += MMQ_NWARPS * MMQ_WARP) {
    const int slab = c / (MMQ_X * 9);
    const int r = c % (MMQ_X * 9);
    const int* g = y + (size_t)(2 * kb0 + slab) * M * MMQ_TILE_Y_K + r * 4;
    cp_async16(sh_y + slab * (MMQ_X * 36) + r * 4, g);
  }
}

__global__ __launch_bounds__(MMQ_WARP* MMQ_NWARPS, MMQ_MIN_BLOCKS) void mul_mat_q_pipe(
    const char* __restrict__ x, const block_q8_1_mmq* __restrict__ y, float* __restrict__ dst,
    const int N, const int K, const int M, const int stride_row) {
  extern __shared__ int smem[];
  int* sh_x = smem;                                   // [MMQ_STAGES][SH_X_INTS]
  int* sh_y = smem + MMQ_STAGES * SH_X_INTS;          // [MMQ_STAGES][SH_Y_INTS]

  const int t = threadIdx.x;
  const int i0 = threadIdx.y * 16;
  const int i_tile = blockIdx.x, j_tile = blockIdx.y;
  const int i_max = N - i_tile * MMQ_Y - 1;
  const int nkb = K / QK_K;

  const char* xb = x + (size_t)i_tile * MMQ_Y * stride_row * sizeof(block_q4_K);
  const int* yb = (const int*)(y + (size_t)j_tile * MMQ_X);

  float sum[MMQ_SUM_NE] = {0.0f};

  // The scale/min unpack is per (row, super-block) and is loop-invariant across j0, so it is done
  // once per stage per lane -- not once per mma, which is where upstream pays for it.
  const int wid_a[MMQ_NA] = {0, 8};

#pragma unroll
  for (int s = 0; s < MMQ_STAGES - 1; s++) {
    if (s < nkb) fetch_stage(sh_x + s * SH_X_INTS, sh_y + s * SH_Y_INTS, xb, yb, s, i_max, stride_row, M);
    cp_async_fence();
  }

  int cur = 0;
  for (int kb = 0; kb < nkb; kb++) {
    cp_async_wait<MMQ_STAGES - 2>();
    __syncthreads();                       // ONE barrier per 256-k iteration (upstream needs four)

    // Issue the next stage BEFORE computing, so the DMA overlaps the mmas. The slot being written is
    // (cur + STAGES-1) % STAGES, which was last read STAGES-1 iterations ago and is therefore already
    // past the barrier above -- that is what makes one barrier enough.
    const int nxt = kb + MMQ_STAGES - 1;
    if (nxt < nkb) {
      const int ns = (cur + MMQ_STAGES - 1) % MMQ_STAGES;
      fetch_stage(sh_x + ns * SH_X_INTS, sh_y + ns * SH_Y_INTS, xb, yb, nxt, i_max, stride_row, M);
    }
    cp_async_fence();

    const int* X = sh_x + cur * SH_X_INTS;
    const int* Y = sh_y + cur * SH_Y_INTS;

    // per-row scales for this super-block: 8 (d*sc_b, -dmin*m_b) pairs, for each of the lane's 2 rows
    half2 dm[MMQ_NA][8];
#pragma unroll
    for (int a = 0; a < MMQ_NA; a++) {
      const int r = i0 + t / 4 + wid_a[a];
      const int* h = X + r * 36;
      const half2 d_nd = __hmul2(*(const half2*)h, make_half2(1.0f, -1.0f));   // (d, -dmin)
      const int sc03 = unpack_scales_q4_K(h + 1, 0), sc47 = unpack_scales_q4_K(h + 1, 1);
      const int m03  = unpack_scales_q4_K(h + 1, 2), m47  = unpack_scales_q4_K(h + 1, 3);
      const uint8_t* s03 = (const uint8_t*)&sc03; const uint8_t* s47 = (const uint8_t*)&sc47;
      const uint8_t* q03 = (const uint8_t*)&m03;  const uint8_t* q47 = (const uint8_t*)&m47;
#pragma unroll
      for (int l = 0; l < 4; l++) {
        dm[a][l]     = __hmul2(d_nd, make_half2((float)s03[l], (float)q03[l]));
        dm[a][l + 4] = __hmul2(d_nd, make_half2((float)s47[l], (float)q47[l]));
      }
    }

#pragma unroll
    for (int b = 0; b < 8; b++) {          // 8 sub-blocks of 32 values == 256 k
      // --- A straight out of the raw nibbles. One LDS + shift + and per register. ---
      uint32_t A[4];
#pragma unroll
      for (int l = 0; l < 4; l++) {
        const int w = X[(i0 + a_row(t, l)) * 36 + 4 + (b / 2) * 8 + a_kw(t, l)];
        A[l] = (uint32_t)(w >> (4 * (b % 2))) & 0x0F0F0F0Fu;
      }
      float dA[MMQ_NA], mA[MMQ_NA];
#pragma unroll
      for (int a = 0; a < MMQ_NA; a++) {
        const float2 v = __half22float2(dm[a][b]);
        dA[a] = v.x; mA[a] = v.y;
      }

      const int* Ys = Y + (b / 4) * (MMQ_X * 36);   // which 128-k slab
      const int kb32 = b % 4;                        // which 32-value block inside it

#pragma unroll
      for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_TILE_N) {
        uint32_t B[MMQ_B_NE];
#pragma unroll
        for (int l = 0; l < MMQ_B_NE; l++)
          B[l] = Ys[(j0 + b_n(t, l)) * 36 + 4 + 8 * kb32 + b_k(t, l)];

        float dB[MMQ_NB], sB[MMQ_NB];
#pragma unroll
        for (int l = 0; l < MMQ_NB; l++) {
          // half2 is 4 B == the width of an int, so a column's half2 index is its int index: 36, not 18.
          const float2 ds = __half22float2(((const half2*)Ys)[(j0 + acc_j(t, l)) * 36 + kb32]);
          dB[l] = ds.x; sB[l] = ds.y;
        }

        int C[MMQ_C_NE] = {0};
        mmq_mma(C, A, B);
#pragma unroll
        for (int l = 0; l < MMQ_C_NE; l++) {
          float* s = &sum[(j0 / MMQ_TILE_N) * MMQ_C_NE + l];
          *s += dA[ACC_A_IDX(l)] * dB[ACC_B_IDX(l)] * (float)C[l];
          *s += mA[ACC_A_IDX(l)] * sB[ACC_B_IDX(l)];
        }
      }
    }
    cur = (cur + 1) % MMQ_STAGES;
  }

#pragma unroll
  for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_TILE_N)
#pragma unroll
    for (int l = 0; l < MMQ_C_NE; l++) {
      const int j = j_tile * MMQ_X + j0 + acc_j(t, l);
      const int i = i_tile * MMQ_Y + i0 + acc_i(t, l);
      if (i < N && j < M) dst[(size_t)j * N + i] = sum[(j0 / MMQ_TILE_N) * MMQ_C_NE + l];
    }
}
#endif  // MMQ_PIPE


// ============================================================================================
// One launcher, so the self-check and the bench cannot drift apart in how they configure the kernel
// -- which is exactly how a "the test passes but the bench is wrong" bug gets in.
// ============================================================================================
struct MMQLaunch { size_t shmem; void (*kern)(const char*, const block_q8_1_mmq*, float*, int, int, int, int); };

static MMQLaunch mmq_launch(bool is_q4_K) {
  MMQLaunch L;
#ifdef MMQ_PIPE
  if (is_q4_K) {
    L.shmem = (size_t)MMQ_STAGES * (SH_X_INTS + SH_Y_INTS) * sizeof(int);
    L.kern = mul_mat_q_pipe;
    CUDA_CHECK(cudaFuncSetAttribute(L.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.shmem));
    return L;
  }
#endif
  L.shmem = (size_t)(MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * sizeof(int);
  L.kern = is_q4_K ? mul_mat_q<true> : mul_mat_q<false>;
  CUDA_CHECK(cudaFuncSetAttribute(L.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.shmem));
  return L;
}

// ============================================================================================
// Host: quantizers + the two references.
// ============================================================================================
static void quantize_q4_0_host(const float* w, block_q4_0* out, int n) {
  for (int b = 0; b < n / QK4_0; b++) {
    const float* s = w + b * QK4_0;
    float amax = 0, vmax = 0;
    for (int i = 0; i < QK4_0; i++)
      if (fabsf(s[i]) > amax) { amax = fabsf(s[i]); vmax = s[i]; }
    const float d = vmax / -8.0f;
    const float id = d ? 1.0f / d : 0.0f;
    out[b].d = __float2half(d);
    for (int i = 0; i < QK4_0 / 2; i++) {
      const int x0 = fminf(15, (int)(s[i] * id + 8.5f));
      const int x1 = fminf(15, (int)(s[i + 16] * id + 8.5f));
      out[b].qs[i] = (uint8_t)(x0 | (x1 << 4));
    }
  }
}

// Q4_K packing. We do NOT need llama.cpp's exact k-means quantizer -- we need a VALID Q4_K block
// with NON-TRIVIAL, VARYING d / dmin / sc_b / m_b, because the kernel's job is to reproduce the
// dequantization, not to be a good quantizer. Flat scales would pass a kernel that ignores them.
static void quantize_q4_K_host(const float* w, block_q4_K* out, int n, std::mt19937& rng) {
  std::uniform_int_distribution<int> d6(1, 63);
  for (int b = 0; b < n / QK_K; b++) {
    const float* s = w + b * QK_K;
    uint8_t sc[8], mn[8];
    for (int g = 0; g < 8; g++) { sc[g] = (uint8_t)d6(rng); mn[g] = (uint8_t)d6(rng); }

    float gmax = 0;
    for (int i = 0; i < QK_K; i++) gmax = fmaxf(gmax, fabsf(s[i]));
    const float d = gmax / (63.0f * 15.0f) + 1e-6f;   // so that d*sc*q spans the data
    const float dmin = d * 0.5f;
    out[b].d = __float2half(d);
    out[b].dmin = __float2half(dmin);

    // pack the 8 six-bit scales and 8 six-bit mins into 12 bytes (llama.cpp's layout)
    uint8_t* q = out[b].scales;
    memset(q, 0, 12);
    for (int g = 0; g < 4; g++) { q[g] = sc[g] & 63; q[g + 4] = mn[g] & 63; }
    for (int g = 4; g < 8; g++) {
      q[g + 4] = (uint8_t)((sc[g] & 0xF) | ((mn[g] & 0xF) << 4));
      q[g - 4] |= (uint8_t)((sc[g] >> 4) << 6);
      q[g - 0] |= (uint8_t)((mn[g] >> 4) << 6);
    }

    // quantize: w ~= d*sc_g*q - dmin*m_g
    for (int g = 0; g < 8; g++) {
      const float ds = d * sc[g], dm = dmin * mn[g];
      for (int i = 0; i < 32; i++) {
        const int k = g * 32 + i;
        int qv = (int)lroundf((s[k] + dm) / (ds > 0 ? ds : 1.0f));
        qv = qv < 0 ? 0 : (qv > 15 ? 15 : qv);
        const int byte = (g / 2) * 32 + i;
        if (g % 2 == 0) out[b].qs[byte] = (uint8_t)qv;
        else            out[b].qs[byte] |= (uint8_t)(qv << 4);
      }
    }
  }
}

// Read back the 8 scales/mins exactly the way unpack_scales_q4_K does, to prove host and device agree.
static void get_scale_min_k4(const uint8_t* q, uint8_t* sc, uint8_t* mn) {
  for (int g = 0; g < 4; g++) { sc[g] = q[g] & 63; mn[g] = q[g + 4] & 63; }
  for (int g = 4; g < 8; g++) {
    sc[g] = (uint8_t)((q[g + 4] & 0xF) | ((q[g - 4] >> 6) << 4));
    mn[g] = (uint8_t)((q[g + 4] >> 4)  | ((q[g - 0] >> 6) << 4));
  }
}

// ref_exact: the kernel's own formula, from the quantized X and quantized Y, in double.
static void ref_exact(bool is_q4_K, const void* xq, const std::vector<block_q8_1_mmq>& yq,
                      std::vector<double>& dst, int N, int K, int M) {
  const int nslab = K / 128;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      double acc = 0;
      for (int kb = 0; kb < K / 32; kb++) {          // 32-value blocks
        const block_q8_1_mmq& yb = yq[(kb / 4) * M + j];
        const float dy = __half2float(__low2half(yb.ds4[kb % 4]));
        const float sy = __half2float(__high2half(yb.ds4[kb % 4]));
        const int8_t* qy = yb.qs + (kb % 4) * 32;

        if (is_q4_K) {
          const block_q4_K* b = (const block_q4_K*)xq + (size_t)i * (K / QK_K) + kb / 8;
          uint8_t sc[8], mn[8];
          get_scale_min_k4(b->scales, sc, mn);
          const int g = kb % 8;
          // ggml forms the per-sub-block scales with a HALF2 multiply (load_tiles_q4_K:
          // x_dm[..] = dm * make_half2(sc8[l], m8[l])), so d*sc and dmin*m are each ROUNDED TO HALF
          // before they ever reach the accumulator. The reference has to round the same way or it
          // is not modelling the kernel: computing these in float leaves a ~2.6e-4 residual, which
          // is half's epsilon, and a tolerance loose enough to swallow that would also swallow a
          // real bug. (Doing the product in float would be strictly more accurate -- but we want
          // bit-compatibility with llama.cpp, so this is faithful, not a defect to fix.)
          const float dx = __half2float(__float2half(__half2float(b->d) * sc[g]));
          const float mx = -__half2float(__float2half(__half2float(b->dmin) * mn[g]));
          int isum = 0;
          for (int t = 0; t < 32; t++) {
            const int byte = (g / 2) * 32 + t;
            const int qv = (g % 2 == 0) ? (b->qs[byte] & 0xF) : (b->qs[byte] >> 4);
            isum += qv * (int)qy[t];
          }
          acc += (double)dx * dy * isum + (double)mx * sy;
        } else {
          const block_q4_0* b = (const block_q4_0*)xq + (size_t)i * (K / QK4_0) + kb;
          const float dx = __half2float(b->d);
          int isum = 0;
          for (int t = 0; t < 32; t++) {
            const int qv = (t < 16) ? (b->qs[t] & 0xF) : (b->qs[t - 16] >> 4);
            isum += (qv - 8) * (int)qy[t];
          }
          acc += (double)dx * dy * isum;
        }
      }
      dst[(size_t)j * N + i] = acc;
    }
  }
  (void)nslab;
}

// ============================================================================================

static bool run_case(bool is_q4_K, int N, int K, int M) {
  const char* name = is_q4_K ? "Q4_K" : "Q4_0";
  const int qk = is_q4_K ? QK_K : QK4_0;
  const int blocks_per_row = K / qk;
  const size_t bsz = is_q4_K ? sizeof(block_q4_K) : sizeof(block_q4_0);

  std::mt19937 rng(12345 + N + K * 7 + M * 13);
  std::normal_distribution<float> nd(0.f, 1.f);

  std::vector<float> W((size_t)N * K), Y((size_t)M * K);
  for (auto& v : W) v = nd(rng);
  for (auto& v : Y) v = nd(rng);

  std::vector<char> Xq((size_t)N * blocks_per_row * bsz);
  for (int i = 0; i < N; i++) {
    if (is_q4_K) quantize_q4_K_host(W.data() + (size_t)i * K, (block_q4_K*)Xq.data() + (size_t)i * blocks_per_row, K, rng);
    else         quantize_q4_0_host(W.data() + (size_t)i * K, (block_q4_0*)Xq.data() + (size_t)i * blocks_per_row, K);
  }

  // Y on device, quantized by the device kernel (so the reference reads back exactly what the GEMM sees).
  const int K_pad = ((K + 511) / 512) * 512;
  const size_t ny = (size_t)(K_pad / 128) * M + MMQ_X;   // + slop: the mainloop copies a full MMQ_X tile
  float* dYf; block_q8_1_mmq* dY; char* dX; float* dC;
  CUDA_CHECK(cudaMalloc(&dYf, Y.size() * 4));
  CUDA_CHECK(cudaMalloc(&dY, ny * sizeof(block_q8_1_mmq)));
  CUDA_CHECK(cudaMalloc(&dX, Xq.size()));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)N * M * 4));
  CUDA_CHECK(cudaMemcpy(dYf, Y.data(), Y.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dX, Xq.data(), Xq.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dY, 0, ny * sizeof(block_q8_1_mmq)));
  CUDA_CHECK(cudaMemset(dC, 0, (size_t)N * M * 4));

  quantize_q8_1_mmq<<<dim3(M, K_pad / 512), 128>>>(dYf, dY, K, K_pad, M);
  CUDA_CHECK(cudaGetLastError());

  const MMQLaunch L = mmq_launch(is_q4_K);
  dim3 grid((N + MMQ_Y - 1) / MMQ_Y, (M + MMQ_X - 1) / MMQ_X);
  L.kern<<<grid, dim3(MMQ_WARP, MMQ_NWARPS), L.shmem>>>(dX, dY, dC, N, K, M, blocks_per_row);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> C((size_t)N * M);
  std::vector<block_q8_1_mmq> Yq(ny);
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Yq.data(), dY, ny * sizeof(block_q8_1_mmq), cudaMemcpyDeviceToHost));
  cudaFree(dYf); cudaFree(dY); cudaFree(dX); cudaFree(dC);

  std::vector<double> Rex((size_t)N * M);
  ref_exact(is_q4_K, Xq.data(), Yq, Rex, N, K, M);

  // vs the kernel's own formula -- any gap here is a KERNEL bug.
  double maxrel = 0; int bi = -1, bj = -1;
  double scale = 0;
  for (size_t i = 0; i < Rex.size(); i++) scale = fmax(scale, fabs(Rex[i]));
  for (int j = 0; j < M; j++)
    for (int i = 0; i < N; i++) {
      const double r = fabs(C[(size_t)j * N + i] - Rex[(size_t)j * N + i]) / (scale + 1e-9);
      if (r > maxrel) { maxrel = r; bi = i; bj = j; }
    }

  // vs the honest fp32 GEMM -- this is the QUANTIZATION error, not a bug. Reported so that a real
  // kernel bug can't hide inside it.
  double qerr = 0, qden = 0;
  for (int j = 0; j < M; j += (M > 8 ? M / 8 : 1))
    for (int i = 0; i < N; i += (N > 8 ? N / 8 : 1)) {
      double t = 0;
      for (int k = 0; k < K; k++) t += (double)W[(size_t)i * K + k] * Y[(size_t)j * K + k];
      qerr += fabs(t - Rex[(size_t)j * N + i]);
      qden += fabs(t);
    }

  const bool bad = !(maxrel < 1e-4);
  printf("  %-4s N=%-6d K=%-6d M=%-4d  vs-kernel-formula rel %.2e %s   (quant err vs fp32: %.1f%%)\n",
         name, N, K, M, maxrel, bad ? "MISMATCH" : "ok", 100.0 * qerr / (qden + 1e-9));
  if (bad) printf("        worst at i=%d j=%d: got %.6f want %.6f\n", bi, bj,
                  C[(size_t)bj * N + bi], Rex[(size_t)bj * N + bi]);
  return bad;
}

// ============================================================================================
// Benchmark. COLD WEIGHTS, and that is not optional: the LLC is 64-128 MB and a single layer's
// weights (a 4096x14336 Q4_K tensor is 33 MB) fit inside it. Timing a loop that re-reads the same
// tensor measures the LLC, not HBM, and flatters the result -- this exact mistake made the decode
// GEMV numbers look good for a day. So we allocate a WORKSET of several weight copies and rotate
// through them, evicting each before it comes back around.
//
// The reported %peak needs a denominator: run mma_peak to get the PPU's achievable back-to-back
// int8 mma throughput. That is the ceiling for a kernel built out of those mmas.
// ============================================================================================
static void bench(bool is_q4_K, int N, int K, int M, double peak_top) {
  const int qk = is_q4_K ? QK_K : QK4_0;
  const int bpr = K / qk;
  const size_t bsz = is_q4_K ? sizeof(block_q4_K) : sizeof(block_q4_0);
  const size_t wbytes = (size_t)N * bpr * bsz;

  const int workset_mb = getenv("MMQ_WORKSET_MB") ? atoi(getenv("MMQ_WORKSET_MB")) : 512;
  int ncopy = (int)(((size_t)workset_mb << 20) / wbytes);
  if (ncopy < 1) ncopy = 1;

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> W((size_t)N * K), Y((size_t)M * K);
  for (auto& v : W) v = nd(rng);
  for (auto& v : Y) v = nd(rng);
  std::vector<char> Xq(wbytes);
  for (int i = 0; i < N; i++) {
    if (is_q4_K) quantize_q4_K_host(W.data() + (size_t)i * K, (block_q4_K*)Xq.data() + (size_t)i * bpr, K, rng);
    else         quantize_q4_0_host(W.data() + (size_t)i * K, (block_q4_0*)Xq.data() + (size_t)i * bpr, K);
  }

  std::vector<char*> dX(ncopy);
  for (int c = 0; c < ncopy; c++) {
    CUDA_CHECK(cudaMalloc(&dX[c], wbytes));
    CUDA_CHECK(cudaMemcpy(dX[c], Xq.data(), wbytes, cudaMemcpyHostToDevice));
  }

  const int K_pad = ((K + 511) / 512) * 512;
  const size_t ny = (size_t)(K_pad / 128) * M + MMQ_X;
  float* dYf; block_q8_1_mmq* dY; float* dC;
  CUDA_CHECK(cudaMalloc(&dYf, Y.size() * 4));
  CUDA_CHECK(cudaMalloc(&dY, ny * sizeof(block_q8_1_mmq)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)N * M * 4));
  CUDA_CHECK(cudaMemcpy(dYf, Y.data(), Y.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dY, 0, ny * sizeof(block_q8_1_mmq)));
  quantize_q8_1_mmq<<<dim3(M, K_pad / 512), 128>>>(dYf, dY, K, K_pad, M);

  const MMQLaunch L = mmq_launch(is_q4_K);
  dim3 grid((N + MMQ_Y - 1) / MMQ_Y, (M + MMQ_X - 1) / MMQ_X);

  const int iters = 50;
  for (int c = 0; c < ncopy; c++)  // warm the instruction cache, not the data
    L.kern<<<grid, dim3(MMQ_WARP, MMQ_NWARPS), L.shmem>>>(dX[c], dY, dC, N, K, M, bpr);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));
  for (int it = 0; it < iters; it++)
    L.kern<<<grid, dim3(MMQ_WARP, MMQ_NWARPS), L.shmem>>>(dX[it % ncopy], dY, dC, N, K, M, bpr);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  CUDA_CHECK(cudaGetLastError());

  const double us = ms * 1e3 / iters;
  const double tops = 2.0 * N * K * M / (us * 1e-6) / 1e12;
  const double gbs = (double)wbytes / (us * 1e-6) / 1e9;

  printf("  %-4s N=%-6d K=%-6d M=%-5d  %8.1f us  %7.1f TOP/s%s  %7.0f GB/s  (workset %d x %.0f MB)\n",
         is_q4_K ? "Q4_K" : "Q4_0", N, K, M, us, tops,
         peak_top > 0 ? "" : "", gbs, ncopy, wbytes / 1048576.0);
  if (peak_top > 0) printf("        -> %.1f%% of the %.0f TOP/s int8 mma peak\n", 100.0 * tops / peak_top, peak_top);

  for (int c = 0; c < ncopy; c++) cudaFree(dX[c]);
  cudaFree(dYf); cudaFree(dY); cudaFree(dC);
}

int main(int argc, char** argv) {
#ifdef MMQ_NV
  printf("=== MMQ standalone: NVIDIA mma.m16n8k32.s32.s8.s8.s32, MMQ_X=%d ===\n", MMQ_X);
#else
  printf("=== MMQ standalone: PPU ppu.mma.m16n16k32.s32.s8.s8.s32, MMQ_X=%d ===\n", MMQ_X);
#endif

  if (argc > 1 && strcmp(argv[1], "bench") == 0) {
    // MMQ_PEAK_TOPS: the int8 mma ceiling from mma_peak, so the %util line means something.
    const double peak = getenv("MMQ_PEAK_TOPS") ? atof(getenv("MMQ_PEAK_TOPS")) : 0.0;
    if (peak <= 0) printf("    (set MMQ_PEAK_TOPS=<int8 TOP/s from ./mma_peak> to get a %% of peak)\n");
    printf("    COLD weights: rotating a %s MB workset so the LLC cannot serve the reads.\n\n",
           getenv("MMQ_WORKSET_MB") ? getenv("MMQ_WORKSET_MB") : "512");
    // llama-ish prefill shapes, same ones the Marlin bench uses so the two are directly comparable.
    bench(true, 4096,  4096, 2048, peak);
    bench(true, 14336, 4096, 2048, peak);
    bench(true, 4096, 14336, 2048, peak);
    bench(false, 4096, 4096, 2048, peak);
    return 0;
  }

  bool bad = false;
  bad |= run_case(false, 128,  256,  64);   // one tile, one k-iteration
  bad |= run_case(false, 256,  512, 128);
  bad |= run_case(false, 512, 4096, 256);
  bad |= run_case(true,  128,  256,  64);
  bad |= run_case(true,  256,  512, 128);
  bad |= run_case(true,  512, 4096, 256);
  printf("\n%s\n", bad ? ">>> FAIL" : ">>> ALL CASES PASS");
  return bad ? 1 : 0;
}
