/*
 * Standalone Marlin INT4 GEMM Kernel
 * Extracted from Marlin with support for group_size=128
 *
 * Features:
 * - INT4 weight quantization with group size = 128
 * - Optimized for m=1 (inference batch size 1)
 * - Tensor Core acceleration with m16n8k16
 * - 4-stage async pipeline
 * - Interleaved weight layout for optimal memory access
 *
 * Walkthrough (M=1,N=2048,K=2048) with tile mapping + gmem/shmem addresses:
 *   docs/MARLIN_KERNEL_WALKTHROUGH_M1_N2048_K2048.md
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

// Disable createpolicy cache hints by default for broader arch compatibility.
#ifndef MARLIN_USE_CREATEPOLICY
#define MARLIN_USE_CREATEPOLICY 0
#endif

// Optional device-side tracing via printf (slow; intended for understanding/debug only).
// Enabled at runtime via the standalone CLI flag `--trace`.
struct MarlinTraceConfig {
  int enabled;  // enable device-side printf tracing
  int block;  // only trace this CTA (blockIdx.x)
  int thread;  // only trace this thread (threadIdx.x)
};  // end MarlinTraceConfig

__device__ __managed__ MarlinTraceConfig g_marlin_trace{0, 0, 0};  // runtime trace controls (managed)

// ============================================================================
// Utility Functions
// ============================================================================
__host__ __device__ constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

// Vector type for organizing registers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

// ============================================================================
// Matrix Fragment Types
// ============================================================================

using I4 = Vec<int, 4>;           // 4x int32 = 128 bits = 32x INT4 values
using FragA = Vec<half2, 4>;      // Input A fragment: 4x half2 = 16x FP16
using FragB = Vec<half2, 2>;      // Weight B fragment: 2x half2 = 4x FP16
using FragC = Vec<float, 4>;      // Accumulator fragment: 4x float
using FragS = Vec<half2, 1>;      // Scale fragment: 1x half2 = 2x FP16

// ============================================================================
// Async Memory Operations
// ============================================================================

// Predicated async global->shared copy for A matrix
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Async global->shared copy with evict_first hint for B matrix
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
#if MARLIN_USE_CREATEPOLICY
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
#else
  asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
#endif
}

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// ============================================================================
// Tensor Core MMA Operations
// ============================================================================

// m16n8k16 tensor core MMA instruction
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}

// m16n16k16 using two m16n8k16 instructions
__device__ inline void mma_m16n16k16(const FragA& a_frag, const FragB& frag_b0, const FragB& frag_b1, FragC& frag_c0, FragC& frag_c1) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b0 = reinterpret_cast<const uint32_t*>(&frag_b0);
  const uint32_t* b1 = reinterpret_cast<const uint32_t*>(&frag_b1);
  float* c0 = reinterpret_cast<float*>(&frag_c0);
  float* c1 = reinterpret_cast<float*>(&frag_c1);

  // First m16n8k16: output columns 0-7
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c0[0]), "=f"(c0[1]), "=f"(c0[2]), "=f"(c0[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b0[0]),  "r"(b0[1]),
       "f"(c0[0]),  "f"(c0[1]),  "f"(c0[2]),  "f"(c0[3])
  );

  // Second m16n8k16: output columns 8-15
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c1[0]), "=f"(c1[1]), "=f"(c1[2]), "=f"(c1[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b1[0]),  "r"(b1[1]),
       "f"(c1[0]),  "f"(c1[1]),  "f"(c1[2]),  "f"(c1[3])
  );
}

// Load 16x16 A matrix fragment from shared memory
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

// ============================================================================
// Dequantization Operations
// ============================================================================

// LOP3 instruction for efficient bit manipulation
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// Dequantize INT4 to FP16
// Input: int32 containing 8x INT4 values (4 bits each)
// Output: FragB with 4x FP16 values
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  // Symmetric INT4 with zero point at -8
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;

  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

// Apply quantization scale (for grouped quantization)
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// ============================================================================
// Barrier Synchronization
// ============================================================================

__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

// ============================================================================
// Main Marlin Kernel
// ============================================================================

template <
  const int threads,           // 256 threads per block
  const int thread_m_blocks,   // M dimension blocks per threadblock
  const int thread_n_blocks,   // N dimension blocks per threadblock
  const int thread_k_blocks,   // K dimension blocks per threadblock
  const int stages,            // Pipeline stages (4)
  const int group_blocks       // Blocks per quantization group (8 for groupsize=128)
>  // end template params
__global__ void Marlin(
  const int4* __restrict__ A,  // A: fp16 row-major, viewed as int4 (16B) for vectorized loads
  const int4* __restrict__ B,  // B: packed INT4 weights in Marlin interleaved layout (int4 = 16B)
        int4* __restrict__ C,  // C: fp16 row-major output, stored via int4 (16B) vector stores
  const int4* __restrict__ s,  // s: fp16 scales; groupsize=16*group_blocks (or per-column when group_blocks==-1)
  int  prob_m,                 // Batch size
  int  prob_n,                 // Output dimension
  int  prob_k,                 // Reduction dimension
  int* locks                   // Barrier synchronization
) {  // kernel entry
  // CTA tile shape:
  //   - M tile: 16 * thread_m_blocks rows
  //   - N tile: 16 * thread_n_blocks cols
  //   - K tile: 16 * thread_k_blocks (accumulated over k_tiles)
  int parallel = 1;  // problem-parallel factor (M slicing)
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);  // problem-parallel factor (M slicing)
    prob_m = 16 * thread_m_blocks;
  }

  // Logical tile grid:
  //   k_tiles = prob_k / (16 * thread_k_blocks)
  //   n_tiles = prob_n / (16 * thread_n_blocks)
  int k_tiles = prob_k / 16 / thread_k_blocks;  // number of K tiles for this launch
  int n_tiles = prob_n / 16 / thread_n_blocks;  // number of N tiles for this launch
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);  // stripe length per CTA (in K-tiles)

  // Ensure iterations align with group boundaries
  if (group_blocks != -1)
    iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));  // keep slices group-aligned for scale loads

  /*
   * Work decomposition ("slices"):
   * We flatten (slice_col_par, slice_row) in row-major order over the tile grid:
   *   linear = blockIdx.x * iters
   *   slice_col_par = linear / k_tiles          (also includes `parallel` when M is sliced)
   *   slice_row     = linear % k_tiles          (K-tile offset within this N-tile)
   *
   * Each CTA processes `slice_iters` consecutive K-tiles within one (slice_col_par) column.
   * If multiple CTAs contribute to the same N tile, we reduce with a global lock array:
   *   locks[slice_col] is the per-(N tile) barrier/reduction slot.
   */
  int slice_row = (iters * blockIdx.x) % k_tiles;  // tile row in K grid (idx%k_tiles)
  int slice_col_par = (iters * blockIdx.x) / k_tiles;  // tile col in N grid (idx/k_tiles, incl parallel)
  int slice_col = slice_col_par;  // tile col (N tile index)
  int slice_iters;  // K-tiles this CTA will process for current N tile
  int slice_count = 0;  // num blocks contributing to this col
  int slice_idx;  // this CTA's index within slice_count (barrier order)

  // Handle parallel execution
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;  // tile col (N tile index)
  }

  // Initialize slice parameters
  auto init_slice = [&] () {  // compute slice_iters/slice_count/slice_idx for this CTA
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);  // tiles this block does in current col
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;  // tiles this block does in current col
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;  // tiles this block does in current col
    slice_count = 1;  // num blocks contributing to this col
    slice_idx = 0;  // this block order within slice (for barrier)

    // If `iters < k_tiles`, multiple CTAs will land in the same slice_col_par and must be reduced.
    // This computes:
    //   - slice_count: total CTAs that touch this N tile
    //   - slice_idx:   this CTA's rank within that set (used by locks[slice_col])
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);  // num blocks contributing to this col
      if (col_off > 0)
        slice_count++;  // account for partial first segment
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;  // this block order within slice (for barrier)
      else {
        slice_idx = slice_count - 1 - delta_first / iters;  // this block order within slice (for barrier)
        if (col_off > 0)
          slice_idx--;  // adjust for non-zero col_off
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;  // tile col (N tile index)
    }
  };  // end init_slice
  init_slice();  // initialize slice state

  if (g_marlin_trace.enabled && blockIdx.x == g_marlin_trace.block && threadIdx.x == g_marlin_trace.thread) {
    printf("[marlin_trace] block=%d thread=%d\\n", (int) blockIdx.x, (int) threadIdx.x);
    printf("[marlin_trace] tm=%d tn=%d tk=%d stages=%d group_blocks=%d threads=%d\\n",
      thread_m_blocks, thread_n_blocks, thread_k_blocks, stages, group_blocks, threads);
    printf("[marlin_trace] prob_m=%d prob_n=%d prob_k=%d parallel=%d\\n", prob_m, prob_n, prob_k, parallel);
    printf("[marlin_trace] k_tiles=%d n_tiles=%d iters=%d\\n", k_tiles, n_tiles, iters);  // num K-tiles (each 128 K elems here)
    if (slice_iters == 0) {
      printf("[marlin_trace] slice: <inactive>\\n");
    } else {
      printf("[marlin_trace] slice_row=%d slice_col_par=%d slice_col=%d slice_iters=%d slice_count=%d slice_idx=%d\\n",
        slice_row, slice_col_par, slice_col, slice_iters, slice_count, slice_idx);
    }
  }

  // ========================================================================
  // Memory Access Parameters - A Matrix
  // ========================================================================
  int a_gl_stride = prob_k / 8;  // A row stride in int4 (K/8)
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;  // A shared stride per row in int4
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;  // A global delta (int4) per tid group
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);  // A global delta (int4) within a tile
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);  // A shared write delta (int4) per iter
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));  // A shared read delta across warps
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;  // A shared read delta across m-blocks
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);  // A shared bytes per stage (in int4)
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);  // A shared write iters per stage

  // ========================================================================
  // Memory Access Parameters - B Matrix (Weights)
  // ========================================================================
  int b_gl_stride = 16 * prob_n / 32;  // B stride in int4 (layout-specific)
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;  // B shared stride in int4
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;  // B global delta (int4) per k-tile
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);  // B global delta (int4) per load-iter
  constexpr int b_sh_wr_delta = threads;  // B shared write delta (int4)
  constexpr int b_sh_rd_delta = threads;  // B shared read delta (int4)
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;  // B shared int4 per stage
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;  // B shared write iters per stage

  // ========================================================================
  // Memory Access Parameters - Scales
  // ========================================================================
  int s_gl_stride = prob_n / 8;  // scale row stride in int4 (N/8)
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;  // scale shared stride in int4
  constexpr int s_sh_stage = s_sh_stride;  // scale shared int4 per stage
  int s_gl_rd_delta = s_gl_stride;  // scale global delta per group

  // ========================================================================
  // Compute Thread-Specific Indices
  // ========================================================================

  // A matrix indices
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A gmem read index (int4)
  a_gl_rd += a_gl_rd_delta_o * slice_row;  // A gmem read index (int4)
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A smem write index (int4)
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;  // A smem read index (int4)
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));  // A smem read index (int4)

  // B matrix indices
  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);  // B gmem read base (int4)
  b_gl_rd += b_sh_stride * slice_col;  // B gmem read base (int4)
  b_gl_rd += b_gl_rd_delta_o * slice_row;  // B gmem read base (int4)
  int b_sh_wr = threadIdx.x;  // B smem write base (int4)
  int b_sh_rd = threadIdx.x;  // B smem read base (int4)

  // Scale indices
  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;  // scale gmem read index (int4)
  int s_sh_wr = threadIdx.x;  // scale smem write index (int4)
  int s_sh_rd;  // scale smem read index (int4)
  if (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;  // scale smem read index (int4)
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;  // scale smem read index (int4)

  bool a_sh_wr_pred[a_sh_wr_iters];  // per-iter pred: A gmem loads are in-bounds
  #pragma unroll  // unroll pred init
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  if (g_marlin_trace.enabled && blockIdx.x == g_marlin_trace.block && threadIdx.x == g_marlin_trace.thread) {
    printf("[marlin_trace] a_gl_stride=%d a_sh_stride=%d\\n", a_gl_stride, a_sh_stride);  // A row stride in int4 (K/8)
    printf("[marlin_trace] a_gl_rd_delta_o=%d a_gl_rd_delta_i=%d a_sh_wr_delta=%d\\n",
      a_gl_rd_delta_o, a_gl_rd_delta_i, a_sh_wr_delta);
    printf("[marlin_trace] a_sh_rd_delta_o=%d a_sh_rd_delta_i=%d a_sh_stage=%d a_sh_wr_iters=%d\\n",
      a_sh_rd_delta_o, a_sh_rd_delta_i, a_sh_stage, a_sh_wr_iters);
    printf("[marlin_trace] a_gl_rd=%d a_sh_wr=%d a_sh_rd=%d a_sh_wr_pred0=%d\\n",
      a_gl_rd, a_sh_wr, a_sh_rd, (int) a_sh_wr_pred[0]);

    printf("[marlin_trace] b_gl_stride=%d b_sh_stride=%d\\n", b_gl_stride, b_sh_stride);  // B stride in int4 (layout-specific)
    printf("[marlin_trace] b_gl_rd_delta_o=%d b_gl_rd_delta_i=%d b_sh_stage=%d b_sh_wr_iters=%d\\n",
      b_gl_rd_delta_o, b_gl_rd_delta_i, b_sh_stage, b_sh_wr_iters);
    printf("[marlin_trace] b_gl_rd=%d b_sh_wr=%d b_sh_rd=%d\\n", b_gl_rd, b_sh_wr, b_sh_rd);  // B gmem read base (int4)

    printf("[marlin_trace] s_gl_stride=%d s_sh_stride=%d s_gl_rd=%d s_sh_rd=%d s_sh_wr_pred=%d\\n",
      s_gl_stride, s_sh_stride, s_gl_rd, s_sh_rd, (int) s_sh_wr_pred);
  }

  // XOR-based layout to avoid bank conflicts
  auto transform_a = [&] (int i) {  // apply XOR swizzle for A shared layout
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };  // end transform_a

  int a_sh_wr_trans[a_sh_wr_iters];  // A smem write indices after swizzle
  #pragma unroll  // unroll swizzle init
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);

  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];  // A smem read indices [k-subtile][m_block] after swizzle
  #pragma unroll  // unroll swizzle init
  for (int i = 0; i < b_sh_wr_iters; i++) {
    #pragma unroll  // unroll m-block init
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // B matrix pointers
  const int4* B_ptr[b_sh_wr_iters];  // per-iter B gmem pointers for cp.async
  #pragma unroll  // unroll pointer init
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  // ========================================================================
  // Shared Memory and Register Allocation
  // ========================================================================
  extern __shared__ int4 sh[];  // shared slab: A|B|S pipelines and temp reduce buffer
  int4* sh_a = sh;  // shared A pipeline base
  int4* sh_b = sh_a + (stages * a_sh_stage);  // shared B pipeline base
  int4* sh_s = sh_b + (stages * b_sh_stage);  // shared scale pipeline base

  FragA frag_a[2][thread_m_blocks];  // regs: A fragments (ping-pong over k)
  I4 frag_b_quant[2];  // regs: packed INT4 B fragments (ping-pong over k)
  FragC frag_c[thread_m_blocks][4][2];  // regs: accumulators [m_block][n_subtile][b_half]
  FragS frag_s[2][4];  // regs: scale fragments (ping-pong over k)

  // ========================================================================
  // Lambda Functions
  // ========================================================================

  // Zero accumulators
  auto zero_accums = [&] () {  // zero f32 accumulators in frag_c
    #pragma unroll  // unroll flat store
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;  // clear one f32 lane
  };  // end zero_accums

  // Fetch tile to shared memory
  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {  // cp.async A/B(/S) into smem stage=pipe
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;  // A stage base in smem
      #pragma unroll  // unroll A cp.async
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
          &sh_a_stage[a_sh_wr_trans[i]],  // smem dst (swizzled)
          &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],  // gmem src
          a_sh_wr_pred[i]  // in-bounds pred
        );  // end cp_async4_pred
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;  // B stage base in smem
      #pragma unroll  // unroll B cp.async
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;  // advance B gmem ptr by one k-tile
      }
      // Fetch scales once per quantization group (groupsize = 16 * group_blocks).
      if constexpr (group_blocks != -1) {
        constexpr int GROUP_PIPES = group_blocks / thread_k_blocks;  // K-tiles per scale row
        if (pipe % GROUP_PIPES == 0) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;  // S stage base in smem
          if (s_sh_wr_pred)
            cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
          s_gl_rd += s_gl_rd_delta;
        }
      }
    }
    cp_async_fence();
  };  // end fetch_to_shared

  // Wait for stage to complete
  auto wait_for_stage = [&] () {  // wait until the needed cp.async group is visible in smem
    cp_async_wait<stages - 2>();
    __syncthreads();  // make stage ready for ldmatrix/loads
  };  // end wait_for_stage

  // Fetch from shared to registers
  auto fetch_to_registers = [&] (int k, int pipe) {  // load A/B(/S) for k-subtile into regs
    if constexpr (group_blocks != -1) {
      // Scales are stored per-group; pin the stage to the group's first pipe.
      constexpr int GROUP_PIPES = group_blocks / thread_k_blocks;
      int4* sh_s_stage = sh_s + s_sh_stage * (GROUP_PIPES * (pipe / GROUP_PIPES));
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];  // load 2x half2 scales
    }
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;  // A stage base in smem
    #pragma unroll  // unroll ldmatrix
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;  // B stage base in smem
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);  // load packed INT4 weights
  };  // end fetch_to_registers

  // Execute matmul
  auto matmul = [&] (int k) {  // dequant + optional scale + mma for one k-subtile
    #pragma unroll  // unroll N subtiles
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j];  // 2x packed INT4 (low 8 nibbles)
      int b_quant_shift = b_quant >> 8;  // 2x packed INT4 (high 8 nibbles)

      FragB frag_b0 = dequant(b_quant);  // dequant low half
      if (group_blocks != -1)
        scale(frag_b0, frag_s[k % 2][j], 0);  // apply scale[0]

      FragB frag_b1 = dequant(b_quant_shift);  // dequant high half
      if (group_blocks != -1)
        scale(frag_b1, frag_s[k % 2][j], 1);  // apply scale[1]

      #pragma unroll  // unroll M blocks
      for (int i = 0; i < thread_m_blocks; i++) {
        mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };  // end matmul

  // Threadblock reduction:
  // Some configs produce multiple "replicas" of the same output fragment within one CTA
  // (due to how B is loaded / partitioned). We reduce those replicas here so that only
  // the first replica (red_idx==0, i.e. threadIdx.x < b_sh_stride) carries the final sums.
  //
  // Conceptually:
  //   - red_idx = threadIdx.x / b_sh_stride identifies which replica group a thread belongs to.
  //   - We do a tree-reduction over red_idx, staging partials in shared (as int4 = 16B = 4 floats).
  auto thread_block_reduce = [&] () {  // CTA-local reduce over red_idx replicas
    constexpr int red_off = threads / b_sh_stride / 2;  // initial tree offset = (num_replicas / 2)
    if (red_off >= 1) {  // only needed when we actually have multiple replicas
      int red_idx = threadIdx.x / b_sh_stride;  // replica id in [0, threads/b_sh_stride)
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;  // int4s per replica buffer in shared (8 * b_sh_stride)
      constexpr int red_sh_delta = b_sh_stride;  // int4 stride between j-slices within one replica buffer
      int red_sh_rd = red_sh_stride * red_idx + (threadIdx.x % b_sh_stride);  // this thread's base int4 slot in shared

      #pragma unroll  // unroll over m-blocks (each m_block is 16 rows)
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {  // reduce per m_block independently
        #pragma unroll  // unroll tree steps
        for (int i = red_off; i > 0; i /= 2) {  // i = reduction distance between source/dest replica groups
          if (i <= red_idx && red_idx < 2 * i) {  // upper-half replica groups write/reduce into lower-half
            #pragma unroll  // unroll over (n_subtile * b_half) = 4 * 2 = 8 fragments
            for (int j = 0; j < 4 * 2; j++) {  // j enumerates the 8 FragC fragments per m_block
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);  // dest shared slot (red_idx -> red_idx-i)
              if (i < red_off) {  // not the first step: shared already contains partials to accumulate
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);  // src partial (this red_idx)
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);  // dst partial (red_idx-i) from previous step
                #pragma unroll  // unroll 4-lane FragC accumulation
                for (int k = 0; k < 4; k++) {  // k enumerates the 4 float lanes inside one FragC
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];  // local += src + dst
                }
              }
              sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];  // write updated partial to shared
            }
          }
          __syncthreads();  // make partials visible before the next tree step
        }
        if (red_idx == 0) {  // replica-0 threads finalize by adding the last partial stored in shared
          #pragma unroll  // unroll over 8 fragments
          for (int i = 0; i < 4 * 2; i++) {  // i enumerates the 8 FragC fragments per m_block
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);  // reduced partial from other replicas
            #pragma unroll  // unroll 4-lane FragC accumulation
            for (int j = 0; j < 4; j++) {  // j enumerates the 4 float lanes inside one FragC
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];  // add reduced partial into replica-0
            }
          }
        }
        __syncthreads();  // keep shared reuse safe across m_block
      }
    }
  };

  // Global reduction across CTAs (when slice_count > 1):
  // - first=true:  first CTA for this (slice_col) => skip load, just write partial
  // - last=true:   last  CTA for this (slice_col) => accumulate but don't write partial
  auto global_reduce = [&] (bool first = false, bool last = false) {  // CTA-to-CTA reduction via global C
    constexpr int active_threads = 32 * thread_n_blocks / 4;  // participating threads (cover N tile as int4 lanes)
    if (threadIdx.x < active_threads) {  // only "writer" threads touch global C (others are inactive here)
      // C is treated as int4 where one int4 packs 8 fp16 values along N (int4 = 16B = 8 * fp16).
      // Thread mapping (tid < active_threads):
      //   warp = tid / 32, lane = tid % 32
      //   row8 = lane / 4   (0..7 rows within an 8-row chunk)
      //   col4 = lane % 4   (0..3 int4 columns within a warp)
      //   col8 = 4*warp+col4 (0..7 int4 columns across 2 warps = 64 fp16 columns)
      //
      // The loop over i below expands this base mapping to cover:
      //   - multiple 8-row chunks (via i/2)
      //   - left/right halves of the N tile (via i%2)
      int c_gl_stride = prob_n / 8;  // global C row stride in int4 (N/8)
      int c_gl_wr_delta_o = 8 * c_gl_stride;  // jump 8 rows in global C (int4 indexing)
      int c_gl_wr_delta_i = 4 * (active_threads / 32);  // jump to the next 64-column half-tile (in int4)
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;  // base (row8,col8)
      c_gl_wr += (2 * thread_n_blocks) * slice_col;  // N-tile offset (tile width = 2*thread_n_blocks int4)
      constexpr int c_sh_wr_delta = active_threads;  // shared staging stride: [i][tid] in int4
      int c_sh_wr = threadIdx.x;  // shared staging column = tid

      int row = (threadIdx.x % 32) / 4;  // row within the current 8-row chunk (for prob_m bounds)

      if (!first) {  // not the first CTA: load previous partials from global C
        #pragma unroll  // unroll staging over i
        for (int i = 0; i < thread_m_blocks * 4; i++) {  // i spans (16*thread_m_blocks) rows as 4 chunks per m_block
          cp_async4_pred(  // stage one int4 (8 fp16) partial into shared
            &sh[c_sh_wr + c_sh_wr_delta * i],  // smem dst: sh[i][tid]
            &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],  // gmem src: (row chunk, N half)
            i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m  // pred: tail rows in last m_block
          );
        }
        cp_async_fence();  // commit cp.async group
        cp_async_wait<0>();  // wait until all staged partials are visible in shared
      }

      #pragma unroll  // unroll accumulation / partial writeback over i
      for (int i = 0; i < thread_m_blocks * 4; i++) {  // i selects (8-row chunk, N half) within the CTA tile
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {  // bounds for the last m_block
          if (!first) {  // accumulate previously-written partial into the local f32 accumulators
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];  // loaded partial: 8 fp16 values packed in int4
            #pragma unroll  // unroll unpack + accumulate of 8 fp16 values
            for (int j = 0; j < 2 * 4; j++) {  // j in [0..7] over the 8 half values inside int4
              reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=  // select matching lane
                __half2float(reinterpret_cast<__half*>(&c_red)[j]);  // fp16 -> f32 and accumulate
            }
          }
          if (!last) {  // not the last CTA: write partial back for the next CTA to consume
            int4 c;  // pack 8 fp16 values back into int4
            #pragma unroll  // unroll pack of 8 halves
            for (int j = 0; j < 2 * 4; j++) {  // j in [0..7]
              reinterpret_cast<__half*>(&c)[j] = __float2half(  // f32 -> fp16
                reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]  // same lane mapping as above
              );
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;  // global partial store (int4)
          }
        }
      }
    }
  };

  // Write final result for this CTA's (M,N) tile.
  // Pack fp16 results into shared (as half2) and stream out as int4 (16B) stores.
  auto write_result = [&] () {  // final writeback of this CTA's tile to global C
    // Shared-memory writeback staging:
    // - First, a subset of threads packs f32 accumulators into shared as half2 (fp16), with padding to avoid bank conflicts.
    // - Then, all threads stream shared -> global as int4 (each int4 = 8 fp16 values = 16 bytes).
    int c_gl_stride = prob_n / 8;  // global C row stride in int4 (N/8)
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;  // shared row stride in int4 (tile width + 1 padding)
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));  // per-iteration global int4 stride for this thread
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));  // per-iteration shared int4 stride for this thread

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));  // base global int4 (row,col)
    c_gl_wr += (2 * thread_n_blocks) * slice_col;  // offset to this N tile (tile width = 2*thread_n_blocks int4)
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;  // base shared half2 index (warp-friendly layout)
    c_sh_wr += 32 * (threadIdx.x / 32);  // separate warps in half2 indexing (each warp writes its own columns)
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));  // base shared int4 index (contiguous stream order)

    int c_gl_wr_end = c_gl_stride * prob_m;  // end of valid global C region (in int4) for prob_m rows

    auto write = [&] (int idx, float c0, float c1, FragS& s) {  // pack 2 f32 -> half2, optionally scale, store to shared
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));  // f32 -> fp16 pair (2 columns)
      if (group_blocks == -1)  // per-column mode: apply scale at the end (grouped mode already scaled during matmul)
        res = __hmul2(res, s[0]);  // scale both halves
      ((half2*) sh)[idx] = res;  // shared write as half2 (4 bytes)
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {  // only the warps that own columns pack frag_c -> shared
      #pragma unroll  // unroll m_block packing
      for (int i = 0; i < thread_m_blocks; i++) {  // i selects the m_block (16 rows each)
        #pragma unroll  // unroll 16-col chunks per warp (4 * 16 = 64 cols per warp)
        for (int j = 0; j < 4; j++) {  // j selects a 16-column chunk within this warp's half-tile
          int wr = c_sh_wr + 8 * j;  // half2 index: 8 half2 = 16 columns per chunk
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);  // rows 0..7,  cols 0..7
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);  // rows 8..15, cols 0..7
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);  // rows 0..7,  cols 8..15
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);  // rows 8..15, cols 8..15
        }
        c_sh_wr += 16 * (4 * c_sh_stride);  // advance shared half2 base by 16 rows for the next m_block
      }
    }
    __syncthreads();  // ensure shared packing is complete before streaming out

    #pragma unroll  // unroll per-thread streaming loop
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {  // cover all rows in the CTA tile
      if (c_gl_wr < c_gl_wr_end) {  // guard the prob_m tail
        C[c_gl_wr] = sh[c_sh_rd];  // global store of one int4 (8 fp16) from shared
        c_gl_wr += c_gl_wr_delta;  // advance global int4 index for the next iteration
        c_sh_rd += c_sh_rd_delta;  // advance shared int4 index for the next iteration
      }
    }
  };

  // ========================================================================
  // Pipeline Initialization and Main Loop
  // ========================================================================

	  // Start pipeline:
	  // Prefetch (stages-1) tiles so the main loop can overlap:
	  //   ldmatrix/dequant/mma on stage p  with  cp.async filling stage (p+stages-1).
	  auto start_pipes = [&] () {  // prologue: fill pipeline and prime registers for k=0
	    #pragma unroll  // unroll stage prefetches
	    for (int i = 0; i < stages - 1; i++)  // prefetch (stages-1) K-tiles into shared
	      fetch_to_shared(i, i, i < slice_iters);  // stage=i, a_off=i, pred guards tail tiles
	    zero_accums();  // clear frag_c accumulators before accumulation
	    wait_for_stage();  // wait until stage 0 data is ready in shared
	    fetch_to_registers(0, 0);  // load A/B(/S) for k=0 from stage 0 into regs
	    a_gl_rd += a_gl_rd_delta_o * (stages - 1);  // A gmem read index (int4)
	  };
	  start_pipes();  // kick off the pipeline for this slice

	  // Main loop
	  while (slice_iters) {  // iterate over K-tiles assigned to this CTA for the current N tile
	    #pragma unroll  // unroll over stages (small fixed count)
	    for (int pipe = 0; pipe < stages;) {  // pipe is the logical K-tile index within this unrolled chunk
	      #pragma unroll  // unroll B-subtiles inside one stage
	      for (int k = 0; k < b_sh_wr_iters; k++) {  // k iterates over B sub-tiles within a stage
	        fetch_to_registers(k + 1, pipe % stages);  // prefetch next k-subtile into regs (ping-pong buffers)
	        if (k == b_sh_wr_iters - 2) {  // when we're about to use the last subtile, rotate the shared stage
	          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);  // cp.async next K-tile into the "oldest" stage
	          pipe++;  // advance logical pipe (stage consumer)
	          wait_for_stage();  // wait until the newly-fetched stage is ready for ldmatrix/loads
	        }
	        matmul(k);  // dequant/scale (if needed) + tensorcore MMA for subtile k
	      }
	      slice_iters--;  // one K-tile fully consumed
	      if (slice_iters == 0)  // slice complete
	        break;  // exit stage loop early
	    }
	    a_gl_rd += a_gl_rd_delta_o * stages;  // A gmem read index (int4)

	    if (slice_iters == 0) {  // finished all K-tiles for this N tile => reduce + writeback
	      cp_async_wait<0>();  // ensure no pending cp.async reads are outstanding
	      bool last = slice_idx == slice_count - 1;  // last CTA for this N tile (does final writeback)

	      if (group_blocks == -1 && last) {  // per-column mode: fetch scales once at the end (applied in write_result)
	        if (s_sh_wr_pred)  // only threads within the scale tile issue the load
	          cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]);  // stage per-column scales into sh_s
	        cp_async_fence();  // commit scale cp.async
	      }

	      thread_block_reduce();  // CTA-local reduce (collapse red_idx replicas into red_idx==0)

	      if (group_blocks == -1 && last) {  // per-column mode: bring scales into regs for write_result()
	        cp_async_wait<0>();  // wait for scale cp.async
	        __syncthreads();  // make sh_s visible to all threads
	        if (threadIdx.x / 32 < thread_n_blocks / 4) {  // only active writer warps need scales
	          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];  // load scales for cols 0..?
	          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];  // load scales for cols +8..?
	        }
	      }

	      if (slice_count > 1) {  // multiple CTAs contribute to the same N tile => global reduction needed
	        barrier_acquire(&locks[slice_col], slice_idx);  // wait until it's this CTA's turn to reduce
	        global_reduce(slice_idx == 0, last);  // first writes partial, middle reduces+writes, last reduces only
	        barrier_release(&locks[slice_col], last);  // signal next CTA (or reset on last)
	      }

	      if (last)  // only last CTA writes the final, fully reduced output tile
	        write_result();  // pack to shared and store to global C

	      slice_row = 0;  // tile row in K grid (idx%k_tiles)
	      slice_col_par++;  // advance to the next N-tile (including parallel slices)
	      slice_col++;  // advance N-tile within this parallel slice
	      init_slice();  // compute next slice_iters/slice_count/slice_idx

	      if (slice_iters) {  // if there is another slice to process, rewind pointers and restart pipeline
	        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A gmem read index (int4)

	        // Move B pointers to the next N tile and rewind K back to slice_row=0 for the new slice.
	        #pragma unroll
	        for (int i = 0; i < b_sh_wr_iters; i++)  // adjust each per-iter B pointer
	          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;  // move to next N tile and rewind K by k_tiles
	        if (slice_col == 0) {  // wrapped around to the next parallel slice
	          #pragma unroll
	          for (int i = 0; i < b_sh_wr_iters; i++)  // adjust each per-iter B pointer
	            B_ptr[i] -= b_gl_stride;  // move B back by one row stride to align slice_col wrap
	        }

	        // Reset scale pointer for the new N tile (grouped mode will advance it inside fetch_to_shared()).
	        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;  // scale gmem read index (int4)
	        start_pipes();  // restart prologue for the next slice
	      }
	    }
	  }
}

// ============================================================================
// Kernel Launch Configuration
// ============================================================================

const int THREADS = 256;  // threads per CTA
const int STAGES = 4;  // cp.async pipeline stages
const int SHARED_MEM = 96 * 1024;  // 96 KB shared memory

// Dispatch helper: select the exact Marlin<...> instantiation for (tm, tn, tk, group_blocks).
#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS)                              \
  else if (thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&                        \
           thread_k_blocks == THREAD_K_BLOCKS && group_blocks == GROUP_BLOCKS) {                              \
    cudaFuncSetAttribute(                                                                                     \
      Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>,               \
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);                                               \
    Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>                  \
      <<<blocks, THREADS, SHARED_MEM, stream>>>(A_ptr, B_ptr, C_ptr, s_ptr, prob_m, prob_n, prob_k, locks);  \
  }

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
) {
  int tot_m = prob_m;
  int tot_m_blocks = ceildiv(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      // Small-M: favor a smaller N tile and larger K tile (better latency for GEMV-ish shapes).
      thread_k = 128;
      thread_n = 128;
    } else {
      // Larger-M: favor a larger N tile (better throughput).
      thread_k = 64;
      thread_n = 256;
    }
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 || (group_blocks != -1 && prob_k % group_blocks != 0))
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4* A_ptr = (const int4*) A;
  const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C;
  const int4* s_ptr = (const int4*) s;

  int* locks = (int*) workspace;

  int ret = 0;
  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par)
        par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }

    if (false) {}
    // Supported instantiations:
    //   (thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks)
    // where group_blocks=-1 means per-column scales and group_blocks=8 means groupsize=128.
    CALL_IF(1,  8,  8, -1)
    CALL_IF(1,  8,  8,  8)
    CALL_IF(1, 16,  4, -1)
    CALL_IF(1, 16,  4,  8)
    CALL_IF(2, 16,  4, -1)
    CALL_IF(2, 16,  4,  8)
    CALL_IF(3, 16,  4, -1)
    CALL_IF(3, 16,  4,  8)
    CALL_IF(4, 16,  4, -1)
    CALL_IF(4, 16,  4,  8)
    else
      ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }

  return ret;
}

void print_usage(const char* prog_name) {
  printf("Usage: %s [OPTIONS]\n", prog_name);
  printf("\nOptions:\n");
  printf("  -m <M>          Batch size (default: 128)\n");
  printf("  -n <N>          Output dimension (default: 256)\n");
  printf("  -k <K>          Input dimension (default: 512)\n");
  printf("  -g <groupsize>  Quantization group size, -1 for per-column (default: -1)\n");
  printf("  -s <sms>        Number of SMs to use, -1 for auto (default: -1)\n");
  printf("  -w <warmup>     Warmup iterations (default: 10)\n");
  printf("  -i <iters>      Timed iterations (default: 100)\n");
  printf("  --ncu           Nsight Compute mode: warmup=0 and default iters=1000\n");  // warmup iters
  printf("  --trace         Print a small device-side trace (printf) for one block/thread (implies warmup=0 iters=1)\n");  // warmup iters
  printf("  --trace_block B Select which CTA (blockIdx.x) to trace (default: 0)\n");
  printf("  --trace_thread T Select which threadIdx.x to trace (default: 0)\n");
  printf("  -h              Show this help message\n");
  printf("\nExample:\n");
  printf("  %s -m 128 -n 4096 -k 4096 -g 128 -s 108\n", prog_name);
}

int main(int argc, char* argv[]) {
  // Default parameters
  int M = 128;           // Batch size
  int N = 256;           // Output dimension
  int K = 512;           // Input dimension
  int groupsize = -1;    // -1 for per-column quantization
  int num_sms = -1;      // -1 for auto-detect
  int warmup = 10;  // warmup iters
  int iters = 100;  // timed iters
  bool iters_set = false;
  bool ncu_mode = false;
  bool trace = false;  // enable device printf trace
  int trace_block = 0;  // CTA to trace
  int trace_thread = 0;  // thread to trace

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      M = atoi(argv[++i]);  // GEMM M
    } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      N = atoi(argv[++i]);  // GEMM N
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      K = atoi(argv[++i]);  // GEMM K
    } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
      groupsize = atoi(argv[++i]);  // quant group size
    } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
      num_sms = atoi(argv[++i]);  // SM count used for launch
    } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
      warmup = atoi(argv[++i]);  // warmup iters
    } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      iters = atoi(argv[++i]);  // timed iters
      iters_set = true;
    } else if (strcmp(argv[i], "--ncu") == 0) {
      ncu_mode = true;
    } else if (strcmp(argv[i], "--trace") == 0) {
      trace = true;  // enable device printf trace
    } else if (strcmp(argv[i], "--trace_block") == 0 && i + 1 < argc) {
      trace_block = atoi(argv[++i]);  // CTA to trace
    } else if (strcmp(argv[i], "--trace_thread") == 0 && i + 1 < argc) {
      trace_thread = atoi(argv[++i]);  // thread to trace
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      printf("Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      return 1;
    }
  }

  if (ncu_mode) {
    // No separate warmup loop; use ncu --launch-skip/--launch-count to pick a single iteration.
    warmup = 0;  // warmup iters
    if (!iters_set) {
      iters = 1000;  // timed iters
    }
  }
  if (trace) {
    // Device printf is very slow; keep this to a single launch by default.
    warmup = 0;  // warmup iters
    iters = 1;  // timed iters
    g_marlin_trace.enabled = 1;
    g_marlin_trace.block = trace_block;
    g_marlin_trace.thread = trace_thread;
  } else {
    g_marlin_trace.enabled = 0;
  }

  // Validate parameters
  if (M <= 0 || N <= 0 || K <= 0) {
    printf("Error: M, N, K must be positive integers\n");
    return 1;
  }
  if (warmup < 0 || iters <= 0) {
    printf("Error: warmup must be >= 0 and iters must be > 0\n");
    return 1;
  }

  // Print configuration
  printf("========================================\n");
  printf("Marlin INT4 GEMM Configuration\n");
  printf("========================================\n");
  printf("Matrix dimensions:\n");
  printf("  M (batch size):      %d\n", M);
  printf("  N (output dim):      %d\n", N);
  printf("  K (input dim):       %d\n", K);
  if (groupsize == -1) {
    printf("  Group size:          per-column\n");
  } else {
    printf("  Group size:          %d\n", groupsize);
  }
  if (num_sms == -1) {
  printf("  Number of SMs:       auto-detect\n");
  } else {
    printf("  Number of SMs:       %d\n", num_sms);
  }
  printf("  Warmup iterations:   %d\n", warmup);
  printf("  Timed iterations:    %d\n", iters);
  printf("========================================\n\n");

  // Allocate device memory
  void *d_A, *d_B, *d_C, *d_s, *d_workspace;
  size_t A_size = M * K * sizeof(half);  // A bytes
  size_t B_size = (K * N) / 2;  // 4-bit packed weights
  size_t C_size = M * N * sizeof(half);  // C bytes
  size_t s_size = (groupsize == -1) ? N * sizeof(half) : (K / groupsize) * N * sizeof(half);
  size_t workspace_size = (N / 128) * 16 * sizeof(int);  // locks bytes

  printf("Allocating device memory...\n");
  printf("  A matrix:      %.2f MB\n", A_size / (1024.0 * 1024.0));
  printf("  B matrix:      %.2f MB\n", B_size / (1024.0 * 1024.0));
  printf("  C matrix:      %.2f MB\n", C_size / (1024.0 * 1024.0));
  printf("  Scales:        %.2f MB\n", s_size / (1024.0 * 1024.0));
  printf("  Workspace:     %.2f KB\n\n", workspace_size / 1024.0);

  cudaMalloc(&d_A, A_size);
  cudaMalloc(&d_B, B_size);
  cudaMalloc(&d_C, C_size);
  cudaMalloc(&d_s, s_size);
  cudaMalloc(&d_workspace, workspace_size);

  // Initialize with deterministic data
  printf("Initializing with deterministic data...\n");
  {
    // A: all 1.0 in FP16
    std::vector<half> h_A(M * K);
    for (auto& v : h_A) v = __float2half(1.0f);
    cudaMemcpy(d_A, h_A.data(), A_size, cudaMemcpyHostToDevice);

    // B: packed INT4 weights, nibbles cycle 0,1,2,...,15,0,1,...
    // Each byte packs two INT4 values: low nibble | (high nibble << 4)
    std::vector<uint8_t> h_B(B_size);
    for (size_t i = 0; i < h_B.size(); i++) {
      uint8_t lo = (2 * i) % 16;       // 0,1,2,...,15,0,1,...
      uint8_t hi = (2 * i + 1) % 16;   // 1,2,3,...,15,0,1,...
      h_B[i] = lo | (hi << 4);
    }
    cudaMemcpy(d_B, h_B.data(), B_size, cudaMemcpyHostToDevice);

    // Scales: all 1.0 in FP16
    int num_scale_rows = (groupsize == -1) ? 1 : K / groupsize;
    std::vector<half> h_s(num_scale_rows * N);
    for (auto& v : h_s) v = __float2half(1.0f);
    cudaMemcpy(d_s, h_s.data(), s_size, cudaMemcpyHostToDevice);
  }
  cudaMemset(d_C, 0, C_size);
  cudaMemset(d_workspace, 0, workspace_size);

  printf("Running kernel...\n");

  for (int i = 0; i < warmup; ++i) {
    marlin_cuda(
      d_A, d_B, d_C, d_s,
      M, N, K,
      d_workspace,
      groupsize,
      0,           // device 0
      0,           // default stream
      -1,          // auto thread_k
      -1,          // auto thread_n
      num_sms,     // custom number of SMs
      16
    );
    cudaError_t warmup_err = cudaGetLastError();
    if (warmup_err != cudaSuccess) {
      printf("Warmup launch failed: %s\n", cudaGetErrorString(warmup_err));
      return 1;
    }
  }
  cudaDeviceSynchronize();

  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  int result = 0;
  for (int i = 0; i < iters; ++i) {
    result = marlin_cuda(
      d_A, d_B, d_C, d_s,
      M, N, K,
      d_workspace,
      groupsize,
      0,           // device 0
      0,           // default stream
      -1,          // auto thread_k
      -1,          // auto thread_n
      num_sms,     // custom number of SMs
      16
    );
    if (result != 0) {
      break;
    }
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      printf("Timed launch failed: %s\n", cudaGetErrorString(launch_err));
      result = launch_err;
      break;
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (result == 0) {
    printf("\n");
    printf("========================================\n");
    printf("SUCCESS!\n");
    printf("========================================\n");
    printf("Kernel executed successfully!\n");
    printf("Matrix multiplication: (%d x %d) * (%d x %d) = (%d x %d)\n", M, K, K, N, M, N);
    printf("Avg kernel time: %.3f us (%d iters, %d warmup)\n",
           (ms * 1000.0f) / (float)iters, iters, warmup);
    printf("========================================\n");
  } else if (result == ERR_PROB_SHAPE) {
    printf("\n");
    printf("========================================\n");
    printf("ERROR: Problem shape incompatible\n");
    printf("========================================\n");
    printf("The problem dimensions are not compatible with kernel constraints.\n");
    printf("Ensure:\n");
    printf("  - N is divisible by thread_n (128 or 256)\n");
    printf("  - K is divisible by thread_k (64 or 128)\n");
    printf("  - If using group quantization, K is divisible by groupsize\n");
    printf("========================================\n");
  } else if (result == ERR_KERN_SHAPE) {
    printf("\n");
    printf("========================================\n");
    printf("ERROR: No kernel implementation\n");
    printf("========================================\n");
    printf("No kernel implementation available for these parameters.\n");
    printf("Try different M, N, K, or groupsize values.\n");
    printf("========================================\n");
  }


  // === Verify output is not all zeros ===
  if (result == 0) {
    std::vector<half> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C, C_size, cudaMemcpyDeviceToHost);

    int nonzero = 0;
    float min_val = 1e30f, max_val = -1e30f, sum = 0.0f;
    for (int i = 0; i < M * N; i++) {
      float v = __half2float(h_C[i]);
      if (v != 0.0f) nonzero++;
      if (v < min_val) min_val = v;
      if (v > max_val) max_val = v;
      sum += v;
    }
    float mean = sum / (M * N);

    printf("\n========================================\n");
    printf("Output Verification\n");
    printf("========================================\n");
    printf("  Total elements:  %d\n", M * N);
    printf("  Non-zero:        %d (%.1f%%)\n", nonzero, 100.0f * nonzero / (M * N));
    printf("  Min value:       %.6f\n", min_val);
    printf("  Max value:       %.6f\n", max_val);
    printf("  Mean value:      %.6f\n", mean);
    if (nonzero > 0) {
      printf("  Status:          PASS (output is non-zero)\n");
    } else {
      printf("  Status:          FAIL (output is all zeros!)\n");
      result = 1;
    }
    printf("========================================\n");
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_s);
  cudaFree(d_workspace);

  return result;
}
