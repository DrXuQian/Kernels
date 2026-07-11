/*
 * INT8 tensor-core smoke test: does the PPU have an m16n16k32 s32.s8.s8.s32 MMA, and what is its
 * fragment layout?
 *
 * This is the FIRST GATE for porting llama.cpp's MMQ (quantized matmul) to the PPU. MMQ is not a
 * fp16 GEMM like Marlin: it dequantizes the INT4 weights to INT8, quantizes the activations to
 * INT8 (Q8_1), and does the whole K-loop in an INTEGER tensor core --
 *     NVIDIA: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
 * If the PPU has no int8 MMA, the entire MMQ approach is dead on arrival and we fall back to dp4a
 * (or abandon MMQ's numerics and dequantize to fp16, which is just Marlin again).
 *
 * Build:
 *   NVIDIA (H800, validates the harness + ggml's layouts on real hw):  nvcc -arch=sm_90 -o s mma_int8_smoke.cu
 *   PPU    (box, ppu001, NO -arch):                                    nvcc -DUSE_PPU -o s mma_int8_smoke.cu
 *
 * Run: ./s          (runs all 3 probes)
 *
 * THE PROBES (they are staged so that a FAILURE LOCALIZES, instead of just saying "nope"):
 *
 *   probe 0  K-INVARIANT.  A[i][k] = a_i and B[j][k] = b_j, constant along k. Then
 *                C[i][j] = sum_k a_i*b_j = 32*a_i*b_j,
 *            which is INDEPENDENT of how k is distributed across a fragment's registers. So any
 *            k-permutation inside A or B still passes. It tests ONLY the m/n row mapping of the
 *            operands and the layout of the accumulator.
 *
 *   probe 1  STRUCTURED.  Chosen so the exact answer is C[i][j] = 16*i + j -- every accumulator
 *            value literally spells out its own (row, col). Also k-invariant-ish, so on a mismatch
 *            the DUMP below reads off the TRUE accumulator layout directly, by eye.
 *
 *   probe 2  RANDOM.  Full random A and B. This is the only probe that exercises the k-layout.
 *
 *   probe 0 pass + probe 2 fail  =>  the k-distribution inside the A or B fragment is wrong.
 *   probe 0 fail                 =>  the m/n mapping or the accumulator layout is wrong; probe 1's
 *                                    dump tells you what the accumulator layout actually is.
 *
 * On mismatch the kernel dumps the raw per-lane registers so the real layout can be recovered
 * rather than guessed at. A smoke test that can only say MATCH/MISMATCH would leave us re-deriving
 * the layout blind.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------
// Fragment layouts.
//
// NVIDIA mma.m16n8k32.s8 -- ESTABLISHED ON HARDWARE by probe 1 below, not copied from mma.cuh:
//   A  4 regs:  i = lane/4 + 8*(l%2)      kint = (lane%4) + 4*(l/2)
//   B  2 regs:  n = lane/4                kint = (lane%4) + 4*l
//   C  4 regs:  i = lane/4 + 8*(l/2)      j    = 2*(lane%4) + (l%2)
// "kint" indexes 32-bit words, i.e. groups of 4 int8: k bytes [4*kint, 4*kint+4).
//
// NOTE, because it cost a debug cycle: ggml's tile<16,8,int>::get_i/get_j give i = 8*(l/2) + lane/4
// and kint = 2*(lane%4) + (l%2). Those describe the ACCUMULATOR (C), not the A operand -- C and A
// happen to share the tile TYPE but not the register distribution, and MMQ fills the A tile with
// ldmatrix (hardware lays it out) rather than through get_i/get_j. Filling A from get_i/get_j
// silently feeds rows 8-15 the wrong data. ggml's own Turing fallback confirms the layout above:
// it splits m16n8k32 into 4x m8n8k16 with A.x[1] and A.x[3] driving D.x[2],D.x[3] (= rows 8-15),
// i.e. the row bit is l%2, and A.x[0..1] pair with B.x[0] (k 0-15), i.e. the k bit is l/2.
//
// PPU HYPOTHESIS -- by analogy with the VERIFIED fp16 ppu.mma.m16n16k16 in marlin_classic_ppu.cuh:
//   frag_i(lane,l) = lane/4 + 8*(l/2)        <- same formula for the A row (m) and the B row (n)
//   frag_j(lane,l) = lane%4 + 4*(l%2)        <- k-group; for fp16 a half2, here a 4x int8 word
//   acc_i(lane,l)  = lane/4 + 8*((l>>2)&1)   \  8 s32 regs, 16x16
//   acc_j(lane,l)  = lane%4 + 4*(l%4)        /
// and the n16 B operand is the two n8 tiles CONCATENATED: {b0[0], b0[1], b1[0], b1[1]}.
// Register counts all line up (A 4, B 4, D 8 = 16*16 s32 / 32 lanes), which is why this is worth
// testing -- but a register count matching is not evidence that llc can select the instruction.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ int nv_a_i(int lane, int l) { return lane / 4 + 8 * (l % 2); }
__device__ __forceinline__ int nv_a_k(int lane, int l) { return (lane % 4) + 4 * (l / 2); }
__device__ __forceinline__ int nv_b_n(int lane, int l) { return lane / 4; }
__device__ __forceinline__ int nv_b_k(int lane, int l) { return 4 * l + (lane % 4); }
__device__ __forceinline__ int nv_c_i(int lane, int l) { return 8 * (l / 2) + lane / 4; }
__device__ __forceinline__ int nv_c_j(int lane, int l) { return 2 * (lane % 4) + (l % 2); }

__device__ __forceinline__ int ppu_frag_i(int lane, int l) { return lane / 4 + 8 * (l / 2); }
__device__ __forceinline__ int ppu_frag_j(int lane, int l) { return lane % 4 + 4 * (l % 2); }
__device__ __forceinline__ int ppu_acc_i(int lane, int l) { return lane / 4 + (((l >> 2) & 1) << 3); }
__device__ __forceinline__ int ppu_acc_j(int lane, int l) { return lane % 4 + ((l % 4) << 2); }

// Pack 4 int8 from row `r` of a 16x32 int8 matrix, starting at k = 4*kint.
__device__ __forceinline__ uint32_t pack4(const int8_t* M, int r, int kint) {
  uint32_t w = 0;
  for (int b = 0; b < 4; b++)
    w |= (uint32_t)(uint8_t)M[r * 32 + 4 * kint + b] << (8 * b);
  return w;
}

// ---------------------------------------------------------------------------------------------
// The kernel. One warp. A is 16x32 int8 (row-major), B is 16x32 int8 (n-major, i.e. ".col"):
//     C[i][j] = sum_{k<32} A[i][k] * B[j][k]
// dump_regs (32 lanes x 8) gets the raw accumulator so a mismatch can be decoded.
// ---------------------------------------------------------------------------------------------
__global__ void mma_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                           int* __restrict__ C, int* __restrict__ dump_regs) {
  const int lane = threadIdx.x;

#ifdef USE_PPU
  uint32_t a[4], b[4];
  int d[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  for (int l = 0; l < 4; l++) a[l] = pack4(A, ppu_frag_i(lane, l), ppu_frag_j(lane, l));
  // B: the two n8 halves of the n16 operand, concatenated -- b[0..1] are cols 0-7, b[2..3] cols 8-15.
  // ppu_frag_i already folds "+8" into l>=2, so this is the same expression as A.
  for (int l = 0; l < 4; l++) b[l] = pack4(B, ppu_frag_i(lane, l), ppu_frag_j(lane, l));

  asm volatile(
      "ppu.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));

  for (int l = 0; l < 8; l++) {
    C[ppu_acc_i(lane, l) * 16 + ppu_acc_j(lane, l)] = d[l];
    dump_regs[lane * 8 + l] = d[l];
  }
#else
  // NVIDIA: two m16n8k32 MMAs, n=0-7 and n=8-15, exactly as MMQ does it.
  uint32_t a[4];
  for (int l = 0; l < 4; l++) a[l] = pack4(A, nv_a_i(lane, l), nv_a_k(lane, l));

  for (int half = 0; half < 2; half++) {
    uint32_t b[2];
    for (int l = 0; l < 2; l++) b[l] = pack4(B, 8 * half + nv_b_n(lane, l), nv_b_k(lane, l));

    int d[4] = {0, 0, 0, 0};
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

    for (int l = 0; l < 4; l++) {
      C[nv_c_i(lane, l) * 16 + (8 * half + nv_c_j(lane, l))] = d[l];
      dump_regs[lane * 8 + 4 * half + l] = d[l];
    }
  }
#endif
}

// ---------------------------------------------------------------------------------------------

static void cpu_ref(const int8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++) {
      int acc = 0;
      for (int k = 0; k < 32; k++) acc += (int)A[i * 32 + k] * (int)B[j * 32 + k];
      C[i * 16 + j] = acc;
    }
}

static const char* PROBE_NAME[3] = {
    "0 K-INVARIANT (m/n mapping + acc layout only; k-permutation-blind)",
    "1 STRUCTURED  (exact answer is C[i][j] = 16*i + j)",
    "2 RANDOM      (the only probe that tests the k-layout)"};

// Returns true on mismatch.
static bool run_probe(int probe) {
  int8_t hA[16 * 32], hB[16 * 32];

  if (probe == 0) {
    // A[i][k] = a_i, B[j][k] = b_j -> C[i][j] = 32*a_i*b_j. Keep |a|,|b| small: 32*3*3 = 288, fits.
    for (int i = 0; i < 16; i++)
      for (int k = 0; k < 32; k++) hA[i * 32 + k] = (int8_t)((i % 7) - 3);
    for (int j = 0; j < 16; j++)
      for (int k = 0; k < 32; k++) hB[j * 32 + k] = (int8_t)((j % 5) - 2);
  } else if (probe == 1) {
    // C[i][j] = A[i][0]*B[j][0] + A[i][1]*B[j][1] = i*16 + 1*j. Every acc value spells out (i,j).
    for (int i = 0; i < 16; i++)
      for (int k = 0; k < 32; k++) hA[i * 32 + k] = (k == 0) ? (int8_t)i : (k == 1 ? 1 : 0);
    for (int j = 0; j < 16; j++)
      for (int k = 0; k < 32; k++) hB[j * 32 + k] = (k == 0) ? 16 : (k == 1 ? (int8_t)j : 0);
  } else {
    srand(1234 + probe);
    for (int i = 0; i < 16 * 32; i++) hA[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < 16 * 32; i++) hB[i] = (int8_t)(rand() % 255 - 127);
  }

  int hRef[256];
  cpu_ref(hA, hB, hRef);

  int8_t *dA, *dB;
  int *dC, *dDump;
  CUDA_CHECK(cudaMalloc(&dA, 16 * 32));
  CUDA_CHECK(cudaMalloc(&dB, 16 * 32));
  CUDA_CHECK(cudaMalloc(&dC, 256 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dDump, 32 * 8 * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dA, hA, 16 * 32, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, 16 * 32, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, 256 * sizeof(int)));

  mma_kernel<<<1, 32>>>(dA, dB, dC, dDump);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int hC[256], hDump[32 * 8];
  CUDA_CHECK(cudaMemcpy(hC, dC, 256 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hDump, dDump, 32 * 8 * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dDump);

  int bad = 0, first = -1;
  for (int i = 0; i < 256; i++)
    if (hC[i] != hRef[i]) { bad++; if (first < 0) first = i; }

  printf("  probe %s\n", PROBE_NAME[probe]);
  if (!bad) {
    printf("    MATCH   (all 256 accumulators)\n");
    return false;
  }

  printf("    MISMATCH  %d/256 wrong.  first: C[%d][%d] got %d want %d\n",
         bad, first / 16, first % 16, hC[first], hRef[first]);

  if (probe == 1) {
    // Under probe 1 the true value of every accumulator is 16*i + j, so the raw registers decode
    // the accumulator layout directly -- IF the operands landed correctly. Print it either way:
    // a value outside [0,256) means the operand (m/n) mapping is wrong too, not just the acc.
    printf("    raw accumulator dump -- each entry decodes as (i,j) = (v/16, v%%16) if operands are OK:\n");
    printf("    lane |");
    for (int l = 0; l < 8; l++) printf("      d%d", l);
    printf("\n");
    for (int lane = 0; lane < 32; lane++) {
      printf("    %4d |", lane);
      for (int l = 0; l < 8; l++) {
        int v = hDump[lane * 8 + l];
        if (v >= 0 && v < 256) printf("  (%2d,%2d)", v / 16, v % 16);
        else                   printf("  %6d", v);
      }
      printf("\n");
    }
  }
  return true;
}

int main() {
#ifdef USE_PPU
  printf("=== INT8 MMA smoke: PPU  ppu.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32 ===\n");
  printf("    (hypothesised layout, by analogy with the verified fp16 ppu.mma.m16n16k16)\n\n");
#else
  printf("=== INT8 MMA smoke: NVIDIA  2x mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ===\n");
  printf("    (reference path: validates the probe harness on hw that definitely has the instruction)\n\n");
#endif
  bool bad = false;
  for (int p = 0; p < 3; p++) bad |= run_probe(p);
  printf("\n%s\n", bad ? ">>> FAIL" : ">>> ALL PROBES PASS");
  return bad ? 1 : 0;
}
