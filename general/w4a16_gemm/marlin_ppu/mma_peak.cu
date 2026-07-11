/*
 * Tensor-core PEAK probe: the DENOMINATOR for every "compute utilization" number.
 *
 * Back-to-back ppu.mma with register-resident operands and zero memory traffic, so the only thing
 * being measured is how fast the tensor core can retire MMAs. Runs both shapes:
 *     ppu.mma.m16n16k16.f32.f16.f16.f32   (what Marlin's prefill uses)   2*16*16*16 =  8192 flop
 *     ppu.mma.m16n16k32.s32.s8.s8.s32     (what MMQ's prefill uses)      2*16*16*32 = 16384 op
 * so the int8:fp16 ratio comes out of the measurement rather than being assumed.
 *
 * Without this, "MMQ hits X TFLOP/s" is a number with no scale. Marlin's 236 TFLOP/s was only ever
 * meaningful because we knew what it was 236 out of.
 *
 * Build:  PPU (ppu001, NO -arch):  nvcc -O3 -std=c++17 -o mma_peak mma_peak.cu
 *         NVIDIA (harness check):  nvcc -O3 -std=c++17 -arch=sm_90 -DPEAK_NV -o mma_peak mma_peak.cu
 *
 * THE TRAP THIS AVOIDS: a peak probe whose MMAs form a dependency chain measures LATENCY, not
 * throughput, and reports a number several times too low. Each thread here keeps NACC independent
 * accumulators, so consecutive MMAs have no RAW dependency and the pipe can stay full. NACC is a
 * knob (-DPEAK_NACC=n) precisely so that "is NACC big enough to saturate?" is answerable by sweeping
 * it instead of by assertion -- if the number is still climbing at NACC=8, the probe is latency-bound
 * and the reported peak is a floor, not a peak.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

#ifndef PEAK_NACC
#define PEAK_NACC 4
#endif
#ifndef PEAK_ITERS
#define PEAK_ITERS 4096
#endif

// ---------------------------------------------------------------------------------------------
template <int NACC>
__global__ void peak_i8(int* out, int iters) {
  uint32_t a[4], b[4];
  int c[NACC][8];
#pragma unroll
  for (int i = 0; i < 4; i++) { a[i] = 0x01010101u + threadIdx.x; b[i] = 0x02020202u ^ threadIdx.x; }
#pragma unroll
  for (int n = 0; n < NACC; n++)
#pragma unroll
    for (int l = 0; l < 8; l++) c[n][l] = n + l;

  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int n = 0; n < NACC; n++)
#ifdef PEAK_NV
      // 2x m16n8k32 == one m16n16k32 worth of work, so the flop count below is comparable.
      for (int h = 0; h < 2; h++)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+r"(c[n][4*h+0]), "+r"(c[n][4*h+1]), "+r"(c[n][4*h+2]), "+r"(c[n][4*h+3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
      asm volatile("ppu.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32 "
                   "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                   : "+r"(c[n][0]), "+r"(c[n][1]), "+r"(c[n][2]), "+r"(c[n][3]),
                     "+r"(c[n][4]), "+r"(c[n][5]), "+r"(c[n][6]), "+r"(c[n][7])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
#endif
  }

  int s = 0;
#pragma unroll
  for (int n = 0; n < NACC; n++)
#pragma unroll
    for (int l = 0; l < 8; l++) s += c[n][l];
  if (s == 0x7fffffff) out[0] = s;   // never true; keeps the whole chain live
}

template <int NACC>
__global__ void peak_f16(float* out, int iters) {
  uint32_t a[4], b[4];
  float c[NACC][8];
#pragma unroll
  for (int i = 0; i < 4; i++) { a[i] = 0x3c003c00u; b[i] = 0x3c003c00u; }
#pragma unroll
  for (int n = 0; n < NACC; n++)
#pragma unroll
    for (int l = 0; l < 8; l++) c[n][l] = 0.0f;

  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int n = 0; n < NACC; n++)
#ifdef PEAK_NV
      for (int h = 0; h < 2; h++)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[n][4*h+0]), "+f"(c[n][4*h+1]), "+f"(c[n][4*h+2]), "+f"(c[n][4*h+3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
      asm volatile("ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
                   "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                   : "+f"(c[n][0]), "+f"(c[n][1]), "+f"(c[n][2]), "+f"(c[n][3]),
                     "+f"(c[n][4]), "+f"(c[n][5]), "+f"(c[n][6]), "+f"(c[n][7])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
#endif
  }

  float s = 0;
#pragma unroll
  for (int n = 0; n < NACC; n++)
#pragma unroll
    for (int l = 0; l < 8; l++) s += c[n][l];
  if (s == 1e30f) out[0] = s;
}

// ---------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  int nacc = PEAK_NACC;
  if (argc > 1) nacc = atoi(argv[1]);

  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  // Enough blocks to fill every CU several times over; the probe has no memory traffic so occupancy
  // is purely a register question.
  const int nblocks = p.multiProcessorCount * 8;
  const int nthreads = 256;
  const int warps = nblocks * (nthreads / 32);

  printf("=== tensor-core PEAK  (%s, %d CUs, %d blocks x %d thr, NACC=%d, iters=%d) ===\n",
         p.name, p.multiProcessorCount, nblocks, nthreads, nacc, PEAK_ITERS);
  printf("    NACC = independent accumulators per thread. If the numbers are still RISING with NACC,\n");
  printf("    the probe is latency-bound and these are FLOORS, not peaks. Sweep it: ./mma_peak 1 2 4 8\n\n");

  void *d;
  CUDA_CHECK(cudaMalloc(&d, 4));

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  double t_i8 = 0, t_f16 = 0;
  for (int rep = 0; rep < 3; rep++) {
    float ms;

#define RUN(KERN, NACC_V) do {                                                        \
      KERN<NACC_V><<<nblocks, nthreads>>>((decltype(out_t))d, PEAK_ITERS);            \
      CUDA_CHECK(cudaDeviceSynchronize());                                            \
      CUDA_CHECK(cudaEventRecord(e0));                                                \
      KERN<NACC_V><<<nblocks, nthreads>>>((decltype(out_t))d, PEAK_ITERS);            \
      CUDA_CHECK(cudaEventRecord(e1));                                                \
      CUDA_CHECK(cudaEventSynchronize(e1));                                           \
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));                                  \
    } while (0)

    {
      int* out_t = nullptr;
      if (nacc == 1) RUN(peak_i8, 1); else if (nacc == 2) RUN(peak_i8, 2);
      else if (nacc == 8) RUN(peak_i8, 8); else RUN(peak_i8, 4);
      // one m16n16k32 per acc per iter per warp
      const double ops = (double)warps * PEAK_ITERS * nacc * 2.0 * 16 * 16 * 32;
      t_i8 = ops / (ms * 1e-3) / 1e12;
    }
    {
      float* out_t = nullptr;
      if (nacc == 1) RUN(peak_f16, 1); else if (nacc == 2) RUN(peak_f16, 2);
      else if (nacc == 8) RUN(peak_f16, 8); else RUN(peak_f16, 4);
      const double ops = (double)warps * PEAK_ITERS * nacc * 2.0 * 16 * 16 * 16;
      t_f16 = ops / (ms * 1e-3) / 1e12;
    }
#undef RUN
  }

  printf("  INT8  m16n16k32 s32.s8.s8.s32 : %8.1f TOP/s\n", t_i8);
  printf("  FP16  m16n16k16 f32.f16.f16.f32: %8.1f TFLOP/s\n", t_f16);
  printf("  ratio int8 / fp16              : %8.2fx\n\n", t_i8 / t_f16);
  printf("  -> use the INT8 number as the denominator for MMQ's compute utilization,\n");
  printf("     and the FP16 number for Marlin's (which measured 236 TFLOP/s at gs=128).\n");
  return 0;
}
