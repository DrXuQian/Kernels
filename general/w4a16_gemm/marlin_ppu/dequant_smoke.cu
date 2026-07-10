// Finest-granularity Marlin microbench #1: the INT4 dequant (lop3 magic) in isolation, vs a plain CPU reference.
// This is the only novel "magic" in the int4 path; everything else (mma, pipeline) reuses known pieces. It only
// needs lop3 + fp16, which run verbatim on the PPU (SM80-compat).
//
// dequant(q) extracts 4 nibbles n0=q[0:4], n1=q[4:8], n2=q[16:20], n3=q[20:24] and returns the signed-int4 fp16
// values {n0-8, n2-8, n1-8, n3-8} (see obsidian/marlin.md: lop3 puts n+1024 in the fp16 mantissa, SUB/ADD remove
// the 1024 and fold the -8 symmetric zero point; the hi nibble also *1/16).
//
// build (box):  nvcc -O3 -o dequant_smoke dequant_smoke.cu   (no -arch: ppu001).  Run: ./dequant_smoke

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// vLLM marlin dequant (kU4B8, signed int4): int32 q -> 4 fp16 (2 half2). Verbatim from marlin dequant.h / obsidian notes.
__device__ inline void dequant(int q, half2 * frag_b) {
    const int LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400;
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);   // (q & LO) | EX  -> {1024+n0, 1024+n2}
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);   // (q & HI) | EX  -> {1024 + (n1<<4), 1024 + (n3<<4)}
    const int SUB = 0x64086408;   // -1024 - 8
    const int MUL = 0x2c002c00;   // 1/16
    const int ADD = 0xd480d480;   // -1024/16 - 8 = -72
    frag_b[0] = __hsub2(*reinterpret_cast<half2 *>(&lo), *reinterpret_cast<const half2 *>(&SUB));   // {n0-8, n2-8}
    frag_b[1] = __hfma2(*reinterpret_cast<half2 *>(&hi), *reinterpret_cast<const half2 *>(&MUL),
                        *reinterpret_cast<const half2 *>(&ADD));                                    // {n1-8, n3-8}
}

__global__ void dequant_kernel(const int * __restrict__ q, float * __restrict__ out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    half2 fb[2];
    dequant(q[i], fb);
    // store as {n0-8, n2-8, n1-8, n3-8}
    out[i * 4 + 0] = __half2float(__low2half(fb[0]));
    out[i * 4 + 1] = __half2float(__high2half(fb[0]));
    out[i * 4 + 2] = __half2float(__low2half(fb[1]));
    out[i * 4 + 3] = __half2float(__high2half(fb[1]));
}

int main() {
    const int N = 64;
    int hq[N]; float hout[N * 4], ref[N * 4];
    srand(1234);
    for (int i = 0; i < N; ++i) {
        int n0 = rand() & 0xf, n1 = rand() & 0xf, n2 = rand() & 0xf, n3 = rand() & 0xf;
        hq[i] = n0 | (n1 << 4) | (n2 << 16) | (n3 << 20);
        ref[i * 4 + 0] = n0 - 8; ref[i * 4 + 1] = n2 - 8; ref[i * 4 + 2] = n1 - 8; ref[i * 4 + 3] = n3 - 8;
    }
    int * dq; float * dout;
    cudaMalloc(&dq, N * sizeof(int)); cudaMalloc(&dout, N * 4 * sizeof(float));
    cudaMemcpy(dq, hq, N * sizeof(int), cudaMemcpyHostToDevice);
    dequant_kernel<<<1, N>>>(dq, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("ERR %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(hout, dout, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    double max_abs = 0; int bad = 0;
    for (int i = 0; i < N * 4; ++i) { double d = fabs((double) hout[i] - ref[i]); if (d > max_abs) max_abs = d; if (d > 1e-3) bad++; }
    printf("INT4 dequant smoke: %d values  max_abs=%.4e  bad=%d -> %s\n", N * 4, max_abs, bad, bad == 0 ? "MATCH" : "MISMATCH");
    printf("  q[0]=%#010x -> got {%.0f %.0f %.0f %.0f}  ref {%.0f %.0f %.0f %.0f}\n", hq[0],
           hout[0], hout[1], hout[2], hout[3], ref[0], ref[1], ref[2], ref[3]);
    return bad == 0 ? 0 : 1;
}
