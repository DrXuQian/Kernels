// W2A16 numeric smoke — validates the NEW device-side 2-bit unpack + affine scale/zero application, the only new
// numeric vs the validated W4A16 path. Plain CUDA (build with bare nvcc -> ppu001; no cutlass/AIU here).
//
//   nvcc -std=c++17 -O3 w2a16_smoke.cu -o w2a16_smoke && ./w2a16_smoke
//
// Layout (forward-compatible with the grouped collective + the Q4_K harness: weight logical [N(out)][K(in)],
// scale [K/gs][N]):
//   packed 2-bit weight : [N][K/4] uint8, 4 K-values/byte, q2 in bits [0-1][2-3][4-5][6-7]  (q2 in [0,3])
//   scale, zero (affine): [K/gs][N] fp16, gs=16.  dequant  W[k][n] = scale[k/gs][n]*q2[n][k] + zero[k/gs][n]
// Tests: (1) device unpack -> fp16 W  vs host golden W;  (2) a naive fp16 GEMM D=A@W (fp32 accum) vs host golden.
// NOTE: this materializes W to check numerics; the REAL collective keeps W 2-bit in HBM and unpacks a tile in the
// mainloop (this file's unpack math is what lifts into transform_B_kblock).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CK(x) do { cudaError_t e=(x); if(e){printf("cuda err %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

static const int M = 64, K = 512, N = 128, GS = 16;   // K,N multiples of 4/16 for clean packing

__global__ void unpack_q2(const uint8_t* __restrict__ packed, const __half* __restrict__ scale,
                          const __half* __restrict__ zero, __half* __restrict__ W, int K_, int N_, int gs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;      // over K*N, W[k][n]
  if (idx >= K_ * N_) return;
  int k = idx / N_, n = idx % N_;
  uint8_t byte = packed[(size_t)n * (K_ / 4) + k / 4];
  int q2 = (byte >> ((k % 4) * 2)) & 3;
  int g = k / gs;
  float w = __half2float(scale[(size_t)g * N_ + n]) * q2 + __half2float(zero[(size_t)g * N_ + n]);
  W[idx] = __float2half(w);
}

__global__ void gemm_naive(const __half* __restrict__ A, const __half* __restrict__ W, __half* __restrict__ D,
                           int M_, int K_, int N_) {
  int m = blockIdx.y * blockDim.y + threadIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M_ || n >= N_) return;
  float acc = 0.f;
  for (int k = 0; k < K_; ++k) acc += __half2float(A[(size_t)m * K_ + k]) * __half2float(W[(size_t)k * N_ + n]);
  D[(size_t)m * N_ + n] = __float2half(acc);
}

int main() {
  srand(1234);
  const int scale_k = K / GS;
  std::vector<int> q2((size_t)N * K);                                  // logical [N][K]
  std::vector<float> hsc((size_t)scale_k * N), hzr((size_t)scale_k * N), hA((size_t)M * K);
  for (auto& q : q2) q = rand() % 4;                                   // [0,3]
  for (auto& s : hsc) s = 0.01f + (rand() % 8) * 0.01f;                // per-16 scale
  for (auto& z : hzr) z = (rand() % 9 - 4) * 0.01f;                    // per-16 affine min
  for (auto& a : hA) a = (rand() % 15 - 7) * 0.5f;

  // pack 2-bit [N][K/4]
  std::vector<uint8_t> packed((size_t)N * (K / 4), 0);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      packed[(size_t)n * (K / 4) + k / 4] |= uint8_t(q2[(size_t)n * K + k] << ((k % 4) * 2));

  // host golden W[K][N] + D[M][N]
  std::vector<float> gW((size_t)K * N), gD((size_t)M * N, 0.f);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) {
      int g = k / GS;
      gW[(size_t)k * N + n] = hsc[(size_t)g * N + n] * q2[(size_t)n * K + k] + hzr[(size_t)g * N + n];
    }
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += hA[(size_t)m * K + k] * gW[(size_t)k * N + n];
      gD[(size_t)m * N + n] = acc;
    }

  // device
  auto to_h = [](std::vector<float> const& f) { std::vector<__half> h(f.size()); for (size_t i=0;i<f.size();++i) h[i]=__float2half(f[i]); return h; };
  auto hsc16 = to_h(hsc), hzr16 = to_h(hzr), hA16 = to_h(hA);
  uint8_t* dP; __half *dSc,*dZr,*dW,*dA,*dD;
  CK(cudaMalloc(&dP, packed.size())); CK(cudaMemcpy(dP, packed.data(), packed.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dSc, hsc16.size()*2)); CK(cudaMemcpy(dSc, hsc16.data(), hsc16.size()*2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dZr, hzr16.size()*2)); CK(cudaMemcpy(dZr, hzr16.data(), hzr16.size()*2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dA, hA16.size()*2));   CK(cudaMemcpy(dA, hA16.data(), hA16.size()*2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dW, (size_t)K*N*2));   CK(cudaMalloc(&dD, (size_t)M*N*2));

  unpack_q2<<<(K*N+255)/256, 256>>>(dP, dSc, dZr, dW, K, N, GS);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  dim3 blk(16,16), grd((N+15)/16,(M+15)/16);
  gemm_naive<<<grd, blk>>>(dA, dW, dD, M, K, N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<__half> hW((size_t)K*N), hD((size_t)M*N);
  CK(cudaMemcpy(hW.data(), dW, (size_t)K*N*2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hD.data(), dD, (size_t)M*N*2, cudaMemcpyDeviceToHost));

  auto report = [](const char* name, std::vector<__half> const& got, std::vector<float> const& gold) {
    double mr = 0; int bad = 0;
    for (size_t i = 0; i < gold.size(); ++i) {
      double gt = __half2float(got[i]), g = gold[i], ad = fabs(gt - g), rel = ad / (fabs(g) + 1e-3);
      if (rel > mr) mr = rel;
      if (ad > 1e-2 + 5e-2 * fabs(g)) ++bad;
    }
    printf("  %-18s max_rel=%.3e bad=%d/%zu -> %s\n", name, mr, bad, gold.size(), bad==0?"MATCH":"MISMATCH");
    return bad == 0;
  };
  printf("[w2a16_smoke] M=%d K=%d N=%d gs=%d\n", M, K, N, GS);
  bool ok = report("unpack W (2b->fp16)", hW, gW);
  ok &= report("gemm D=A@W", hD, gD);
  printf("  gW[0..3]="); for (int i=0;i<4;++i) printf(" %.4f", gW[i]);
  printf("\n  dW[0..3]="); for (int i=0;i<4;++i) printf(" %.4f", __half2float(hW[i]));
  printf("\n");
  return ok ? 0 : 1;
}
