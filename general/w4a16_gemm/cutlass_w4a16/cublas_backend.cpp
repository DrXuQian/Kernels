// cublas fp16 GEMM, ISOLATED in its own host translation unit.
//
// WHY A SEPARATE .cpp: cublas_v2.h drags in CUDA's cuda_fp16.h, whose ::__half2 clashes with the PPU SDK's
// hggc_fp8.hpp ::__half2 -- they cannot coexist in one TU (hgcc auto-includes the hggc headers). So the
// cublas call lives here, compiled by g++ (host), seeing ONLY cublas/CUDA and never the hggc/cute headers.
// test_fpA_intB_ppu.cu (the actlize/hgcc TU) calls this through the extern "C" prototype and does all the
// timing itself, so the dequant+cublas candidate still lives in the one unified sweep.
//
// D[m,n] = A[m,k] * W[n,k]^T  in fp16, with cublas column-major: we compute (n,m) = W(n,k) * A(m,k)^T style
// exactly as the marlin bench does: cublasHgemm(OP_T, OP_N, n, m, k, W[k-major], A[k-major] -> D[n-major]).
// Pointers are opaque void* (hggc device memory from the caller's cutlass DeviceAllocation); cublas on this
// SDK is the CUDA-API shim over the same device runtime, so the pointers are valid here.
#include <cublas_v2.h>
#include <cstring>
#include <cstdio>

static cublasHandle_t g_handle = nullptr;

extern "C" void cublas_hgemm_once(const void* W, const void* A, void* D, int m, int n, int k) {
  if (!g_handle) { cublasCreate(&g_handle); cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH); }
  // half constants without __float2half (that is __device__-leaning; build the bit patterns directly).
  __half one, zero;
  unsigned short o = 0x3C00 /*1.0*/, z = 0x0000 /*0.0*/;
  std::memcpy(&one, &o, 2); std::memcpy(&zero, &z, 2);
  cublasStatus_t st = cublasHgemm(g_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
      &one, reinterpret_cast<const __half*>(W), k,
            reinterpret_cast<const __half*>(A), k,
      &zero, reinterpret_cast<__half*>(D), n);
  if (st != CUBLAS_STATUS_SUCCESS) std::printf("[cublas] Hgemm status %d\n", (int)st);
}
