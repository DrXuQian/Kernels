#pragma once
// Standalone replacements for ggml CUDA helpers used by llama.cpp kernels.
// Extracted from ggml/src/ggml-cuda/common.cuh and unary.cuh

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define GGML_UNUSED(x) (void)(x)
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "GGML_ASSERT(%s) failed at %s:%d\n", #x, __FILE__, __LINE__); abort(); } } while(0)
#define GGML_ABORT(msg) do { fprintf(stderr, "GGML_ABORT: %s at %s:%d\n", msg, __FILE__, __LINE__); abort(); } while(0)

static constexpr __device__ int ggml_cuda_get_physical_warp_size() { return 32; }

template <int width = 32>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset, width);
    return x;
}

// Fast integer division helpers (from ggml common.cuh)
static const uint3 init_fastdiv_values(uint64_t d_64) {
    uint32_t d = (uint32_t)d_64;
    if (d == 1) return make_uint3(0, 0, d);
    uint32_t L = 32u - __builtin_clz(d - 1);
    uint64_t mp = ((1ULL << (32 + L)) + d - 1) / d;
    return make_uint3((uint32_t)mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 v) {
    uint32_t hi = __umulhi(n, v.x);
    return (hi + n) >> v.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 v) {
    return n - fastdiv(n, v) * v.z;
}

// SiLU activation (from unary.cuh)
__device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

// ggml_cuda_info stub for warp_size query
struct ggml_cuda_device_info {
    int warp_size;
    int cc;
};
struct ggml_cuda_info_t {
    ggml_cuda_device_info devices[1];
};
static inline ggml_cuda_info_t ggml_cuda_info() {
    ggml_cuda_info_t info;
    info.devices[0].warp_size = 32;
    info.devices[0].cc = 800;
    return info;
}
static inline int ggml_cuda_get_device() { return 0; }
