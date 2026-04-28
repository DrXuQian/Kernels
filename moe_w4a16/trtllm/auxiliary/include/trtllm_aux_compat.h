#pragma once

// Standalone compatibility helpers for TensorRT-LLM MoE auxiliary kernels.

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32

#define TRTLLM_AUX_CHECK(cond, ...)                                                                                   \
    do                                                                                                                \
    {                                                                                                                 \
        if (!(cond))                                                                                                  \
        {                                                                                                             \
            std::fprintf(stderr, "CHECK failed: %s at %s:%d\n", #cond, __FILE__, __LINE__);                          \
            std::fprintf(stderr, __VA_ARGS__);                                                                        \
            std::fprintf(stderr, "\n");                                                                               \
            std::abort();                                                                                             \
        }                                                                                                             \
    } while (0)

inline void trtllm_aux_check_cuda(cudaError_t status, char const* expr, char const* file, int line)
{
    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error at %s:%d: %s: %s (%d)\n", file, line, expr, cudaGetErrorString(status),
            static_cast<int>(status));
        std::abort();
    }
}

#define TRTLLM_AUX_CUDA_CHECK(expr) trtllm_aux_check_cuda((expr), #expr, __FILE__, __LINE__)

inline int trtllm_aux_next_power_of_two(int num)
{
    if (num <= 1)
    {
        return 1;
    }
    int power = 1;
    while (power < num && power <= INT_MAX / 2)
    {
        power <<= 1;
    }
    return power;
}

template <typename T>
T trtllm_aux_from_float(float value);

template <>
inline float trtllm_aux_from_float<float>(float value)
{
    return value;
}

template <>
inline half trtllm_aux_from_float<half>(float value)
{
    return __float2half(value);
}

#if ENABLE_BF16
template <>
inline __nv_bfloat16 trtllm_aux_from_float<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}
#endif
