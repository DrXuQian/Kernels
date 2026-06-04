/*
 * Minimal CUDA utilities for the standalone FP8 block-scale MoE build.
 */
#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
inline void check_cuda_error(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s (%d)\n", cudaGetErrorString(status), static_cast<int>(status));
        std::abort();
    }
}

inline int getSMVersion()
{
    int device = 0;
    check_cuda_error(cudaGetDevice(&device));
    int major = 0;
    int minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major * 10 + minor;
}

inline int getMultiProcessorCount()
{
    int device = 0;
    check_cuda_error(cudaGetDevice(&device));
    int count = 0;
    check_cuda_error(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return count;
}

template <bool Value>
using ConstBool = std::integral_constant<bool, Value>;
} // namespace common

TRTLLM_NAMESPACE_END
