/*
 * Minimal CUDA utilities for standalone fpA_intB build.
 */
#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>
#include <cstdio>

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
    int major = 0, minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    int sm = major * 10 + minor;

    // Print device info once
    static bool printed = false;
    if (!printed)
    {
        printed = true;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::fprintf(stderr, "[fpA_intB] device=%d name=%s sm=%d.%d (sm_version=%d)\n",
            device, prop.name, major, minor, sm);

        // Check compiled arch
#ifdef __CUDA_ARCH__
        // This only works in device code, so we print the compile-time flags instead
#endif
#if defined(COMPILE_HOPPER_TMA_GEMMS) && COMPILE_HOPPER_TMA_GEMMS
        std::fprintf(stderr, "[fpA_intB] compiled with: COMPILE_HOPPER_TMA_GEMMS=1\n");
#else
        std::fprintf(stderr, "[fpA_intB] compiled with: COMPILE_HOPPER_TMA_GEMMS=0 (SM90 TMA path DISABLED)\n");
#endif
    }
    return sm;
}
} // namespace common

TRTLLM_NAMESPACE_END
