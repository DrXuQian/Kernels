#pragma once
// Standalone replacements for vLLM helpers (cuda_compat.h, cub_helpers.h, etc.)

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <climits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>

// --- cuda_compat.h ---
#define WARP_SIZE 32

#define VLLM_LDG(arg) __ldg(arg)

#define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)

#define VLLM_SHFL_SYNC(var, src_lane) \
    __shfl_sync(uint32_t(-1), var, src_lane)

#define VLLM_SHFL_DOWN_SYNC(var, lane_delta) \
    __shfl_down_sync(uint32_t(-1), var, lane_delta)

// --- cub_helpers.h ---
#if CUB_VERSION >= 200800
  #include <cuda/std/functional>
  using CubAddOp = cuda::std::plus<>;
  using CubMaxOp = cuda::maximum<>;
#else
  using CubAddOp = cub::Sum;
  using CubMaxOp = cub::Max;
#endif

// --- core/math.hpp ---
template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) { return (a + b - 1) / b; }

inline constexpr uint32_t next_pow_2(uint32_t num) {
    if (num <= 1) return num;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

// --- CEILDIV macro used in align kernels ---
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// --- TORCH_CHECK replacement ---
#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...) \
    do { if (!(cond)) { fprintf(stderr, "CHECK failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); abort(); } } while(0)
#endif

// --- Type conversion helpers ---
namespace vllm { namespace moe {

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) return value;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(value);
    else if constexpr (std::is_same_v<T, __half>) return __half2float(value);
}

enum ScoringFunc { SCORING_SOFTMAX = 0, SCORING_SIGMOID = 1 };

template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray { T data[N]; };

}} // namespace vllm::moe
