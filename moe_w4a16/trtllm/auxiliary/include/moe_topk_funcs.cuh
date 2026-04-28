#pragma once

// Extracted and minimized from TensorRT-LLM:
//   cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh

#include "trtllm_aux_compat.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace trtllm_aux
{
namespace reduce_topk
{
namespace cg = cooperative_groups;

static constexpr int kWARP_SIZE = 32;

template <typename T_>
struct TopKRedType
{
    using T = T_;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>
            || std::is_same_v<T, int>,
        "Top K reduction only supports int, float, fp16, and bf16");

    using TypeCmp = std::conditional_t<sizeof(T) == 4, uint64_t, uint32_t>;
    using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int16_t>;

    static constexpr int kMoveBits = (sizeof(T) == 4) ? 32 : 16;
    static constexpr int kMaxIdx = 65535;

    TypeCmp compValIdx{};

    static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0)
    {
        auto valueBits = cub::Traits<T>::TwiddleIn(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
        TypeCmp compactTmp = valueBits;
        compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
        return compactTmp;
    }

    static __host__ __device__ void unpack(T& value, int32_t& index, TypeCmp cmp)
    {
        index = kMaxIdx - static_cast<int32_t>(cmp & 0xFFFF);
        auto compactTmp = cmp >> kMoveBits;
        auto valueBits = cub::Traits<T>::TwiddleOut(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(compactTmp));
        value = reinterpret_cast<T&>(valueBits);
    }

    __host__ __device__ TopKRedType() = default;

    __host__ __device__ TopKRedType(T val, int32_t idx)
        : compValIdx(makeCmpVal(val, idx))
    {
    }

    __host__ __device__ operator TypeCmp() const noexcept
    {
        return compValIdx;
    }

    __device__ inline TypeCmp reduce(cg::thread_block_tile<kWARP_SIZE> const& warp)
    {
        return cg::reduce(warp, compValIdx, cg::greater<TypeCmp>{});
    }
};

#define TOPK_SWAP(I, J)                                                                                                \
    {                                                                                                                  \
        auto pairMin = min(topK[I].compValIdx, topK[J].compValIdx);                                                    \
        auto pairMax = max(topK[I].compValIdx, topK[J].compValIdx);                                                    \
        topK[I].compValIdx = pairMax;                                                                                  \
        topK[J].compValIdx = pairMin;                                                                                  \
    }

template <int N, typename RedType>
struct Sort;

template <typename RedType>
struct Sort<1, RedType>
{
    static __device__ void run(RedType*) {}
};

template <typename RedType>
struct Sort<2, RedType>
{
    static __device__ void run(RedType* topK)
    {
        TOPK_SWAP(0, 1);
    }
};

template <typename RedType>
struct Sort<3, RedType>
{
    static __device__ void run(RedType* topK)
    {
        TOPK_SWAP(0, 1);
        TOPK_SWAP(1, 2);
        TOPK_SWAP(0, 1);
    }
};

template <typename RedType>
struct Sort<4, RedType>
{
    static __device__ void run(RedType* topK)
    {
        TOPK_SWAP(0, 2);
        TOPK_SWAP(1, 3);
        TOPK_SWAP(0, 1);
        TOPK_SWAP(2, 3);
        TOPK_SWAP(1, 2);
    }
};

template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopKFunc(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type (&value)[N], int32_t (&idx)[N], Type minValue, int actualK = K)
{
    static_assert(K > 0 && K < kWARP_SIZE, "Top K must be in (0, warp_size)");
    static_assert(N > 0 && N < 5, "Only supports up to 128 candidates in this stage");
    using RedType = TopKRedType<Type>;

    RedType topK[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
    {
        topK[nn] = RedType{value[nn], idx[nn]};
    }

    if constexpr (!IsSorted)
    {
        Sort<N, RedType>::run(topK);
    }

    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < actualK; ++kk)
    {
        bool update = kk > 0 && packedMax == topK[0].compValIdx;
#pragma unroll
        for (int nn = 0; nn < N; ++nn)
        {
            topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]} : update ? topK[nn + 1] : topK[nn];
        }
        packedMax = topK[0].reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
}

template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type (&value)[N], int32_t (&idx)[N], Type const minValue, int actualK = K)
{
    static_assert(K > 0 && K < kWARP_SIZE, "Top K must be in (0, warp_size)");
    static_assert(N > 0 && N <= 16, "Only supports up to 512 candidates");
    static_assert(N <= 4 || N % 4 == 0, "Candidate chunks must be <=4 or a multiple of 4");

    if constexpr (N <= 4)
    {
        reduceTopKFunc<K, Type, N>(warp, out, outIdx, value, idx, minValue, actualK);
    }
    else
    {
        constexpr int numLoops = N / 4;
        constexpr int numResults = (numLoops * K - 1) / kWARP_SIZE + 1;

        Type topKBufferValue[numResults];
        int32_t topKBufferIdx[numResults];
        int32_t laneIdx = threadIdx.x % kWARP_SIZE;

#pragma unroll
        for (int ii = 0; ii < numResults; ++ii)
        {
            topKBufferValue[ii] = minValue;
            topKBufferIdx[ii] = ii * kWARP_SIZE - 1;
        }

#pragma unroll
        for (int loop = 0; loop < numLoops; ++loop)
        {
            int start = loop * 4;
            Type topKValue[K];
            int32_t topKIdx[K];
            Type inValue[4];
            int32_t inIdx[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                inValue[i] = value[start + i];
                inIdx[i] = idx[start + i];
            }
            reduceTopKFunc<K, Type, 4>(warp, topKValue, topKIdx, inValue, inIdx, minValue, actualK);
            int inOffset = laneIdx % K;
            if (laneIdx >= loop * K && laneIdx < (loop + 1) * K)
            {
                topKBufferValue[0] = topKValue[inOffset];
                topKBufferIdx[0] = topKIdx[inOffset];
            }
            if (loop == numLoops - 1 && (laneIdx < (numLoops * K - kWARP_SIZE)))
            {
                topKBufferValue[1] = topKValue[inOffset];
                topKBufferIdx[1] = topKIdx[inOffset];
            }
        }

        reduceTopKFunc<K, Type, numResults>(warp, out, outIdx, topKBufferValue, topKBufferIdx, minValue, actualK);
    }
}

#undef TOPK_SWAP

} // namespace reduce_topk
} // namespace trtllm_aux
