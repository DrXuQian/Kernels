/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "moe_cuda_core_gemv.h"

#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/details.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/utility.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{
namespace
{

namespace wo = tensorrt_llm::kernels::weight_only;

template <typename Details, int CtaN, int Threads, int GroupSize, bool SingleExpert,
    typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void moe_cuda_core_m1_kernel(TypeA const* act, int64_t const* total_tokens_including_expert,
    uint8_t const* weight, TypeA const* scales, TypeA* out, int64_t num_rows, int num_experts, int n, int k)
{
    using AccessTypeA = typename Details::AccessTypeA;
    using AccessTypeW = typename Details::AccessTypeW;

    static constexpr int CtaM = 1;
    static constexpr bool Mandatory = true;
    static constexpr bool EnableZero = false;
    static constexpr bool EnableBias = false;
    static constexpr bool ApplyAlphaInAdvance = false;
    static constexpr int StepK = Details::kStepK;
    static constexpr int CtaK = StepK * Threads;

    static_assert(CtaN % 2 == 0);
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);

    int const expert = SingleExpert ? 0 : blockIdx.x;
    int const tile_id_n = blockIdx.y;
    int const tid = threadIdx.x;
    if constexpr (!SingleExpert)
    {
        if (expert >= num_experts)
        {
            return;
        }
    }

    int64_t row_begin = 0;
    if constexpr (!SingleExpert)
    {
        row_begin = expert == 0 ? 0 : total_tokens_including_expert[expert - 1];
        int64_t const row_end = total_tokens_including_expert[expert];
        if (row_end - row_begin != 1 || row_begin < 0 || row_begin >= num_rows)
        {
            return;
        }
    }

    int const origin_k = k;
    int const interleaved_k = k * Details::kInterleave;
    int const interleaved_offset_n = tile_id_n * CtaN;
    int const real_offset_n = interleaved_offset_n * Details::kInterleave
        + ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
    int const real_offset_k
        = (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize
        + ((tid * StepK) % Details::LayoutDetails::kTileSize);

    bool constexpr scale_zero_ldg128 = Details::kInterleave == 1 && CtaN == 8;
    using AccessTypeScaleZero = std::conditional_t<scale_zero_ldg128, AccessTypeA, TypeA>;

    auto* act_ptr = const_cast<TypeA*>(act);
    auto* weight_ptr = const_cast<uint8_t*>(weight);
    auto* scales_ptr = const_cast<TypeA*>(scales);
    if constexpr (!SingleExpert)
    {
        weight_ptr += static_cast<int64_t>(expert) * static_cast<int64_t>(k) * static_cast<int64_t>(n) / 2;
        scales_ptr += static_cast<int64_t>(expert) * (static_cast<int64_t>(k) / GroupSize) * static_cast<int64_t>(n);
    }

    wo::GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
        act_ptr, static_cast<int>(row_begin * origin_k + real_offset_k), CtaK / Details::kInterleave, origin_k);
    wo::GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(weight_ptr,
        (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW, CtaK / Details::kElemsPerByteW,
        interleaved_k / Details::kElemsPerByteW);
    wo::GMemIterator<Mandatory, AccessTypeScaleZero, CtaN, 1, TypeA> scales_iterator(scales_ptr,
        real_offset_k / GroupSize * n + real_offset_n, CtaK / Details::kInterleave / GroupSize * n,
        Details::kInterleave);

    out += row_begin * n + tile_id_n * CtaN * Details::kInterleave;

    TypeA tile_acc[CtaM * CtaN];
    wo::fill<CtaM * CtaN>(tile_acc, static_cast<TypeA>(0.f));

    for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter)
    {
        TypeA vec_scale[CtaN], vec_zero[CtaN];
        TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
        uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
        if constexpr (scale_zero_ldg128)
        {
            scales_iterator.load(vec_scale, iter);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < CtaN; ++i)
            {
                scales_iterator.load(vec_scale + i, iter, i);
            }
        }

#pragma unroll
        for (int i = 0; i < CtaN; ++i)
        {
            weight_iterator.load(tile_w_quantized, iter, i);
            wo::dequantize<Details, 1, StepK, EnableZero, ApplyAlphaInAdvance>(
                tile_w, tile_w_quantized, vec_scale + i, vec_zero + i, 1.0f);
            wo::pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
        }

        act_iterator.load(tile_a, iter, 0);
        wo::mma<Details, CtaM, CtaN, StepK, EnableZero, ApplyAlphaInAdvance>(tile_acc, tile_w_pack2, tile_a, vec_scale);
    }

    wo::epilogue<Details, CtaM, CtaN, Threads, EnableBias, ApplyAlphaInAdvance>(out, n, tile_acc, nullptr, 1.0f);
}

template <typename T>
using Int4Details = wo::KernelDetails<T, wo::Int4DetailsW, wo::ColumnMajorInterleaved, true, 64>;

template <typename T>
void launchMoeCudaCoreGemvM1(GroupedGemmInput<typename T::Type, cutlass::uint4b_t, typename T::Type,
    typename T::Type> const& input)
{
    using Details = Int4Details<T>;
    using TypeA = typename T::Type;
    int const n = static_cast<int>(input.n);
    int const k = static_cast<int>(input.k);
    bool const single_expert = input.num_experts == 1;
    int const cta_n = single_expert ? 4 : 8;
    dim3 const grid(input.num_experts, n / (cta_n * Details::kInterleave));
    dim3 const block(128);

    auto* weight = reinterpret_cast<uint8_t const*>(input.B);
    if (input.groupwise_quant_group_size == 64)
    {
        if (single_expert)
        {
            moe_cuda_core_m1_kernel<Details, 4, 128, 64, true><<<grid, block, 0, input.stream>>>(input.A,
                input.total_tokens_including_expert, weight, input.scales, input.C, input.num_rows, input.num_experts,
                n, k);
        }
        else
        {
            moe_cuda_core_m1_kernel<Details, 8, 128, 64, false><<<grid, block, 0, input.stream>>>(input.A,
                input.total_tokens_including_expert, weight, input.scales, input.C, input.num_rows, input.num_experts,
                n, k);
        }
    }
    else if (input.groupwise_quant_group_size == 128)
    {
        if (single_expert)
        {
            moe_cuda_core_m1_kernel<Details, 4, 128, 128, true><<<grid, block, 0, input.stream>>>(input.A,
                input.total_tokens_including_expert, weight, input.scales, input.C, input.num_rows, input.num_experts,
                n, k);
        }
        else
        {
            moe_cuda_core_m1_kernel<Details, 8, 128, 128, false><<<grid, block, 0, input.stream>>>(input.A,
                input.total_tokens_including_expert, weight, input.scales, input.C, input.num_rows, input.num_experts,
                n, k);
        }
    }
    else
    {
        throw std::runtime_error("CUDA-core MoE GEMV supports group_size 64 or 128 only");
    }
    auto const status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

} // namespace

template <typename T, typename OutputType>
bool moeCudaCoreGemvM1IsSupported(GroupedGemmInput<T, cutlass::uint4b_t, T, OutputType> const& input)
{
    if constexpr (!std::is_same_v<T, OutputType>)
    {
        return false;
    }
    return input.gemm_config.enableCudaKernel && input.A != nullptr && input.total_tokens_including_expert != nullptr
        && input.B != nullptr && input.scales != nullptr && input.zeros == nullptr && input.biases == nullptr
        && input.C != nullptr && input.alpha_scales == nullptr && input.activation_type == ActivationType::Identity
        && input.num_rows <= input.num_experts && input.num_rows > 0 && input.n > 0 && input.k > 0
        && (input.groupwise_quant_group_size == 64 || input.groupwise_quant_group_size == 128)
        && input.k % input.groupwise_quant_group_size == 0 && input.k % 64 == 0
        && input.n % (4 * Int4Details<wo::FP16DetailsA>::kInterleave) == 0;
}

template <typename T, typename OutputType>
void dispatchMoeCudaCoreGemvM1(GroupedGemmInput<T, cutlass::uint4b_t, T, OutputType> const& input)
{
    if (!moeCudaCoreGemvM1IsSupported(input))
    {
        throw std::runtime_error(
            "CUDA-core MoE GEMV requires Identity, scale-only INT4, group_size 64/128, and <=1 row per expert");
    }

    if constexpr (std::is_same_v<T, half> && std::is_same_v<OutputType, half>)
    {
        launchMoeCudaCoreGemvM1<wo::FP16DetailsA>(input);
    }
#ifdef ENABLE_BF16
    else if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<OutputType, __nv_bfloat16>)
    {
        launchMoeCudaCoreGemvM1<wo::BF16DetailsA>(input);
    }
#endif
    else
    {
        throw std::runtime_error("CUDA-core MoE GEMV supports FP16 and BF16 activation/output only");
    }
}

template bool moeCudaCoreGemvM1IsSupported<half, half>(
    GroupedGemmInput<half, cutlass::uint4b_t, half, half> const& input);
template void dispatchMoeCudaCoreGemvM1<half, half>(
    GroupedGemmInput<half, cutlass::uint4b_t, half, half> const& input);

#ifdef ENABLE_BF16
template bool moeCudaCoreGemvM1IsSupported<__nv_bfloat16, __nv_bfloat16>(
    GroupedGemmInput<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16> const& input);
template void dispatchMoeCudaCoreGemvM1<__nv_bfloat16, __nv_bfloat16>(
    GroupedGemmInput<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16> const& input);
#endif

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
