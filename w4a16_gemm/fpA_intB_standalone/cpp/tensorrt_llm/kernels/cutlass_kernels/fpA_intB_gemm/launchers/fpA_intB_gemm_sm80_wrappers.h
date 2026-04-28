/*
 * Standalone wrappers for fpA_intB GEMM (FP16 x INT4, GPTQ).
 */
#pragma once

#include "tensorrt_llm/common/config.h"
#include "cutlass_extensions/gemm_configs.h"
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels_oss
{
using CutlassGemmConfig = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

size_t fpA_intB_get_all_configs(CutlassGemmConfig const** configs);

bool fpA_intB_is_supported_config(CutlassGemmConfig const& config);

bool fpA_intB_select_config_fp16_int4_gptq(CutlassGemmConfig& out_config, int m, int n, int k, int group_size,
    int multi_processor_count, bool enable_cuda_fallback = true);

size_t fpA_intB_get_workspace_size(int m, int n, int k);

void fpA_intB_gemm_fp16_int4_gptq_with_config(half const* A, int8_t const* B, half const* weight_scales,
    half const* weight_zero_points, half* C, int m, int n,
    int k, int group_size, cudaStream_t stream, void* workspace, size_t workspace_bytes,
    CutlassGemmConfig const& config);

void fpA_intB_gemm_fp16_int4_gptq(half const* A, int8_t const* B, half const* weight_scales,
    half const* weight_zero_points, half* C, int m, int n, int k, int group_size,
    cudaStream_t stream, void* workspace, size_t workspace_bytes, CutlassGemmConfig* selected_config = nullptr);
} // namespace kernels::cutlass_kernels_oss

TRTLLM_NAMESPACE_END
