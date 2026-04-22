// Explicit template instantiation for delta rule prefill kernel (bf16 only)
// Only instantiates the most common variant: no GVA, with alpha, no beta, no init_state, no checkpointing
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"

namespace flat {

// DeltaNet: is_gva=false, needs_beta=false, needs_alpha=true, init_state=false, enable_ckpt=false
template void launch_delta_rule_prefill_kernel_gbai<false, false, true, false, false,
    cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, uint8_t*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t, float*, int64_t const*, int32_t);

// Also: no alpha no beta (simplest case)
template void launch_delta_rule_prefill_kernel_gbai<false, false, false, false, false,
    cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, uint8_t*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t, float*, int64_t const*, int32_t);

// With init_state
template void launch_delta_rule_prefill_kernel_gbai<false, false, true, true, false,
    cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, uint8_t*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t, float*, int64_t const*, int32_t);

}  // namespace flat
