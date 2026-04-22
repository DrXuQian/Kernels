// Explicit template instantiation for FlashInfer GDN prefill kernel
// Only bf16, Qwen3.5-122B DeltaNet config:
//   num_k_heads=16, num_v_heads=64, head_dim=128 → is_gva=true
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"

namespace flat {

// Macro for explicit instantiation
#define INST(is_gva, needs_beta, needs_alpha, init_state, enable_ckpt) \
template void launch_delta_rule_prefill_kernel_gbai< \
    is_gva, needs_beta, needs_alpha, init_state, enable_ckpt, \
    cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>( \
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*, \
    float const*, float const*, float const*, int64_t const*, uint8_t*, int32_t, int32_t, \
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t, float*, int64_t const*, int32_t);

// DeltaNet GVA: is_gva=true, needs_alpha=true, needs_beta=true (most common)
// Enumerate init_state × enable_ckpt = 4 combos
INST(true, true, true, false, false)
INST(true, true, true, true,  false)
INST(true, true, true, false, true)
INST(true, true, true, true,  true)

// Without beta (simpler delta rule)
INST(true, false, true, false, false)
INST(true, false, true, true,  false)

// Without alpha (no gate)
INST(true, true, false, false, false)
INST(true, false, false, false, false)

#undef INST

}  // namespace flat
