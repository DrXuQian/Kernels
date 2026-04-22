// Dispatch for FlashInfer GDN prefill — bf16 only, standalone (no PyTorch)
#include <cuda_bf16.h>
#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"

namespace flat {

using namespace cute;

void launch_gdn_prefill_bf16(
    cudaStream_t stream,
    nv_bfloat16* output,
    float* output_state,
    nv_bfloat16 const* q,
    nv_bfloat16 const* k,
    nv_bfloat16 const* v,
    float const* input_state,
    float const* alpha,     // gate, can be null
    float const* beta,      // update gate, can be null
    int64_t const* cu_seqlens,
    uint8_t* workspace,
    int32_t num_seqs,
    int32_t num_q_heads,
    int32_t num_k_heads,
    int32_t num_v_heads,
    int32_t num_o_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    int32_t sm_count)
{
    bool is_gva = num_v_heads > num_q_heads;
    bool needs_beta = (beta != nullptr);
    bool needs_alpha = (alpha != nullptr);
    bool init_state = (input_state != nullptr);

#define LAUNCH(gva, nb, na, is) \
    launch_delta_rule_prefill_kernel_gbai<gva, nb, na, is, false, \
        cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>( \
        stream, output, output_state, q, k, v, input_state, alpha, beta, \
        cu_seqlens, workspace, num_seqs, num_q_heads, num_k_heads, \
        num_v_heads, num_o_heads, head_size, total_seqlen, scale, sm_count, \
        nullptr, nullptr, 0);

    // Dispatch on runtime booleans → compile-time template params
    if (is_gva) {
        if (needs_beta && needs_alpha) {
            if (init_state) { LAUNCH(true, true, true, true); }
            else            { LAUNCH(true, true, true, false); }
        } else if (!needs_beta && needs_alpha) {
            if (init_state) { LAUNCH(true, false, true, true); }
            else            { LAUNCH(true, false, true, false); }
        } else if (needs_beta && !needs_alpha) {
            LAUNCH(true, true, false, false);
        } else {
            LAUNCH(true, false, false, false);
        }
    } else {
        // Non-GVA fallback (not instantiated — will linker-error if reached)
        fprintf(stderr, "ERROR: non-GVA not instantiated in this build\n");
    }
#undef LAUNCH
}

}  // namespace flat
