#include <cuda_bf16.h>

#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"

namespace gdn_splitseq_study {

using T = nv_bfloat16;

void launch_gdn_prefill_bf16_gva_checkpoint(
    cudaStream_t stream, T* output, float* output_state, T const* q, T const* k, T const* v,
    float const* alpha, float const* beta, int64_t const* cu_seqlens, uint8_t* workspace,
    int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads,
    int32_t num_o_heads, int32_t head_size, int64_t total_seqlen, float scale, int32_t sm_count,
    float* state_checkpoints, int64_t const* checkpoint_cu_starts,
    int32_t checkpoint_every_n_tokens) {
  flat::launch_delta_rule_prefill_kernel_gbai<
      true, true, true, false, true, cutlass::arch::Sm90, T, T, float>(
      stream, output, output_state, q, k, v,
      nullptr, alpha, beta, cu_seqlens, workspace, num_seqs, num_q_heads, num_k_heads,
      num_v_heads, num_o_heads, head_size, total_seqlen, scale, sm_count, state_checkpoints,
      checkpoint_cu_starts, checkpoint_every_n_tokens);
}

void launch_gdn_prefill_bf16_gva_initstate(
    cudaStream_t stream, T* output, float* output_state, T const* q, T const* k, T const* v,
    float const* input_state, float const* alpha, float const* beta, int64_t const* cu_seqlens,
    uint8_t* workspace, int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads,
    int32_t num_v_heads, int32_t num_o_heads, int32_t head_size, int64_t total_seqlen,
    float scale, int32_t sm_count) {
  flat::launch_delta_rule_prefill_kernel_gbai<
      true, true, true, true, false, cutlass::arch::Sm90, T, T, float>(
      stream, output, output_state, q, k, v, input_state, alpha, beta, cu_seqlens, workspace,
      num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen,
      scale, sm_count, nullptr, nullptr, 0);
}

}  // namespace gdn_splitseq_study
