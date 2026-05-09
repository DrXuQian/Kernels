#include <stdexcept>

#include "gdn_tile_study_kernel.cuh"

namespace gdn_tile_study {

void launch_gdn_prefill_bf16_gva_tile(
    int tile_tokens, int variant_id, cudaStream_t stream, nv_bfloat16* output, float* output_state,
    nv_bfloat16 const* q, nv_bfloat16 const* k, nv_bfloat16 const* v, float const* alpha,
    float const* beta, int64_t const* cu_seqlens, uint8_t* workspace, int32_t num_seqs,
    int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads, int32_t num_o_heads,
    int32_t head_size, int64_t total_seqlen, float scale, int32_t sm_count) {
  if (tile_tokens != 64) {
    throw std::runtime_error("tile 128 does not satisfy this FlashInfer GDN collective; use --tile 64");
  }

  if (variant_id == 0) {
    launch_gdn_prefill_bf16_gva_tile<64, 2, 3, 2>(
        stream, output, output_state, q, k, v, alpha, beta, cu_seqlens, workspace, num_seqs,
        num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen, scale,
        sm_count);
    return;
  }
  if (variant_id == 1) {
    launch_gdn_prefill_bf16_gva_tile<64, 2, 2, 2>(
        stream, output, output_state, q, k, v, alpha, beta, cu_seqlens, workspace, num_seqs,
        num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen, scale,
        sm_count);
    return;
  }
  if (variant_id == 2) {
    launch_gdn_prefill_bf16_gva_tile<64, 3, 3, 2>(
        stream, output, output_state, q, k, v, alpha, beta, cu_seqlens, workspace, num_seqs,
        num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen, scale,
        sm_count);
    return;
  }
  if (variant_id == 3) {
    launch_gdn_prefill_bf16_gva_tile<64, 2, 3, 3>(
        stream, output, output_state, q, k, v, alpha, beta, cu_seqlens, workspace, num_seqs,
        num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen, scale,
        sm_count);
    return;
  }
  throw std::runtime_error("unsupported variant; use default, k2, q3, or v3");
}

}  // namespace gdn_tile_study
