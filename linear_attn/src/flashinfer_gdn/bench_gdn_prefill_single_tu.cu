// Single-translation-unit build for the Qwen3.5 GVA FlashInfer GDN prefill path.
//
// The SM90 warp-specialized GDN kernel depends on setmaxnreg/warpgroup register
// allocation. Building the launcher and kernel template in separate device-link
// objects can make ptxas ignore setmaxnreg and serialize WGMMA.
//
// This file intentionally instantiates only the path used by bench_all:
//   GVA=true, alpha=true, beta=true, init_state=false, checkpoint=false.
// Use bench_gdn_prefill_separable for the slower generic runtime-dispatch build.

#include <stdexcept>

#include <cuda_bf16.h>

#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"

namespace flat {

void launch_gdn_prefill_bf16(
    cudaStream_t stream, nv_bfloat16* output, float* output_state,
    nv_bfloat16 const* q, nv_bfloat16 const* k, nv_bfloat16 const* v,
    float const* input_state, float const* alpha, float const* beta, int64_t const* cu_seqlens,
    uint8_t* workspace, int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads,
    int32_t num_v_heads, int32_t num_o_heads, int32_t head_size, int64_t total_seqlen,
    float scale, int32_t sm_count) {
  if (!(num_v_heads > num_q_heads)) {
    throw std::runtime_error("single-TU GDN prefill supports only the GVA path");
  }
  if (input_state != nullptr) {
    throw std::runtime_error("single-TU GDN prefill expects no input state");
  }
  if (alpha == nullptr || beta == nullptr) {
    throw std::runtime_error("single-TU GDN prefill expects alpha and beta");
  }

  launch_delta_rule_prefill_kernel_gbai<
      true, true, true, false, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
      stream, output, output_state, q, k, v, input_state, alpha, beta, cu_seqlens, workspace,
      num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen,
      scale, sm_count, nullptr, nullptr, 0);
}

}  // namespace flat

#ifdef CHECK
#undef CHECK
#endif

#include "bench_gdn_prefill.cu"
