// Minimal pybind11 binding for marlin_moe_wna16
#include <torch/all.h>
#include <torch/python.h>
#include "core/scalar_type.hpp"

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    vllm::ScalarTypeId const& b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, int64_t thread_k, int64_t thread_n,
    int64_t blocks_per_sm);

// Wrapper that takes b_type_id as int64_t
torch::Tensor moe_wna16_marlin_gemm_wrapper(
    torch::Tensor a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor b_q_weight,
    std::optional<torch::Tensor> b_bias_or_none, torch::Tensor b_scales,
    std::optional<torch::Tensor> a_scales_or_none,
    std::optional<torch::Tensor> global_scale_or_none,
    std::optional<torch::Tensor> b_zeros_or_none,
    std::optional<torch::Tensor> g_idx_or_none,
    std::optional<torch::Tensor> perm_or_none, torch::Tensor workspace,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_past_padded, torch::Tensor topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t b_type_id_val, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, int64_t thread_k, int64_t thread_n,
    int64_t blocks_per_sm) {
  vllm::ScalarTypeId b_type_id = b_type_id_val;
  return moe_wna16_marlin_gemm(
      a, c_or_none, b_q_weight, b_bias_or_none, b_scales,
      a_scales_or_none, global_scale_or_none, b_zeros_or_none,
      g_idx_or_none, perm_or_none, workspace,
      sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights,
      moe_block_size, top_k, mul_topk_weights,
      b_type_id, size_m, size_n, size_k, is_k_full,
      use_atomic_add, use_fp32_reduce, is_zp_float,
      thread_k, thread_n, blocks_per_sm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm_wrapper);
  // Export type IDs
  m.attr("U4B8_TYPE_ID") = vllm::kU4B8.id();
  m.attr("U4_TYPE_ID") = vllm::kU4.id();
}
