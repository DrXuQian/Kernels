# Operator Coverage

This file lists the Qwen3.5-122B-A10B operators currently covered by the
standalone benchmark folders and `bench_all.sh`.

Scope:
- Include operators discussed for Linear-Attn, Flash-Attn, MoE-FFN, Sampling,
  and shared general kernels.
- Include CUDA standalone implementations in this repo.
- Include Python only for the FlashAttention core path.
- Skip operators where no CUDA standalone source was found.

`bench_all.sh` covers every operator listed below.

## General

| Operator | Shape | Implementation | `bench_all.sh` usage |
|---|---:|---|---|
| FP16/BF16 GEMM | `M,N,K` | `general/bench_cublas_gemm` | Used by dense FP16 cases below |
| RMSNorm | `(tokens, hidden)` or `(tokens*heads, head_dim)` | `general/bench_rmsnorm.cu`, built into category folders | `*_rmsnorm`, `flash_attn_*_q_norm`, `flash_attn_*_k_norm` |

## Linear-Attn

| Order | Operator | Shape | Implementation | `bench_all.sh` case |
|---:|---|---:|---|---|
| 1 | RMSNorm decode | `(1,3072)` | `linear_attn/bench_rmsnorm` | `linear_attn_decode_rmsnorm` |
| 1 | RMSNorm prefill | `(3823,3072)` | `linear_attn/bench_rmsnorm` | `linear_attn_prefill_rmsnorm` |
| 2 | `in_proj_a` FP16 GEMM decode | `(1,3072)->(1,64)` | `general/bench_cublas_gemm` | `linear_attn_decode_in_proj_a_cublas` |
| 2 | `in_proj_b` FP16 GEMM decode | `(1,3072)->(1,64)` | `general/bench_cublas_gemm` | `linear_attn_decode_in_proj_b_cublas` |
| 2 | `in_proj_a` FP16 GEMM prefill | `(3823,3072)->(3823,64)` | `general/bench_cublas_gemm` | `linear_attn_prefill_in_proj_a_cublas` |
| 2 | `in_proj_b` FP16 GEMM prefill | `(3823,3072)->(3823,64)` | `general/bench_cublas_gemm` | `linear_attn_prefill_in_proj_b_cublas` |
| 3 | `in_proj_qkv` W4A16 prefill | `(3823,3072)->(3823,12288)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_linear_attn_in_proj_qkv_cutlass55` |
| 3 | `in_proj_qkv` W4A16 decode | `(1,3072)->(1,12288)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_linear_attn_in_proj_qkv_fpA_intB` |
| 4 | `in_proj_z` W4A16 prefill | `(3823,3072)->(3823,8192)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_linear_attn_in_proj_z_cutlass55` |
| 4 | `in_proj_z` W4A16 decode | `(1,3072)->(1,8192)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_linear_attn_in_proj_z_fpA_intB` |
| 5 | Conv1d update decode | `(1,12288)` state update | `linear_attn/bench_conv1d_update` | `linear_decode_conv1d_update` |
| 5 | Conv1d forward prefill | `(3823,12288)` | `linear_attn/bench_conv1d_fwd` | `linear_prefill_conv1d_fwd` |
| 6 | Gated Delta Net decode | `Q,K,V:(1,64,128)` | `linear_attn/bench_gated_delta_net` | `linear_decode_gdn` |
| 6 | Gated Delta Net prefill | `tokens=3823, q_heads=16, v_heads=64, head_dim=128` | `linear_attn/bench_gdn_prefill` | `linear_prefill_flashinfer_gdn` |
| 7 | Output projection W4A16 prefill | `(3823,8192)->(3823,3072)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_linear_attn_out_proj_cutlass55` |
| 7 | Output projection W4A16 decode | `(1,8192)->(1,3072)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_linear_attn_out_proj_fpA_intB` |
| 8 | Residual add decode | `(1,3072)+(1,3072)` | `linear_attn/bench_linear_ops` | `linear_attn_decode_residual_add` |
| 8 | Residual add prefill | `(3823,3072)+(3823,3072)` | `linear_attn/bench_linear_ops` | `linear_attn_prefill_residual_add` |

## Flash-Attn

| Order | Operator | Shape | Implementation | `bench_all.sh` case |
|---:|---|---:|---|---|
| 1 | RMSNorm decode | `(1,3072)` | `flash_attn/bench_rmsnorm` | `flash_attn_decode_rmsnorm` |
| 1 | RMSNorm prefill | `(3823,3072)` | `flash_attn/bench_rmsnorm` | `flash_attn_prefill_rmsnorm` |
| 2 | `q_proj + gate` W4A16 prefill | `(3823,3072)->(3823,16384)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_full_attn_q_proj_gate_cutlass55` |
| 2 | `q_proj + gate` W4A16 decode | `(1,3072)->(1,16384)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_full_attn_q_proj_gate_fpA_intB` |
| 3 | `k_proj` W4A16 prefill | `(3823,3072)->(3823,512)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_full_attn_k_proj_cutlass55` |
| 3 | `k_proj` W4A16 decode | `(1,3072)->(1,512)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_full_attn_k_proj_fpA_intB` |
| 4 | `v_proj` W4A16 prefill | `(3823,3072)->(3823,512)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_full_attn_v_proj_cutlass55` |
| 4 | `v_proj` W4A16 decode | `(1,3072)->(1,512)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_full_attn_v_proj_fpA_intB` |
| 5 | Q RMSNorm prefill | `(3823*32,256)` | `flash_attn/bench_rmsnorm` | `flash_attn_prefill_q_norm` |
| 5 | K RMSNorm prefill | `(3823*2,256)` | `flash_attn/bench_rmsnorm` | `flash_attn_prefill_k_norm` |
| 5 | Q RMSNorm decode | `(32,256)` | `flash_attn/bench_rmsnorm` | `flash_attn_decode_q_norm` |
| 5 | K RMSNorm decode | `(2,256)` | `flash_attn/bench_rmsnorm` | `flash_attn_decode_k_norm` |
| 6 | FlashAttention core prefill | `Q:(3823,32,256), KV:(3823,2,256)` | `flash_attn/bench_flash_attn.py` | `flash_attn_prefill_full_attn` |
| 6 | FlashAttention core decode | `Q:(1,32,256), KV:(3823,2,256)` | `flash_attn/bench_flash_attn.py` | `flash_attn_decode_full_attn` |
| 7 | Output projection W4A16 prefill | `(3823,8192)->(3823,3072)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_full_attn_o_proj_cutlass55` |
| 7 | Output projection W4A16 decode | `(1,8192)->(1,3072)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_full_attn_o_proj_fpA_intB` |
| 8 | Residual add decode | `(1,3072)+(1,3072)` | `linear_attn/bench_linear_ops` | `flash_attn_decode_residual_add` |
| 8 | Residual add prefill | `(3823,3072)+(3823,3072)` | `linear_attn/bench_linear_ops` | `flash_attn_prefill_residual_add` |

Skipped from the table: MRoPE and attention output gate. vLLM has MRoPE as
Triton, and vLLM/TRT-LLM express the output gate as Torch elementwise ops. No
standalone CUDA implementation was found in the local extracted sources.

## MoE-FFN

| Order | Operator | Shape | Implementation | `bench_all.sh` case |
|---:|---|---:|---|---|
| 1 | RMSNorm decode | `(1,3072)` | `moe_ffn/bench_rmsnorm` | `moe_ffn_decode_rmsnorm` |
| 1 | RMSNorm prefill | `(3823,3072)` | `moe_ffn/bench_rmsnorm` | `moe_ffn_prefill_rmsnorm` |
| 2 | Router gate FP16 GEMM decode | `(1,3072)->(1,256)` | `general/bench_cublas_gemm` | `moe_router_gate_decode_cublas` |
| 2 | Router gate FP16 GEMM prefill | `(3823,3072)->(3823,256)` | `general/bench_cublas_gemm` | `moe_router_gate_prefill_cublas` |
| 3 | TRT-LLM routing prefill | `(3823,256)->topk` | `moe_ffn/w4a16/trtllm/auxiliary/bench_custom_moe_routing` | `moe_routing_prefill_trtllm` |
| 3 | vLLM routing decode | `(1,256)->topk` | `moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating` | `moe_routing_decode_vllm` |
| 4 | TRT-LLM expert map prefill | topk/expert metadata | `moe_ffn/w4a16/trtllm/auxiliary/bench_expert_map` | `moe_expert_map_prefill_trtllm` |
| 4 | vLLM MoE align decode | topk/expert metadata | `moe_ffn/w4a16/vllm/auxiliary/bench_moe_align` | `moe_align_decode_vllm` |
| 5 | TRT-LLM expand input rows prefill | `(3823,3072)->(3823*8,3072)` | `moe_ffn/w4a16/trtllm/auxiliary/bench_expand_input_rows` | `moe_expand_prefill_trtllm` |
| 6 | TRT-LLM MoE gate/up W4A16 prefill | per expert `(3823,3072)->(3823,2048)` | `moe_ffn/w4a16/trtllm/moe_w4a16_standalone` | `moe_gate_up_prefill_trtllm` |
| 6 | vLLM Marlin gate/up W4A16 decode | routed `(8,1,3072)->(8,1,2048)` | `moe_ffn/w4a16/vllm/marlin/bench_marlin_moe` | `moe_gate_up_decode_vllm` |
| 7 | TRT-LLM gated activation prefill | `(3823*8,2048)->(3823*8,1024)` | `moe_ffn/w4a16/trtllm/auxiliary/bench_gated_activation` | `moe_gated_prefill_trtllm` |
| 7 | vLLM SiLU and mul decode | `(8,1,2048)->(8,1,1024)` | `moe_ffn/w4a16/vllm/auxiliary/bench_silu_and_mul` | `moe_gated_decode_vllm` |
| 8 | TRT-LLM MoE down W4A16 prefill | per expert `(3823,1024)->(3823,3072)` | `moe_ffn/w4a16/trtllm/moe_w4a16_standalone` | `moe_down_prefill_trtllm` |
| 8 | vLLM Marlin down W4A16 decode | routed `(8,1,1024)->(8,1,3072)` | `moe_ffn/w4a16/vllm/marlin/bench_marlin_moe` | `moe_down_decode_vllm` |
| 9 | TRT-LLM finalize routing prefill | `(3823*8,3072)->(3823,3072)` | `moe_ffn/w4a16/trtllm/auxiliary/bench_finalize_moe_routing` | `moe_finalize_prefill_trtllm` |
| 9 | vLLM MoE sum decode | `(8,1,3072)->(1,3072)` | `moe_ffn/w4a16/vllm/auxiliary/bench_moe_sum` | `moe_finalize_decode_vllm` |
| 10 | Shared expert gate FP16 GEMM decode | `(1,3072)->(1,1)` | `general/bench_cublas_gemm` | `moe_shared_expert_gate_decode_cublas` |
| 10 | Shared expert gate FP16 GEMM prefill | `(3823,3072)->(3823,1)` | `general/bench_cublas_gemm` | `moe_shared_expert_gate_prefill_cublas` |
| 11 | Shared expert fusion decode | `routed + sigmoid(gate) * shared`, `(1,3072)` | `moe_ffn/bench_shared_expert` | `moe_shared_expert_fusion_decode` |
| 11 | Shared expert fusion prefill | `routed + sigmoid(gate) * shared`, `(3823,3072)` | `moe_ffn/bench_shared_expert` | `moe_shared_expert_fusion_prefill` |
| 12 | Consistent/shared expert up W4A16 decode | `(1,2048)->(1,3072)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_consistent_expert_up_fpA_intB` |
| 12 | Consistent/shared expert up W4A16 prefill | `(3823,2048)->(3823,3072)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_consistent_expert_up_cutlass55` |
| 13 | Consistent/shared expert down W4A16 decode | `(1,3072)->(1,1024)` | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_consistent_expert_down_fpA_intB` |
| 13 | Consistent/shared expert down W4A16 prefill | `(3823,3072)->(3823,1024)` | `general/w4a16_gemm/machete_standalone` CUTLASS55 backend | `w4a16_prefill_consistent_expert_down_cutlass55` |
| 14 | Residual add decode | `(1,3072)+(1,3072)` | `linear_attn/bench_linear_ops` | `moe_ffn_decode_residual_add` |
| 14 | Residual add prefill | `(3823,3072)+(3823,3072)` | `linear_attn/bench_linear_ops` | `moe_ffn_prefill_residual_add` |

## Sampling

| Order | Operator | Shape | Implementation | `bench_all.sh` case |
|---:|---|---:|---|---|
| 1 | lm_head FP16 GEMM | `(1,3072)->(1,248320)` | `general/bench_cublas_gemm`, fp32 output | `sampling_lm_head_gemm` |
| 2 | Top-K mask logits | `(1,248320)->(1,248320)` | `sampling/bench_sampling`, FlashInfer kernel | `sampling_topk_mask_logits` |
| 3 | Softmax | `(1,248320)->(1,248320)` | `sampling/bench_sampling`, FlashInfer kernel | `sampling_softmax` |
| 4 | Top-P sample | `(1,248320)->(1,)` | `sampling/bench_sampling`, FlashInfer kernel | `sampling_top_p` |

## Coverage Answer

Yes: for the operator list above, `bench_all.sh --list` contains a case for
every row.

Not included by design:
- helper/study executables under `studies/`
- standalone comparison/demo binaries such as raw Marlin or CUTLASS55 examples
- Triton-only Linear-Attn scripts
- Flash-Attn MRoPE and output gate, because no CUDA standalone source was found
  locally
