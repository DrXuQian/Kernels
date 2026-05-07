# Operator Coverage

This file lists the kernels needed by the Qwen3.5-122B-A10B standalone profiling
path. It is organized by module, and each module is split into prefill and
decode in execution order.

Coverage status meanings:

| Status | Meaning |
|---|---|
| `covered` | A standalone CUDA/Python benchmark exists in this repo and is wired into `bench_all.sh`. |
| `missing` | The model needs this kernel, but this repo does not currently have a CUDA standalone benchmark for it. |

Module order:

1. Flash-Attn
2. Linear-Attn
3. MoE-FFN
4. Sampling

## Flash-Attn

### Prefill

| Order | Kernel | Shape | Status | Implementation | `bench_all.sh` case / note |
|---:|---|---:|---|---|---|
| 1 | RMSNorm | `(3823,3072)` | covered | `flash_attn/bench_rmsnorm` | `flash_attn_prefill_rmsnorm` |
| 2 | `q_proj + gate` W4A16 GEMM | `(3823,3072)->(3823,16384)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_full_attn_q_proj_gate_cutlass55` |
| 3 | `k_proj` W4A16 GEMM | `(3823,3072)->(3823,512)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_full_attn_k_proj_cutlass55` |
| 4 | `v_proj` W4A16 GEMM | `(3823,3072)->(3823,512)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_full_attn_v_proj_cutlass55` |
| 5 | Q RMSNorm | `(3823*32,256)` | covered | `flash_attn/bench_rmsnorm` | `flash_attn_prefill_q_norm` |
| 6 | K RMSNorm | `(3823*2,256)` | covered | `flash_attn/bench_rmsnorm` | `flash_attn_prefill_k_norm` |
| 7 | MRoPE | `Q:(3823,32,256), K:(3823,2,256)` | missing | none | missing: no CUDA standalone in repo |
| 8 | FlashAttention core | `Q:(3823,32,256), KV:(3823,2,256)` | covered | `flash_attn/bench_flash_attn.py` | `flash_attn_prefill_full_attn` |
| 9 | Output gate, `sigmoid(gate) * attn` | `(3823,8192)` | missing | none | missing: no CUDA standalone in repo |
| 10 | `o_proj` W4A16 GEMM | `(3823,8192)->(3823,3072)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_full_attn_o_proj_cutlass55` |
| 11 | Residual add | `(3823,3072)+(3823,3072)` | covered | `linear_attn/bench_linear_ops` | `flash_attn_prefill_residual_add` |

### Decode

| Order | Kernel | Shape | Status | Implementation | `bench_all.sh` case / note |
|---:|---|---:|---|---|---|
| 1 | RMSNorm | `(1,3072)` | covered | `flash_attn/bench_rmsnorm` | `flash_attn_decode_rmsnorm` |
| 2 | `q_proj + gate` W4A16 GEMM | `(1,3072)->(1,16384)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_full_attn_q_proj_gate_fpA_intB` |
| 3 | `k_proj` W4A16 GEMM | `(1,3072)->(1,512)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_full_attn_k_proj_fpA_intB` |
| 4 | `v_proj` W4A16 GEMM | `(1,3072)->(1,512)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_full_attn_v_proj_fpA_intB` |
| 5 | Q RMSNorm | `(32,256)` | covered | `flash_attn/bench_rmsnorm` | `flash_attn_decode_q_norm` |
| 6 | K RMSNorm | `(2,256)` | covered | `flash_attn/bench_rmsnorm` | `flash_attn_decode_k_norm` |
| 7 | MRoPE | `Q:(1,32,256), K:(1,2,256)` | missing | none | missing: no CUDA standalone in repo |
| 8 | FlashAttention core | `Q:(1,32,256), KV:(3823,2,256)` | covered | `flash_attn/bench_flash_attn.py` | `flash_attn_decode_full_attn` |
| 9 | Output gate, `sigmoid(gate) * attn` | `(1,8192)` | missing | none | missing: no CUDA standalone in repo |
| 10 | `o_proj` W4A16 GEMM | `(1,8192)->(1,3072)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_full_attn_o_proj_fpA_intB` |
| 11 | Residual add | `(1,3072)+(1,3072)` | covered | `linear_attn/bench_linear_ops` | `flash_attn_decode_residual_add` |

## Linear-Attn

### Prefill

| Order | Kernel | Shape | Status | Implementation | `bench_all.sh` case / note |
|---:|---|---:|---|---|---|
| 1 | RMSNorm | `(3823,3072)` | covered | `linear_attn/bench_rmsnorm` | `linear_attn_prefill_rmsnorm` |
| 2 | `in_proj_a` FP16 GEMM | `(3823,3072)->(3823,64)` | covered | `general/bench_cublas_gemm` | `linear_attn_prefill_in_proj_a_cublas` |
| 3 | `in_proj_b` FP16 GEMM | `(3823,3072)->(3823,64)` | covered | `general/bench_cublas_gemm` | `linear_attn_prefill_in_proj_b_cublas` |
| 4 | `in_proj_qkv` W4A16 GEMM | `(3823,3072)->(3823,12288)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_linear_attn_in_proj_qkv_cutlass55` |
| 5 | `in_proj_z` W4A16 GEMM | `(3823,3072)->(3823,8192)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_linear_attn_in_proj_z_cutlass55` |
| 6 | Gate prep, `g = -exp(A) * softplus(a + dt_bias)` | `(3823,64)` | missing | none | missing: no CUDA standalone in repo |
| 7 | Conv1d forward | `(3823,12288)` | covered | `linear_attn/bench_conv1d_fwd` | `linear_prefill_conv1d_fwd` |
| 8 | Gated Delta Net prefill | `tokens=3823, q_heads=16, v_heads=64, head_dim=128` | covered | `linear_attn/bench_gdn_prefill` | `linear_prefill_flashinfer_gdn` |
| 9 | Fused RMSNorm gate | `(3823*64,128)` | covered | `linear_attn/bench_fused_rms_norm_gate` | `linear_attn_prefill_fused_rms_norm_gate` |
| 10 | Output projection W4A16 GEMM | `(3823,8192)->(3823,3072)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_linear_attn_out_proj_cutlass55` |
| 11 | Residual add | `(3823,3072)+(3823,3072)` | covered | `linear_attn/bench_linear_ops` | `linear_attn_prefill_residual_add` |

### Decode

| Order | Kernel | Shape | Status | Implementation | `bench_all.sh` case / note |
|---:|---|---:|---|---|---|
| 1 | RMSNorm | `(1,3072)` | covered | `linear_attn/bench_rmsnorm` | `linear_attn_decode_rmsnorm` |
| 2 | `in_proj_a` FP16 GEMM | `(1,3072)->(1,64)` | covered | `general/bench_cublas_gemm` | `linear_attn_decode_in_proj_a_cublas` |
| 3 | `in_proj_b` FP16 GEMM | `(1,3072)->(1,64)` | covered | `general/bench_cublas_gemm` | `linear_attn_decode_in_proj_b_cublas` |
| 4 | `in_proj_qkv` W4A16 GEMM | `(1,3072)->(1,12288)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_linear_attn_in_proj_qkv_fpA_intB` |
| 5 | `in_proj_z` W4A16 GEMM | `(1,3072)->(1,8192)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_linear_attn_in_proj_z_fpA_intB` |
| 6 | Gate prep, `g = -exp(A) * softplus(a + dt_bias)` | `(1,64)` | missing | none | missing: no CUDA standalone in repo |
| 7 | Conv1d update | `(1,12288)` | covered | `linear_attn/bench_conv1d_update` | `linear_decode_conv1d_update` |
| 8 | Gated Delta Net decode | `Q,K,V:(1,64,128)` | covered | `linear_attn/bench_gated_delta_net` | `linear_decode_gdn` |
| 9 | Fused RMSNorm gate | `(64,128)` | covered | `linear_attn/bench_fused_rms_norm_gate` | `linear_attn_decode_fused_rms_norm_gate` |
| 10 | Output projection W4A16 GEMM | `(1,8192)->(1,3072)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_linear_attn_out_proj_fpA_intB` |
| 11 | Residual add | `(1,3072)+(1,3072)` | covered | `linear_attn/bench_linear_ops` | `linear_attn_decode_residual_add` |

## MoE-FFN

### Prefill

| Order | Kernel | Shape | Status | Implementation | `bench_all.sh` case / note |
|---:|---|---:|---|---|---|
| 1 | RMSNorm | `(3823,3072)` | covered | `moe_ffn/bench_rmsnorm` | `moe_ffn_prefill_rmsnorm` |
| 2 | Router gate FP16 GEMM | `(3823,3072)->(3823,256)` | covered | `general/bench_cublas_gemm` | `moe_router_gate_prefill_cublas` |
| 3 | Routing top-k | `(3823,256)->topk` | covered | `moe_ffn/w4a16/trtllm/auxiliary/bench_custom_moe_routing` | `moe_routing_prefill_trtllm` |
| 4 | Expert map / prefix metadata | topk/expert metadata | covered | `moe_ffn/w4a16/trtllm/auxiliary/bench_expert_map` | `moe_expert_map_prefill_trtllm` |
| 5 | Expand input rows | `(3823,3072)->(3823*8,3072)` | covered | `moe_ffn/w4a16/trtllm/auxiliary/bench_expand_input_rows` | `moe_expand_prefill_trtllm` |
| 6 | MoE gate/up W4A16 GEMM | per expert `(3823,3072)->(3823,2048)` | covered | `moe_ffn/w4a16/trtllm/moe_w4a16_standalone` | `moe_gate_up_prefill_trtllm` |
| 7 | Gated activation | `(3823*8,2048)->(3823*8,1024)` | covered | `moe_ffn/w4a16/trtllm/auxiliary/bench_gated_activation` | `moe_gated_prefill_trtllm` |
| 8 | MoE down W4A16 GEMM | per expert `(3823,1024)->(3823,3072)` | covered | `moe_ffn/w4a16/trtllm/moe_w4a16_standalone` | `moe_down_prefill_trtllm` |
| 9 | Finalize routing | `(3823*8,3072)->(3823,3072)` | covered | `moe_ffn/w4a16/trtllm/auxiliary/bench_finalize_moe_routing` | `moe_finalize_prefill_trtllm` |
| 10 | Shared expert gate FP16 GEMM | `(3823,3072)->(3823,1)` | covered | `general/bench_cublas_gemm` | `moe_shared_expert_gate_prefill_cublas` |
| 11 | Shared/consistent expert up W4A16 GEMM | `(3823,2048)->(3823,3072)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_consistent_expert_up_cutlass55` |
| 12 | Shared expert activation | `(3823,2048)->(3823,1024)` | missing | none | missing: no CUDA standalone in repo |
| 13 | Shared/consistent expert down W4A16 GEMM | `(3823,3072)->(3823,1024)` | covered | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | `w4a16_prefill_consistent_expert_down_cutlass55` |
| 14 | Shared expert fusion, `routed + sigmoid(gate) * shared` | `(3823,3072)` | covered | `moe_ffn/bench_shared_expert` | `moe_shared_expert_fusion_prefill` |
| 15 | Residual add | `(3823,3072)+(3823,3072)` | covered | `linear_attn/bench_linear_ops` | `moe_ffn_prefill_residual_add` |

### Decode

| Order | Kernel | Shape | Status | Implementation | `bench_all.sh` case / note |
|---:|---|---:|---|---|---|
| 1 | RMSNorm | `(1,3072)` | covered | `moe_ffn/bench_rmsnorm` | `moe_ffn_decode_rmsnorm` |
| 2 | Router gate FP16 GEMM | `(1,3072)->(1,256)` | covered | `general/bench_cublas_gemm` | `moe_router_gate_decode_cublas` |
| 3 | Routing top-k | `(1,256)->topk` | covered | `moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating` | `moe_routing_decode_vllm` |
| 4 | MoE align metadata | topk/expert metadata | covered | `moe_ffn/w4a16/vllm/auxiliary/bench_moe_align` | `moe_align_decode_vllm` |
| 5 | MoE gate/up W4A16 GEMM | routed `(8,1,3072)->(8,1,2048)` | covered | `moe_ffn/w4a16/vllm/marlin/bench_marlin_moe` | `moe_gate_up_decode_vllm` |
| 6 | Gated activation | `(8,1,2048)->(8,1,1024)` | covered | `moe_ffn/w4a16/vllm/auxiliary/bench_silu_and_mul` | `moe_gated_decode_vllm` |
| 7 | MoE down W4A16 GEMM | routed `(8,1,1024)->(8,1,3072)` | covered | `moe_ffn/w4a16/vllm/marlin/bench_marlin_moe` | `moe_down_decode_vllm` |
| 8 | Finalize / sum experts | `(8,1,3072)->(1,3072)` | covered | `moe_ffn/w4a16/vllm/auxiliary/bench_moe_sum` | `moe_finalize_decode_vllm` |
| 9 | Shared expert gate FP16 GEMM | `(1,3072)->(1,1)` | covered | `general/bench_cublas_gemm` | `moe_shared_expert_gate_decode_cublas` |
| 10 | Shared/consistent expert up W4A16 GEMM | `(1,2048)->(1,3072)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_consistent_expert_up_fpA_intB` |
| 11 | Shared expert activation | `(1,2048)->(1,1024)` | missing | none | missing: no CUDA standalone in repo |
| 12 | Shared/consistent expert down W4A16 GEMM | `(1,3072)->(1,1024)` | covered | `general/w4a16_gemm/fpA_intB_standalone` | `w4a16_decode_consistent_expert_down_fpA_intB` |
| 13 | Shared expert fusion, `routed + sigmoid(gate) * shared` | `(1,3072)` | covered | `moe_ffn/bench_shared_expert` | `moe_shared_expert_fusion_decode` |
| 14 | Residual add | `(1,3072)+(1,3072)` | covered | `linear_attn/bench_linear_ops` | `moe_ffn_decode_residual_add` |

## Sampling

Sampling is decode-only.

### Decode

| Order | Kernel | Shape | Status | Implementation | `bench_all.sh` case / note |
|---:|---|---:|---|---|---|
| 1 | `lm_head` FP16 GEMM | `(1,3072)->(1,248320)` | covered | `general/bench_cublas_gemm`, fp32 output | `sampling_lm_head_gemm` |
| 2 | Top-K mask logits | `(1,248320)->(1,248320)` | covered | `sampling/bench_sampling`, vendored FlashInfer kernel headers | `sampling_topk_mask_logits` |
| 3 | Softmax | `(1,248320)->(1,248320)` | covered | `sampling/bench_sampling`, vendored FlashInfer kernel headers | `sampling_softmax` |
| 4 | Top-P sample | `(1,248320)->(1,)` | covered | `sampling/bench_sampling`, vendored FlashInfer kernel headers | `sampling_top_p` |

### Prefill

No Sampling kernels are used in prefill.

## Shared Standalone Backends

| Backend | Implementation | Used by |
|---|---|---|
| FP16/BF16 GEMM | `general/bench_cublas_gemm` | Linear-Attn, MoE-FFN, Sampling |
| RMSNorm | `general/bench_rmsnorm.cu`, built into category folders | Flash-Attn, Linear-Attn, MoE-FFN |
| W4A16 prefill GEMM | `general/w4a16_gemm/machete_standalone`, CUTLASS55 backend | Flash-Attn, Linear-Attn, MoE-FFN |
| W4A16 decode GEMM | `general/w4a16_gemm/fpA_intB_standalone` | Flash-Attn, Linear-Attn, MoE-FFN |
