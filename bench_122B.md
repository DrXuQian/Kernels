# Qwen3.5-122B-A10B Benchmark Results

This file records the current H800 Nsight Compute result set for the
standalone Qwen3.5-122B-A10B kernel mix. Operator shapes and source mappings
are kept in [OPERATOR_COVERAGE.md](OPERATOR_COVERAGE.md).

## Measurement Notes

The H800 table below is from an NCU capture. Columns mean:

| Column | Meaning |
|---|---|
| `kernels` | Number of profiled CUDA kernel rows for this benchmark case. |
| `sum_cycles_avg` | Sum of `sm__cycles_elapsed.avg` across kernel rows. |
| `sum_cycles_max` | Sum of `sm__cycles_elapsed.max` across kernel rows. |
| `sum_duration_ns` | Sum of profiled GPU kernel durations in ns. |

For multi-kernel cases, `sum_duration_ns` is the sum of kernel durations, not
host-side wall time between kernels. It intentionally excludes CPU launch cost
and cuBLAS/Python dispatch gaps.

Default model-repeat assumptions used by `bench_all.sh` and the summary tools:

| Parameter | Value |
|---|---:|
| hidden_size | 3072 |
| full_attention_layers | 12 |
| linear_attention_layers | 36 |
| moe_ffn_layers | 48 |
| prefill tokens | 3823 |
| decode tokens | 1 |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 1024 |
| shared_expert_intermediate_size | 1024 |
| vocab_size | 248320 |

## Missing From This H800 NCU Capture

These cases are present in `./bench_all.sh --list` but are not present in the
NCU table below:

| Case | Note |
|---|---|
| `flash_attn_prefill_full_attn` | FlashAttention core was not captured in this NCU result set. |
| `flash_attn_decode_full_attn` | FlashAttention core was not captured in this NCU result set. |
| `linear_attn_decode_fused_rms_norm_gate` | Not captured in this NCU result set. |
| `linear_attn_prefill_fused_rms_norm_gate` | Not captured in this NCU result set. |
| `moe_shared_expert_activation_prefill_trtllm` | Not captured in this NCU result set. |
| `moe_shared_expert_activation_decode_trtllm` | Not captured in this NCU result set. |
| `moe_align_decode_vllm` | Not captured in this NCU result set. |

No other `bench_all.sh` case is missing from the provided H800 NCU table.

## Flash-Attn

### Decode

| Case | kernels | sum_cycles_avg | sum_cycles_max | sum_duration_ns |
|---|---:|---:|---:|---:|
| `flash_attn_decode_k_norm` | 1 | 6758.820 | 6762 | 4288 |
| `flash_attn_decode_q_norm` | 1 | 6778.700 | 6781 | 4320 |
| `flash_attn_decode_residual_add` | 3 | 12562.160 | 12572 | 7968 |
| `flash_attn_decode_rmsnorm` | 1 | 9450.480 | 9455 | 5984 |
| `w4a16_decode_full_attn_k_proj_fpA_intB` | 1 | 12218.560 | 12223 | 7712 |
| `w4a16_decode_full_attn_o_proj_fpA_intB` | 1 | 22983.910 | 22996 | 14496 |
| `w4a16_decode_full_attn_q_proj_gate_fpA_intB` | 1 | 22746.830 | 22866 | 14496 |
| `w4a16_decode_full_attn_v_proj_fpA_intB` | 1 | 12111.210 | 12116 | 7648 |

### Prefill

| Case | kernels | sum_cycles_avg | sum_cycles_max | sum_duration_ns |
|---|---:|---:|---:|---:|
| `flash_attn_prefill_k_norm` | 1 | 23561.180 | 23614 | 14912 |
| `flash_attn_prefill_q_norm` | 1 | 281965.590 | 282058 | 177440 |
| `flash_attn_prefill_residual_add` | 3 | 95314.050 | 95500 | 60192 |
| `flash_attn_prefill_rmsnorm` | 1 | 84347.420 | 84429 | 53120 |
| `w4a16_prefill_full_attn_k_proj_cutlass55` | 1 | 62231.580 | 62491 | 39776 |
| `w4a16_prefill_full_attn_o_proj_cutlass55` | 1 | 415034.970 | 419432 | 273440 |
| `w4a16_prefill_full_attn_q_proj_gate_cutlass55` | 1 | 801631.560 | 806852 | 518656 |
| `w4a16_prefill_full_attn_v_proj_cutlass55` | 1 | 61907.740 | 62104 | 39424 |

## Linear-Attn

### Decode

| Case | kernels | sum_cycles_avg | sum_cycles_max | sum_duration_ns |
|---|---:|---:|---:|---:|
| `linear_attn_decode_in_proj_a_cublas` | 4 | 25091.230 | 25165 | 15968 |
| `linear_attn_decode_in_proj_b_cublas` | 4 | 25722.710 | 25803 | 16320 |
| `linear_attn_decode_residual_add` | 3 | 13016.630 | 13027 | 8288 |
| `linear_attn_decode_rmsnorm` | 1 | 9529.970 | 9535 | 6016 |
| `linear_decode_conv1d_update` | 1 | 8852.610 | 8856 | 5600 |
| `linear_decode_gdn` | 1 | 11079.980 | 11095 | 7040 |
| `w4a16_decode_linear_attn_in_proj_qkv_fpA_intB` | 1 | 18461.210 | 18547 | 11744 |
| `w4a16_decode_linear_attn_in_proj_z_fpA_intB` | 1 | 16530.320 | 16573 | 10464 |
| `w4a16_decode_linear_attn_out_proj_fpA_intB` | 1 | 23071.790 | 23087 | 14560 |

### Prefill

| Case | kernels | sum_cycles_avg | sum_cycles_max | sum_duration_ns |
|---|---:|---:|---:|---:|
| `linear_attn_prefill_in_proj_a_cublas` | 3 | 51980.320 | 52232 | 33248 |
| `linear_attn_prefill_in_proj_b_cublas` | 3 | 48630.880 | 48747 | 30944 |
| `linear_attn_prefill_residual_add` | 3 | 99180.530 | 99354 | 62624 |
| `linear_attn_prefill_rmsnorm` | 1 | 81508.710 | 81666 | 51424 |
| `linear_prefill_conv1d_fwd` | 1 | 164492.940 | 164812 | 103744 |
| `linear_prefill_flashinfer_gdn` | 1 | 868840.360 | 869178 | 547712 |
| `w4a16_prefill_linear_attn_in_proj_qkv_cutlass55` | 1 | 592269.180 | 597914 | 388704 |
| `w4a16_prefill_linear_attn_in_proj_z_cutlass55` | 1 | 432636.390 | 436858 | 285888 |
| `w4a16_prefill_linear_attn_out_proj_cutlass55` | 1 | 415357.500 | 419797 | 273632 |

## MoE-FFN

### Decode

| Case | kernels | sum_cycles_avg | sum_cycles_max | sum_duration_ns |
|---|---:|---:|---:|---:|
| `moe_down_decode_vllm` | 1 | 20520.790 | 20600 | 13216 |
| `moe_ffn_decode_residual_add` | 3 | 12621.350 | 12633 | 8032 |
| `moe_ffn_decode_rmsnorm` | 1 | 9374.740 | 9379 | 5920 |
| `moe_finalize_decode_vllm` | 1 | 5712.940 | 5717 | 3648 |
| `moe_gate_up_decode_vllm` | 1 | 33898.550 | 33981 | 21600 |
| `moe_gated_decode_vllm` | 1 | 7848.830 | 7852 | 4960 |
| `moe_router_gate_decode_cublas` | 4 | 28075.140 | 28181 | 17888 |
| `moe_routing_decode_vllm` | 1 | 12063.230 | 12068 | 7616 |
| `moe_shared_expert_fusion_decode` | 4 | 17599.690 | 17618 | 11168 |
| `moe_shared_expert_gate_decode_cublas` | 4 | 21053.620 | 21063 | 13408 |
| `w4a16_decode_consistent_expert_down_fpA_intB` | 1 | 11608.620 | 11616 | 7328 |
| `w4a16_decode_consistent_expert_up_fpA_intB` | 1 | 10474.330 | 10490 | 6656 |

### Prefill

| Case | kernels | sum_cycles_avg | sum_cycles_max | sum_duration_ns |
|---|---:|---:|---:|---:|
| `moe_down_prefill_trtllm` | 1 | 948383.710 | 948616 | 596896 |
| `moe_expand_prefill_trtllm` | 1 | 382169.980 | 382181 | 240416 |
| `moe_expert_map_prefill_trtllm` | 3 | 53195.590 | 53239 | 33600 |
| `moe_ffn_prefill_residual_add` | 3 | 103954.570 | 104169 | 65664 |
| `moe_ffn_prefill_rmsnorm` | 1 | 81199.940 | 81232 | 51104 |
| `moe_finalize_prefill_trtllm` | 1 | 149599.710 | 149702 | 94176 |
| `moe_gate_up_prefill_trtllm` | 1 | 1765701.500 | 1765981 | 1110944 |
| `moe_gated_prefill_trtllm` | 1 | 484353.550 | 485097 | 305248 |
| `moe_router_gate_prefill_cublas` | 3 | 61616.820 | 61830 | 39264 |
| `moe_routing_prefill_trtllm` | 1 | 19613.080 | 19628 | 12384 |
| `moe_shared_expert_fusion_prefill` | 4 | 159174.630 | 159375 | 100416 |
| `moe_shared_expert_gate_prefill_cublas` | 3 | 52362.730 | 52457 | 33088 |
| `w4a16_prefill_consistent_expert_down_cutlass55` | 1 | 69605.480 | 70121 | 45216 |
| `w4a16_prefill_consistent_expert_up_cutlass55` | 1 | 134150.820 | 135479 | 88128 |

## Sampling

Sampling is decode-only in the default `bench_all.sh` model summary.

| Case | kernels | sum_cycles_avg | sum_cycles_max | sum_duration_ns |
|---|---:|---:|---:|---:|
| `sampling_lm_head_gemm` | 3 | 1764629.610 | 1765727 | 1111744 |
| `sampling_softmax` | 3 | 80546.770 | 80566 | 50752 |
| `sampling_top_p` | 2 | 49576.780 | 49589 | 31264 |
| `sampling_topk_mask_logits` | 2 | 81894.170 | 81903 | 51616 |

## Interpretation Notes

- Decode cuBLAS small GEMMs (`M=1`) are represented by several cuBLASLt kernel
  rows. Their NCU kernel-only totals exclude host dispatch gaps, so they should
  be compared against nsys/NCU kernel durations rather than PyTorch eager event
  times.
- `sampling_lm_head_gemm` dominates sampling and is much larger than the other
  decode FP16 cuBLAS rows because it reads the full vocab projection.
- The current H800 NCU table still lacks the FlashAttention core rows. Any
  module-level Flash-Attn total from this file is therefore incomplete until
  `flash_attn_prefill_full_attn` and `flash_attn_decode_full_attn` are rerun.
- The current H800 NCU table also lacks `moe_align_decode_vllm` and the TRT-LLM
  shared-expert activation rows, so MoE decode/prefill totals from this file are
  incomplete by those small components.
