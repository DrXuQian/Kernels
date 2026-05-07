# Qwen3.5-122B-A10B Benchmark Results

This file is organized by module:

1. Flash-Attn
2. Linear-Attn
3. MoE-FFN
4. Sampling

`not tested` means this repo does not currently have a recorded result for that
device/method. PPU latency is converted from `compute_cycles / 1.5GHz`.

## Measurement Notes

Primary H800 reference:

```bash
RUN_ID=h800_nsys_all_$(date +%Y%m%d_%H%M%S)
BENCH_RUN_ID="$RUN_ID" \
PERFRAWLOG_POSTPROCESS=0 \
nsys profile \
  --force-overwrite=true \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --output=".bench_profiles/$RUN_ID" \
  ./bench_all.sh

nsys stats ".bench_profiles/$RUN_ID.nsys-rep" \
  --report cuda_gpu_trace \
  --format csv \
  --output ".bench_profiles/${RUN_ID}_trace"
```

Unless stated otherwise, H800 values below are from nsys `cuda_gpu_trace` and
sum only CUDA kernel rows. PPU values are from per-case `perfstatistics.log`.
All benchmark cases use single-run settings (`warmup=0`, `iters=1`, or
`--bench 0 1`).

Some late-added smoke results are CUDA-event single-run numbers rather than
nsys traces; those rows are marked in the `Method` column.

## Model Parameters

| Parameter | Value |
|---|---:|
| hidden_size | 3072 |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 64 |
| linear_key/value_head_dim | 128 |
| linear_conv_kernel_dim | 4 |
| linear conv_dim (Q+K+V) | 12288 |
| full attention heads | 32 |
| full attention KV heads | 2 |
| full attention head_dim | 256 |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 1024 |
| shared_expert_intermediate_size | 1024 |
| layers | 48 total in the profiled standalone shape mix |

## Flash-Attn

Build and run:

```bash
./compile.sh build flash_attn w4a16-machete w4a16-fpa linear_attn
./bench_all.sh --case flash_attn
./bench_all.sh --case w4a16_prefill_full_attn,w4a16_decode_full_attn
```

### Decode

| Case | Shape | Method | H800 (us) | PPU (us) | PPU cycles |
|---|---:|---|---:|---:|---:|
| `flash_attn_decode_rmsnorm` | `(1,3072)` | not tested by nsys | not tested | not tested | not tested |
| `w4a16_decode_full_attn_q_proj_gate_fpA_intB` | `(1,16384,3072)` | nsys | 12.960 | 15.875 | 23813 |
| `w4a16_decode_full_attn_k_proj_fpA_intB` | `(1,512,3072)` | nsys | 3.936 | 5.347 | 8020 |
| `w4a16_decode_full_attn_v_proj_fpA_intB` | `(1,512,3072)` | nsys | 3.935 | 5.285 | 7927 |
| `flash_attn_decode_q_norm` | `(32,256)` | not tested by nsys | not tested | not tested | not tested |
| `flash_attn_decode_k_norm` | `(2,256)` | not tested by nsys | not tested | not tested | not tested |
| `flash_attn_decode_full_attn` | `Q:(1,32,256), KV:(3823,2,256)` | nsys / perfstatistics | 24.738 | 46.405 | 69608 |
| `w4a16_decode_full_attn_o_proj_fpA_intB` | `(1,3072,8192)` | nsys | 8.672 | 14.235 | 21352 |
| `flash_attn_decode_residual_add` | `(1,3072)` | CUDA event single-run | 15.9 | not tested | not tested |

### Prefill

| Case | Shape | Method | H800 (us) | PPU (us) | PPU cycles |
|---|---:|---|---:|---:|---:|
| `flash_attn_prefill_rmsnorm` | `(3823,3072)` | not tested by nsys | not tested | not tested | not tested |
| `w4a16_prefill_full_attn_q_proj_gate_cutlass55` | `(3823,16384,3072)` | nsys | 694.155 | 599.523 | 899284 |
| `w4a16_prefill_full_attn_k_proj_cutlass55` | `(3823,512,3072)` | nsys | 37.727 | 42.915 | 64373 |
| `w4a16_prefill_full_attn_v_proj_cutlass55` | `(3823,512,3072)` | nsys | 37.343 | 43.018 | 64527 |
| `flash_attn_prefill_q_norm` | `(3823*32,256)` | not tested by nsys | not tested | not tested | not tested |
| `flash_attn_prefill_k_norm` | `(3823*2,256)` | not tested by nsys | not tested | not tested | not tested |
| `flash_attn_prefill_full_attn` | `Q:(3823,32,256), KV:(3823,2,256)` | nsys / perfstatistics | 1061.617 | 768.187 | 1152280 |
| `w4a16_prefill_full_attn_o_proj_cutlass55` | `(3823,3072,8192)` | nsys | 389.781 | 327.592 | 491388 |
| `flash_attn_prefill_residual_add` | `(3823,3072)` | CUDA event single-run | 64.1 | not tested | not tested |

Interpretation:

- Decode FlashAttention is slower on PPU (`46.4us` vs `24.7us`) because single-token attention is dominated by KV scan, reduction, and latency-sensitive scalar work.
- Prefill FlashAttention is faster on PPU (`768.2us` vs `1061.6us`) because large prefill attention has more tensor-core work and HBM traffic.
- Small `k_proj/v_proj` prefill GEMMs have weak PPU advantage because `N=512` limits saturation.

## Linear-Attn

Build and run:

```bash
./compile.sh build general linear_attn flashinfer-gdn w4a16-machete w4a16-fpa
./bench_all.sh --case linear_attn
./bench_all.sh --case linear_decode
./bench_all.sh --case linear_prefill
./bench_all.sh --case w4a16_prefill_linear_attn,w4a16_decode_linear_attn
```

### Decode

| Case | Shape | Method | H800 (us) | PPU (us) | PPU cycles |
|---|---:|---|---:|---:|---:|
| `linear_attn_decode_rmsnorm` | `(1,3072)` | not tested by nsys | not tested | not tested | not tested |
| `linear_attn_decode_in_proj_a_cublas` | `(1,64,3072)` | not tested by nsys | not tested | not tested | not tested |
| `linear_attn_decode_in_proj_b_cublas` | `(1,64,3072)` | not tested by nsys | not tested | not tested | not tested |
| `w4a16_decode_linear_attn_in_proj_qkv_fpA_intB` | `(1,12288,3072)` | nsys | 9.376 | 15.254 | 22881 |
| `w4a16_decode_linear_attn_in_proj_z_fpA_intB` | `(1,8192,3072)` | nsys | 6.976 | 9.091 | 13637 |
| `linear_decode_conv1d_update` | `(1,12288)` | nsys | 2.528 | 2.948 | 4422 |
| `linear_decode_gdn` | `Q,K,V:(1,64,128)` | nsys | 4.480 | 4.661 | 6991 |
| `linear_attn_decode_fused_rms_norm_gate` | `(64,128)` | CUDA event single-run | 185.6 | not tested | not tested |
| `w4a16_decode_linear_attn_out_proj_fpA_intB` | `(1,3072,8192)` | nsys | 8.672 | 14.235 | 21352 |
| `linear_attn_decode_residual_add` | `(1,3072)` | CUDA event single-run | 17.4 | not tested | not tested |

### Prefill

| Case | Shape | Method | H800 (us) | PPU (us) | PPU cycles |
|---|---:|---|---:|---:|---:|
| `linear_attn_prefill_rmsnorm` | `(3823,3072)` | not tested by nsys | not tested | not tested | not tested |
| `linear_attn_prefill_in_proj_a_cublas` | `(3823,64,3072)` | not tested by nsys | not tested | not tested | not tested |
| `linear_attn_prefill_in_proj_b_cublas` | `(3823,64,3072)` | not tested by nsys | not tested | not tested | not tested |
| `w4a16_prefill_linear_attn_in_proj_qkv_cutlass55` | `(3823,12288,3072)` | nsys | 528.311 | 440.939 | 661408 |
| `w4a16_prefill_linear_attn_in_proj_z_cutlass55` | `(3823,8192,3072)` | nsys | 365.882 | 321.688 | 482532 |
| `linear_prefill_conv1d_fwd` | `(3823,12288)` | nsys | 127.208 | 118.904 | 178356 |
| `linear_prefill_flashinfer_gdn` | `tokens=3823, q=16, v=64, d=128` | nsys | 520.926 | 1212.643 | 1818965 |
| `linear_attn_prefill_fused_rms_norm_gate` | `(3823*64,128)` | CUDA event single-run | 752.1 | not tested | not tested |
| `w4a16_prefill_linear_attn_out_proj_cutlass55` | `(3823,3072,8192)` | nsys | 391.321 | 327.592 | 491388 |
| `linear_attn_prefill_residual_add` | `(3823,3072)` | CUDA event single-run | 64.8 | not tested | not tested |

Interpretation:

- Dense W4A16 prefill GEMMs are faster on PPU because large-M GEMMs benefit from higher bandwidth and enough tensor-core work.
- Decode W4A16 GEMMs are slower on PPU because `M=1` behaves closer to GEMV and exposes scalar/dequant/address overhead.
- `linear_prefill_flashinfer_gdn` is an issue: PPU is `2.33x` slower. Earlier metrics pointed to sleep, memory-dependency, and sync stalls, consistent with a persistent/pipeline-heavy kernel.

## MoE-FFN

Build and run:

```bash
./compile.sh build moe-ffn moe-trtllm moe-trtllm-auxiliary moe-vllm w4a16-machete w4a16-fpa
./bench_all.sh --case moe
./bench_all.sh --case prefill_trtllm
./bench_all.sh --case decode_vllm
./bench_all.sh --case consistent_expert
```

### Decode

| Case | Shape | Method | H800 (us) | PPU (us) | PPU cycles |
|---|---:|---|---:|---:|---:|
| `moe_ffn_decode_rmsnorm` | `(1,3072)` | not tested by nsys | not tested | not tested | not tested |
| `moe_router_gate_decode_cublas` | `(1,256,3072)` | not tested by nsys | not tested | not tested | not tested |
| `moe_routing_decode_vllm` | `(1,256)->topk` | nsys | 4.352 | 5.867 | 8801 |
| `moe_align_decode_vllm` | topk/expert metadata | nsys | 10.017 | 10.299 | 15449 |
| `moe_gate_up_decode_vllm` | routed `(8,1,2048,3072)` | nsys | 21.185 | 31.368 | 47052 |
| `moe_gated_decode_vllm` | `(8,1,2048)->(8,1,1024)` | nsys | 2.720 | 2.943 | 4414 |
| `moe_down_decode_vllm` | routed `(8,1,1024,3072)` | nsys | 14.561 | 20.510 | 30765 |
| `moe_finalize_decode_vllm` | `(8,1,3072)->(1,3072)` | nsys | 1.696 | 1.772 | 2658 |
| `w4a16_decode_consistent_expert_up_fpA_intB` | `(1,3072,2048)` | nsys | 3.616 | 5.028 | 7542 |
| `w4a16_decode_consistent_expert_down_fpA_intB` | `(1,1024,3072)` | nsys | 4.032 | 5.349 | 8023 |
| `moe_shared_expert_gate_decode_cublas` | `(1,1,3072)` | not tested by nsys | not tested | not tested | not tested |
| `moe_shared_expert_fusion_decode` | `(1,3072)` | CUDA event single-run | 20.3 | not tested | not tested |
| `moe_ffn_decode_residual_add` | `(1,3072)` | CUDA event single-run | 17.0 | not tested | not tested |

### Prefill

| Case | Shape | Method | H800 (us) | PPU (us) | PPU cycles |
|---|---:|---|---:|---:|---:|
| `moe_ffn_prefill_rmsnorm` | `(3823,3072)` | not tested by nsys | not tested | not tested | not tested |
| `moe_router_gate_prefill_cublas` | `(3823,256,3072)` | not tested by nsys | not tested | not tested | not tested |
| `moe_routing_prefill_trtllm` | `(3823,256)->topk` | nsys | 5.824 | 3.443 | 5164 |
| `moe_expert_map_prefill_trtllm` | topk/expert metadata | nsys, 3 kernels | 10.657 | 10.815 | 16222 |
| `moe_expand_prefill_trtllm` | `(3823,3072)->(3823*8,3072)` | nsys | 284.848 | 188.691 | 283036 |
| `moe_gate_up_prefill_trtllm` | per expert `(3823,3072)->(3823,2048)` | nsys | 1314.123 | 963.023 | 1444534 |
| `moe_gated_prefill_trtllm` | `(3823*8,2048)->(3823*8,1024)` | nsys | 390.486 | 523.552 | 785328 |
| `moe_down_prefill_trtllm` | per expert `(3823,1024)->(3823,3072)` | nsys | 675.654 | 491.889 | 737833 |
| `moe_finalize_prefill_trtllm` | `(3823*8,3072)->(3823,3072)` | nsys | 98.534 | 172.975 | 259463 |
| `w4a16_prefill_consistent_expert_up_cutlass55` | `(3823,3072,2048)` | nsys | 111.613 | 113.129 | 169694 |
| `w4a16_prefill_consistent_expert_down_cutlass55` | `(3823,1024,3072)` | nsys | 69.726 | 57.661 | 86491 |
| `moe_shared_expert_gate_prefill_cublas` | `(3823,1,3072)` | not tested by nsys | not tested | not tested | not tested |
| `moe_shared_expert_fusion_prefill` | `(3823,3072)` | CUDA event single-run | 89.5 | not tested | not tested |
| `moe_ffn_prefill_residual_add` | `(3823,3072)` | CUDA event single-run | 65.2 | not tested | not tested |

Interpretation:

- MoE prefill grouped W4A16 GEMMs are faster on PPU (`~0.73x` PPU/H800).
- Decode MoE kernels are generally slower on PPU because `M=1`, topk is small, and scalar/latency overhead dominates.
- `moe_finalize_prefill_trtllm` is an issue: PPU is `1.76x` slower, likely due to irregular finalize routing and synchronization behavior.
- The isolated finalize study is under `studies/ppu_finalize_moe_routing/` and is intentionally not included in `bench_all.sh`.

## Sampling

Build and run:

```bash
./compile.sh build general sampling
./bench_all.sh --case sampling
```

Sampling has H800 CUDA-event smoke results only. It has not been re-run under
nsys, and PPU has not been tested.

| Case | Shape | Method | H800 (us) | PPU (us) | PPU cycles |
|---|---:|---|---:|---:|---:|
| `sampling_lm_head_gemm` | `(1,248320,3072)` | CUDA event single-run | 52401.0 | not tested | not tested |
| `sampling_topk_mask_logits` | `(1,248320)` | CUDA event single-run | 105.1 | not tested | not tested |
| `sampling_softmax` | `(1,248320)` | CUDA event single-run | 123.3 | not tested | not tested |
| `sampling_top_p` | `(1,248320)->(1)` | CUDA event single-run | 71.0 | not tested | not tested |

Sampling caveat:

- The `lm_head` row is cuBLAS and the current value is a single CUDA-event smoke
  number. Use nsys or ncu before comparing it with other modules.
- PPU sampling data is `not tested`.

## Cross-Module Summary

| Group | H800 subtotal (us) | PPU subtotal (us) | Notes |
|---|---:|---:|---|
| Linear-Attn decode in-repo core | 7.008 | 7.609 | conv1d update + GDN only |
| Linear-Attn prefill in-repo core | 648.134 | 1331.547 | conv1d fwd + FlashInfer GDN |
| MoE-FFN decode vLLM path | 54.531 | 72.759 | routing + align + Marlin + activation + sum |
| MoE-FFN prefill TRT-LLM path | 2780.126 | 2354.388 | routing + expert map + expand + gate/up + activation + down + finalize |

Hardware interpretation:

- PPU has similar FP16 tensor throughput to H800 but lower scalar/CUDA-core
  throughput and higher HBM bandwidth.
- Large prefill GEMM/attention cases tend to favor PPU.
- Decode, reductions, elementwise, small-N GEMM, and persistent/sync-heavy
  kernels tend to be neutral or worse on PPU.
