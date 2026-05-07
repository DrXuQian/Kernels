# Qwen3.5-122B-A10B Standalone Kernel Benchmark

H800 PCIe (SM 9.0, HBM 2.0 TB/s)

## H800 `bench_all.sh` nsys reference

Run: `h800_nsys_all_20260429_084402`

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

The table below is from `cuda_gpu_trace` and sums only CUDA kernel rows. The whole-process nsys report also contains setup H2D/memset rows, which are intentionally excluded here. All benchmark commands used `warmup=0` and `iters=1`; every case maps to one benchmark kernel except `moe_expert_map_prefill_trtllm`, which is the TRT-LLM three-kernel prefix-sum path. PPU rows are from `perfstatistics` reports; `PPU latency` is converted with a 1.5 GHz clock (`compute_cycles / 1500`).

| Case | Impl | Phase | Kernel(s) | nsys kernels | H800 GPU time (us) | H800 cycles @1.5GHz | PPU latency (us) | PPU cycles |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `linear_decode_conv1d_update` | linear | decode | conv1d_update | 1 | 2.528 | 3792 | 2.948 | 4422 |
| `linear_decode_gdn` | linear | decode | gated_delta_net | 1 | 4.480 | 6720 | 4.661 | 6991 |
| `linear_prefill_conv1d_fwd` | linear | prefill | conv1d_fwd | 1 | 127.208 | 190812 | 118.904 | 178356 |
| `linear_prefill_flashinfer_gdn` | linear | prefill | flashinfer_gdn | 1 | 520.926 | 781389 | 1212.643 | 1818965 |
| `moe_routing_prefill_trtllm` | moe/trtllm | prefill | custom_moe_routing | 1 | 5.824 | 8736 | 3.443 | 5164 |
| `moe_expert_map_prefill_trtllm` | moe/trtllm | prefill | block/global/merge expert prefix sum | 3 | 10.657 | 15986 | 10.815 | 16222 |
| `moe_expand_prefill_trtllm` | moe/trtllm | prefill | expand_input_rows | 1 | 284.848 | 427272 | 188.691 | 283036 |
| `moe_gate_up_prefill_trtllm` | moe/trtllm | prefill | MoeFCGemm gate_up | 1 | 1314.123 | 1971184 | 963.023 | 1444534 |
| `moe_gated_prefill_trtllm` | moe/trtllm | prefill | gated_activation | 1 | 390.486 | 585729 | 523.552 | 785328 |
| `moe_down_prefill_trtllm` | moe/trtllm | prefill | MoeFCGemm down | 1 | 675.654 | 1013481 | 491.889 | 737833 |
| `moe_finalize_prefill_trtllm` | moe/trtllm | prefill | finalize_moe_routing | 1 | 98.534 | 147801 | 172.975 | 259463 |
| `moe_routing_decode_vllm` | moe/vllm | decode | topk_gating | 1 | 4.352 | 6528 | 5.867 | 8801 |
| `moe_align_decode_vllm` | moe/vllm | decode | moe_align small batch | 1 | 10.017 | 15026 | 10.299 | 15449 |
| `moe_gate_up_decode_vllm` | moe/vllm | decode | Marlin MoE gate_up | 1 | 21.185 | 31777 | 31.368 | 47052 |
| `moe_gated_decode_vllm` | moe/vllm | decode | silu_and_mul | 1 | 2.720 | 4080 | 2.943 | 4414 |
| `moe_down_decode_vllm` | moe/vllm | decode | Marlin MoE down | 1 | 14.561 | 21842 | 20.510 | 30765 |
| `moe_finalize_decode_vllm` | moe/vllm | decode | moe_sum | 1 | 1.696 | 2544 | 1.772 | 2658 |

Subtotals from these rows:

| Group | Included rows | H800 GPU time (us) | PPU latency (us) |
|---|---|---:|---:|
| Linear attention decode in-repo | conv1d_update + GDN | 7.008 | 7.609 |
| Linear attention prefill in-repo | conv1d_fwd + FlashInfer GDN | 648.134 | 1331.547 |
| MoE prefill | TRT-LLM routing + expert map + expand + gate_up + gated + down + finalize | 2780.126 | 2354.388 |
| MoE decode | vLLM routing + align + gate_up + gated + down + finalize | 54.531 | 72.759 |

### Dense W4A16 GEMM 主表

这些 case 都由根目录 `bench_all.sh` 直接调度。Prefill dense GEMM 默认走
`test_machete_gemm --backend=cutlass55 --offline_prepack --profile_gemm_only`，
decode dense GEMM 默认走 TensorRT-LLM `fpA_intB` standalone。所有 shape 均为
`(M,N,K)`，group size 为 128。

| Case | Stage | Logical op | Shape (M,N,K) | Backend | Cached config | H800 nsys GPU time (us) | PPU latency (us) | PPU cycles |
|---|---|---|---:|---|---|---:|---:|---:|
| `w4a16_prefill_linear_attn_in_proj_qkv_cutlass55` | prefill | linear-attn in_proj QKV | 3823,12288,3072 | machete cutlass55 | `128x256x64_2x1x1` | 528.311 | 440.939 | 661408 |
| `w4a16_prefill_linear_attn_in_proj_z_cutlass55` | prefill | linear-attn in_proj z | 3823,8192,3072 | machete cutlass55 | `128x256x64_2x1x1` | 365.882 | 321.688 | 482532 |
| `w4a16_prefill_linear_attn_out_proj_cutlass55` | prefill | linear-attn out_proj | 3823,3072,8192 | machete cutlass55 | `256x128x64_1x1x1` | 391.321 | 327.592 | 491388 |
| `w4a16_decode_linear_attn_in_proj_qkv_fpA_intB` | decode | linear-attn in_proj QKV | 1,12288,3072 | fpA_intB | `cuda` | 9.376 | 15.254 | 22881 |
| `w4a16_decode_linear_attn_in_proj_z_fpA_intB` | decode | linear-attn in_proj z | 1,8192,3072 | fpA_intB | `cuda` | 6.976 | 9.091 | 13637 |
| `w4a16_decode_linear_attn_out_proj_fpA_intB` | decode | linear-attn out_proj | 1,3072,8192 | fpA_intB | `cuda` | 8.672 | 14.235 | 21352 |
| `w4a16_prefill_full_attn_q_proj_gate_cutlass55` | prefill | full-attn q_proj + gate | 3823,16384,3072 | machete cutlass55 | `128x256x64_2x1x1` | 694.155 | 599.523 | 899284 |
| `w4a16_prefill_full_attn_k_proj_cutlass55` | prefill | full-attn k_proj | 3823,512,3072 | machete cutlass55 | `128x256x64_2x1x1` | 37.727 | 42.915 | 64373 |
| `w4a16_prefill_full_attn_v_proj_cutlass55` | prefill | full-attn v_proj | 3823,512,3072 | machete cutlass55 | `128x256x64_2x1x1` | 37.343 | 43.018 | 64527 |
| `w4a16_prefill_full_attn_o_proj_cutlass55` | prefill | full-attn o_proj | 3823,3072,8192 | machete cutlass55 | `256x128x64_1x1x1` | 389.781 | 327.592 | 491388 |
| `w4a16_decode_full_attn_q_proj_gate_fpA_intB` | decode | full-attn q_proj + gate | 1,16384,3072 | fpA_intB | `cuda` | 12.960 | 15.875 | 23813 |
| `w4a16_decode_full_attn_k_proj_fpA_intB` | decode | full-attn k_proj | 1,512,3072 | fpA_intB | `cuda` | 3.936 | 5.347 | 8020 |
| `w4a16_decode_full_attn_v_proj_fpA_intB` | decode | full-attn v_proj | 1,512,3072 | fpA_intB | `cuda` | 3.935 | 5.285 | 7927 |
| `w4a16_decode_full_attn_o_proj_fpA_intB` | decode | full-attn o_proj | 1,3072,8192 | fpA_intB | `cuda` | 8.672 | 14.235 | 21352 |
| `w4a16_prefill_consistent_expert_up_cutlass55` | prefill | consistent expert gate_up | 3823,3072,2048 | machete cutlass55 | `128x128x64_1x1x1` | 111.613 | 113.129 | 169694 |
| `w4a16_prefill_consistent_expert_down_cutlass55` | prefill | consistent expert down | 3823,1024,3072 | machete cutlass55 | `128x128x64_1x1x1` | 69.726 | 57.661 | 86491 |
| `w4a16_decode_consistent_expert_up_fpA_intB` | decode | consistent expert gate_up | 1,3072,2048 | fpA_intB | `cuda` | 3.616 | 5.028 | 7542 |
| `w4a16_decode_consistent_expert_down_fpA_intB` | decode | consistent expert down | 1,1024,3072 | fpA_intB | `cuda` | 4.032 | 5.349 | 8023 |

Additional dense W4A16 projection cases were added after the full `bench_all.sh` run above. They are guarded by tactic-cache checks, so a missing shape fails early instead of falling back to a default config.

Run: `h800_nsys_w4a16_dense_20260429_091047`

| Case | Shape (M,N,K) | Backend | Cached config | nsys GPU time (us) |
|---|---:|---|---|---:|
| `w4a16_prefill_linear_attn_in_proj_qkv_cutlass55` | 3823,12288,3072 | machete cutlass55 | `128x256x64_2x1x1` | 528.311 |
| `w4a16_prefill_linear_attn_in_proj_z_cutlass55` | 3823,8192,3072 | machete cutlass55 | `128x256x64_2x1x1` | 365.882 |
| `w4a16_prefill_linear_attn_out_proj_cutlass55` | 3823,3072,8192 | machete cutlass55 | `256x128x64_1x1x1` | 391.321 |
| `w4a16_decode_linear_attn_in_proj_qkv_fpA_intB` | 1,12288,3072 | fpA_intB | `cuda` | 9.376 |
| `w4a16_decode_linear_attn_in_proj_z_fpA_intB` | 1,8192,3072 | fpA_intB | `cuda` | 6.976 |
| `w4a16_decode_linear_attn_out_proj_fpA_intB` | 1,3072,8192 | fpA_intB | `cuda` | 8.672 |

Run: `h800_nsys_consistent_expert_20260429_094031`

| Case | Shape (M,N,K) | Backend | Cached config | nsys GPU time (us) |
|---|---:|---|---|---:|
| `w4a16_prefill_consistent_expert_up_cutlass55` | 3823,3072,2048 | machete cutlass55 | `128x128x64_1x1x1` | 111.613 |
| `w4a16_prefill_consistent_expert_down_cutlass55` | 3823,1024,3072 | machete cutlass55 | `128x128x64_1x1x1` | 69.726 |
| `w4a16_decode_consistent_expert_up_fpA_intB` | 1,3072,2048 | fpA_intB | `cuda` | 3.616 |
| `w4a16_decode_consistent_expert_down_fpA_intB` | 1,1024,3072 | fpA_intB | `cuda` | 4.032 |

Run: `h800_nsys_full_attn_proj_20260429_094647`

| Case | Shape (M,N,K) | Backend | Cached config | nsys GPU time (us) |
|---|---:|---|---|---:|
| `w4a16_prefill_full_attn_q_proj_gate_cutlass55` | 3823,16384,3072 | machete cutlass55 | `128x256x64_2x1x1` | 694.155 |
| `w4a16_prefill_full_attn_k_proj_cutlass55` | 3823,512,3072 | machete cutlass55 | `128x256x64_2x1x1` | 37.727 |
| `w4a16_prefill_full_attn_v_proj_cutlass55` | 3823,512,3072 | machete cutlass55 | `128x256x64_2x1x1` | 37.343 |
| `w4a16_prefill_full_attn_o_proj_cutlass55` | 3823,3072,8192 | machete cutlass55 | `256x128x64_1x1x1` | 389.781 |
| `w4a16_decode_full_attn_q_proj_gate_fpA_intB` | 1,16384,3072 | fpA_intB | `cuda` | 12.960 |
| `w4a16_decode_full_attn_k_proj_fpA_intB` | 1,512,3072 | fpA_intB | `cuda` | 3.936 |
| `w4a16_decode_full_attn_v_proj_fpA_intB` | 1,512,3072 | fpA_intB | `cuda` | 3.935 |
| `w4a16_decode_full_attn_o_proj_fpA_intB` | 1,3072,8192 | fpA_intB | `cuda` | 8.672 |

## 模型参数

| 参数 | 值 |
|------|-----|
| hidden_size | 3072 |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 64 |
| linear_key/value_head_dim | 128 |
| linear_conv_kernel_dim | 4 |
| conv_dim (Q+K+V) | 12288 (2048+2048+8192) |
| sab_heads | 64 |
| num_attention_heads | 32 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 1024 |
| shared_expert_intermediate_size | 1024 |
| layers | 48 (36 linear_attn + 12 full_attention) |

## DeltaNet Linear Attention (×36 layers)

### Decode (batch=1, single token)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | RMSNorm | (1,3072)→(1,3072) | — | `./bench_all.sh --case flash_attn_decode_rmsnorm` |
| 2 | in_proj_qkv (W4A16) | (1,3072)→(1,12288) | **9.4 μs** | `./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB` |
| 3 | in_proj_z (W4A16) | (1,3072)→(1,8192) | **7.0 μs** | `./bench_all.sh --case w4a16_decode_linear_attn_in_proj_z_fpA_intB` |
| 4 | in_proj_b (FP16) | (1,3072)→(1,64) | — | not in repo (cuBLAS GEMV) |
| 5 | in_proj_a (FP16) | (1,3072)→(1,64) | — | not in repo (cuBLAS GEMV) |
| 6 | **conv1d decode** | dim=12288, w=4 | **5.4 μs** | `./bench_conv1d_update 12288 4 1 --bench 20 100` |
| 7 | **GDN decode** (llama.cpp CUDA) | Q,K,V:(1,64,128) state:(64,128,128) | **5.4 μs** | `./bench_gated_delta_net 1 64 128 1 --bench 20 100` |
| 8 | **FusedRMSNormGated** | (64,128)→(64,128) | **5.3 μs** | `./bench_fused_rms_norm_gate 64 128 --bench 20 100` |
| 9 | out_proj (W4A16) | (1,8192)→(1,3072) | **8.7 μs** | `./bench_all.sh --case w4a16_decode_linear_attn_out_proj_fpA_intB` |

### Prefill (seq=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | in_proj_qkv (W4A16) | (3823,3072)→(3823,12288) | **528.3 μs** | `./bench_all.sh --case w4a16_prefill_linear_attn_in_proj_qkv_cutlass55` |
| 2 | in_proj_z (W4A16) | (3823,3072)→(3823,8192) | **365.9 μs** | `./bench_all.sh --case w4a16_prefill_linear_attn_in_proj_z_cutlass55` |
| 3 | **conv1d prefill** | (1,12288,3823) | **134.0 μs** | `./bench_conv1d_fwd 3823 12288 4 1 --bench 10 50` |
| 4 | **FlashInfer GDN prefill** (CUTLASS SM90) | Q:(3823,16,128) K:(3823,16,128) V:(3823,64,128) | **525.9 μs** | `./bench_gdn_prefill 3823 16 64 128 1 --bench 10 50` |
| 5 | **FusedRMSNormGated** | (3823×64,128)→(3823×64,128) | — | `./bench_fused_rms_norm_gate 245472 128 --bench 10 50` |
| 6 | out_proj (W4A16) | (3823,8192)→(3823,3072) | **391.3 μs** | `./bench_all.sh --case w4a16_prefill_linear_attn_out_proj_cutlass55` |

## MoE FFN (×48 layers, 256 experts, topk=8)

### Decode (batch=1, single token)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | RMSNorm | (1,3072)→(1,3072) | — | not in repo (generic) |
| 2 | Router gate (FP16) | (1,3072)→(1,256) | — | not in repo (cuBLAS GEMV) |
| 3 | **topk_gating** | (1,256)→w:(1,8) idx:(1,8) | **8.4 μs** | `moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating 1 256 8 --bench 20 100` |
| 4 | **moe_align** | (1,8)→sorted_ids, expert_ids | **3.3 μs** | `moe_ffn/w4a16/vllm/auxiliary/bench_moe_align 1 256 8 16 --bench 20 100` |
| 5 | **Marlin MoE gate_up** (W4A16) | (8,1,3072)→(8,1,2048) | **11.5 μs** | `moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 1 256 8 3072 1024 --bench 20 100` |
| 6 | **silu_and_mul** | (8,1,2048)→(8,1,1024) | **5.0 μs** | `moe_ffn/w4a16/vllm/auxiliary/bench_silu_and_mul 1 8 1024 --bench 20 100` |
| 7 | **Marlin MoE down** (W4A16) | (8,1,1024)→(8,1,3072) | **12.3 μs** | `moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 1 256 8 1024 3072 --bench 20 100` |
| 8 | **moe_sum** | (8,1,3072)→(1,3072) | **5.4 μs** | `moe_ffn/w4a16/vllm/auxiliary/bench_moe_sum 1 8 3072 --bench 20 100` |
| 9 | Shared/consistent gate_up (W4A16) | (1,2048)→(1,3072) | **3.6 μs** | `./bench_all.sh --case w4a16_decode_consistent_expert_up_fpA_intB` |
| 10 | Shared SwiGLU | (1,2048)→(1,1024) | — | not in repo |
| 11 | Shared/consistent down (W4A16) | (1,3072)→(1,1024) | **4.0 μs** | `./bench_all.sh --case w4a16_decode_consistent_expert_down_fpA_intB` |

### Prefill (seq=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 3 | **topk_gating** | (3823,256)→... | — | `moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating 3823 256 8 --bench 10 50` |
| 4 | **moe_align** | (3823,8)→... | — | `moe_ffn/w4a16/vllm/auxiliary/bench_moe_align 3823 256 8 16 --bench 10 50` |
| 5 | **Marlin MoE gate_up** (W4A16) | (8,3823,3072)→(8,3823,2048) | — | `moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 3823 256 8 3072 1024 --bench 10 50` |
| 6 | **silu_and_mul** | (8,3823,2048)→(8,3823,1024) | — | `moe_ffn/w4a16/vllm/auxiliary/bench_silu_and_mul 3823 8 1024 --bench 10 50` |
| 7 | **Marlin MoE down** (W4A16) | (8,3823,1024)→(8,3823,3072) | — | `moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 3823 256 8 1024 3072 --bench 10 50` |
| 8 | **moe_sum** | (8,3823,3072)→(3823,3072) | — | `moe_ffn/w4a16/vllm/auxiliary/bench_moe_sum 3823 8 3072 --bench 10 50` |
| 9 | Shared/consistent gate_up (W4A16) | (3823,2048)→(3823,3072) | **111.6 μs** | `./bench_all.sh --case w4a16_prefill_consistent_expert_up_cutlass55` |
| 10 | Shared SwiGLU | (3823,2048)→(3823,1024) | — | not in repo |
| 11 | Shared/consistent down (W4A16) | (3823,3072)→(3823,1024) | **69.7 μs** | `./bench_all.sh --case w4a16_prefill_consistent_expert_down_cutlass55` |

## Full Attention (×12 layers)

### Decode (batch=1, ctx=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | RMSNorm | (1,3072)→(1,3072) | — | not in repo (generic) |
| 2 | q_proj + gate (W4A16) | (1,3072)→(1,16384) | **13.0 μs** | `./bench_all.sh --case w4a16_decode_full_attn_q_proj_gate_fpA_intB` |
| 3 | k_proj (W4A16) | (1,3072)→(1,512) | **3.9 μs** | `./bench_all.sh --case w4a16_decode_full_attn_k_proj_fpA_intB` |
| 4 | v_proj (W4A16) | (1,3072)→(1,512) | **3.9 μs** | `./bench_all.sh --case w4a16_decode_full_attn_v_proj_fpA_intB` |
| 5 | q/k RMSNorm | Q:(32,256), K:(2,256) | — | `./bench_all.sh --case flash_attn_decode_q_norm,flash_attn_decode_k_norm` |
| 6 | MRoPE | Q:(1,32,256) K:(1,2,256) | — | not in repo (vLLM has Triton; no standalone CUDA / local TRT-LLM CUDA source found) |
| 7 | **FlashAttention v3 decode** | Q:(1,32,256) KV:(ctx,2,256) | **H800 24.7 μs / PPU 46.4 μs** | `./bench_all.sh --case flash_attn_decode_full_attn` |
| 8 | output gate | attn,gate:(1,8192)→(1,8192) | — | not in repo (vLLM/TRT-LLM use Torch elementwise; no standalone CUDA source found) |
| 9 | o_proj (W4A16) | (1,8192)→(1,3072) | **8.7 μs** | `./bench_all.sh --case w4a16_decode_full_attn_o_proj_fpA_intB` |
| + | MoE FFN (same as above) | | | |

### Prefill (seq=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | RMSNorm | (3823,3072)→(3823,3072) | — | `./bench_all.sh --case flash_attn_prefill_rmsnorm` |
| 2 | q_proj + gate (W4A16) | (3823,3072)→(3823,16384) | **694.2 μs** | `./bench_all.sh --case w4a16_prefill_full_attn_q_proj_gate_cutlass55` |
| 3 | k_proj (W4A16) | (3823,3072)→(3823,512) | **37.7 μs** | `./bench_all.sh --case w4a16_prefill_full_attn_k_proj_cutlass55` |
| 4 | v_proj (W4A16) | (3823,3072)→(3823,512) | **37.3 μs** | `./bench_all.sh --case w4a16_prefill_full_attn_v_proj_cutlass55` |
| 5 | q/k RMSNorm | Q:(3823×32,256), K:(3823×2,256) | — | `./bench_all.sh --case flash_attn_prefill_q_norm,flash_attn_prefill_k_norm` |
| 6 | MRoPE | Q:(3823,32,256) K:(3823,2,256) | — | not in repo (vLLM has Triton; no standalone CUDA / local TRT-LLM CUDA source found) |
| 7 | **FlashAttention v3 prefill** | Q:(3823,32,256) KV:(3823,2,256) | **H800 1061.6 μs / PPU 768.2 μs** | `./bench_all.sh --case flash_attn_prefill_full_attn` |
| 8 | output gate | attn,gate:(3823,8192)→(3823,8192) | — | not in repo (vLLM/TRT-LLM use Torch elementwise; no standalone CUDA source found) |
| 9 | o_proj (W4A16) | (3823,8192)→(3823,3072) | **389.8 μs** | `./bench_all.sh --case w4a16_prefill_full_attn_o_proj_cutlass55` |

> FlashAttention bench 使用 `flash_attn` 库（需 pip install flash-attn）。H800 数字来自 nsys `cuda_gpu_trace`，只汇总 captured range 内 CUDA kernel rows；PPU 数字来自 `perfstatistics`，按 1.5 GHz 换算。

### Attention Python/JIT inference nsys 对比

H800 run: `attention_nsys_20260430_165220`, command pattern:
`nsys profile --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop ... ./bench_attention_inference.sh --exact-case <case>` with `BENCH_WARMUP=1 BENCH_ITERS=1`. H800 latency is the sum of captured CUDA kernel rows.

| Case | Shape | H800 nsys kernels | H800 nsys latency (us) | PPU latency (us) | PPU cycles | PPU/H800 |
|---|---|---:|---:|---:|---:|---:|
| `linear_triton_decode_gdn` | B=1, q=16, v=64, d=128 | 1 | 6.560 | - | - | - |
| `linear_triton_prefill_gdn` | seq=3823, q=16, v=64, d=128 | 8 | 938.595 | - | - | - |
| `linear_triton_prefill_gdn_core_only` | seq=3823, q=16, v=64, d=128 | 7 | 831.561 | - | - | - |
| `flashinfer_decode_full_attn` | Q:(1,32,256), KV:(3823,2,256) | 2 | 12.577 | - | - | - |
| `flashinfer_prefill_full_attn` | Q:(3823,32,256), KV:(3823,2,256) | 1 | 572.283 | - | - | - |
| `flash_attn_decode_full_attn` | Q:(1,32,256), KV:(3823,2,256) | 2 | 24.738 | 46.405 | 69608 | 1.88x |
| `flash_attn_prefill_full_attn` | Q:(3823,32,256), KV:(3823,2,256) | 1 | 1061.617 | 768.187 | 1152280 | 0.72x |

### PPU/H800 对比解读

硬件前提：PPU 的 FP16 tensor 算力按 H800 同级理解，但 CUDA core / scalar
计算吞吐约为 H800 的一半；PPU HBM 带宽约 6 TB/s，高于 H800 PCIe 的约
2 TB/s。因此，large prefill GEMM / attention 这类 tensor core + HBM 占比高的
case 应该更容易在 PPU 上变快；decode、小 M GEMV、softmax/reduction、elementwise、
scatter/gather 这类 CUDA core / latency / synchronization 占比高的 case 则可能在
PPU 上变慢。下表 `PPU/H800 > 1` 表示 PPU 更慢，`< 1` 表示 PPU 更快。

| Group | Cases | PPU/H800 | Interpretation |
|---|---|---:|---|
| Linear decode CUDA kernels | `linear_decode_conv1d_update`, `linear_decode_gdn` | 1.17x, 1.04x | 符合预期。decode 是小 batch / latency-sensitive，CUDA core 和调度开销占比高，PPU 没有带宽优势可发挥。 |
| Linear prefill conv1d | `linear_prefill_conv1d_fwd` | 0.93x | 符合预期。序列长后访存规模更大，PPU 带宽优势能抵消部分 scalar 开销。 |
| Linear prefill FlashInfer GDN | `linear_prefill_flashinfer_gdn` | 2.33x | **[issue]** 不符合“tensor 算力相同 + 带宽更高”的直觉。这个 kernel 更像 persistent / pipeline-heavy kernel，之前 PPU metrics 也看到 sleep、memory dependency、sync stall。需要继续看 block/warp latency balance、pipeline wait、SM/CE occupancy 和具体指令 mix。 |
| MoE prefill routing/expand | `moe_routing_prefill_trtllm`, `moe_expand_prefill_trtllm` | 0.59x, 0.66x | 基本符合预期。这类 prefill routing/expand 更偏大规模 gather/scatter/copy，PPU 带宽高会有收益。 |
| MoE prefill expert map | `moe_expert_map_prefill_trtllm` | 1.01x | 中性结果。该路径是三个 prefix-sum/merge 小 kernel，launch/同步/整数控制开销占比高，硬件带宽差异不明显。 |
| MoE prefill W4A16 GEMM | `moe_gate_up_prefill_trtllm`, `moe_down_prefill_trtllm` | 0.73x, 0.73x | 符合预期。大 M grouped GEMM 有较高 tensor core 和 HBM 占比，PPU 更快但不到 3x，因为地址计算、scale/dequant、调度和非 tensor 指令仍然存在。 |
| MoE prefill activation/finalize | `moe_gated_prefill_trtllm`, `moe_finalize_prefill_trtllm` | 1.34x, 1.76x | `gated_activation` 偏 elementwise/SFU/CUDA core，PPU 变慢可以解释；`finalize_moe_routing` 更慢较多，属于 **[issue]**，可能是 irregular scatter/reduction/sync 对 PPU 不友好，需要 metrics 确认。 |
| MoE decode vLLM pipeline | routing, align, Marlin gate/up/down, silu, sum | 1.03x-1.48x | 符合预期。decode M=1/topk=8，Marlin MoE 和 auxiliary kernels 都很小，tensor core 利用率有限，CUDA core / launch / memory latency 占比高，所以 PPU 普遍慢一些。 |
| Dense W4A16 prefill, large N/K | linear qkv/z/out, full q/o, consistent down | 0.83x-0.88x | 符合预期。prefill GEMM 规模足够大，PPU tensor 算力不吃亏且带宽更高，因此整体更快。 |
| Dense W4A16 prefill, small N | full-attn k/v proj `(N=512)`, consistent expert up | 1.01x-1.15x | **[caveat]** 小 N/较小输出维度降低并行度和 tensor core 饱和度，CUDA core/调度/epilogue 开销占比上升，PPU 带宽优势不明显。 |
| Dense W4A16 decode fpA_intB | all M=1 projection GEMMs | 1.22x-1.64x | 符合预期。M=1 更接近 GEMV，tensor core 很难充分利用，scale/dequant/address/epilogue 和 latency 开销主导，PPU CUDA core 半吞吐会体现为更慢。 |
| FlashAttention decode | `flash_attn_decode_full_attn` | 1.88x | 符合预期。单 token attention 主要是 KV scan、online softmax/reduction 和 split-KV combine，tensor core 占比低；PPU CUDA core 半吞吐和同步/latency 更敏感。 |
| FlashAttention prefill | `flash_attn_prefill_full_attn` | 0.72x | 符合预期。prefill 有大 QK/PV tensor core 计算和较大 HBM 流量，PPU 带宽优势能体现；没有达到 3x 是因为 online softmax、shared memory、同步、mask/bounds 和地址计算仍占不少比例。 |

需要继续确认的重点：

- **[issue] `linear_prefill_flashinfer_gdn`**：PPU 2.33x 慢，和硬件预期相反，应优先看 pipeline/sleep/sync stall、persistent kernel 占用、CE/SM latency balance。
- **[issue] `moe_finalize_prefill_trtllm`**：PPU 1.76x 慢，可能是 finalize 的 irregular scatter/reduction 或同步结构导致，需要 kernel-level metrics。
- **[caveat] 小 N prefill GEMM**：`full_attn_k/v_proj` 等小输出维度 case 不应按大 GEMM 预期解读，PPU 略慢或接近 H800 不一定代表 GEMM 主路径有问题。

## Decode 单层时间估算

### DeltaNet 层 (per layer)

| Kernel | Latency | 占比 |
|--------|---------|------|
| GDN decode (llama.cpp CUDA) | 5.4 μs | 33.5% |
| conv1d decode | 5.4 μs | 33.5% |
| FusedRMSNormGated | 5.3 μs | 32.9% |
| **Subtotal (in-repo)** | **16.1 μs** | 100% |

> GDN decode nsys GPU kernel time: llama.cpp 4.5μs, fla Triton 4.9μs（vLLM 用 fla 版本 6.4μs）。
> 不含 in_proj_qkv/z/out_proj (Marlin GEMV) 和 RMSNorm，这些额外 ~30-50μs。

### MoE FFN 层 (per layer)

| Kernel | Latency | 占比 |
|--------|---------|------|
| Marlin MoE gate_up | 11.5 μs | 25.0% |
| Marlin MoE down | 12.3 μs | 26.7% |
| topk_gating | 8.4 μs | 18.3% |
| silu_and_mul | 5.0 μs | 10.9% |
| moe_sum | 5.4 μs | 11.7% |
| moe_align | 3.3 μs | 7.2% |
| **Subtotal (in-repo)** | **45.9 μs** | 100% |

> 不含 Router gate (cuBLAS GEMV) 和 Shared expert，这些额外 ~20-30μs。

## 命令速查

```bash
cd Kernels

# ── DeltaNet Decode ──
linear_attn/bench_conv1d_update 12288 4 1 --bench 20 100
linear_attn/bench_gated_delta_net 1 64 128 1 --bench 20 100
linear_attn/bench_fused_rms_norm_gate 64 128 --bench 20 100

# ── DeltaNet Prefill (seq=3823) ──
linear_attn/bench_conv1d_fwd 3823 12288 4 1 --bench 10 50
linear_attn/bench_gdn_prefill 3823 16 64 128 1 --bench 10 50

# ── MoE FFN Decode ──
moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating 1 256 8 --bench 20 100
moe_ffn/w4a16/vllm/auxiliary/bench_moe_align 1 256 8 16 --bench 20 100
moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 1 256 8 3072 1024 --bench 20 100
moe_ffn/w4a16/vllm/auxiliary/bench_silu_and_mul 1 8 1024 --bench 20 100
moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 1 256 8 1024 3072 --bench 20 100
moe_ffn/w4a16/vllm/auxiliary/bench_moe_sum 1 8 3072 --bench 20 100

# ── MoE FFN Prefill (seq=3823) ──
moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating 3823 256 8 --bench 10 50
moe_ffn/w4a16/vllm/auxiliary/bench_moe_align 3823 256 8 16 --bench 10 50
moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 3823 256 8 3072 1024 --bench 10 50
moe_ffn/w4a16/vllm/auxiliary/bench_silu_and_mul 3823 8 1024 --bench 10 50
moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 3823 256 8 1024 3072 --bench 10 50
moe_ffn/w4a16/vllm/auxiliary/bench_moe_sum 3823 8 3072 --bench 10 50

# ── Full Attention (FlashAttn, Python) ──
python3 flash_attn/bench_flash_attn.py decode 3823
python3 flash_attn/bench_flash_attn.py prefill 3823
python3 flash_attn/bench_flash_infer.py decode 3823
python3 flash_attn/bench_flash_infer.py prefill 3823

# ── nsys (纯 GPU kernel time) ──
nsys profile --trace=cuda -o trace ./bench_xxx [args]
nsys stats trace.nsys-rep --report cuda_gpu_kern_sum
```
