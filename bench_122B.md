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

The table below is from `cuda_gpu_trace` and sums only CUDA kernel rows. The whole-process nsys report also contains setup H2D/memset rows, which are intentionally excluded here. All benchmark commands used `warmup=0` and `iters=1`; every case maps to one benchmark kernel except `moe_expert_map_prefill_trtllm`, which is the TRT-LLM three-kernel prefix-sum path.

| Case | Impl | Phase | Kernel(s) | nsys kernels | GPU time (us) | cycles @1.5GHz |
|---|---:|---:|---|---:|---:|---:|
| `linear_decode_conv1d_update` | linear | decode | conv1d_update | 1 | 2.528 | 3792 |
| `linear_decode_gdn` | linear | decode | gated_delta_net | 1 | 4.480 | 6720 |
| `linear_prefill_conv1d_fwd` | linear | prefill | conv1d_fwd | 1 | 127.208 | 190812 |
| `linear_prefill_flashinfer_gdn` | linear | prefill | flashinfer_gdn | 1 | 520.926 | 781389 |
| `moe_routing_prefill_trtllm` | moe/trtllm | prefill | custom_moe_routing | 1 | 5.824 | 8736 |
| `moe_expert_map_prefill_trtllm` | moe/trtllm | prefill | block/global/merge expert prefix sum | 3 | 10.657 | 15986 |
| `moe_expand_prefill_trtllm` | moe/trtllm | prefill | expand_input_rows | 1 | 284.848 | 427272 |
| `moe_gate_up_prefill_trtllm` | moe/trtllm | prefill | MoeFCGemm gate_up | 1 | 1314.123 | 1971184 |
| `moe_gated_prefill_trtllm` | moe/trtllm | prefill | gated_activation | 1 | 390.486 | 585729 |
| `moe_down_prefill_trtllm` | moe/trtllm | prefill | MoeFCGemm down | 1 | 675.654 | 1013481 |
| `moe_finalize_prefill_trtllm` | moe/trtllm | prefill | finalize_moe_routing | 1 | 98.534 | 147801 |
| `moe_routing_decode_vllm` | moe/vllm | decode | topk_gating | 1 | 4.352 | 6528 |
| `moe_align_decode_vllm` | moe/vllm | decode | moe_align small batch | 1 | 10.017 | 15026 |
| `moe_gate_up_decode_vllm` | moe/vllm | decode | Marlin MoE gate_up | 1 | 21.185 | 31777 |
| `moe_gated_decode_vllm` | moe/vllm | decode | silu_and_mul | 1 | 2.720 | 4080 |
| `moe_down_decode_vllm` | moe/vllm | decode | Marlin MoE down | 1 | 14.561 | 21842 |
| `moe_finalize_decode_vllm` | moe/vllm | decode | moe_sum | 1 | 1.696 | 2544 |

Subtotals from these rows:

| Group | Included rows | GPU time (us) |
|---|---|---:|
| Linear attention decode in-repo | conv1d_update + GDN | 7.008 |
| Linear attention prefill in-repo | conv1d_fwd + FlashInfer GDN | 648.134 |
| MoE prefill | TRT-LLM routing + expert map + expand + gate_up + gated + down + finalize | 2780.126 |
| MoE decode | vLLM routing + align + gate_up + gated + down + finalize | 54.531 |

Additional dense W4A16 projection cases were added after the full `bench_all.sh` run above. They are guarded by tactic-cache checks, so a missing shape fails early instead of falling back to a default config.

Run: `h800_nsys_w4a16_dense_20260429_091047`

| Case | Shape (M,N,K) | Backend | Cached config | nsys GPU time (us) |
|---|---:|---|---|---:|
| `w4a16_prefill_linear_qkv_cutlass55` | 3823,12288,3072 | machete cutlass55 | `128x256x64_2x1x1` | 528.311 |
| `w4a16_prefill_linear_z_cutlass55` | 3823,8192,3072 | machete cutlass55 | `128x256x64_2x1x1` | 365.882 |
| `w4a16_prefill_linear_out_cutlass55` | 3823,3072,8192 | machete cutlass55 | `256x128x64_1x1x1` | 391.321 |
| `w4a16_decode_linear_qkv_fpA_intB` | 1,12288,3072 | fpA_intB | `cuda` | 9.376 |
| `w4a16_decode_linear_z_fpA_intB` | 1,8192,3072 | fpA_intB | `cuda` | 6.976 |
| `w4a16_decode_linear_out_fpA_intB` | 1,3072,8192 | fpA_intB | `cuda` | 8.672 |

Run: `h800_nsys_consistent_expert_20260429_094031`

| Case | Shape (M,N,K) | Backend | Cached config | nsys GPU time (us) |
|---|---:|---|---|---:|
| `w4a16_prefill_consistent_expert_up_cutlass55` | 3823,3072,2048 | machete cutlass55 | `128x128x64_1x1x1` | 111.613 |
| `w4a16_prefill_consistent_expert_down_cutlass55` | 3823,1024,3072 | machete cutlass55 | `128x128x64_1x1x1` | 69.726 |
| `w4a16_decode_consistent_expert_up_fpA_intB` | 1,3072,2048 | fpA_intB | `cuda` | 3.616 |
| `w4a16_decode_consistent_expert_down_fpA_intB` | 1,1024,3072 | fpA_intB | `cuda` | 4.032 |

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
| layers | 48 (36 linear_attention + 12 full_attention) |

## DeltaNet Linear Attention (×36 layers)

### Decode (batch=1, single token)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | RMSNorm | (1,3072)→(1,3072) | — | not in repo (generic) |
| 2 | in_proj_qkv (W4A16) | (1,3072)→(1,12288) | **9.4 μs** | `./bench_all.sh --case w4a16_decode_linear_qkv_fpA_intB` |
| 3 | in_proj_z (W4A16) | (1,3072)→(1,8192) | **7.0 μs** | `./bench_all.sh --case w4a16_decode_linear_z_fpA_intB` |
| 4 | in_proj_b (FP16) | (1,3072)→(1,64) | — | not in repo (cuBLAS GEMV) |
| 5 | in_proj_a (FP16) | (1,3072)→(1,64) | — | not in repo (cuBLAS GEMV) |
| 6 | **conv1d decode** | dim=12288, w=4 | **5.4 μs** | `./bench_conv1d_update 12288 4 1 --bench 20 100` |
| 7 | **GDN decode** (llama.cpp CUDA) | Q,K,V:(1,64,128) state:(64,128,128) | **5.4 μs** | `./bench_gated_delta_net 1 64 128 1 --bench 20 100` |
| 8 | **FusedRMSNormGated** | (64,128)→(64,128) | **5.3 μs** | `./bench_fused_rms_norm_gate 64 128 --bench 20 100` |
| 9 | out_proj (W4A16) | (1,8192)→(1,3072) | **8.7 μs** | `./bench_all.sh --case w4a16_decode_linear_out_fpA_intB` |

### Prefill (seq=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | in_proj_qkv (W4A16) | (3823,3072)→(3823,12288) | **528.3 μs** | `./bench_all.sh --case w4a16_prefill_linear_qkv_cutlass55` |
| 2 | in_proj_z (W4A16) | (3823,3072)→(3823,8192) | **365.9 μs** | `./bench_all.sh --case w4a16_prefill_linear_z_cutlass55` |
| 3 | **conv1d prefill** | (1,12288,3823) | **134.0 μs** | `./bench_conv1d_fwd 3823 12288 4 1 --bench 10 50` |
| 4 | **FlashInfer GDN prefill** (CUTLASS SM90) | Q:(3823,16,128) K:(3823,16,128) V:(3823,64,128) | **525.9 μs** | `./bench_gdn_prefill 3823 16 64 128 1 --bench 10 50` |
| 5 | **FusedRMSNormGated** | (3823×64,128)→(3823×64,128) | — | `./bench_fused_rms_norm_gate 245472 128 --bench 10 50` |
| 6 | out_proj (W4A16) | (3823,8192)→(3823,3072) | **391.3 μs** | `./bench_all.sh --case w4a16_prefill_linear_out_cutlass55` |

## MoE FFN (×48 layers, 256 experts, topk=8)

### Decode (batch=1, single token)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | RMSNorm | (1,3072)→(1,3072) | — | not in repo (generic) |
| 2 | Router gate (FP16) | (1,3072)→(1,256) | — | not in repo (cuBLAS GEMV) |
| 3 | **topk_gating** | (1,256)→w:(1,8) idx:(1,8) | **8.4 μs** | `moe_w4a16/vllm/auxiliary/bench_topk_gating 1 256 8 --bench 20 100` |
| 4 | **moe_align** | (1,8)→sorted_ids, expert_ids | **3.3 μs** | `moe_w4a16/vllm/auxiliary/bench_moe_align 1 256 8 16 --bench 20 100` |
| 5 | **Marlin MoE gate_up** (W4A16) | (8,1,3072)→(8,1,2048) | **11.5 μs** | `moe_w4a16/vllm/marlin/bench_marlin_moe 1 256 8 3072 1024 --bench 20 100` |
| 6 | **silu_and_mul** | (8,1,2048)→(8,1,1024) | **5.0 μs** | `moe_w4a16/vllm/auxiliary/bench_silu_and_mul 1 8 1024 --bench 20 100` |
| 7 | **Marlin MoE down** (W4A16) | (8,1,1024)→(8,1,3072) | **12.3 μs** | `moe_w4a16/vllm/marlin/bench_marlin_moe 1 256 8 1024 3072 --bench 20 100` |
| 8 | **moe_sum** | (8,1,3072)→(1,3072) | **5.4 μs** | `moe_w4a16/vllm/auxiliary/bench_moe_sum 1 8 3072 --bench 20 100` |
| 9 | Shared/consistent gate_up (W4A16) | (1,2048)→(1,3072) | **3.6 μs** | `./bench_all.sh --case w4a16_decode_consistent_expert_up_fpA_intB` |
| 10 | Shared SwiGLU | (1,2048)→(1,1024) | — | not in repo |
| 11 | Shared/consistent down (W4A16) | (1,3072)→(1,1024) | **4.0 μs** | `./bench_all.sh --case w4a16_decode_consistent_expert_down_fpA_intB` |

### Prefill (seq=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 3 | **topk_gating** | (3823,256)→... | — | `moe_w4a16/vllm/auxiliary/bench_topk_gating 3823 256 8 --bench 10 50` |
| 4 | **moe_align** | (3823,8)→... | — | `moe_w4a16/vllm/auxiliary/bench_moe_align 3823 256 8 16 --bench 10 50` |
| 5 | **Marlin MoE gate_up** (W4A16) | (8,3823,3072)→(8,3823,2048) | — | `moe_w4a16/vllm/marlin/bench_marlin_moe 3823 256 8 3072 1024 --bench 10 50` |
| 6 | **silu_and_mul** | (8,3823,2048)→(8,3823,1024) | — | `moe_w4a16/vllm/auxiliary/bench_silu_and_mul 3823 8 1024 --bench 10 50` |
| 7 | **Marlin MoE down** (W4A16) | (8,3823,1024)→(8,3823,3072) | — | `moe_w4a16/vllm/marlin/bench_marlin_moe 3823 256 8 1024 3072 --bench 10 50` |
| 8 | **moe_sum** | (8,3823,3072)→(3823,3072) | — | `moe_w4a16/vllm/auxiliary/bench_moe_sum 3823 8 3072 --bench 10 50` |
| 9 | Shared/consistent gate_up (W4A16) | (3823,2048)→(3823,3072) | **111.6 μs** | `./bench_all.sh --case w4a16_prefill_consistent_expert_up_cutlass55` |
| 10 | Shared SwiGLU | (3823,2048)→(3823,1024) | — | not in repo |
| 11 | Shared/consistent down (W4A16) | (3823,3072)→(3823,1024) | **69.7 μs** | `./bench_all.sh --case w4a16_prefill_consistent_expert_down_cutlass55` |

## Full Attention (×12 layers)

### Decode (batch=1, ctx=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | RMSNorm | (1,3072)→(1,3072) | — | not in repo (generic) |
| 2 | q_proj (W4A16) | (1,3072)→(1,16384) | — | not in repo (Marlin GEMV, ×2 for gate) |
| 3 | k_proj (W4A16) | (1,3072)→(1,512) | — | not in repo (Marlin GEMV) |
| 4 | v_proj (W4A16) | (1,3072)→(1,512) | — | not in repo (Marlin GEMV) |
| 5 | q/k RMSNorm | (1,32,256) / (1,2,256) | — | not in repo (generic) |
| 6 | MRoPE | Q:(1,32,256) K:(1,2,256) | — | not in repo (generic) |
| 7 | **FlashAttention v3 decode** | Q:(1,32,256) KV:(ctx,2,256) | — | `python3 bench_flash_attn.py decode 3823` |
| 8 | output gate | sigmoid × (1,8192) | — | elementwise |
| 9 | o_proj (W4A16) | (1,8192)→(1,3072) | — | not in repo (Marlin GEMV) |
| + | MoE FFN (same as above) | | | |

### Prefill (seq=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 7 | **FlashAttention v3 prefill** | Q:(3823,32,256) KV:(3823,2,256) | — | `python3 bench_flash_attn.py prefill 3823` |

> FlashAttention bench 使用 `flash_attn` 库（需 pip install flash-attn），如未安装自动 fallback 到 torch SDPA。

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
linear_attention/bench_conv1d_update 12288 4 1 --bench 20 100
linear_attention/bench_gated_delta_net 1 64 128 1 --bench 20 100
linear_attention/bench_fused_rms_norm_gate 64 128 --bench 20 100

# ── DeltaNet Prefill (seq=3823) ──
linear_attention/bench_conv1d_fwd 3823 12288 4 1 --bench 10 50
linear_attention/bench_gdn_prefill 3823 16 64 128 1 --bench 10 50

# ── MoE FFN Decode ──
moe_w4a16/vllm/auxiliary/bench_topk_gating 1 256 8 --bench 20 100
moe_w4a16/vllm/auxiliary/bench_moe_align 1 256 8 16 --bench 20 100
moe_w4a16/vllm/marlin/bench_marlin_moe 1 256 8 3072 1024 --bench 20 100
moe_w4a16/vllm/auxiliary/bench_silu_and_mul 1 8 1024 --bench 20 100
moe_w4a16/vllm/marlin/bench_marlin_moe 1 256 8 1024 3072 --bench 20 100
moe_w4a16/vllm/auxiliary/bench_moe_sum 1 8 3072 --bench 20 100

# ── MoE FFN Prefill (seq=3823) ──
moe_w4a16/vllm/auxiliary/bench_topk_gating 3823 256 8 --bench 10 50
moe_w4a16/vllm/auxiliary/bench_moe_align 3823 256 8 16 --bench 10 50
moe_w4a16/vllm/marlin/bench_marlin_moe 3823 256 8 3072 1024 --bench 10 50
moe_w4a16/vllm/auxiliary/bench_silu_and_mul 3823 8 1024 --bench 10 50
moe_w4a16/vllm/marlin/bench_marlin_moe 3823 256 8 1024 3072 --bench 10 50
moe_w4a16/vllm/auxiliary/bench_moe_sum 3823 8 3072 --bench 10 50

# ── Full Attention (FlashAttn, Python) ──
python3 flash_attn/bench_flash_attn.py decode 3823
python3 flash_attn/bench_flash_attn.py prefill 3823
python3 flash_attn/bench_flash_infer.py decode 3823
python3 flash_attn/bench_flash_infer.py prefill 3823

# ── nsys (纯 GPU kernel time) ──
nsys profile --trace=cuda -o trace ./bench_xxx [args]
nsys stats trace.nsys-rep --report cuda_gpu_kern_sum
```
