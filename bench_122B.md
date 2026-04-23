# Qwen3.5-122B-A10B Standalone Kernel Benchmark

H800 PCIe (SM 9.0, HBM 2.0 TB/s)

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
| 2 | in_proj_qkv (W4A16) | (1,3072)→(1,12288) | — | not in repo (Marlin GEMV) |
| 3 | in_proj_z (W4A16) | (1,3072)→(1,8192) | — | not in repo (Marlin GEMV) |
| 4 | in_proj_b (FP16) | (1,3072)→(1,64) | — | not in repo (cuBLAS GEMV) |
| 5 | in_proj_a (FP16) | (1,3072)→(1,64) | — | not in repo (cuBLAS GEMV) |
| 6 | **conv1d decode** | dim=12288, w=4 | **5.4 μs** | `./bench_conv1d_update 12288 4 1 --bench 20 100` |
| 7 | **GDN decode** (llama.cpp CUDA) | Q,K,V:(1,64,128) state:(64,128,128) | **5.4 μs** | `./bench_gated_delta_net 1 64 128 1 --bench 20 100` |
| 8 | **FusedRMSNormGated** | (64,128)→(64,128) | **5.3 μs** | `./bench_fused_rms_norm_gate 64 128 --bench 20 100` |
| 9 | out_proj (W4A16) | (1,8192)→(1,3072) | — | not in repo (Marlin GEMV) |

### Prefill (seq=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | in_proj_qkv (W4A16) | (3823,3072)→(3823,12288) | — | not in repo (Marlin GEMM) |
| 2 | in_proj_z (W4A16) | (3823,3072)→(3823,8192) | — | not in repo (Marlin GEMM) |
| 3 | **conv1d prefill** | (1,12288,3823) | **134.0 μs** | `./bench_conv1d_fwd 3823 12288 4 1 --bench 10 50` |
| 4 | **FlashInfer GDN prefill** (CUTLASS SM90) | Q:(3823,16,128) K:(3823,16,128) V:(3823,64,128) | **525.9 μs** | `./bench_gdn_prefill 3823 16 64 128 1 --bench 10 50` |
| 5 | **FusedRMSNormGated** | (3823×64,128)→(3823×64,128) | — | `./bench_fused_rms_norm_gate 245472 128 --bench 10 50` |
| 6 | out_proj (W4A16) | (3823,8192)→(3823,3072) | — | not in repo (Marlin GEMM) |

## MoE FFN (×48 layers, 256 experts, topk=8)

### Decode (batch=1, single token)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 1 | RMSNorm | (1,3072)→(1,3072) | — | not in repo (generic) |
| 2 | Router gate (FP16) | (1,3072)→(1,256) | — | not in repo (cuBLAS GEMV) |
| 3 | **topk_gating** | (1,256)→w:(1,8) idx:(1,8) | **8.4 μs** | `./bench_topk_gating 1 256 8 --bench 20 100` |
| 4 | **moe_align** | (1,8)→sorted_ids, expert_ids | **3.3 μs** | `./bench_moe_align 1 256 8 16 --bench 20 100` |
| 5 | **Marlin MoE gate_up** (W4A16) | (8,1,3072)→(8,1,2048) | **11.5 μs** | `./bench_marlin_moe 1 256 8 3072 1024 --bench 20 100` |
| 6 | **silu_and_mul** | (8,1,2048)→(8,1,1024) | **5.0 μs** | `./bench_silu_and_mul 1 8 1024 --bench 20 100` |
| 7 | **Marlin MoE down** (W4A16) | (8,1,1024)→(8,1,3072) | **12.3 μs** | `./bench_marlin_moe 1 256 8 1024 3072 --bench 20 100` |
| 8 | **moe_sum** | (8,1,3072)→(1,3072) | **5.4 μs** | `./bench_moe_sum 1 8 3072 --bench 20 100` |
| 9 | Shared gate_up (W4A16) | (1,3072)→(1,2048) | — | not in repo (Marlin GEMV) |
| 10 | Shared SwiGLU | (1,2048)→(1,1024) | — | not in repo |
| 11 | Shared down (W4A16) | (1,1024)→(1,3072) | — | not in repo (Marlin GEMV) |

### Prefill (seq=3823)

| # | Kernel | Shape | Latency | Command |
|---|--------|-------|---------|---------|
| 3 | **topk_gating** | (3823,256)→... | — | `./bench_topk_gating 3823 256 8 --bench 10 50` |
| 4 | **moe_align** | (3823,8)→... | — | `./bench_moe_align 3823 256 8 16 --bench 10 50` |
| 5 | **Marlin MoE gate_up** (W4A16) | (8,3823,3072)→(8,3823,2048) | — | `./bench_marlin_moe 3823 256 8 3072 1024 --bench 10 50` |
| 6 | **silu_and_mul** | (8,3823,2048)→(8,3823,1024) | — | `./bench_silu_and_mul 3823 8 1024 --bench 10 50` |
| 7 | **Marlin MoE down** (W4A16) | (8,3823,1024)→(8,3823,3072) | — | `./bench_marlin_moe 3823 256 8 1024 3072 --bench 10 50` |
| 8 | **moe_sum** | (8,3823,3072)→(3823,3072) | — | `./bench_moe_sum 3823 8 3072 --bench 10 50` |

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
moe_w4a16/auxiliary/bench_topk_gating 1 256 8 --bench 20 100
moe_w4a16/auxiliary/bench_moe_align 1 256 8 16 --bench 20 100
moe_w4a16/marlin/bench_marlin_moe 1 256 8 3072 1024 --bench 20 100
moe_w4a16/auxiliary/bench_silu_and_mul 1 8 1024 --bench 20 100
moe_w4a16/marlin/bench_marlin_moe 1 256 8 1024 3072 --bench 20 100
moe_w4a16/auxiliary/bench_moe_sum 1 8 3072 --bench 20 100

# ── MoE FFN Prefill (seq=3823) ──
moe_w4a16/auxiliary/bench_topk_gating 3823 256 8 --bench 10 50
moe_w4a16/auxiliary/bench_moe_align 3823 256 8 16 --bench 10 50
moe_w4a16/marlin/bench_marlin_moe 3823 256 8 3072 1024 --bench 10 50
moe_w4a16/auxiliary/bench_silu_and_mul 3823 8 1024 --bench 10 50
moe_w4a16/marlin/bench_marlin_moe 3823 256 8 1024 3072 --bench 10 50
moe_w4a16/auxiliary/bench_moe_sum 3823 8 3072 --bench 10 50

# ── Full Attention (FlashAttn, Python) ──
python3 flash_attn/bench_flash_attn.py decode 3823
python3 flash_attn/bench_flash_attn.py prefill 3823

# ── nsys (纯 GPU kernel time) ──
nsys profile --trace=cuda -o trace ./bench_xxx [args]
nsys stats trace.nsys-rep --report cuda_gpu_kern_sum
```
