# Kernels

Standalone CUDA kernel bench for Qwen3.5 inference profiling.

## 目录

```
linear_attention/                DeltaNet layer
  src/flashinfer_gdn/            GDN chunked prefill (FlashInfer CUTLASS SM90, GVA)
  src/kda/                       KDA chunked prefill (cuLA CUTLASS SM90)
  src/causal_conv1d_*.cu         conv1d fwd/update (Dao-AILab)
  src/gated_delta_net.cu         GDN decode recurrent (llama.cpp CUDA)
  src/bench_gdn_decode.py        GDN decode recurrent (fla Triton, matches vLLM)
moe_w4a16/                       MoE FFN layer
  marlin/                        W4A16 Marlin GEMM (vLLM)
  auxiliary/                     topk, align, silu_and_mul, sum (vLLM)
w4a16_gemm/                      Single GEMM bench: Marlin vs cuBLAS (Python)
third_party/cutlass/             CUTLASS (git submodule)
```

## Kernel 来源

| 目录 | Kernel | 来源 | 提取方式 |
|------|--------|------|---------|
| `linear_attention/src/flashinfer_gdn/` | GDN chunked prefill (GVA) | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | 删 TVM-FFI/PyTorch, C++20→C++17 fix |
| `linear_attention/src/kda/` | KDA chunked prefill | [cuLA](https://github.com/inclusionAI/cuLA) | 零修改复制 |
| `linear_attention/` | causal_conv1d fwd/update | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | 删 c10 include |
| `linear_attention/` | gated_delta_net decode | [llama.cpp](https://github.com/ggml-org/llama.cpp) | 删 ggml wrapper |
| `moe_w4a16/marlin/` | Marlin MoE GEMM | [vLLM](https://github.com/vllm-project/vllm) | 删 PyTorch wrapper |
| `moe_w4a16/auxiliary/` | topk, align, silu, sum | vLLM | 删 torch/cub wrapper |

## Qwen3.5 DeltaNet 参数

| 参数 | 35B-A3B | 122B-A10B |
|------|---------|-----------|
| linear_num_key_heads | 16 | 16 |
| linear_num_value_heads | 32 | 64 |
| linear_key_head_dim | 128 | 128 |
| linear_value_head_dim | 128 | 128 |
| linear_conv_kernel_dim | 4 | 4 |
| num_hidden_layers | 40 (30 linear + 10 full) | 80 (60 linear + 20 full) |
| MoE experts | 64 | 64 |
| MoE top_k | 8 | 8 |
| MoE hidden (K) | 2048 | 2048 |
| MoE intermediate (N) | 5632 | 5632 |

## 编译

```bash
git clone --recursive https://github.com/DrXuQian/Kernels.git
cd Kernels

# 全部编译
make

# 或单独编译
cd linear_attention && make
cd linear_attention/src/flashinfer_gdn && make    # FlashInfer GDN prefill (~5min)
cd moe_w4a16/marlin && make
cd moe_w4a16/auxiliary && make
```

## Benchmark

所有 bench 支持两种模式：

```bash
# 单次执行（用于 ncu/nsys profiling）
./bench_gdn_prefill 3823 16 64 128

# nsys 抓 GPU kernel time
nsys profile --trace=cuda --output=trace ./bench_gdn_prefill 3823 16 64 128
nsys stats trace.nsys-rep --report cuda_gpu_kern_sum

# CUDA event 计时（含 host launch overhead）
./bench_gdn_prefill 3823 16 64 128 1 --bench 10 50
```

## Benchmark 结果 (H800 PCIe, nsys GPU kernel time)

### Standalone vs vLLM 对比验证

使用 nsys 测量纯 GPU kernel 执行时间（不含 host launch overhead），验证提取的 standalone kernel 与 vLLM production 运行一致。

vLLM 配置：Qwen3.5-35B-A3B-GPTQ-Int4, gptq_marlin, enforce_eager, prefill ~205 tokens + decode 32 tokens。

#### DeltaNet Linear Attention

| Kernel | Standalone (μs) | vLLM nsys (μs) | 匹配 |
|--------|----------------|----------------|------|
| **FlashInfer GDN prefill** (q=16,v=32,seq=64,1chunk) | 12.0 | 13.0 (avg) | **YES** |
| **FlashInfer GDN prefill** (q=16,v=32,seq=205,4chunks) | 42.0 | 52.0 (4×13) | **YES** |
| conv1d_fwd prefill (seq=3823,dim=12288) | 128.6 | — | |
| conv1d_update decode (dim=12288) | 2.5 | 2.7 | **YES** |
| GDN decode fla Triton (h=64,d=128) | 4.9 | 6.4 | **YES** |
| GDN decode llama.cpp CUDA (h=64,d=128) | 4.5 | — | (不同 kernel) |

#### MoE FFN (decode, M=1, 64 experts, topk=8)

| Kernel | Standalone (μs) | vLLM nsys (μs) | 匹配 |
|--------|----------------|----------------|------|
| Marlin MoE GEMM (K=2048,N=5632) | 11.3 | 9.9-14.6 | **YES** |
| topk_gating | 4.4 | 5.7 | **YES** |
| moe_align | 10.1 | 4.4 | ~2x |
| silu_and_mul | 3.6 | 5.2 | ~YES |
| moe_sum | 2.2 | — | |

> **moe_align 差异说明**：standalone bench 使用 M=1 (decode)，`numel=8 < 1024` 走 `small_batch_expert_kernel`（10.1μs）。
> vLLM 使用 chunked prefill，prefill+decode 混合 batch 导致 `numel > 1024`，走 generic `moe_align_block_size_kernel`（4.4μs）。
> 不同输入走不同 kernel variant，非 kernel 不一致。
>
> **GDN decode 说明**：vLLM 使用 fla 的 `fused_recurrent_gated_delta_rule_packed_decode_kernel`（Triton JIT）。
> 原 standalone 使用 llama.cpp 的 `gated_delta_net_cuda`（纯 CUDA）。已添加 fla Triton 版 bench (`bench_gdn_decode.py`) 对齐。
> fla Triton 4.9μs vs vLLM 6.4μs — 接近（vLLM 用 packed 变体，略有差异）。

### GDN Prefill 三路对比 (122B config: q=16, v=64, dim=128)

nsys GPU kernel time, 单次 per-layer call：

| seqlen | FlashInfer CUTLASS (μs) | Triton fla (μs) | KDA cuLA (μs) |
|--------|------------------------|-----------------|---------------|
| 64 | **13** | 456 | 87 |
| 256 | **42** | 456 | 87 |
| 1024 | **150** | 459 | 305 |
| 2048 | **285** | 458 | 599 |
| 3823 | **527** | 517 | 1,109 |
| 8192 | **1,088** | 1,049 | 2,348 |

FlashInfer CUTLASS 全面最优。短 seq 比 Triton 快 **35x**（Triton 有 ~450μs 固定开销）。
KDA 不支持 GVA（只有单一 num_heads），实际使用时计算量偏大。

### GDN Prefill：Triton sub-kernel 拆分 (seq=3823, q=16, v=64)

vLLM Triton/FLA 路径将 GDN prefill 拆成 7 个 kernel：

| Triton Kernel | H800 (μs) | 占比 |
|--------------|-----------|------|
| chunk_fwd_kernel_o | 150.6 | 29% |
| chunk_gated_delta_rule_fwd_kernel_h ⭐ | 139.5 | 27% |
| chunk_local_cumsum_vector | 98.3 | 19% |
| recompute_w_u_fwd | 74.0 | 14% |
| chunk_gated_delta_rule_fwd_kkt_solve | 53.2 | 10% |
| **Total** | **519.8** | 100% |

FlashInfer CUTLASS 将以上全部 fuse 成 **1 个 kernel**：527 μs。

### MoE FFN Decode 完整 pipeline

| Kernel | nsys (μs) | 占比 |
|--------|-----------|------|
| topk_gating | 4.4 | 14% |
| moe_align | 10.1 | 31% |
| Marlin MoE GEMM | 11.3 | 35% |
| silu_and_mul | 3.6 | 11% |
| moe_sum | 2.2 | 7% |
| **Total** | **~31.6** | 100% |

## ncu Profiling

```bash
# FlashInfer GDN prefill
cd linear_attention/src/flashinfer_gdn
ncu --set full ./bench_gdn_prefill 3823 16 64 128

# Conv1d prefill
cd linear_attention
ncu --set full --kernel-name "causal_conv1d_fwd" ./bench_conv1d_fwd 3823 12288 4 1

# MoE Marlin GEMM decode
cd moe_w4a16/marlin
ncu --set full --kernel-name "Marlin" ./bench_marlin_moe 1 64 8 2048 5632
```
