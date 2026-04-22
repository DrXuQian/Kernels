# Kernels

Standalone CUDA kernel bench for Qwen3.5-122B inference profiling (ncu).

## 目录

```
linear_attention/                DeltaNet layer
  src/flashinfer_gdn/            GDN chunked prefill (FlashInfer CUTLASS SM90, GVA)
  src/kda/                       KDA chunked prefill (cuLA CUTLASS SM90)
  src/causal_conv1d_*.cu         conv1d fwd/update (Dao-AILab)
  src/gated_delta_net.cu         GDN decode recurrent (llama.cpp)
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

## Qwen3.5-122B DeltaNet 参数

| 参数 | 值 |
|------|-----|
| linear_num_key_heads | 16 |
| linear_num_value_heads | 64 |
| linear_key_head_dim | 128 |
| linear_value_head_dim | 128 |
| linear_conv_kernel_dim | 4 |
| MoE experts | 64 |
| MoE top_k | 8 |
| MoE hidden (K) | 2048 |
| MoE intermediate (N) | 5632 |

## 编译

```bash
git clone --recursive https://github.com/DrXuQian/Kernels.git
cd Kernels

# 全部编译（KDA 编译慢 ~10min，FlashInfer GDN ~5min）
make

# 或单独编译
cd linear_attention && make
cd linear_attention/src/flashinfer_gdn && make    # FlashInfer GDN prefill
cd moe_w4a16/marlin && make
cd moe_w4a16/auxiliary && make
```

## Benchmark (--bench mode)

所有 bench 二进制支持 `--bench [warmup] [iters]` 参数，使用 CUDA event 精确计时：

```bash
# FlashInfer GDN prefill (推荐，支持 GVA)
cd linear_attention/src/flashinfer_gdn
./bench_gdn_prefill 3823 --bench 10 50

# KDA prefill (cuLA，不支持 GVA)
cd linear_attention
./bench_kda_prefill 3823 16 128 1 --bench 10 50

# conv1d
./bench_conv1d_fwd 3823 12288 4 1 --bench 10 50
./bench_conv1d_update 12288 4 1 --bench 20 100

# MoE
cd moe_w4a16/marlin
./bench_marlin_moe 1 64 8 2048 5632 --bench 20 100
cd moe_w4a16/auxiliary
./bench_topk_gating 1 64 8 --bench 20 100
```

不加 `--bench` 默认单次执行，用于 ncu profiling。

## Benchmark 结果 (H800 PCIe)

### GDN Prefill (Qwen3.5-122B: q=16h, v=64h, dim=128)

| seqlen | FlashInfer CUTLASS | Triton (fla) | KDA (cuLA, h=16) |
|--------|-------------------|--------------|-------------------|
| 256 | **0.042 ms** | 0.528 ms | 0.087 ms |
| 1024 | **0.146 ms** | 0.549 ms | 0.305 ms |
| 2048 | **0.286 ms** | 0.533 ms | 0.593 ms |
| 3823 | **0.522 ms** | 0.553 ms | 1.109 ms |
| 8192 | **1.088 ms** | 1.047 ms | 2.348 ms |

FlashInfer CUTLASS 全面最优：原生 GVA 支持，短 seq 比 Triton 快 12x。

### Linear Attention Decode (batch=1)

| Kernel | Time (μs) |
|--------|-----------|
| conv1d_update (dim=12288) | 5.6 |
| gated_delta_net (64h, d=128) | 7.8 |

### MoE FFN Decode (M=1, 64 experts, topk=8)

| Kernel | Time (μs) | 占比 |
|--------|-----------|------|
| topk_gating | 7.6 | 16% |
| moe_align | 13.3 | 28% |
| Marlin MoE GEMM | 14.3 | 30% |
| silu_and_mul | 6.4 | 14% |
| moe_sum | 5.3 | 11% |
| **Total** | **~46.9** | 100% |

## ncu Profiling

```bash
# FlashInfer GDN prefill
cd linear_attention/src/flashinfer_gdn
ncu --set full ./bench_gdn_prefill 3823

# DeltaNet conv1d prefill
cd linear_attention
ncu --set full --kernel-name "causal_conv1d_fwd" ./bench_conv1d_fwd 3823 12288 4 1

# MoE Marlin GEMM decode
cd moe_w4a16/marlin
ncu --set full --kernel-name "Marlin" ./bench_marlin_moe 1 64 8 2048 5632
```
