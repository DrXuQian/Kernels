# Kernels

Standalone CUDA kernel bench for Qwen3.5 inference profiling (ncu).

## 目录

```
linear_attention/       DeltaNet layer: conv1d + gated delta rule
moe_w4a16/              MoE FFN layer: W4A16 Marlin GEMM + auxiliary kernels
w4a16_gemm/             Single GEMM bench: Marlin vs cuBLAS (Python)
third_party/cutlass/    CUTLASS (git submodule)
```

## Kernel 来源

| 目录 | Kernel | 来源 | 提取方式 |
|------|--------|------|---------|
| `linear_attention/` | causal_conv1d fwd/update | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | 删 c10 include |
| `linear_attention/` | gated_delta_net | [llama.cpp](https://github.com/ggml-org/llama.cpp) | 删 ggml wrapper |
| `linear_attention/` | KDA chunked prefill | [cuLA](https://github.com/inclusionAI/cuLA) | 零修改复制 |
| `moe_w4a16/marlin/` | Marlin MoE GEMM | [vLLM](https://github.com/vllm-project/vllm) | 删 PyTorch wrapper |
| `moe_w4a16/auxiliary/` | topk, align, silu, sum | vLLM | 删 torch/cub wrapper |

## 编译

```bash
git clone --recursive https://github.com/DrXuQian/Kernels.git
cd Kernels

# 全部编译
make

# 或单独编译
cd linear_attention && make
cd moe_w4a16/marlin && make
cd moe_w4a16/auxiliary && make
```

## 快速 ncu 示例

```bash
# DeltaNet conv1d prefill
cd linear_attention
ncu --set full --kernel-name "causal_conv1d_fwd" ./bench_conv1d_fwd 3823 12288 4 1

# MoE Marlin GEMM decode
cd moe_w4a16/marlin
ncu --set full --kernel-name "Marlin" ./bench_marlin_moe 1 64 8 2048 5632
```
