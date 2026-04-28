# Linear Attention (DeltaNet) Kernels

Qwen3.5 DeltaNet layer 的 standalone CUDA kernel bench。

## Kernel 列表

| Bench | Kernel | 来源 | GPU | 阶段 |
|-------|--------|------|-----|------|
| `bench_conv1d_fwd` | `causal_conv1d_fwd` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | SM80+ | Prefill |
| `bench_conv1d_update` | `causal_conv1d_update` | Dao-AILab/causal-conv1d | SM80+ | Decode |
| `bench_gated_delta_net` | `gated_delta_net_cuda` | [llama.cpp](https://github.com/ggml-org/llama.cpp) | SM80+ | Prefill + Decode |
| `bench_kda_prefill` | `launch_kda_fwd_prefill_kernel` | [cuLA](https://github.com/inclusionAI/cuLA) | SM90 | Prefill (chunked) |

## 编译

```bash
make                        # 全部
make bench_conv1d_fwd       # 单个
```

也可以从 repo 根目录使用统一编译脚本选择目标，并让脚本检查当前环境：

```bash
./compile.sh env
./compile.sh build linear_attention
./compile.sh build flashinfer-gdn
```

## 运行

```bash
# Conv1d prefill
./bench_conv1d_fwd [seq_len] [dim] [width] [batch]         # default: 3823 12288 4 1

# Conv1d decode
./bench_conv1d_update [dim] [width] [batch]                 # default: 12288 4 1

# GDN 递推 (decode: n_tokens=1, prefill: n_tokens>1)
./bench_gated_delta_net [n_tokens] [heads] [head_dim] [n_seqs]  # default: 1 64 128 1

# cuLA chunked prefill (Hopper only)
./bench_kda_prefill [seq_len] [num_heads] [head_dim] [num_seqs] # default: 3823 64 128 1
```

## ncu Profile

```bash
ncu --set full --kernel-name "causal_conv1d_fwd"    -o conv_fwd    ./bench_conv1d_fwd 3823 12288 4 1
ncu --set full --kernel-name "causal_conv1d_update"  -o conv_update ./bench_conv1d_update 12288 4 1
ncu --set full --kernel-name "gated_delta_net"       -o gdn         ./bench_gated_delta_net 1 64 128 1
ncu --set full                                       -o kda         ./bench_kda_prefill 3823 64 128 1
```
