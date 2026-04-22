# DeltaNet CUDA Kernels

Qwen3.5 DeltaNet layer 的 standalone CUDA kernel bench。

## Kernel 列表

| Kernel | 文件 | 来源 | GPU | 阶段 |
|--------|------|------|-----|------|
| `causal_conv1d_fwd` | `src/causal_conv1d_fwd.cu` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | SM80+ | Prefill |
| `causal_conv1d_update` | `src/causal_conv1d_update.cu` | Dao-AILab/causal-conv1d | SM80+ | Decode |
| `gated_delta_net_cuda` | `src/gated_delta_net.cu` | [llama.cpp](https://github.com/ggml-org/llama.cpp) PR #19504 | SM80+ | Prefill + Decode |
| `launch_kda_fwd_prefill_kernel` | `src/kda/sm90/` | [cuLA](https://github.com/inclusionAI/cuLA) | **SM90** | Prefill (chunked) |

## 依赖

- CUDA Toolkit 12.x
- [CUTLASS](https://github.com/NVIDIA/cutlass)（cuLA kernel 需要，Makefile 中设 `CUTLASS` 路径）

## 编译

```bash
make                        # 全部（conv1d + GDN + KDA prefill）
make bench_conv1d_fwd       # 只编译 conv1d prefill
make bench_conv1d_update    # 只编译 conv1d decode
make bench_gated_delta_net  # 只编译 GDN 递推
make bench_kda_prefill      # 只编译 cuLA chunked prefill (需 sm_90a)
```

## 运行

### Conv1d Prefill (Dao-AILab, 128-bit vectorized) — SM80+

```bash
./bench_conv1d_fwd [seq_len] [dim] [width] [batch]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `seq_len` | 3823 | prefill 长度 |
| `dim` | 12288 | CONV_DIM |
| `width` | 4 | 卷积核大小（2/3/4） |
| `batch` | 1 | batch size |

### Conv1d Decode (Dao-AILab, state update) — SM80+

```bash
./bench_conv1d_update [dim] [width] [batch]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `dim` | 12288 | CONV_DIM |
| `width` | 4 | 卷积核大小（2/3/4） |
| `batch` | 1 | batch size |

### Gated Delta Net 递推 (llama.cpp) — SM80+

```bash
./bench_gated_delta_net [n_tokens] [heads] [head_dim] [n_seqs]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `n_tokens` | 1 | 1=decode, >1=prefill |
| `heads` | 64 | num_v_heads |
| `head_dim` | 128 | 支持 16/32/64/128 |
| `n_seqs` | 1 | batch |

### cuLA Chunked KDA Prefill — SM90 (Hopper) only

```bash
./bench_kda_prefill [seq_len] [num_heads] [head_dim] [num_seqs]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `seq_len` | 3823 | prefill 长度 |
| `num_heads` | 64 | Q=K=V heads |
| `head_dim` | 128 | head_dim |
| `num_seqs` | 1 | batch |

## ncu Profile

```bash
# Conv1d
ncu --set full --kernel-name "causal_conv1d_fwd" -o conv_prefill ./bench_conv1d_fwd 3823 12288 4 1
ncu --set full --kernel-name "causal_conv1d_update" -o conv_decode ./bench_conv1d_update 12288 4 1

# GDN 递推
ncu --set full --kernel-name "gated_delta_net" -o gdn_decode ./bench_gated_delta_net 1 64 128 1
ncu --set full --kernel-name "gated_delta_net" -o gdn_prefill ./bench_gated_delta_net 3823 64 128 1

# KDA chunked prefill（需 Hopper）
ncu --set full -o kda_prefill ./bench_kda_prefill 3823 64 128 1
```

## 提取说明

| 来源 | Kernel | 修改 |
|------|--------|------|
| [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | conv1d fwd + update | 替换 `c10` include → `<cstdint>`，删 `C10_CUDA_CHECK` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | gated_delta_net | 替换 ggml include → `ggml_compat.h`，删 ggml_tensor wrapper |
| [cuLA](https://github.com/inclusionAI/cuLA) | kda sm90 prefill | **零修改**复制 `csrc/kda/` + `csrc/kerutils/`，仅新写 bench main |
