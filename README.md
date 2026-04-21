# DeltaNet CUDA Kernels (extracted from llama.cpp)

Qwen3.5 DeltaNet layer 的 CUDA kernel，从 [llama.cpp](https://github.com/ggml-org/llama.cpp) 提取，零依赖 standalone bench。

## Kernel 列表

| Kernel | 文件 | 来源 | 说明 |
|--------|------|------|------|
| `ssm_conv_f32` | `src/ssm_conv.cu` | `ggml/src/ggml-cuda/ssm-conv.cu` | 因果 1D 卷积（prefill + decode 共用） |
| `gated_delta_net_cuda` | `src/gated_delta_net.cu` | `ggml/src/ggml-cuda/gated_delta_net.cu` | 门控 delta rule 递推（prefill + decode 共用） |

## 编译

```bash
# 修改 Makefile 中 ARCH:
#   A100: -arch=sm_80    H100: -arch=sm_90    5090: -arch=sm_120

make
```

## 运行

### SSM Conv（因果卷积）

```bash
./bench_ssm_conv [n_tokens] [d_inner] [d_conv] [n_seqs]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_tokens` | 1 | token 数（1=decode, >1=prefill） |
| `d_inner` | 12288 | CONV_DIM = KEY_DIM×2 + VALUE_DIM |
| `d_conv` | 4 | 卷积核大小（支持 3/4/5/9） |
| `n_seqs` | 1 | batch size |

```bash
# Decode
ncu --set full --kernel-name "ssm_conv" -o conv_decode ./bench_ssm_conv 1 12288 4 1

# Prefill
ncu --set full --kernel-name "ssm_conv" -o conv_prefill ./bench_ssm_conv 3823 12288 4 1
```

### Gated Delta Net（门控 delta rule）

```bash
./bench_gated_delta_net [n_tokens] [heads] [head_dim] [n_seqs]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_tokens` | 1 | token 数（1=decode, >1=prefill） |
| `heads` | 64 | NUM_V_HEADS（扩展后） |
| `head_dim` | 128 | HEAD_DIM（支持 16/32/64/128） |
| `n_seqs` | 1 | batch size |

```bash
# Decode
ncu --set full --kernel-name "gated_delta_net" -o gdn_decode ./bench_gated_delta_net 1 64 128 1

# Prefill
ncu --set full --kernel-name "gated_delta_net" -o gdn_prefill ./bench_gated_delta_net 3823 64 128 1
```

## 提取说明

kernel 代码从 llama.cpp 零修改复制，仅做了：
1. 替换 `#include "common.cuh"` → `include/ggml_compat.h`（内联 `warp_reduce_sum`、`fastdiv` 等小函数）
2. 删除 `ggml_cuda_op_*` host wrapper（依赖 `ggml_tensor` 类型）
3. 添加 `#ifdef BENCH` standalone main

## 来源

- `ssm_conv`: llama.cpp 的 SSM 因果卷积 kernel，支持 kernel_size=3/4/5/9，长序列自动切换 shared memory 优化版
- `gated_delta_net`: llama.cpp PR #19504，递推式实现（sequential over tokens），warp-level reduce，支持 GDN 和 KDA 两种 gate 模式
- 注释 `//TODO: Add chunked kernel for even faster pre-fill` — llama.cpp 尚未实现 chunked 版本
