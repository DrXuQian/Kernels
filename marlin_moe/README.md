# MoE Kernel Standalone Bench

从 vLLM 提取的完整 MoE pipeline kernel，零 PyTorch 依赖，可直接 ncu profile。

## MoE FFN Pipeline

```
topk_gating → moe_align → Marlin GEMM (gate_up) → silu_and_mul → Marlin GEMM (down) → moe_sum
```

## 编译

```bash
# 修改 Makefile 中 ARCH:
#   A100: -arch=sm_80    H100: -arch=sm_90    5090: -arch=sm_120

make        # 编译全部 4 个 bench
make bench_marlin_moe   # 只编译 GEMM
```

## Kernel 列表

### 1. TopK Gating — `bench_topk_gating`

融合 softmax + topK 选 expert，warp butterfly reduce。

```bash
./bench_topk_gating [num_tokens] [num_experts] [topk]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_tokens` | 1 | token 数（decode=1, prefill>1） |
| `num_experts` | 64 | expert 数（须为 8/16/32/64/128/256） |
| `topk` | 8 | 每个 token 选几个 expert |

```bash
# ncu
ncu --set full --kernel-name "topkGating" -o topk ./bench_topk_gating 1 64 8
```

### 2. MoE Align — `bench_moe_align`

按 expert 排序 token，对齐到 block_size 边界，生成 sorted_token_ids 和 expert_ids。

```bash
./bench_moe_align [num_tokens] [num_experts] [topk] [block_size]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_tokens` | 1 | token 数 |
| `num_experts` | 64 | expert 数 |
| `topk` | 8 | top-k |
| `block_size` | 16 | Marlin 的 moe_block_size（8 或 16） |

```bash
ncu --set full --kernel-name "moe_align" -o align ./bench_moe_align 1 64 8 16
```

### 3. Marlin MoE GEMM — `bench_marlin_moe`

W4A16 量化 MoE GEMM（FP16 activation + INT4 weight + group_size=128）。

```bash
./bench_marlin_moe [M] [num_experts] [top_k] [K] [N]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `M` | 1 | token 数（decode=1） |
| `num_experts` | 64 | expert 数 |
| `top_k` | 8 | top-k |
| `K` | 2048 | hidden_size（须 % 16 == 0） |
| `N` | 5632 | intermediate_size（须 % 64 == 0） |

```bash
# Decode
ncu --set full --kernel-name "Marlin" -o gemm_decode ./bench_marlin_moe 1 64 8 2048 5632

# Prefill
ncu --set full --kernel-name "Marlin" -o gemm_prefill ./bench_marlin_moe 128 64 8 2048 5632
```

### 4. SiLU + Mul — `bench_silu_and_mul`

Marlin GEMM (gate_up) 输出 `[gate, up]` 拼接，此 kernel 计算 `SiLU(gate) * up`。

```bash
./bench_silu_and_mul [num_tokens] [top_k] [hidden_size]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_tokens` | 1 | token 数 |
| `top_k` | 8 | 每个 token 选的 expert 数，实际行数 = M × top_k |
| `hidden_size` | 5632 | intermediate_size（N），input 宽度为 2*N |

```bash
ncu --set full --kernel-name "silu_and_mul" -o silu ./bench_silu_and_mul 1 8 5632
```

### 5. MoE Sum — `bench_moe_sum`

聚合 topk 个 expert 输出（element-wise sum）。

```bash
./bench_moe_sum [num_tokens] [topk] [hidden_size]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_tokens` | 1 | token 数 |
| `topk` | 8 | top-k（支持 2/4/8） |
| `hidden_size` | 5632 | 输出维度 |

```bash
ncu --set full --kernel-name "moe_sum" -o sum ./bench_moe_sum 1 8 5632
```

## 目录结构

```
src/
  topk_gating.cu         # K1: fused softmax + topK
  moe_align.cu           # K2: token alignment/sorting
  kernel_fp16_u4.cu      # K3: Marlin GEMM 模板实例化
  dispatch.cu            # K3: Marlin GEMM host dispatch
  silu_and_mul.cu        # K4: SiLU activation + multiply
  moe_sum.cu             # K5: output aggregation
include/
  moe_compat.h           # K1/K2/K4 的 standalone compat layer
  compat.h               # K3 (Marlin) 的 TORCH_CHECK polyfill
  marlin_template.h      # Marlin GEMM kernel (2241 行, 零修改)
  kernel_selector.h      # FP16+U4+group128 only (15 变体)
  ...                    # 其他 Marlin headers
bench_marlin_moe.cu      # K3 bench main
```

## 配置说明

- **Marlin GEMM**: FP16 act + INT4 weight + FP16 output, group_size=128, 无 zero-point, 无 act_order
- **TopK Gating**: float32 gating scores, int32 indices, softmax scoring
- **MoE Align**: int32 topk_ids, small-batch-expert 模式（< 1024 tokens, ≤ 64 experts）
- **MoE Sum**: FP16 input/output

## 来源

所有 kernel 代码从 [vllm-project/vllm](https://github.com/vllm-project/vllm) `csrc/moe/` 提取，kernel 函数零修改，仅删除 PyTorch/Python 依赖。
