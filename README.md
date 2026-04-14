# DeltaNet CUDA Kernels

Qwen3.5-122B-A10B DeltaNet layer 的 5 个 CUDA kernel 实现。

## 模型参数

| 参数 | 值 |
|------|-----|
| BATCH | 1 |
| NUM_K_HEADS | 16 |
| NUM_V_HEADS | 64 |
| HEAD_DIM | 128 |
| KEY_DIM | 2048 |
| VALUE_DIM | 8192 |
| CONV_DIM | 12288 |
| CONV_KERNEL | 4 |

## Kernel 列表

| ID | 函数 | 阶段 | 说明 |
|----|------|------|------|
| K1 | `causal_conv1d_fn` | prefill | 因果卷积，seq_len 可配 |
| K2 | `causal_conv1d_update` | decode | 单步卷积更新 |
| K3 | `chunk_gated_delta_rule` | prefill | 门控 delta rule 注意力，seq_len 可配 |
| K4 | `fused_recurrent_gated_delta_rule` | decode | 单步递推 delta rule |
| K5 | `fused_rms_norm_gate` | 共用 | RMS norm + sigmoid gate |

## 目录结构

```
include/
  deltanet.h                  # 公共接口
  naive_reference.h           # CPU 参考实现（验证用）
src/                          # 每个 kernel 一个 .cu 文件
  causal_conv1d_fwd.cu        # K1 — prefill conv
  causal_conv1d_update.cu     # K2 — decode conv
  chunk_gated_delta_rule.cu   # K3 — prefill delta rule
  fused_recurrent.cu          # K4 — decode delta rule
  fused_rms_norm_gate.cu      # K5 — RMS norm + gate
bench/                        # 每个 kernel 独立 bench 入口
  bench_k1.cu                 # ./bench_k1 [seq_len]
  bench_k2.cu                 # ./bench_k2
  bench_k3.cu                 # ./bench_k3 [seq_len]
  bench_k4.cu                 # ./bench_k4
  bench_k5.cu                 # ./bench_k5 [N]
main.cu                       # 全流程测试 + CPU 数值验证
Makefile
```

## 编译

```bash
# 修改 Makefile 中 ARCH 为你的 GPU 架构，例如：
#   A100:  -arch=sm_80
#   H100:  -arch=sm_90
#   5090:  -arch=sm_120

# 编译全流程测试（含数值验证）
make all

# 编译全部 bench 二进制（每个 kernel 独立）
make bench

# 只编译单个 kernel 的 bench
make bench_k3
```

## 运行测试

```bash
# 全流程：prefill → decode × 10 → CPU 数值验证
./deltanet_test
```

## ncu Profile

每个 bench 二进制只包含对应 kernel 的一次调用，直接用 ncu 即可。

### Prefill（K1 + K3 + K5，seq_len 可配）

```bash
# K1: causal_conv1d_fn
ncu --set full -o k1_seq4096 ./bench_k1 4096
ncu --set full -o k1_seq8192 ./bench_k1 8192

# K3: chunk_gated_delta_rule
ncu --set full -o k3_seq4096 ./bench_k3 4096
ncu --set full -o k3_seq8192 ./bench_k3 8192

# K5: fused_rms_norm_gate (prefill, N = seq_len)
ncu --set full -o k5_prefill4096 ./bench_k5 4096
```

### Decode（K2 + K4 + K5，固定单步）

```bash
# K2: causal_conv1d_update
ncu --set full -o k2_decode ./bench_k2

# K4: fused_recurrent_gated_delta_rule
ncu --set full -o k4_decode ./bench_k4

# K5: fused_rms_norm_gate (decode, N = 1)
ncu --set full -o k5_decode ./bench_k5 1
```

### 只 profile 目标 kernel（跳过 init kernel）

bench 二进制中包含数据初始化用的 `fill_*` kernel。如果只想看目标 kernel：

```bash
# 按 kernel 名过滤
ncu --set full --kernel-name "causal_conv1d_fn_kernel" -o k1 ./bench_k1 4096
ncu --set full --kernel-name "chunk_gated_delta_rule_kernel" -o k3 ./bench_k3 4096
ncu --set full --kernel-name "fused_recurrent_gated_delta_rule_kernel" -o k4 ./bench_k4
ncu --set full --kernel-name "fused_rms_norm_gate_kernel" -o k5 ./bench_k5 4096

# 或者按 launch 序号跳过前面的 init kernel
ncu --set full --launch-skip 3 --launch-count 1 -o k3 ./bench_k3 4096
```

### 快速查看 summary（不生成 .ncu-rep）

```bash
ncu --kernel-name "chunk_gated_delta_rule_kernel" ./bench_k3 4096
```
