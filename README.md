# CUDA Kernels

CUDA kernel 提取、编译与性能对比。

## 项目列表

| 目录 | 内容 | 来源 |
|------|------|------|
| [`w4a16_gemm_bench/`](w4a16_gemm_bench/) | W4A16 GEMM: Marlin vs CUTLASS SM90 vs BF16 cuBLAS | IST-DASLab/marlin + CUTLASS example 55 |
| [`moe_w4a16_bench/`](moe_w4a16_bench/) | MoE W4A16 GEMM: moe_wna16 vs Marlin MoE vs BF16 | vLLM `csrc/moe/` |
| [`marlin_moe/`](marlin_moe/) | MoE 全 pipeline standalone (topk + align + Marlin + silu + sum) | vLLM `csrc/moe/` |
| `src/` + `include/` | DeltaNet CUDA Kernels (Qwen3.5-122B) | — |

---

## W4A16 GEMM Benchmark 摘要 (H800 PCIe, 1024x1024x1024)

| Kernel | 延迟 (ms) | TFLOPS | 说明 |
|--------|-----------|--------|------|
| BF16 cuBLAS | 0.020 | 106.5 | baseline |
| Marlin W4A16 | 0.019 | 112.6 | Ampere MMA |
| CUTLASS SM90 W4A16 | 0.017 | 123.3 | Hopper WGMMA + TMA |

## MoE W4A16 GEMM Benchmark 摘要 (H800, Mixtral-8x7B shape)

| Config | moe_wna16 | Marlin MoE | BF16 x8 | Marlin/BF16 |
|--------|-----------|------------|---------|-------------|
| bs=1 (decode) | 0.132ms | 0.300ms | 0.547ms | **0.55x** |
| bs=16 | 0.602ms | 0.312ms | 0.552ms | **0.56x** |
| bs=128 (prefill) | 3.537ms | 0.336ms | 0.545ms | **0.62x** |
| bs=512 | 14.12ms | 0.635ms | 0.582ms | 1.09x |

详见各子目录 README。

---

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

| ID | 函数 | 阶段 | 源文件 |
|----|------|------|--------|
| K1 | `causal_conv1d_fn` | prefill | `src/causal_conv1d_fwd.cu` |
| K2 | `causal_conv1d_update` | decode | `src/causal_conv1d_update.cu` |
| K3 | `chunk_gated_delta_rule` | prefill | `src/chunk_gated_delta_rule.cu` |
| K4 | `fused_recurrent_gated_delta_rule` | decode | `src/fused_recurrent.cu` |
| K5 | `fused_rms_norm_gate` | 共用 | `src/fused_rms_norm_gate.cu` |

每个 `.cu` 文件底部有 `#ifdef BENCH` 块，加 `-DBENCH` 编译即为独立可执行文件。

## 目录结构

```
include/
  deltanet.h            # 公共接口
  bench_utils.h         # host 端初始化工具（无 GPU kernel）
  naive_reference.h     # CPU 参考实现（验证用）
src/                    # 每个 kernel 一个 .cu
  causal_conv1d_fwd.cu        # K1
  causal_conv1d_update.cu     # K2
  chunk_gated_delta_rule.cu   # K3
  fused_recurrent.cu          # K4
  fused_rms_norm_gate.cu      # K5
main.cu                 # 全流程测试 + CPU 数值验证
Makefile
```

## 编译

```bash
# 修改 Makefile 中 ARCH，例如：
#   A100: -arch=sm_80    H100: -arch=sm_90    5090: -arch=sm_120

# 全流程测试（含数值验证）
make all

# 全部 bench 二进制
make bench

# 单个 bench
make bench_k3
```

编译产物：

| 目标 | 编译方式 | 说明 |
|------|----------|------|
| `deltanet_test` | 所有 kernel `.o` + `main.o` 链接 | 全流程测试 |
| `bench_k1` ~ `bench_k5` | 单个 `.cu` 加 `-DBENCH` 编译 | 独立可执行，含 kernel + main |

## 运行测试

```bash
./deltanet_test
```

## ncu Profile

bench 二进制内无 GPU 初始化 kernel（全部 host 端 `cudaMemcpy`），ncu 只会看到目标 kernel。

### Prefill（seq_len 可配）

```bash
ncu --set full -o k1_seq4096 ./bench_k1 4096
ncu --set full -o k3_seq4096 ./bench_k3 4096
ncu --set full -o k5_prefill ./bench_k5 4096
```

### Decode（固定单步）

```bash
ncu --set full -o k2_decode ./bench_k2
ncu --set full -o k4_decode ./bench_k4
ncu --set full -o k5_decode ./bench_k5 1
```

### 按 kernel 名过滤（可选）

```bash
ncu --set full --kernel-name "causal_conv1d_fn_kernel" -o k1 ./bench_k1 4096
ncu --set full --kernel-name "chunk_gated_delta_rule_kernel" -o k3 ./bench_k3 4096
ncu --set full --kernel-name "fused_recurrent_gated_delta_rule_kernel" -o k4 ./bench_k4
ncu --set full --kernel-name "fused_rms_norm_gate_kernel" -o k5 ./bench_k5 4096
```
