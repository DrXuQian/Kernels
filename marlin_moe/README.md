# Marlin MoE W4A16 Standalone Bench

从 vLLM 提取的 Marlin MoE kernel，零 PyTorch 依赖，可直接 ncu profile。

## 配置

- **Activation**: FP16 (half)
- **Weight**: INT4 (U4, unsigned 4-bit)
- **Output/Scales**: FP16
- **Group size**: 128 (`group_blocks=8`)
- **无 zero-point, 无 act_order**

## 编译

```bash
# 修改 Makefile 中 ARCH:
#   A100: -arch=sm_80    H100: -arch=sm_90    5090: -arch=sm_120

make
```

## 运行

```bash
# Usage: ./bench_marlin_moe [M] [num_experts] [top_k] [K] [N]
#   M: token 数 (1=decode, >1=prefill)
#   K: hidden_size (must % 16 == 0)
#   N: intermediate_size (must % 64 == 0)

# Decode
./bench_marlin_moe 1 64 8 2048 5632

# Prefill
./bench_marlin_moe 128 64 8 2048 5632
```

## ncu Profile

```bash
# Decode
ncu --set full --kernel-name "Marlin" -o marlin_decode ./bench_marlin_moe 1 64 8 2048 5632

# Prefill
ncu --set full --kernel-name "Marlin" -o marlin_prefill ./bench_marlin_moe 128 64 8 2048 5632
```

## 从 vLLM 提取的文件

| 文件 | 来源 | 修改 |
|------|------|------|
| `include/marlin_template.h` | `csrc/moe/marlin_moe_wna16/` | include 路径扁平化 |
| `include/marlin.cuh` | `csrc/quantization/marlin/` | 删 `torch/all.h` |
| `include/scalar_type.hpp` | `csrc/core/` | `TORCH_CHECK` → `assert` |
| `include/dequant.h` | `csrc/quantization/marlin/` | 无修改 |
| `include/marlin_mma.h` | `csrc/quantization/marlin/` | 无修改 |
| `include/marlin_dtypes.cuh` | `csrc/quantization/marlin/` | include 路径 |
| `include/kernel.h` | `csrc/moe/marlin_moe_wna16/` | include 路径 |
| `include/kernel_selector.h` | 重写 | 只保留 FP16+U4+group128 |
| `src/kernel_fp16_u4.cu` | 重写 | 15 个模板实例化（原 130+）|
| `src/dispatch.cu` | `csrc/moe/marlin_moe_wna16/ops.cu` | 删 PyTorch wrapper |
| `include/compat.h` | 新建 | `TORCH_CHECK` polyfill |

## Kernel 变体说明

15 个实例化覆盖所有需要的 thread block 配置：

| 场景 | thread_m_blocks | m_block_size_8 | thread configs |
|------|----------------|----------------|----------------|
| Decode (BS≤8) | 1 | true | 3 种 |
| Decode (BS 9-16) | 1 | false | 3 种 |
| Prefill (small) | 2 | false | 3 种 |
| Prefill (medium) | 3 | false | 3 种 |
| Prefill (large) | 4 | false | 3 种 |

运行时 `determine_exec_config()` 自动选最优配置。
