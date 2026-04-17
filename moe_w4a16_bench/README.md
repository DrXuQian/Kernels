# MoE W4A16 GEMM Benchmark

从 vLLM 提取的 production MoE GEMM kernel 对比：moe_wna16 vs Marlin MoE vs BF16

## Kernel 来源

| Kernel | vLLM 文件 | 类型 | 特点 |
|--------|-----------|------|------|
| **moe_wna16** | `csrc/moe/moe_wna16.cu` | W4A16 CUDA | 标量 `__hfma2`，fused，单次 launch |
| **Marlin MoE** | `csrc/moe/marlin_moe_wna16/` | W4A16 Marlin | Tensor core `mma.sync`，fused |
| **BF16 sequential** | cuBLAS `torch.mm` x N_experts | BF16 baseline | 非 grouped GEMM |

### vLLM MoE Kernel 选择策略 (Hopper sm_90)

```
W4A16 MoE dispatch:
  tokens/experts <= 6 且 group_size ∈ [32,64,128]
    → moe_wna16 CUDA kernel (fused, 不用 tensor core)
  否则
    → Triton fused_moe kernel

W4A16 MoE (MarlinExperts path):
    → marlin_moe_wna16 (tensor core, fused)

BF16 MoE (unquantized):
  Hopper → FlashInfer CUTLASS grouped GEMM (production)
  Fallback → Triton fused_moe
```

**注意**: Machete MoE **不存在**。Machete 只用于非 MoE linear 层。

## 编译

```bash
# 1. moe_wna16
cd kernels/moe_wna16
python3 setup.py build_ext --inplace

# 2. Marlin MoE (需要先生成 kernel)
cd kernels/marlin_moe
# 生成 kernel variants (需要 jinja2)
pip install jinja2
python3 generate_kernels.py "9.0"
# 复制 deps 到 include path
cp -r deps/* .
# 编译 (慢，14 个 kernel 文件)
TORCH_CUDA_ARCH_LIST="8.0 9.0" python3 setup.py build_ext --inplace
```

## 运行

```bash
# 固定 shape 对比
python3 bench_moe_full.py

# 多尺寸 sweep (含 Mixtral/DeepSeek-V2 真实 shape)
python3 bench_moe_large.py
```

## Benchmark 结果 (H800 PCIe)

### Sweep 1: N=K=4096, 8 experts, top_k=2

| Tokens | M/expert | moe_wna16 (ms) | Marlin MoE (ms) | BF16 x8 (ms) | wna16/BF16 | Marlin/BF16 |
|--------|----------|----------------|-----------------|---------------|------------|-------------|
| 1 | 1 | 0.069 | 0.082 | 0.177 | **0.39x** | **0.46x** |
| 4 | 1 | 0.115 | 0.083 | 0.175 | **0.65x** | **0.47x** |
| 16 | 4 | 0.173 | 0.083 | 0.178 | **0.97x** | **0.47x** |
| 64 | 16 | 0.502 | 0.098 | 0.179 | 2.80x | **0.54x** |
| 256 | 64 | 1.931 | 0.112 | 0.187 | 10.31x | **0.60x** |
| 1024 | 256 | 8.203 | 0.357 | 0.212 | 38.63x | 1.68x |
| 4096 | 1024 | 32.52 | 1.355 | 0.631 | 51.54x | 2.15x |

### Sweep 2: Mixtral-8x7B 真实 shape (gate_proj: N=14336, K=4096)

| Config | moe_wna16 | Marlin MoE | BF16 x8 | Marlin/BF16 |
|--------|-----------|------------|---------|-------------|
| bs=1 | 0.132ms | 0.300ms | 0.547ms | **0.55x** |
| bs=16 | 0.602ms | 0.312ms | 0.552ms | **0.56x** |
| bs=128 | 3.537ms | 0.336ms | 0.545ms | **0.62x** |
| bs=512 | 14.12ms | 0.635ms | 0.582ms | 1.09x |

### DeepSeek-V2 (64 experts, N=1536, K=2048)

| Config | moe_wna16 | BF16 x8 | wna16/BF16 |
|--------|-----------|---------|------------|
| bs=1 | 0.029ms | 0.752ms | **0.04x** |
| bs=16 | 0.093ms | 0.751ms | **0.12x** |
| bs=128 | 0.253ms | 0.750ms | **0.34x** |

### 分析

1. **Marlin MoE 在 memory-bound 区间最优**：bs<=256 时比 BF16 快 40-55%，weight 读取量 1/4 + tensor core
2. **moe_wna16 在极小 batch + 大 expert 数时最优**：DeepSeek-V2 (64 experts, bs=1) 时快 25 倍，因为每 expert 只有 ~0.03 个 token，fused kernel launch overhead 优势巨大
3. **大 batch 时 BF16 反超**：compute-bound 区间 tensor core 满载 BF16 效率更高
4. **BF16 baseline 是 sequential cuBLAS，非 grouped GEMM**：production 用 FlashInfer CUTLASS grouped GEMM 会更快

## 文件

```
bench_moe_full.py             # 固定 shape benchmark (8 experts, 1024x1024x1024)
bench_moe_large.py            # 多尺寸 sweep (含 Mixtral/DeepSeek 真实 shape)
kernels/
  moe_wna16/
    moe_wna16.cu              # vLLM production MoE W4A16 kernel (342 lines)
    moe_wna16_utils.h         # dequant + type utils (200 lines)
    moe_wna16_binding.cpp     # pybind11 binding
    setup.py
  marlin_moe/
    ops.cu                    # Marlin MoE host dispatch (874 lines)
    kernel.h                  # kernel params + template declaration
    marlin_template.h         # Marlin MoE kernel template (2241 lines)
    generate_kernels.py       # 生成 kernel_selector.h + sm80_kernel_*.cu
    binding.cpp               # pybind11 binding
    setup.py
    deps/                     # 从 vLLM csrc/ 复制的头文件
      quantization/marlin/    # marlin.cuh, marlin_dtypes.cuh, marlin_mma.h, dequant.h
      core/                   # scalar_type.hpp, registration.h
```

## 来源

所有 kernel 代码从 [vllm-project/vllm](https://github.com/vllm-project/vllm) 提取，GEMM kernel 函数零修改，仅替换 `TORCH_LIBRARY_IMPL` 注册为 pybind11 binding。
