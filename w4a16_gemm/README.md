# W4A16 GEMM Benchmark

Single GEMM: Marlin W4A16 vs CUTLASS SM90 W4A16 vs BF16 cuBLAS

NVIDIA H800 PCIe (SM 9.0, BF16 dense peak 756 TFLOPS)

## Kernel 来源

| Kernel | 来源 | 说明 |
|--------|------|------|
| **Marlin W4A16** | [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | FP16 act + INT4 weight, `mma.sync.aligned.m16n8k16` |
| **CUTLASS SM90 W4A16** | CUTLASS example 55 (`55_hopper_int4_bf16_gemm`) | BF16 act + INT4 weight, WGMMA + TMA |
| **BF16 cuBLAS** | standalone cuBLAS | cuBLAS `cublasGemmEx` with BF16/FP16 |

## vLLM Kernel 选择策略

在 Hopper (sm_90) 上，vLLM 对非 MoE linear 层的 W4A16 kernel 选择优先级：

```
CutlassW4A8 → MacheteLinearKernel (sm_90+, CUTLASS) → MarlinLinearKernel (fallback)
```

Machete 本质是 CUTLASS 3.x collective builder 构建的 SM90 mixed-type GEMM，与 example 55 底层相同。

## 编译

```bash
# 1. Marlin (standalone, 无需 PyTorch)
nvcc -O2 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
  -diag-suppress 177,179,39 marlin_standalone.cu -o marlin_standalone

# 2. CUTLASS SM90 (需要 CUTLASS headers)
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
  -I<CUTLASS>/include -I<CUTLASS>/tools/util/include -I<CUTLASS>/examples/common \
  -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
  <CUTLASS>/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu \
  -o cutlass_w4a16_bench -lcublas

# 3. cuBLAS BF16/FP16
nvcc -O3 -std=c++17 -arch=sm_90 cublas_bf16_bench.cu -o cublas_bf16_bench -lcublas
```

## 运行

```bash
# Marlin W4A16 (standalone)
./marlin_standalone -m 4096 -n 4096 -k 4096 -w 50 -i 200
./marlin_standalone -m 4096 -n 4096 -k 4096 -g 128 -w 50 -i 200

# CUTLASS SM90 W4A16
./cutlass_w4a16_bench --m=4096 --n=4096 --k=4096 --g=128 --mode=1 --shuffle=true --iterations=200
./cutlass_w4a16_bench --m=4096 --n=4096 --k=4096 --g=128 --mode=1 --shuffle=false --iterations=200

# cuBLAS BF16/FP16
./cublas_bf16_bench 4096 4096 4096
```

## Benchmark 结果 (H800 PCIe, 4096x4096x4096)

All standalone CUDA, no Python overhead.

| Kernel | 延迟 (ms) | TFLOPS | 利用率 | vs BF16 |
|--------|-----------|--------|--------|---------|
| BF16 cuBLAS | 0.238 | 578.3 | 76.5% | 1.00x |
| FP16 cuBLAS | 0.239 | 576.2 | 76.2% | 1.00x |
| CUTLASS SM90 W4A16 (shuffle ON) | 0.396 | 346.6 | 45.8% | 0.60x |
| Marlin W4A16 (per-column) | 0.510 | 269.6 | 35.7% | 0.47x |
| CUTLASS SM90 W4A16 (shuffle OFF) | 0.524 | 262.1 | 34.7% | 0.45x |

### 分析

- **4096x4096x4096 是 compute-bound**，W4A16 的带宽优势无法体现
- **CUTLASS SM90 > Marlin 29%**：WGMMA + TMA 比 `mma.sync` 吞吐更高
- **Shuffle ON 提速 32%**：离线 weight reorder 减少 shared memory load 指令
- **Marlin 对大 M 会拆为多次 launch**（M=4096 → 4 次），cuBLAS 仅 1 次
- W4A16 的真正优势在 **memory-bound 场景**（小 batch, 大 weight），见 MoE benchmark

## 文件

```
marlin_standalone.cu          # Marlin W4A16 standalone (含正确性验证)
cublas_bf16_bench.cu          # cuBLAS BF16/FP16 standalone benchmark
kernels/
  marlin/
    marlin_cuda_kernel.cu     # 原始 Marlin CUDA kernel
    marlin_cuda.cpp           # PyTorch binding (可选)
    setup.py
```
