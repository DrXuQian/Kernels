# W4A16 GEMM Benchmark

Single GEMM: Marlin W4A16 vs CUTLASS SM90 W4A16 vs BF16 cuBLAS

Shape: M=1024, N=1024, K=1024 on NVIDIA H800 PCIe (SM 9.0)

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
# 1. Marlin (需要 PyTorch)
cd kernels/marlin && python3 setup.py build_ext --inplace

# 2. CUTLASS SM90 (需要 CUTLASS headers)
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
  -I<CUTLASS>/include -I<CUTLASS>/tools/util/include -I<CUTLASS>/examples/common \
  -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
  <CUTLASS>/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu \
  -o cutlass_w4a16_bench -lcublas

# 3. cuBLAS BF16
nvcc -O3 -std=c++17 -arch=sm_90 cublas_bf16_bench.cu -o cublas_bf16_bench -lcublas
```

## 运行

```bash
# Marlin + BF16 (PyTorch)
python3 bench_marlin.py

# CUTLASS (standalone, mode=1 for scaled W4A16, g=128 group size)
./cutlass_w4a16_bench --m=1024 --n=1024 --k=1024 --g=128 --mode=1 --shuffle=true --iterations=200

# cuBLAS BF16
./cublas_bf16_bench
```

## Benchmark 结果 (H800 PCIe, 1024x1024x1024)

| Kernel | 延迟 (ms) | TFLOPS | vs BF16 |
|--------|-----------|--------|---------|
| BF16 cuBLAS (torch.mm) | 0.0202 | 106.5 | 1.00x |
| FP16 cuBLAS (torch.mm) | 0.0204 | 105.5 | 0.99x |
| Marlin W4A16 (FP16 act) | 0.0191 | 112.6 | **1.06x** |
| CUTLASS SM90 W4A16 (shuffle ON) | 0.0174 | 123.3 | **1.16x** |
| CUTLASS SM90 W4A16 (shuffle OFF) | 0.0219 | 98.1 | 0.84x |

### 分析

- **1024x1024x1024 接近 compute-bound**，W4A16 的带宽优势不大
- **CUTLASS SM90 > Marlin**：WGMMA + TMA 比 `mma.sync` 高效 ~9%
- **Shuffle ON 提速 25%**：离线 weight reorder 减少 shared memory load 指令
- W4A16 的真正优势在 **memory-bound 场景**（小 batch, 大 weight），见 MoE benchmark

## 文件

```
cublas_bf16_bench.cu          # cuBLAS BF16/FP16 standalone benchmark
bench_marlin.py               # Marlin + torch.mm benchmark
kernels/
  marlin/
    marlin_cuda_kernel.cu     # Marlin CUDA kernel (822 lines)
    marlin_cuda.cpp           # PyTorch binding
    setup.py
  cutlass_sm90/               # 使用 CUTLASS repo example 55 编译
```
