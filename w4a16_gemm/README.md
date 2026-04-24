# W4A16 GEMM Benchmark

Single GEMM: fpAIntB (TRT-LLM) vs Marlin vs CUTLASS SM90 vs BF16 cuBLAS

NVIDIA H800 PCIe (SM 9.0, BF16 dense peak 756 TFLOPS)

## Kernel 来源

| Kernel | 来源 | 说明 |
|--------|------|------|
| **fpAIntB (TRT-LLM)** | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) extracted | Prefill: CUTLASS 3.x SM90 WGMMA; Decode: CUDA core GEMV |
| **Marlin W4A16** | [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | FP16 act + INT4 weight, `mma.sync.aligned.m16n8k16` |
| **CUTLASS SM90 W4A16** | CUTLASS example 55 (`55_hopper_int4_bf16_gemm`) | BF16 act + INT4 weight, WGMMA + TMA |
| **BF16 cuBLAS** | standalone cuBLAS | cuBLAS `cublasGemmEx` with BF16/FP16 |

## vLLM Kernel 选择策略

在 Hopper (sm_90) 上，vLLM 对非 MoE linear 层的 W4A16 kernel 选择优先级：

```
CutlassW4A8 → MacheteLinearKernel (sm_90+, CUTLASS) → MarlinLinearKernel (fallback)
```

Machete 本质是 CUTLASS 3.x collective builder 构建的 SM90 mixed-type GEMM，与 example 55 底层相同。

TRT-LLM 的 fpAIntB 则自带 dispatch：M < 16 走 CUDA core GEMV，M >= 16 走 CUTLASS WGMMA。

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

# 4. fpAIntB (TRT-LLM extracted, 需要 CUTLASS headers)
# See https://github.com/DrXuQian/w4a16 for standalone extraction
cd fpA_intB_standalone/build
cmake .. -DCUTLASS_DIR=<CUTLASS> -DCMAKE_CUDA_ARCHITECTURES="90a-real"
make -j$(nproc)
```

## 运行

```bash
# Marlin W4A16 (standalone)
./marlin_standalone -m 3823 -n 12288 -k 3072 -g 128 -w 10 -i 100

# CUTLASS SM90 W4A16
./cutlass_w4a16_bench --m=3823 --n=12288 --k=3072 --g=128 --mode=1 --shuffle=true --iterations=100

# cuBLAS BF16/FP16
./cublas_bf16_bench 3823 12288 3072

# fpAIntB (TRT-LLM)
./test_fpA_intB_gemm --m=3823 --n=12288 --k=3072 --group_size=128 --warmup=10 --iters=100
```

## Benchmark 结果 (H800 PCIe)

All standalone CUDA, no Python overhead. groupsize=128.

### Prefill (M=3823, Qwen3.5-122B shapes)

| Shape (M×N×K) | fpAIntB SM90 (us) | Marlin (us) | cuBLAS BF16 (us) | CUTLASS ex55 (us) | best W4A16 vs BF16 |
|---------------|-------------------|-------------|------------------|-------------------|--------------------|
| 3823×12288×3072 | **738** | 1148 | 465 | 839 | 0.63x |
| 3823×8192×3072 | **482** | 763 | 309 | 549 | 0.64x |
| 3823×3072×8192 | **469** | 758 | 323 | 555 | 0.69x |
| 3823×16384×3072 | **1006** | 1604 | 620 | 1135 | 0.62x |
| 3823×512×3072 | **50** | 150 | 31 | 66 | 0.62x |

### Decode (M=1)

| Shape (M×N×K) | fpAIntB CUDA (us) | Marlin (us) | cuBLAS BF16 (us) | CUTLASS ex55 (us) | best W4A16 vs BF16 |
|---------------|-------------------|-------------|------------------|-------------------|--------------------|
| 1×12288×3072 | **10.0** | 15.0 | 48.6 | 38.4 | **4.86x** |
| 1×8192×3072 | **7.8** | 12.5 | 36.0 | 35.3 | **4.62x** |
| 1×3072×8192 | **9.1** | 15.9 | 39.9 | 79.5 | **4.38x** |
| 1×16384×3072 | **12.8** | 17.7 | 60.9 | 65.6 | **4.76x** |

### 4096x4096x4096

| Kernel | 延迟 (us) | TFLOPS | 利用率 | vs BF16 |
|--------|-----------|--------|--------|---------|
| BF16 cuBLAS | 238 | 578.3 | 76.5% | 1.00x |
| FP16 cuBLAS | 239 | 576.2 | 76.2% | 1.00x |
| CUTLASS SM90 W4A16 (shuffle ON) | 396 | 346.6 | 45.8% | 0.60x |
| Marlin W4A16 | 510 | 269.6 | 35.7% | 0.47x |
| CUTLASS SM90 W4A16 (shuffle OFF) | 524 | 262.1 | 34.7% | 0.45x |

### 分析

- **fpAIntB (TRT-LLM) 全场景最优 W4A16 kernel**
  - Decode (M=1): CUDA core GEMV 比 Marlin 快 35-45%，比 cuBLAS BF16 快 **4.4-4.9x**
  - Prefill (M=3823): SM90 WGMMA 比 Marlin 快 35-67%，比 CUTLASS ex55 快 10-25%
- **Prefill compute-bound**：所有 W4A16 kernel 仍慢于 cuBLAS BF16（dequant 开销）
- **Decode memory-bound**：INT4 权重体积减半，带宽优势直接转化为 4-5x 加速
- **Marlin 多次 launch**（M=3823 → 4 次），fpAIntB/cuBLAS 单次 launch
- **CUTLASS ex55 shuffle ON 提速 ~35%**（vs shuffle OFF），但仍不如 fpAIntB 的 tile heuristic

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

fpAIntB standalone 提取见 [DrXuQian/w4a16](https://github.com/DrXuQian/w4a16)
