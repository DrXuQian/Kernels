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

## 编译

```bash
# From Kernels/w4a16_gemm: initialize the CUTLASS submodule kept by the parent repo.
git -C .. submodule update --init third_party/cutlass

# 1. Marlin (standalone, 无需 PyTorch)
nvcc -O2 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
  -diag-suppress 177,179,39 marlin_standalone.cu -o marlin_standalone

# 2. CUTLASS SM90
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
  -I../third_party/cutlass/include -I../third_party/cutlass/tools/util/include \
  -I../third_party/cutlass/examples/common \
  -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
  ../third_party/cutlass/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu \
  -o cutlass_w4a16_bench -lcublas

# 3. cuBLAS BF16/FP16
nvcc -O3 -std=c++17 -arch=sm_90 cublas_bf16_bench.cu -o cublas_bf16_bench -lcublas

# 4. fpAIntB (TRT-LLM extracted)
cmake -S fpA_intB/fpA_intB_standalone \
  -B fpA_intB/fpA_intB_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/../third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build fpA_intB/fpA_intB_standalone/build_cmake_release \
  --target test_fpA_intB_gemm -j$(nproc)
```

## 运行

```bash
# Marlin W4A16
./marlin_standalone -m 3823 -n 12288 -k 3072 -g 128 -w 10 -i 100

# CUTLASS SM90 W4A16
./cutlass_w4a16_bench --m=3823 --n=12288 --k=3072 --g=128 --mode=1 --shuffle=true --iterations=100

# cuBLAS BF16/FP16
./cublas_bf16_bench 3823 12288 3072

# fpAIntB (with tactic cache to skip online profiling)
fpA_intB/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
    --m=3823 --n=12288 --k=3072 --group_size=128 \
    --tactic=fpA_intB/fpA_intB_standalone/tactics_h800.cache --warmup=10 --iters=100

# Single inference (no profiling, no warmup)
fpA_intB/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
    --m=1 --n=12288 --k=3072 --group_size=128 \
    --tactic=fpA_intB/fpA_intB_standalone/tactics_h800.cache --warmup=0 --iters=1
```

fpAIntB 详细用法见 [fpA_intB/README.md](fpA_intB/README.md)。

## Benchmark 结果 (H800 PCIe)

All standalone CUDA, no Python overhead. groupsize=128.

### Prefill (M=3823, Qwen3.5-122B shapes)

| Shape (M×N×K) | fpAIntB SM90 (us) | Marlin (us) | cuBLAS BF16 (us) | CUTLASS ex55 (us) | best W4A16 vs BF16 |
|---------------|-------------------|-------------|------------------|-------------------|--------------------|
| 3823×12288×3072 | **646** | 1148 | 465 | 839 | 0.72x |
| 3823×8192×3072 | **439** | 763 | 309 | 549 | 0.70x |
| 3823×3072×8192 | **425** | 758 | 323 | 555 | 0.76x |
| 3823×16384×3072 | **894** | 1604 | 620 | 1135 | 0.69x |
| 3823×512×3072 | **39** | 150 | 31 | 66 | 0.79x |

### Decode (M=1)

| Shape (M×N×K) | fpAIntB CUDA (us) | Marlin (us) | cuBLAS BF16 (us) | CUTLASS ex55 (us) | best W4A16 vs BF16 |
|---------------|-------------------|-------------|------------------|-------------------|--------------------|
| 1×12288×3072 | **9.5** | 15.0 | 48.6 | 38.4 | **5.13x** |
| 1×8192×3072 | **7.7** | 12.5 | 36.0 | 35.3 | **4.67x** |
| 1×3072×8192 | **8.7** | 15.9 | 39.9 | 79.5 | **4.60x** |
| 1×16384×3072 | **12.6** | 17.7 | 60.9 | 65.6 | **4.84x** |

### 4096x4096x4096

| Kernel | 延迟 (us) | TFLOPS | 利用率 | vs BF16 |
|--------|-----------|--------|--------|---------|
| BF16 cuBLAS | 238 | 578.3 | 76.5% | 1.00x |
| FP16 cuBLAS | 239 | 576.2 | 76.2% | 1.00x |
| **fpAIntB SM90** | **308** | **445.5** | **58.9%** | **0.77x** |
| CUTLASS SM90 W4A16 (shuffle ON) | 396 | 346.6 | 45.8% | 0.60x |
| Marlin W4A16 | 510 | 269.6 | 35.7% | 0.47x |
| CUTLASS SM90 W4A16 (shuffle OFF) | 524 | 262.1 | 34.7% | 0.45x |

### 分析

- **fpAIntB (TRT-LLM) 全场景最优 W4A16 kernel**
  - Decode (M=1): CUDA core GEMV 比 Marlin 快 30-45%，比 cuBLAS BF16 快 **4.6-5.1x**
  - Prefill (M=3823): SM90 WGMMA 比 Marlin 快 40-74%，比 CUTLASS ex55 快 20-40%
- **Prefill compute-bound**：所有 W4A16 kernel 仍慢于 cuBLAS BF16（dequant 开销）
- **Decode memory-bound**：INT4 权重体积减半，带宽优势直接转化为 4-5x 加速
- **Marlin 多次 launch**（M=3823 → 4 次），fpAIntB/cuBLAS 单次 launch
- **CUTLASS ex55 shuffle ON 提速 ~35%**（vs shuffle OFF），但仍不如 fpAIntB 的 tile heuristic
- fpAIntB 4096x4096x4096 的完整 SM90 config sweep 见
  [fpA_intB/fpA_intB_standalone/README.md](fpA_intB/fpA_intB_standalone/README.md)。

## 文件结构

```
w4a16_gemm/
├── README.md                 # 本文件（benchmark 入口）
├── marlin_standalone.cu      # Marlin W4A16 standalone (含正确性验证)
├── cublas_bf16_bench.cu      # cuBLAS BF16/FP16 standalone benchmark
├── fpA_intB/                 # extracted standalone W4A16 GEMM projects
│   ├── fpA_intB_standalone/  # TensorRT-LLM dense fpA_intB
│   ├── moe_w4a16_standalone/ # TensorRT-LLM MoE grouped GEMM
│   ├── machete_standalone/   # vLLM Machete + CUTLASS55 backend
│   └── cutlass55_standalone/ # CUTLASS example 55 standalone
└── kernels/
    └── marlin/
        ├── marlin_cuda_kernel.cu
        ├── marlin_cuda.cpp
        └── setup.py
```
