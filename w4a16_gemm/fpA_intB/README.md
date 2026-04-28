# W4A16 standalone kernels

This directory collects standalone W4A16 GEMM extractions (FP16/BF16
activations, INT4 weights). It is checked into the main `Kernels` repo; only
`../../third_party/cutlass` remains a git submodule.

Included projects
-----------------
- `fpA_intB_standalone/`: TensorRT-LLM extracted fpA_intB dense GEMM
  (FP16 x INT4, GPTQ scale+zero)
  - CUTLASS path + CUDA fallback for small M (e.g. M=1)
  - TRT-LLM-style config profiling (`--list_configs`, `--config=...`, `FPA_INTB_PROFILE_LOG=1`)
  - Tactic caching (`--tactic=<file>`) to skip profiling on repeated runs
- `moe_w4a16_standalone/`: TensorRT-LLM extracted MoE W4A16 grouped GEMM
  - FP16/BF16 activations with INT4 weights
  - Standalone tactic caching for Qwen MoE shapes
- `machete_standalone/`: vLLM Machete standalone GEMM
  - vLLM schedule heuristics and explicit schedule forcing
  - additional CUTLASS example 55 backend and tactic cache helper
- `cutlass55_standalone/`: standalone CUTLASS example 55 kernel
  - FP16/BF16 activations with INT4 weights on SM90/SM90a
  - `--single_kernel` path to launch exactly one target GEMM kernel

## Build

From `w4a16_gemm/fpA_intB`:

```bash
cmake -S fpA_intB_standalone -B fpA_intB_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/../../third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build fpA_intB_standalone/build_cmake_release \
  --target test_fpA_intB_gemm -j$(nproc)

cmake -S moe_w4a16_standalone -B moe_w4a16_standalone/build_cmake_release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCUTLASS_DIR=$PWD/../../third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build moe_w4a16_standalone/build_cmake_release \
  --target test_moe_w4a16_gemm -j$(nproc)

cmake -S cutlass55_standalone -B cutlass55_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/../../third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cutlass55_standalone/build_cmake_release \
  --target cutlass55_fp16_gemm cutlass55_bf16_gemm -j$(nproc)
```

## Usage

```bash
# Basic run (profiles all configs on first call, slow ~5s)
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128

# With tactic cache (recommended):
#   First run: profiles and saves best config to file
#   Subsequent runs with same (m,n,k,gs): loads directly, no profiling
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128 \
    --tactic=tactics.cache --warmup=10 --iters=100

# Force a specific config (skip profiling entirely)
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128 --config=cuda
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m=3823 --n=12288 --k=3072 --group_size=128 --config=sm90:128x256x128:2x1x1
# On an SM80 build:
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m=1 --n=2048 --k=3584 --group_size=128 --config=sm80:128x128x64:3:7

# List all available configs
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --list_configs

# Debug profiling (print per-config timing)
FPA_INTB_PROFILE_LOG=1 fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m=3823 --n=4096 --k=4096 --group_size=128

# Correctness verification (small shapes only)
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m=2 --n=128 --k=128 --group_size=128 --verify

# Single inference, no profiling, no warmup (GPU only runs the GEMM kernel)
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128 \
    --tactic=tactics.cache --warmup=0 --iters=1

# CUTLASS example 55 standalone: launch exactly one target GEMM kernel
cutlass55_standalone/build_cmake_release/cutlass55_fp16_gemm \
    --m=4096 --n=4096 --k=4096 --g=128 \
    --mode=1 --shuffle=true --single_kernel --profile_gemm_only
```

### Tactic cache

The `--tactic=<file>` option saves/loads kernel configs to skip online profiling:

- **First run** with a new (m,n,k,gs): profiles all candidate configs (~5s), saves best to file
- **Subsequent runs** with same shape: loads config directly, zero overhead
- Pre-profiled cache for H800 PCIe included: `tactics_h800.cache`

File format (portable across builds, human-readable):

```
3823,12288,3072,128|cuda=0,tma=1,sm=90,tile90=128256128,ml=0,el=0,cl=2001001
1,12288,3072,128|cuda=1
```

Fields: `m,n,k,gs|` followed by config key-value pairs:
- `cuda=1`: CUDA core GEMV path (M < 16)
- `tma=1,sm=90,tile90=...,ml=...,el=...,cl=...`: SM90 TMA WGMMA path
- `tma=0,tile80=...,stages=...,splitk=...`: SM80 CUTLASS path

Delete the cache file to force re-profiling (e.g. when switching GPU arch).

## Notes

- Build directories are intentionally ignored (see `.gitignore`).
- `fpA_intB_standalone` uses `-DGPU_ARCH=sm_90a` for Hopper TMA kernels.
- `moe_w4a16_standalone` uses `-DCMAKE_CUDA_ARCHITECTURES=90` on Hopper because
  this extraction uses the SM80 grouped GEMM fallback. Use `80` for Ampere.
