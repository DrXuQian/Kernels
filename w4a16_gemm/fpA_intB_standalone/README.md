Standalone fpA_intB GEMM (FP16 x INT4, GPTQ) for SM80 + SM90
============================================================

This directory contains a minimal, standalone build of TensorRT-LLM's fpA_intB
(weight-only) GEMM. It supports:

- FP16 activations
- INT4 weights (GPTQ-style, scale + zero)
- CUTLASS fpA_intB path
- CUDA kernel fallback for small M (best performance for M=1)
- SM80 (Ampere) and SM90 (Hopper) CUTLASS paths (build-arch dependent)

What I extracted
----------------
- fpA_intB CUTLASS runner (`fpA_intB_gemm_template.h` + instantiation)
- CUTLASS config selection via TRT-LLM profiling-style timing
- weight-only CUDA kernel fallback (`weightOnlyBatchedGemv`)
- C++ weight preprocessing (`cutlass_preprocessors.*`)
- Standalone CLI test (`test_fpA_intB_gemm.cu`)

What is NOT included (limitations)
----------------------------------
- SM100/SM110/SM120 specialized CUTLASS paths
- BF16/FP8/FP4 or INT8 weights
- bias/alpha/act-scale fusion (GPTQ scale+zero only)
- non-groupwise quantization (group_size must be 64 or 128)

Config selection (aligned with TRT-LLM)
---------------------------------------
The selector uses TRT-LLM's profiling method:

- Generate candidate CUTLASS configs via `get_candidate_configs` (weight-only).
- Include the CUDA kernel as an extra "config" with `enableCudaKernel=true`.
- Time each candidate with CUDA events (warmup + runs) and pick the fastest.
- M<16 is where the CUDA fallback typically wins.

You can list configs or force a config via CLI.

Input/weight layout assumptions
-------------------------------
- A: FP16 row-major, shape [M, K].
- B: INT4 packed (2 values per byte), then preprocessed with
  `preprocess_weights_for_mixed_gemm(..., QuantType::W4_A16, force_interleave=false)`.
  (If you need to force the SM80 interleaved layout on Hopper, pass `force_interleave=true`.)
- Scales and zeros: FP16 row-major, shape [K / group_size, N].
- GPTQ zeros are passed as `zero_x_scale = (-qzeros + 7) * scale` (same as TRT-LLM).

Build
-----
From the repo root:

```
cmake -S w4a16_gemm/fpA_intB_standalone \
  -B w4a16_gemm/fpA_intB_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build w4a16_gemm/fpA_intB_standalone/build_cmake_release \
  --target test_fpA_intB_gemm -j$(nproc)
```

The output binary is
`w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm`.
Use `-DGPU_ARCH=sm_80` for an Ampere build.

Run
---
Example:

```
w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128
```

List configs:

```
w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --list_configs
```

Force CUDA kernel:

```
w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128 \
  --config=cuda
```

Force an SM90 TMA CUTLASS config:

```
w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=3823 --n=12288 --k=3072 --group_size=128 \
  --config=sm90:128x256x128:2x1x1
```

Force an SM80 CUTLASS config:

```
w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=2048 --k=3584 --group_size=128 \
  --config=sm80:128x128x64:3:7
```

Notes:
- `split_k` must divide K and keep `K / split_k` a multiple of 64 (weight-only fpA_intB requirement).
  For example `K=2048` cannot use `split_k=7` (not divisible).
- The old SM80 spelling `--config=128x128x64x3x7` is still accepted for compatibility, but new commands should use
  `--config=sm80:128x128x64:3:7`.

Interpreting `--list_configs` output
------------------------------------
`--list_configs` prints values that can be passed back directly to `--config=`:

```
0: cuda
1: sm90:64x16x128:1x1x1
2: sm90:128x256x128:2x1x1
```

On an SM80 build the CUTLASS entries use the same aligned spelling:

```
sm80:128x128x64:3:1
```

Debug profile logging (config search)
-------------------------------------
Set `FPA_INTB_PROFILE_LOG=1` to print each candidate config, its timing, and failure reasons:

```
FPA_INTB_PROFILE_LOG=1 \
w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128
```

Traverse all configs without search
-----------------------------------
Use the helper script to run every candidate configuration directly:

```
w4a16_gemm/fpA_intB_standalone/tools/run_all_configs.sh \
  --bin w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m 1 --n 4096 --k 4096 --group 128 --warmup 10 --iters 100 --out results.txt
```

Skip the CUDA fallback:

```
w4a16_gemm/fpA_intB_standalone/tools/run_all_configs.sh \
  --bin w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m 1 --n 4096 --k 4096 --group 128 --warmup 10 --iters 100 --skip-cuda
```

H800 SM90 all-config sweep: 4096x4096x4096
------------------------------------------
Measured on H800 PCIe with `build_cmake_release`, `group_size=128`, `warmup=100`, and `iters=1000`.
The reported time is the benchmark's CUDA event time around the timed GEMM loop. Effective dense TFLOPS is
`2*M*N*K / time`. The `cuda` fallback config is not included because it is only valid for small M and reports
`unsupported m` for M=4096.

Command:

```
w4a16_gemm/fpA_intB_standalone/tools/run_all_configs.sh \
  --bin w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m 4096 --n 4096 --k 4096 --group 128 \
  --warmup 100 --iters 1000 --skip-cuda --out /tmp/fpa_4096_all_configs.txt
```

| Index | Config | Avg time (us) | Effective TFLOPS |
|------:|--------|--------------:|-----------------:|
| 0 | `sm90:64x16x128:1x1x1` | 2051.252 | 67.0 |
| 1 | `sm90:64x32x128:1x1x1` | 1079.957 | 127.3 |
| 2 | `sm90:64x64x128:1x1x1` | 723.060 | 190.1 |
| 3 | `sm90:64x128x128:1x1x1` | 519.368 | 264.6 |
| 4 | `sm90:64x128x128:1x2x1` | 518.203 | 265.2 |
| 5 | `sm90:64x256x128:1x1x1` | 451.721 | 304.3 |
| 6 | `sm90:64x256x128:1x2x1` | 455.438 | 301.8 |
| 7 | `sm90:128x16x128:1x1x1` | 1439.797 | 95.5 |
| 8 | `sm90:128x16x128:2x1x1` | 1390.696 | 98.8 |
| 9 | `sm90:128x32x128:1x1x1` | 781.428 | 175.9 |
| 10 | `sm90:128x32x128:2x1x1` | 795.908 | 172.7 |
| 11 | `sm90:128x64x128:1x1x1` | 498.425 | 275.7 |
| 12 | `sm90:128x64x128:2x1x1` | 487.549 | 281.9 |
| 13 | `sm90:128x128x128:1x1x1` | 401.109 | 342.6 |
| 14 | `sm90:128x128x128:2x1x1` | 384.894 | 357.1 |
| 15 | `sm90:128x128x128:1x2x1` | 397.587 | 345.7 |
| 16 | `sm90:128x128x128:2x2x1` | 409.044 | 336.0 |
| 17 | `sm90:128x256x128:1x1x1` | 334.338 | 411.1 |
| 18 | `sm90:128x256x128:2x1x1` | **308.499** | **445.5** |
| 19 | `sm90:128x256x128:1x2x1` | 330.743 | 415.5 |
| 20 | `sm90:128x256x128:2x2x1` | 334.082 | 411.4 |

CPU reference (small shapes only):

```
w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=128 --k=128 --group_size=128 --verify
```

Notes
-----
- K must be a multiple of 64 (SM80 fpA_intB requirement) and group_size.
- The CUDA fallback only supports M in [1, 15].
- CPU reference is a simple sanity check (not a full correctness suite).

SM110/SM120 behavior
--------------------
This standalone extraction includes SM80 and SM90 CUTLASS fpA_intB kernel paths (plus the CUDA fallback).
On SM110/SM120 GPUs, the runner intentionally dispatches to the SM80 implementation (i.e. no SM100/SM110/SM120
specialized CUTLASS kernels are included here).
