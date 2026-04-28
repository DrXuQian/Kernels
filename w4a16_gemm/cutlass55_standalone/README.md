CUTLASS Example 55 Standalone
=============================

This directory contains a standalone copy of CUTLASS example 55 for Hopper
FP16/BF16 activation x INT4 weight GEMM. The kernel configuration is the
example-55 shape:

- `TileShape=128x128x64` for FP16/BF16
- `ClusterShape=1x1x1`
- `KernelTmaWarpSpecializedCooperative`
- `TmaWarpSpecializedCooperative` epilogue
- optional `LayoutB_Reordered` value shuffle

The wrapper adds profiler-friendly controls to skip setup/reference kernels and
to launch exactly one target GEMM kernel.

Build
-----

```
cmake -S w4a16_gemm/cutlass55_standalone \
  -B w4a16_gemm/cutlass55_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release

cmake --build w4a16_gemm/cutlass55_standalone/build_cmake_release \
  --target cutlass55_fp16_gemm cutlass55_bf16_gemm -j$(nproc)
```

Direct `nvcc` build for FP16:

```
CUTLASS_DIR=$PWD/third_party/cutlass
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
  -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
  -DCUTLASS_USE_FP16=1 \
  -Iw4a16_gemm/cutlass55_standalone \
  -I$CUTLASS_DIR/include \
  -I$CUTLASS_DIR/tools/util/include \
  -I$CUTLASS_DIR/examples/common \
  -I$CUTLASS_DIR/examples/55_hopper_mixed_dtype_gemm \
  w4a16_gemm/cutlass55_standalone/cutlass55_gemm.cu \
  -o /tmp/cutlass55_fp16_gemm
```

Run
---

Steady-state GEMM-only timing. This skips the GPU random fill, dequantize, and
reorder setup kernels, then initializes A/B/C/scales with deterministic host
synthetic data before timing. It also skips the correctness warmup GEMM,
reference GEMM, and comparison kernel:

```
w4a16_gemm/cutlass55_standalone/build_cmake_release/cutlass55_fp16_gemm \
  --m=4096 --n=4096 --k=4096 --g=128 \
  --mode=1 --shuffle=true \
  --skip_setup_kernels --skip_verify \
  --warmup=100 --iterations=1000
```

The skipped-setup modes are for profiler isolation. Output values are not
meaningful because B is synthetic bytes in the layout consumed by the selected
kernel path.

Launch exactly one target GEMM kernel:

```
w4a16_gemm/cutlass55_standalone/build_cmake_release/cutlass55_fp16_gemm \
  --m=4096 --n=4096 --k=4096 --g=128 \
  --mode=1 --shuffle=true \
  --single_kernel --profile_gemm_only
```

`--single_kernel` implies `--skip_setup_kernels`, `--skip_verify`,
`--warmup=0`, and `--iterations=1`.

H800 Validation
---------------
FP16 mode 1, `4096x4096x4096`, group size 128, `shuffle=true`,
`warmup=100`, `iterations=1000`:

| Setup path | Avg time (us) | Effective TFLOPS |
|---|---:|---:|
| `--skip_setup_kernels --skip_verify` | 351.389 | 391.1 |
| initialized + reorder, `--skip_verify` | 345.001 | 398.4 |

`nsys` capture with `--skip_setup_kernels --skip_verify --profile_gemm_only
--warmup=0 --iterations=10` contains only the target GEMM kernel in the capture
range. The kernel average was 296.305 us, with no GPU memory operations reported
inside the captured range.
