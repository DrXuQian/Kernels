Standalone vLLM Machete GEMM for SM90
=====================================

This directory contains a minimal, standalone extraction of vLLM's Machete
weight-only GEMM path. It keeps the Machete prepacked weight layout, SM90 TMA
warp-specialized CUTLASS kernel, and the H100/H800 heuristic schedule selector
from vLLM.

Source
------
- vLLM tree used for extraction:
  local vLLM checkout
- Source commit observed locally:
  `e3126cd107460444d7fd9a1445b8d4f4393a06b2`
- Main extracted files:
  - `csrc/quantization/machete/*`
  - `csrc/cutlass_extensions/*`

Scope
-----
Supported:

- SM90/SM90a Hopper builds
- FP16 and BF16 activations
- INT4 weights:
  - GPTQ-style `u4b8` with group scales
  - AWQ-style `u4` with group scales and group zeros
- vLLM Machete prepack kernel
- vLLM H100 heuristic schedule selection
- CUTLASS example-55 mode-1 scale-only backend for comparison

Not included:

- PyTorch/ATen bindings
- u8/u8b128 kernels
- channel scales or token scales
- SM80/SM100+ specializations
- CUTLASS example-55 mode 0 or mode 2 through the Machete test binary

Build
-----
From the repo root:

```
cmake -S w4a16_gemm/machete_standalone \
  -B w4a16_gemm/machete_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release

cmake --build w4a16_gemm/machete_standalone/build_cmake_release \
  --target test_machete_gemm -j$(nproc)
```

If the machine has multiple CUDA-compatible SDKs in `PATH`, configure CMake with the
intended compiler explicitly:

```
CUDA_ROOT=/path/to/CUDA_SDK
CUDACXX=$CUDA_ROOT/bin/nvcc \
cmake -S w4a16_gemm/machete_standalone \
  -B w4a16_gemm/machete_standalone/build_cmake_release \
  -DCUDAToolkit_ROOT=$CUDA_ROOT \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
```

Run
---
Default run, using heuristic schedule selection:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=gptq_u4b8 \
  --warmup=100 --iters=1000
```

CUTLASS example-55 backend inside the same test binary. This path supports
mode 1 only: signed INT4 weights with group scales and the example-55 value
shuffle layout. The default config is the original example-55
`128x128x64_1x1x1` tile/cluster shape.

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --backend=cutlass55 \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=cutlass_s4 \
  --warmup=100 --iters=1000
```

List CUTLASS55 configs:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --list_cutlass55_configs
```

Force one CUTLASS55 config:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --backend=cutlass55 \
  --cutlass55_config=256x128x64_1x1x1 \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --no_checksum --warmup=100 --iters=1000
```

Search all compiled CUTLASS55 configs sequentially:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --search_cutlass55_configs \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --no_checksum --warmup=100 --iters=1000
```

Save the best searched config into a tactic cache:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --search_cutlass55_configs \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --no_checksum --warmup=100 --iters=1000 \
  --save_cutlass55_tactic=w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache
```

Load a cached config directly:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --backend=cutlass55 \
  --cutlass55_tactic=w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --no_checksum --warmup=100 --iters=1000
```

The cache is exact-match text:

```
m,n,k,group,act|config=<cutlass55_config>,avg_us=<measured_time>
```

Batch-search a list of shapes and emit both a cache and a markdown table:

```
w4a16_gemm/machete_standalone/tools/search_cutlass55_tactics.sh \
  --bin w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --out-cache w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache \
  --out-md w4a16_gemm/machete_standalone/CUTLASS55_TACTICS_H800.md \
  --warmup 100 --iters 1000
```

For a custom shape file, each non-comment line is:

```
M N K GROUP ACT
```

AWQ-style u4:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=bf16 --quant=awq_u4 \
  --warmup=100 --iters=1000
```

List schedules:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm --list_schedules
```

Force a schedule:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --m=128 --n=4096 --k=4096 --group_size=128 \
  --schedule=128x32_2x1x1_TmaMI_TmaCoop_streamK
```

Save a Machete-prepacked weight file once:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=gptq_u4b8 \
  --save_prepacked=/tmp/machete_b_4096x4096_fp16_gptq_u4b8.bin \
  --warmup=1 --iters=1
```

Run later with the weight already prepacked offline. This skips the runtime GPU
prepack kernel and loads B from the saved file:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=gptq_u4b8 \
  --offline_prepack=/tmp/machete_b_4096x4096_fp16_gptq_u4b8.bin \
  --warmup=0 --iters=1
```

For performance-only profiling where the actual B contents do not matter, omit
the path. The test creates deterministic nonzero prepacked data in host code and
copies it to the device before timing. This does not run runtime prepack or file
IO, and avoids the unrealistic all-zero/unwritten-buffer timing path:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=gptq_u4b8 \
  --offline_prepack --no_checksum --warmup=0 --iters=1
```

For profiler captures focused only on the timed GEMM loop, use CUDA profiler
range capture:

```
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -f true -o /tmp/machete_gemm_only \
  w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
    --m=4096 --n=4096 --k=4096 --group_size=128 \
    --act=fp16 --quant=gptq_u4b8 \
    --offline_prepack \
    --profile_gemm_only --no_checksum --warmup=0 --iters=1
```

Summarize the captured GPU kernels:

```
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  /tmp/machete_gemm_only.nsys-rep
```

Run all schedules:

```
w4a16_gemm/machete_standalone/tools/run_all_schedules.sh \
  --bin w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --m 4096 --n 4096 --k 4096 --group 128 \
  --act fp16 --quant gptq_u4b8 \
  --warmup 100 --iters 1000 --out /tmp/machete_4096_all_schedules.txt
```

H800 SM90 performance: 4096x4096x4096
-------------------------------------
Measured on H800 PCIe with `build_cmake_release`, `group_size=128`,
`warmup=100`, and `iters=1000`. The reported time is CUDA event time around the
timed GEMM loop; Machete prepack is run before timing. Effective dense TFLOPS is
`2*M*N*K / time`.

Command:

```
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=gptq_u4b8 \
  --warmup=100 --iters=1000
```

| Activation | Quant | Selected schedule | Avg time (us) | Effective TFLOPS |
|---|---|---|---:|---:|
| FP16 | GPTQ `u4b8` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 354.820 | 387.3 |
| FP16 | AWQ `u4` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 370.910 | 370.5 |
| BF16 | GPTQ `u4b8` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 358.621 | 383.2 |
| BF16 | AWQ `u4` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 360.375 | 381.4 |

CUTLASS example 55 comparison
-----------------------------
For reference, CUTLASS example 55 FP16 mode 1 was measured on the same shape
with initialized inputs and offline reorder before timing:

```
w4a16_gemm/cutlass55_standalone/build_cmake_release/cutlass55_fp16_gemm \
  --m=4096 --n=4096 --k=4096 --g=128 \
  --mode=1 --shuffle=true --skip_verify \
  --warmup=100 --iterations=1000
```

| Kernel | Mode | Avg time (us) | Effective TFLOPS |
|---|---|---:|---:|
| example 55 FP16 | mode 1, group scale | 353.483 | 388.8 |

Machete and example 55 do not use the same kernel even when both have a
`128x128x64` threadblock tile:

- vLLM's heuristic selects `128x128_2x1x1_TmaMI_TmaCoop_streamK` for 4096^3,
  while example 55 is hard-coded to `TileShape=128x128x64` and
  `ClusterShape=1x1x1`.
- Machete uses `PrepackedLayoutBTemplate` and `MacheteCollectiveMma`, so the
  int4 weights are prepacked to match the WGMMA register-load pattern more
  directly.
- Example 55 uses CUTLASS's generic mixed-input `CollectiveBuilder` with
  `LayoutB_Reordered` and `ValueShuffle`. Its `--shuffle=true` path is an
  offline value reorder, but it is still not Machete's prepacked layout or
  custom mainloop.
- Machete launches the transposed formulation internally (`Y^T = W^T X^T`) to
  feed the decompressed weight operand through the GMMA register-sourced side.
  This changes the memory layout and schedule interpretation relative to the
  example 55 benchmark.

CUTLASS55 backend inside Machete test binary
--------------------------------------------
The `--backend=cutlass55` path embeds the example-55 mode-1 kernel in the
Machete standalone test binary without changing the original Machete backend.
It initializes the CUTLASS adapter once before timing; the timed loop only calls
`gemm.run()`.

Compiled config set:

| Config | Notes |
|---|---|
| `128x128x64_1x1x1` | CUTLASS example-55 default |
| `128x64x64_1x1x1` | Smaller M tile |
| `128x256x64_1x1x1` | Larger M tile |
| `256x128x64_1x1x1` | Larger N tile |
| `128x128x64_2x1x1` | Cluster-N 2 |
| `128x64x64_2x1x1` | Smaller M tile, Cluster-N 2 |
| `128x256x64_2x1x1` | Larger M tile, Cluster-N 2 |
| `256x128x64_2x1x1` | Larger N tile, Cluster-N 2 |

`64x*` as the first tile dimension is intentionally not included: the
cooperative SM90 kernel has a CUTLASS static assertion requiring tile M
`>= 128`.

H800 PCIe event-time measurements for
`4096x4096x4096`, FP16, group size 128, synthetic prepacked data,
`warmup=100`, `iters=1000`:

| Backend/config | Avg time (us) | Effective TFLOPS |
|---|---:|---:|
| Machete `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 356.764 | 385.2 |
| CUTLASS55 `128x128x64_1x1x1` | 345.916 | 397.3 |
| CUTLASS55 `128x64x64_1x1x1` | 462.896 | 296.9 |
| CUTLASS55 `128x256x64_1x1x1` | 331.355 | 414.8 |
| CUTLASS55 `256x128x64_1x1x1` | 312.123 | 440.3 |
| CUTLASS55 `128x128x64_2x1x1` | 349.728 | 393.0 |
| CUTLASS55 `128x64x64_2x1x1` | 462.586 | 297.1 |
| CUTLASS55 `128x256x64_2x1x1` | 317.692 | 432.6 |
| CUTLASS55 `256x128x64_2x1x1` | 315.717 | 435.3 |

`nsys` GPU-kernel summaries with `--profile_gemm_only --warmup=0 --iters=10`
show that the captured range contains only the target GEMM kernel and no GPU
memcpy/memset. These are kernel durations, not CUDA event elapsed time across
the host submission loop:

| Binary/backend | GPU kernels | Avg kernel time (us) |
|---|---:|---:|
| Machete backend | 10 | 298.246 |
| CUTLASS55 in Machete binary | 10 | 302.873 |
| CUTLASS55 standalone, initialized + reorder before timing | 10 | 300.853 |

Heuristic
---------
This is copied from vLLM `csrc/quantization/machete/generate.py`.
The condition order matters.

| Condition | Schedule |
|---|---|
| `M > 256 && K <= 16384 && N <= 4096` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 256` | `128x256_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 128 && K <= 4096 && N <= 4096` | `128x64_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 128 && K <= 8192 && N <= 8192` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 128` | `128x256_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 64 && K <= 4069 && N <= 4069` | `128x32_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 64 && K <= 4069 && N <= 8192` | `128x64_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 64 && K >= 8192 && N >= 12288` | `256x128_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 64` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 32 && K <= 6144 && N <= 6144` | `128x16_1x1x1_TmaMI_TmaCoop_streamK` |
| `M > 32 && K >= 16384 && N >= 12288` | `256x64_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 32` | `128x64_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 16 && K <= 12288 && N <= 8192` | `128x32_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 16` | `256x32_2x1x1_TmaMI_TmaCoop_streamK` |
| `N >= 26624` | `256x16_1x1x1_TmaMI_TmaCoop_streamK` |
| fallback | `128x16_1x1x1_TmaMI_TmaCoop_streamK` |

Supported schedule set:

```
128x128_2x1x1_TmaMI_TmaCoop_streamK
128x256_2x1x1_TmaMI_TmaCoop_streamK
128x64_2x1x1_TmaMI_TmaCoop_streamK
128x32_2x1x1_TmaMI_TmaCoop_streamK
256x128_2x1x1_TmaMI_TmaCoop_streamK
128x16_1x1x1_TmaMI_TmaCoop_streamK
256x64_2x1x1_TmaMI_TmaCoop_streamK
256x32_2x1x1_TmaMI_TmaCoop_streamK
256x16_1x1x1_TmaMI_TmaCoop_streamK
```

Smoke Test
----------
Verified locally on H800 PCIe, compute capability 9.0:

```
test_machete_gemm --m=128 --n=128 --k=128 --group_size=128 --warmup=1 --iters=1
```

The four combinations below launch successfully:

- `--act=fp16 --quant=gptq_u4b8`
- `--act=bf16 --quant=gptq_u4b8`
- `--act=fp16 --quant=awq_u4`
- `--act=bf16 --quant=awq_u4`

For `4096x4096x4096`, the heuristic selects
`128x128_2x1x1_TmaMI_TmaCoop_streamK`.
