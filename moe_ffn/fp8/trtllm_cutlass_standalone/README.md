# TRT-LLM CUTLASS FP8 Block-Scale MoE GEMM

This directory is the standalone extraction track for MiniMax-style FP8 MoE
GEMM. It does not depend on FlashInfer, Python, TVM, or TensorRT plugins at
runtime.

Source lineage:

- `cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/`:
  TensorRT-LLM CUTLASS block-FP8 GEMM runner, vendored from the TRT-LLM kernel
  layer shipped inside the installed FlashInfer package.
- `cpp/tensorrt_llm/deep_gemm/`: header-only/JIT helper dependency used by
  the same TRT-LLM block-FP8 runner. The benchmark disables DeepGEMM JIT by
  default with `TRTLLM_DG_ENABLED=0`, so the timed kernel is the static CUTLASS
  grouped-with-offset fallback unless `--deep-gemm` is passed.
- Common TensorRT-LLM compatibility headers are reused from
  `moe_ffn/w4a16/trtllm/moe_w4a16_standalone/cpp`.

The benchmark covers only the grouped MoE GEMM call:

```text
A: grouped FP8 rows, [experts * m_per_expert, K]
B: FP8 expert weights, [experts, N, K]
D: BF16 grouped output, [experts * m_per_expert, N]
scale_a: 1x128 activation scales
scale_b: 128x128 weight block scales
```

Routing, expand, activation, and finalize remain separate auxiliary kernels in
the model wrappers. This is intentional: the first standalone target is the
actual block-FP8 grouped GEMM kernel with no FlashInfer binding dependency.

## Build

```bash
./compile.sh build moe-fp8-trtllm
```

Manual CMake equivalent:

```bash
cmake -S moe_ffn/fp8/trtllm_cutlass_standalone \
  -B moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/third_party/cutlass

cmake --build moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release \
  --target bench_moe_fp8_blockscale_gemm -j
```

`compile.sh` also injects the CUDA CCCL include path when `CUDA_ROOT` is set,
which avoids `cuda/std/utility` include issues on SDK layouts that keep CCCL
under `targets/x86_64-linux/include/cccl`.

## Run

MiniMax-M2.7 TP=1 gate/up:

```bash
moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release/bench_moe_fp8_blockscale_gemm \
  --experts=8 --m_per_expert=3823 --n=3072 --k=3072 --bench 0 1
```

MiniMax-M2.7 TP=1 down:

```bash
moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release/bench_moe_fp8_blockscale_gemm \
  --experts=8 --m_per_expert=3823 --n=3072 --k=1536 --bench 0 1
```

Decode-one-token variants use `--m_per_expert=1`.

Load a serialized static-tile tactic:

```bash
moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release/bench_moe_fp8_blockscale_gemm \
  --experts=8 --m_per_expert=1 --n=3072 --k=3072 \
  --tactic=moe_ffn/fp8/trtllm_cutlass_standalone/tactics_h800_minimax.cache \
  --bench 0 1
```

Search the small static-TMA tile space and append the best config to a tactic
cache:

```bash
moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release/bench_moe_fp8_blockscale_gemm \
  --experts=8 --m_per_expert=1 --n=3072 --k=3072 \
  --sweep_configs --tactic=/tmp/fp8_moe.cache --bench 0 3
```

For bandwidth utilization, run the binary under NCU on H800 with counter
permissions. Local `nsys` is still useful for kernel latency and launch-count
checks, but it cannot prove DRAM bandwidth utilization.

## Current Fixed Config

The static CUTLASS fallback in this extraction uses:

```text
tile = 128x64x128
activation scale = 1x128
weight scale = 128x128
output = BF16
layout A = row-major
layout B = column-major
layout D = row-major
```

This matches the fixed block-FP8 config selected in the FlashInfer/TRT-LLM
DeepSeek block-scale binding path before any optional DeepGEMM JIT override.

The standalone static fallback now exposes a small tactic space:

```text
tile_m = 64 or 128
tile_n = 64 or 128
tile_k = 128
```

`--tactic` loads entries with the same cache format used by
`tactics_h800_minimax.cache`; `--sweep_configs` measures the four static
configs and can append the best entry. DeepGEMM remains a separate optional
path selected only by `--deep-gemm`; it is not part of this static tactic
search.

Without an explicit `--tile_*` or tactic entry, the static fallback uses the
same default heuristic as the recorded best MiniMax path:

- decode-like `m_per_expert <= 64`: `64x128x128`
- prefill-like `m_per_expert > 64`: `128x64x128`

## TP1 Decode Static-Tile Result

On the local H800 PCIe, using nsys kernel duration and the PCIe H800 2.0 TB/s
bandwidth reference:

| MiniMax-M2.7 TP1 decode GEMM | old tile | new tile | old nsys latency | new nsys latency | effective bandwidth |
|---|---|---|---:|---:|---:|
| gate/up `(experts=8,m=1,n=3072,k=3072)` | `128x64x128` | `64x128x128` | 60.992 us | 50.879 us | 1.486 TB/s |
| down `(experts=8,m=1,n=3072,k=1536)` | `128x64x128` | `64x128x128` | 30.783 us | 20.352 us | 1.859 TB/s |

For TP1 prefill (`m_per_expert=3823`), the sweep kept the original
`128x64x128` tile for both gate/up and down.

An extra decode-only study tried adding `tile_n=32` to the grouped-GEMM static
search space. It was slower for both TP1 decode shapes on the same H800:

| candidate | gate/up median | down median |
|---|---:|---:|
| `64x32x128` | 66.4 us | 40.9 us |
| `64x128x128` | 51.8 us | 24.2 us |

So `tile_n=32` is intentionally omitted from the committed tactic search space.
