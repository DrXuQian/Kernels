# PPU finalizeMoeRoutingKernel Study

This folder is an isolated source-level compensation study for TensorRT-LLM's
`finalizeMoeRoutingKernel`. It is intentionally not wired into the repository
root `compile.sh` or `bench_all.sh`.

The study binary contains two implementations:

- `baseline`: direct standalone copy of the current auxiliary finalize kernel.
- `optimized`: source-level compensation for compiler gaps:
  - hoist per `(row, k)` scale/base pointer metadata into shared memory
  - fold invalid expert checks into `scale = 0`
  - four-column vectorized load/store for aligned hidden sizes
  - compile-time top-k unroll for the common `topk <= 8` path
  - separated preload and compute phases inside the vectorized loop

## Build

```bash
cd studies/ppu_finalize_moe_routing
make clean
make
```

For SDKs that require explicit CUDA/CCCL include paths:

```bash
make clean
make NVCC="$CUDA_ROOT/bin/nvcc" \
  ARCH=-arch=sm_90a \
  CUDA_CCCL_INC="-I$CUDA_ROOT/targets/x86_64-linux/include/cccl"
```

## Run

Default shape mirrors the TRT-LLM auxiliary standalone default:

```bash
./bench_finalize_moe_routing_study
./bench_finalize_moe_routing_study 3823 8 1024 fp16 --bench 20 100
```

Compare both implementations:

```bash
./bench_finalize_moe_routing_study 3823 8 1024 fp16 --mode both --bench 20 100
```

Run one implementation only:

```bash
./bench_finalize_moe_routing_study 3823 8 1024 fp16 --mode baseline --bench 20 100
./bench_finalize_moe_routing_study 3823 8 1024 fp16 --mode optimized --bench 20 100
```

Other options:

```text
--no-scales       Use SCALE=false template path.
--no-check        Skip baseline-vs-optimized correctness check.
--mode MODE       baseline, optimized, or both.
```

## Profiling

```bash
nsys profile --trace=cuda -o finalize_study \
  ./bench_finalize_moe_routing_study 3823 8 1024 fp16 --mode optimized --bench 20 100
nsys stats finalize_study.nsys-rep --report cuda_gpu_trace
```

## H800 sanity results

Measured on H800 with CUDA event timing. These numbers are only a local sanity
check before running the same study on PPU.

| Shape / mode | Baseline median | Optimized median | Speedup |
|---|---:|---:|---:|
| `tokens=1 topk=8 hidden=1024 fp16 scales=1` | 0.0133 ms | 0.0061 ms | 2.18x |
| `tokens=3823 topk=8 hidden=1024 fp16 scales=1` | 0.1104 ms | 0.0473 ms | 2.33x |
| `tokens=3823 topk=8 hidden=1024 fp16 scales=0` | 0.0865 ms | 0.0470 ms | 1.84x |
| `tokens=128 topk=8 hidden=1024 bf16 scales=1` | 0.0152 ms | 0.0067 ms | 2.27x |

Correctness checks passed with `max_abs=0` for the rows above. Additional
fallback coverage also passed for `tokens=7 topk=10 hidden=1025 fp16`, which
exercises both `topk > 8` and non-4-divisible hidden size paths.
