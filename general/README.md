# General kernels

Standalone benchmarks for kernels that are not specific to one model block.

## Kernels

| Bench | Kernel | Source | Notes |
|-------|--------|--------|-------|
| `bench_layernorm` | LayerNorm | OneFlow | Header-style CUDA extraction with a small benchmark harness |

## Build

```bash
make -C general

# PPU/H800 example
make -C general clean
make -C general bench_layernorm CUDA_ROOT=$CUDA_ROOT ARCH=-arch=sm_90a
```

## Run

```bash
general/bench_layernorm --batch 13824 --embed 1152 --dtype float16
general/bench_layernorm --batch 13824 --embed 1152 --dtype float32
```
