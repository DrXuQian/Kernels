# Marlin standalone

Standalone Marlin W4A16 GEMM benchmarks.

## Build

```bash
make -C w4a16_gemm/marlin_standalone
```

## Run

```bash
w4a16_gemm/marlin_standalone/marlin_standalone \
  -m 3823 -n 12288 -k 3072 -g 128 -w 10 -i 100
```

`bench_marlin.py` and `kernels/marlin/` are the older PyTorch extension bench
path kept for reference; the standalone CUDA binary above is the preferred
path for repo-local benchmarking.
