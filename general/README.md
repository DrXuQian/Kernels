# General kernels

Standalone benchmarks for kernels that are not specific to one model block.

## Kernels

| Bench | Kernel | Source | Notes |
|-------|--------|--------|-------|
| `bench_layernorm` | LayerNorm | OneFlow | Header-style CUDA extraction with a small benchmark harness |
| `bench_rmsnorm` | RMSNorm | TensorRT-LLM | Standalone extraction of the non-quantized `generalRmsNorm` path |
| `bench_cublas_gemm` | FP16/BF16 GEMM | cuBLAS | Generic row-major `C[M,N] = A[M,K] * B[K,N]` benchmark |

## Build

```bash
make -C general

# Or from the repo root:
./compile.sh build general
```

## Run

```bash
general/bench_layernorm --batch 13824 --embed 1152 --dtype float16
general/bench_layernorm --batch 13824 --embed 1152 --dtype float32
general/bench_rmsnorm --batch 1 --embed 3072 --dtype fp16 --bench 20 100
general/bench_rmsnorm --batch 3823 --embed 3072 --dtype bf16 --no-check --bench 0 1
general/bench_cublas_gemm --m=3823 --n=64 --k=3072 --dtype=fp16 --bench 0 1
general/bench_cublas_gemm --m=1 --n=248320 --k=3072 --dtype=fp16 --out-dtype=fp32 --bench 0 1
```

`bench_rmsnorm` supports `--dtype fp16|bf16|fp32`, `--eps`, `--beta`,
`--no-check`, and `--bench warmup iters`.

`bench_cublas_gemm` supports `--m`, `--n`, `--k`, `--dtype fp16|bf16`,
`--out-dtype same|fp16|bf16|fp32`, and `--bench warmup iters`.
