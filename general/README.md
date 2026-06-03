# General kernels

Standalone benchmarks for kernels that are not specific to one model block.

## Kernels

| Bench | Kernel | Source | Notes |
|-------|--------|--------|-------|
| `bench_layernorm` | LayerNorm | OneFlow | Header-style CUDA extraction with a small benchmark harness |
| `bench_rmsnorm` | RMSNorm | TensorRT-LLM | Standalone extraction of the non-quantized `generalRmsNorm` path |
| `bench_cublas_gemm` | FP16/BF16 GEMM, dense FP8 baseline | cuBLAS | Generic row-major `C[M,N] = A[M,K] * B[K,N]` benchmark |
| `bench_cutlass_block_fp8_gemm` | W8A8 block-FP8 dense GEMM | vLLM/CUTLASS | Standalone SM90 extraction of `cutlass_scaled_mm_blockwise_sm90_fp8` |

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
general/bench_cublas_gemm --m=3823 --n=6144 --k=3072 --dtype=fp8 --out-dtype=fp16 --bench 0 1
general/bench_cublas_gemm --m=1 --n=248320 --k=3072 --dtype=fp16 --out-dtype=fp32 --bench 0 1
general/bench_cutlass_block_fp8_gemm --m=3823 --n=6144 --k=3072 --out-dtype=fp16 --bench 0 1
```

`bench_rmsnorm` supports `--dtype fp16|bf16|fp32`, `--eps`, `--beta`,
`--no-check`, and `--bench warmup iters`.

`bench_cublas_gemm` supports `--m`, `--n`, `--k`, `--dtype fp16|bf16|fp8`,
`--out-dtype same|fp16|bf16|fp32`, and `--bench warmup iters`.

The `fp8` mode is a dense E4M3 cuBLAS baseline. It is not a replacement for
block-wise FP8 weight-only kernels that require scale tensors.

`bench_cutlass_block_fp8_gemm` is the block-wise FP8 path used for MiniMax
dense projections. It follows vLLM's SM90 CUTLASS configuration:

- A and B are FP8 E4M3.
- A scales use block shape `(1, 128)`.
- B scales use block shape `(128, 128)`.
- Hopper `M % 4` padding is handled allocation-side, so only the GEMM kernel is
  timed by profilers.
- This is not MXFP8; MiniMax-M2.7 config uses ordinary block-wise FP8 E4M3
  (`weight_block_size=[128,128]`), not E8M0 microscaling.
