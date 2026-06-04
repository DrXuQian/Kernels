# Sampling

Standalone sampling-stage benchmarks for Qwen3.5 decode.

## Kernels

| Bench op | Logical stage | Shape |
|---|---|---|
| `lm_head` | TMA warp-specialized GEMV via `studies/lm_head_gemv_bw/bench_lm_head_gemv` | `(1,3072) x (248320,3072)^T -> (1,248320)` |
| `topk_mask` | FlashInfer radix Top-K logits mask (`RadixTopKMaskLogitsMultiCTA`) | `(1,248320) -> (1,248320)` |
| `softmax` | FlashInfer online softmax (`OnlineSoftmax`) | `(1,248320) -> (1,248320)` |
| `top_p` | FlashInfer top-p sampling from probabilities (`TopPSamplingFromProb`) | `(1,248320) -> (1,)` |

The model benchmark lm-head case uses the local TMA GEMV study kernel:
`studies/lm_head_gemv_bw/bench_lm_head_gemv --op=ptx_tma_ws --k-unroll=8`.
`bench_sampling` still includes a direct `--op=lm_head` cuBLAS comparison for
local sampling-only runs. Sampling stages call FlashInfer CUDA header kernels vendored under
`sampling/third_party/flashinfer/include`, so
building this folder does not require installing the FlashInfer Python package.
The benchmark keeps only the single-row profiling setup; it does not model
higher-level request state, finished flags, or tokenizer/runtime control flow.

## Build

```bash
# Builds both sampling/bench_sampling and the lm_head TMA GEMV study binary:
./compile.sh build sampling

# Or build the two parts directly:
make -C sampling
make -C studies/lm_head_gemv_bw
```

## Run

```bash
studies/lm_head_gemv_bw/bench_lm_head_gemv --op=ptx_tma_ws --n 248320 --k 3072 --k-unroll=8 --warmup=0 --iters=1 --no-verify
sampling/bench_sampling --op=topk_mask --vocab=248320 --top-k=50 --bench 0 1
sampling/bench_sampling --op=softmax --vocab=248320 --bench 0 1
sampling/bench_sampling --op=top_p --vocab=248320 --top-k=50 --top-p=0.9 --bench 0 1
```

Use `--op=all` to run all four stages in sequence.
