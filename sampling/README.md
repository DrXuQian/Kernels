# Sampling

Standalone sampling-stage benchmarks for Qwen3.5 decode.

## Kernels

| Bench op | Logical stage | Shape |
|---|---|---|
| `lm_head` | FP16 GEMM via cuBLAS `cublasGemmEx` | `(1,3072) x (3072,248320) -> (1,248320)` |
| `topk_mask` | FlashInfer radix Top-K logits mask (`RadixTopKMaskLogitsMultiCTA`) | `(1,248320) -> (1,248320)` |
| `softmax` | FlashInfer online softmax (`OnlineSoftmax`) | `(1,248320) -> (1,248320)` |
| `top_p` | FlashInfer top-p sampling from probabilities (`TopPSamplingFromProb`) | `(1,248320) -> (1,)` |

FP16 GEMM stages use cuBLAS. Sampling stages call FlashInfer CUDA header kernels
vendored under `sampling/third_party/flashinfer/include`, so building this
folder does not require installing the FlashInfer Python package. The benchmark
keeps only the single-row profiling setup; it does not model higher-level
request state, finished flags, or tokenizer/runtime control flow.

## Build

```bash
make -C sampling

# Or from the repo root:
./compile.sh build sampling
```

## Run

```bash
sampling/bench_sampling --op=lm_head --hidden=3072 --vocab=248320 --bench 0 1
sampling/bench_sampling --op=topk_mask --vocab=248320 --top-k=50 --bench 0 1
sampling/bench_sampling --op=softmax --vocab=248320 --bench 0 1
sampling/bench_sampling --op=top_p --vocab=248320 --top-k=50 --top-p=0.9 --bench 0 1
```

Use `--op=all` to run all four stages in sequence.
