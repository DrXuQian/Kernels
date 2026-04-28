# TensorRT-LLM vs vLLM MoE W4A16 Breakdown

This note compares the standalone MoE W4A16 paths currently extracted in this
repo. Timings are Nsight Systems GPU-kernel summaries from single-iteration
runs (`warmup=0`, `iters=1`), so they exclude process launch and most CUDA event
overhead.

## Status

The vLLM path is a complete standalone component pipeline:

```
topk_gating -> moe_align -> Marlin gate/up GEMM -> silu_and_mul -> Marlin down GEMM -> moe_sum
```

The TensorRT-LLM path is not a complete end-to-end standalone MoE pipeline yet.
The extracted pieces currently cover:

```
custom_moe_routing -> moe_align -> CUTLASS grouped gate/up GEMM -> CUTLASS grouped down GEMM
```

The TRT-LLM `expandInputRows`, gated activation, and `finalizeMoeRouting`
helpers from `cutlass_kernels/moe_gemm/moe_kernels.cu` are not extracted yet.
Therefore the TRT-LLM totals below are "available component" lower bounds, not
full production end-to-end times.

## Shapes

Qwen3.5-122B-A10B MoE settings used here:

| Item | Value |
|------|-------|
| prefill tokens | 3823 |
| decode tokens | 1 |
| experts | 64 for routing/align, 8 active experts for grouped GEMM bench |
| topk | 8 |
| group size | 128 |
| gate/up GEMM | K=2048, N=3072 |
| down GEMM | K=3072, N=1024 |
| activation dtype | FP16 |
| weight dtype | INT4 |

The TRT-LLM GEMM harness takes already grouped rows per expert. The routing and
align kernels are measured as standalone kernels and are not wired into the GEMM
input buffers in the benchmark.

## Decode

| Path | Stage | GPU launches | Time (us) |
|------|-------|--------------|-----------|
| vLLM | topk_gating | 1 | 4.416 |
| vLLM | moe_align | 1 | 10.016 |
| vLLM | Marlin gate/up GEMM | 1 | 21.055 |
| vLLM | silu_and_mul | 1 | 2.560 |
| vLLM | Marlin down GEMM | 1 | 14.719 |
| vLLM | moe_sum | 1 | 1.728 |
| vLLM | total | 6 | 54.494 |
| TRT-LLM | custom_moe_routing | 1 | 2.304 |
| TRT-LLM | moe_align | 1 | 10.271 |
| TRT-LLM | CUTLASS grouped gate/up GEMM | 1 | 20.799 |
| TRT-LLM | CUTLASS grouped down GEMM | 1 | 27.070 |
| TRT-LLM | available component total | 4 | 60.444 |

Decode GEMM-only:

| Path | gate/up (us) | down (us) | total (us) |
|------|--------------|-----------|------------|
| vLLM Marlin | 21.055 | 14.719 | 35.774 |
| TRT-LLM CUTLASS grouped GEMM | 20.799 | 27.070 | 47.869 |

For decode, vLLM is currently better in the down projection. The TRT-LLM
available total is also missing activation and final reduction, so it should not
be read as a complete TRT-LLM MoE decode time.

## Prefill

| Path | Stage | GPU launches | Time (us) |
|------|-------|--------------|-----------|
| vLLM | topk_gating | 1 | 8.063 |
| vLLM | moe_align | 1 | 367.590 |
| vLLM | Marlin gate/up GEMM | 1 | 2862.930 |
| vLLM | silu_and_mul | 1 | 411.170 |
| vLLM | Marlin down GEMM | 1 | 1418.235 |
| vLLM | moe_sum | 1 | 39.357 |
| vLLM | total | 6 | 5107.345 |
| TRT-LLM | custom_moe_routing | 1 | 6.527 |
| TRT-LLM | moe_align generic count/sort | 2 | 165.876 |
| TRT-LLM | CUTLASS grouped gate/up GEMM | 1 | 1325.764 |
| TRT-LLM | CUTLASS grouped down GEMM | 1 | 677.777 |
| TRT-LLM | available component total | 5 | 2175.944 |

Prefill GEMM-only:

| Path | gate/up (us) | down (us) | total (us) |
|------|--------------|-----------|------------|
| vLLM Marlin | 2862.930 | 1418.235 | 4281.165 |
| TRT-LLM CUTLASS grouped GEMM | 1325.764 | 677.777 | 2003.541 |

For prefill, TRT-LLM's grouped CUTLASS GEMM is about 2.14x faster than the
current vLLM Marlin standalone GEMM sum on these shapes. The available TRT-LLM
component sum is about 2.35x lower than the vLLM standalone pipeline, but this
is a lower bound because the TRT-LLM activation/finalize/expand helpers are not
included.

## Kernel Differences

vLLM path:

- Routing uses vLLM `topkGating`, producing top-k ids and weights.
- Alignment uses the vLLM `moe_align_block_size` layout expected by Marlin.
- GEMM uses vLLM Marlin MoE WNA16 kernels with Marlin-packed INT4 weights.
- Activation is a separate `silu_and_mul` kernel.
- Top-k expert outputs are reduced by a separate `moe_sum` kernel.

TensorRT-LLM path:

- Routing uses TRT-LLM `customMoeRoutingKernel`, including optional
  softmax-before-topk support in the extracted source.
- Alignment uses TRT-LLM `moe_align_block_size`; prefill selects the generic
  count/sort path, which launches two kernels.
- GEMM uses TRT-LLM's CUTLASS `MoeFCGemm` grouped-GEMM path with a tactic cache.
- The standalone GEMM harness expects rows already grouped by expert and uses
  identity activation.
- The production TRT-LLM path has extra helpers for input expansion, gated
  activation, and finalize/reduction that are not yet represented here.

## Commands

The breakdown was generated with Nsight Systems `cuda_gpu_kern_sum` reports.
Representative component commands:

```bash
moe_w4a16/vllm/auxiliary/bench_topk_gating 3823 64 8 --bench 0 1
moe_w4a16/vllm/auxiliary/bench_moe_align 3823 64 8 16 --bench 0 1
moe_w4a16/vllm/marlin/bench_marlin_moe 3823 64 8 2048 3072 --balanced --no-topk-weights --bench 0 1
moe_w4a16/vllm/auxiliary/bench_silu_and_mul 3823 8 3072 --bench 0 1
moe_w4a16/vllm/marlin/bench_marlin_moe 3823 64 8 3072 1024 --balanced --bench 0 1
moe_w4a16/vllm/auxiliary/bench_moe_sum 3823 8 1024 --bench 0 1

moe_w4a16/trtllm/auxiliary/bench_custom_moe_routing 3823 64 8 fp16 --bench 0 1
moe_w4a16/trtllm/auxiliary/bench_moe_align 3823 64 8 16 auto --bench 0 1
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 --n=3072 --k=2048 \
  --group_size=128 --tactic=moe_w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 --n=1024 --k=3072 \
  --group_size=128 --tactic=moe_w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
```

Use the same commands with `3823` replaced by `1` for decode.
