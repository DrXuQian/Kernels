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

The TensorRT-LLM path is now represented as standalone component kernels:

```
custom_moe_routing -> moe_align -> expandInputRows -> CUTLASS grouped gate/up GEMM
-> gated activation -> CUTLASS grouped down GEMM -> finalizeMoeRouting
```

The `expandInputRows`, gated activation, and `finalizeMoeRouting` helpers are
FP16/BF16 standalone specializations of the corresponding kernels in
`cutlass_kernels/moe_gemm/moe_kernels.cu`. This is still a component-level
benchmark: it does not use TRT-LLM's fused-finalize GEMM epilogue or run the
whole production runner in a single process.

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
| TRT-LLM | custom_moe_routing | 1 | 2.240 |
| TRT-LLM | moe_align | 1 | 10.272 |
| TRT-LLM | expandInputRows | 1 | 3.999 |
| TRT-LLM | CUTLASS grouped gate/up GEMM | 1 | 20.607 |
| TRT-LLM | gated activation | 1 | 6.560 |
| TRT-LLM | CUTLASS grouped down GEMM | 1 | 27.455 |
| TRT-LLM | finalizeMoeRouting | 1 | 11.360 |
| TRT-LLM | component total | 7 | 82.493 |

Decode GEMM-only:

| Path | gate/up (us) | down (us) | total (us) |
|------|--------------|-----------|------------|
| vLLM Marlin | 21.055 | 14.719 | 35.774 |
| TRT-LLM CUTLASS grouped GEMM | 20.607 | 27.455 | 48.062 |

For decode, vLLM is currently better overall. TRT-LLM's gate/up GEMM is about
flat, but its down projection and the extra standalone expand/finalize helpers
make the component sum slower for batch=1.

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
| TRT-LLM | custom_moe_routing | 1 | 6.656 |
| TRT-LLM | moe_align generic count/sort | 2 | 165.945 |
| TRT-LLM | expandInputRows | 1 | 281.428 |
| TRT-LLM | CUTLASS grouped gate/up GEMM | 1 | 1327.658 |
| TRT-LLM | gated activation | 1 | 391.216 |
| TRT-LLM | CUTLASS grouped down GEMM | 1 | 681.477 |
| TRT-LLM | finalizeMoeRouting | 1 | 99.164 |
| TRT-LLM | component total | 8 | 2953.544 |

Prefill GEMM-only:

| Path | gate/up (us) | down (us) | total (us) |
|------|--------------|-----------|------------|
| vLLM Marlin | 2862.930 | 1418.235 | 4281.165 |
| TRT-LLM CUTLASS grouped GEMM | 1327.658 | 681.477 | 2009.135 |

For prefill, TRT-LLM's grouped CUTLASS GEMM is about 2.14x faster than the
current vLLM Marlin standalone GEMM sum on these shapes. Including standalone
expand, gated activation, and finalize helpers, the TRT-LLM component pipeline
is about 1.73x faster than the vLLM standalone pipeline.

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
- `expandInputRows` duplicates the input activation rows according to the
  permuted expert layout before the first grouped GEMM.
- Gated activation is represented as a standalone FP16/BF16 Swiglu helper.
- `finalizeMoeRouting` unpermutes, applies top-k scales, and reduces the expert
  outputs back to one row per token.
- The standalone GEMM harness itself still expects rows already grouped by
  expert; the helper benchmarks measure the missing component kernels
  separately.

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
moe_w4a16/trtllm/auxiliary/bench_expand_input_rows 3823 8 2048 fp16 --bench 0 1
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 --n=3072 --k=2048 \
  --group_size=128 --tactic=moe_w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
moe_w4a16/trtllm/auxiliary/bench_gated_activation 3823 8 3072 fp16 --bench 0 1
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 --n=1024 --k=3072 \
  --group_size=128 --tactic=moe_w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
moe_w4a16/trtllm/auxiliary/bench_finalize_moe_routing 3823 8 1024 fp16 --bench 0 1
```

Use the same commands with `3823` replaced by `1` for decode.
