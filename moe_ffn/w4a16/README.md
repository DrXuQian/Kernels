# MoE W4A16 kernels

Standalone MoE kernels are split by upstream implementation. The vLLM and
TensorRT-LLM paths use different MoE GEMM implementations and different
auxiliary kernels, so they live in separate subtrees.

For the current Nsight Systems component breakdown, see
`TRTLLM_VS_VLLM_BREAKDOWN.md`.

The root `bench_all.sh` default uses the faster measured path for each phase:
TensorRT-LLM components for prefill and vLLM components for decode.

Machete is not a MoE GEMM path here. It is used by dense W4A16 linear layers
under `general/w4a16_gemm/machete_standalone`. The vLLM MoE decode path in this
repo is Marlin MoE plus vLLM auxiliary kernels. Older PyTorch-extension
benchmark scripts remain under `vllm/bench_python/` for historical reference,
but their old README/results are intentionally not restored because they are not
part of the current standalone build or `bench_all.sh` path.

## Layout

```
moe_ffn/w4a16/
├── vllm/
│   ├── marlin/       # vLLM Marlin MoE W4A16 GEMM
│   ├── auxiliary/    # vLLM topk, align, silu_and_mul, sum
│   └── bench_python/ # older PyTorch extension benchmarks from vLLM
└── trtllm/
    ├── moe_w4a16_standalone/ # TensorRT-LLM MoE grouped W4A16 GEMM
    └── auxiliary/            # TensorRT-LLM routing, expert-map, and pipeline helpers
```

## vLLM Pipeline

```
topk_gating -> moe_align -> Marlin GEMM (gate_up) -> silu_and_mul -> Marlin GEMM (down) -> moe_sum
[auxiliary]    [auxiliary]   [marlin]                [auxiliary]      [marlin]             [auxiliary]
```

Build the vLLM standalone CUDA pieces:

```bash
make -C moe_ffn/w4a16/vllm/marlin
make -C moe_ffn/w4a16/vllm/auxiliary

# Or from the repo root:
./compile.sh build moe-vllm
```

Qwen3.5-122B-A10B decode commands:

```bash
moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating 1 256 8 --bench 0 1
moe_ffn/w4a16/vllm/auxiliary/bench_moe_align 1 256 8 16 --bench 0 1
moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 1 256 8 2048 3072 --balanced --no-topk-weights --bench 0 1
moe_ffn/w4a16/vllm/auxiliary/bench_silu_and_mul 1 8 3072 --bench 0 1
moe_ffn/w4a16/vllm/marlin/bench_marlin_moe 1 256 8 3072 1024 --balanced --bench 0 1
moe_ffn/w4a16/vllm/auxiliary/bench_moe_sum 1 8 1024 --bench 0 1
```

## TensorRT-LLM Pipeline

`trtllm/moe_w4a16_standalone` contains the TensorRT-LLM grouped MoE W4A16 GEMM
extraction. It includes its own TRT-LLM-style tactic cache and does not share
the vLLM Marlin/auxiliary kernel path.

The extracted TensorRT-LLM component pipeline is:

```
custom_moe_routing -> expert_map -> expandInputRows -> grouped GEMM (gate/up)
-> gated activation -> grouped GEMM (down) -> finalizeMoeRouting
[auxiliary]          [auxiliary]  [auxiliary]        [moe_w4a16_standalone]
```

Build the TensorRT-LLM grouped GEMM and auxiliary kernels:

```bash
./compile.sh build moe-trtllm moe-trtllm-auxiliary

# Or build the auxiliary Makefile directly:
make -C moe_ffn/w4a16/trtllm/auxiliary
```

Qwen3.5-122B-A10B prefill commands:

```bash
moe_ffn/w4a16/trtllm/auxiliary/bench_custom_moe_routing 3823 256 8 fp16 --bench 0 1
moe_ffn/w4a16/trtllm/auxiliary/bench_expert_map 3823 256 8 auto --bench 0 1
moe_ffn/w4a16/trtllm/auxiliary/bench_expand_input_rows 3823 8 2048 fp16 --bench 0 1
moe_ffn/w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 --n=3072 --k=2048 \
  --group_size=128 --tactic=moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
moe_ffn/w4a16/trtllm/auxiliary/bench_gated_activation 3823 8 3072 fp16 --bench 0 1
moe_ffn/w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 --n=1024 --k=3072 \
  --group_size=128 --tactic=moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
moe_ffn/w4a16/trtllm/auxiliary/bench_finalize_moe_routing 3823 8 1024 fp16 --bench 0 1
```
