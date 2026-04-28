# TensorRT-LLM MoE Auxiliary Kernels

Standalone TensorRT-LLM MoE helper kernels extracted from `TensorRT-LLM`.

## Included

| Binary | Source kernel | Upstream file |
|--------|---------------|---------------|
| `bench_custom_moe_routing` | custom MoE topK routing with optional softmax-before-topK | `cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu` |
| `bench_moe_align` | `moe_align_block_size` small-batch and generic paths | `cpp/tensorrt_llm/kernels/moeAlignKernels.cu` |

The TensorRT-LLM MoE GEMM standalone target is in `../moe_w4a16_standalone`.
The heavier GEMM-adjacent helpers from `cutlass_kernels/moe_gemm/moe_kernels.cu`
(`expandInputRows`, gated activation, `finalizeMoeRouting`) are not extracted
in this folder yet.

## Build

From this directory:

```bash
make
```

From repo root:

```bash
./compile.sh build moe-trtllm-auxiliary
```

## Run

```bash
./bench_custom_moe_routing [tokens] [experts] [topk] [dtype] [--softmax-before-topk] [--bench warmup iters]
./bench_moe_align          [tokens] [experts] [topk] [block_size] [auto|small|generic] [--bench warmup iters]
```

Examples:

```bash
./bench_custom_moe_routing 1 64 8 fp16 --bench 100 1000
./bench_custom_moe_routing 3823 64 8 fp16 --bench 100 1000
./bench_moe_align 1 64 8 16 auto --bench 100 1000
./bench_moe_align 3823 64 8 16 auto --bench 100 1000
```
