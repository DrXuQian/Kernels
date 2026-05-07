# TensorRT-LLM MoE Auxiliary Kernels

Standalone TensorRT-LLM MoE helper kernels extracted from `TensorRT-LLM`.

## Included

| Binary | Source kernel | Upstream file |
|--------|---------------|---------------|
| `bench_custom_moe_routing` | custom MoE topK routing with optional softmax-before-topK | `cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu` |
| `bench_moe_align` | `moe_align_block_size` small-batch and generic paths | `cpp/tensorrt_llm/kernels/moeAlignKernels.cu` |
| `bench_expert_map` | `fusedBuildExpertMapsSortFirstTokenKernel` and `blockExpertPrefixSumKernel -> globalExpertPrefixSumKernel -> mergeExpertPrefixSumKernel` | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` |
| `bench_expand_input_rows` | `expandInputRowsKernel` FP16/BF16 subset | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` |
| `bench_gated_activation` | `doGatedActivationKernel` FP16/BF16 Swiglu subset | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` |
| `bench_finalize_moe_routing` | `finalizeMoeRoutingKernel` FP16/BF16 subset | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` |

The TensorRT-LLM MoE GEMM standalone target is in `../moe_w4a16_standalone`.
The `moe_kernels.cu` helpers are specialized here for the FP16/BF16 activation
path used by W4A16 benchmarks; FP8/FP4 scaling variants are intentionally not
included.

`bench_expert_map` matches the expert-map prologue inside TensorRT-LLM
`runMoe`: `auto` mode selects the fused single-kernel path for decode-sized
`num_tokens <= 256`, and the three-kernel prefix-sum path for prefill-sized
batches.

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
./bench_expert_map         [tokens] [experts] [topk] [auto|fused|three_step] [--bench warmup iters]
./bench_expand_input_rows  [tokens] [topk] [hidden_size] [dtype] [--bench warmup iters]
./bench_gated_activation   [tokens] [topk] [inter_size] [dtype] [--bench warmup iters]
./bench_finalize_moe_routing [tokens] [topk] [hidden_size] [dtype] [--no-scales] [--bench warmup iters]
```

Examples:

```bash
./bench_custom_moe_routing 1 256 8 fp16 --bench 100 1000
./bench_custom_moe_routing 3823 256 8 fp16 --bench 100 1000
./bench_moe_align 1 256 8 16 auto --bench 100 1000
./bench_moe_align 3823 256 8 16 auto --bench 100 1000
./bench_expert_map 1 256 8 auto --bench 100 1000
./bench_expert_map 3823 256 8 auto --bench 100 1000
./bench_expand_input_rows 3823 8 2048 fp16 --bench 100 1000
./bench_gated_activation 3823 8 3072 fp16 --bench 100 1000
./bench_finalize_moe_routing 3823 8 1024 fp16 --bench 100 1000
```
