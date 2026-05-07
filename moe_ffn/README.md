# MoE-FFN

Standalone kernels for MoE/FFN layer components.

## Layout

| Path | Contents |
|---|---|
| `bench_rmsnorm` | Category-local build of `general/bench_rmsnorm.cu` |
| `bench_shared_expert` | Router gate GEMM, shared expert gate GEMV, and `sigmoid(gate) * shared + routed` fusion |
| `w4a16/trtllm/moe_w4a16_standalone/` | TensorRT-LLM MoE W4A16 grouped GEMM |
| `w4a16/trtllm/auxiliary/` | TensorRT-LLM routing, expand, activation, finalize helpers |
| `w4a16/vllm/marlin/` | vLLM Marlin MoE W4A16 GEMM |
| `w4a16/vllm/auxiliary/` | vLLM topk, align, activation, finalize helpers |

## Build

```bash
make -C moe_ffn

# CMake MoE GEMM is built through the repo-level script:
./compile.sh build moe-trtllm
```

`./compile.sh build moe` builds the category-local Makefile targets plus the
TensorRT-LLM MoE GEMM CMake target.

## Shared Expert

Qwen3-Next style MoE also has a shared expert path in addition to the routed
experts:

```text
router_gate:        (tokens, 3072) x (3072, 256) -> (tokens, 256)
shared_expert_gate: (tokens, 3072) x (3072, 1) -> (tokens, 1)
shared_fusion:      routed + sigmoid(shared_expert_gate) * shared_expert
```

The formula matches the TensorRT-LLM/vLLM Qwen3-Next model path. The router gate
and shared expert gate projections are dense FP16/BF16 cuBLAS GEMMs; the fusion
is isolated as one standalone CUDA kernel so it can be profiled separately.

```bash
moe_ffn/bench_shared_expert --op=router_gate_gemm --tokens=3823 --hidden=3072 --out-dim=256 --dtype fp16 --bench 0 1
moe_ffn/bench_shared_expert --op=gate_gemv --tokens=3823 --hidden=3072 --dtype fp16 --bench 0 1
moe_ffn/bench_shared_expert --op=sigmoid_mul_add --tokens=3823 --hidden=3072 --dtype fp16 --bench 0 1
```
