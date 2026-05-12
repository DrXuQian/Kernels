# MoE-FFN

Standalone kernels for MoE/FFN layer components.

## Layout

| Path | Contents |
|---|---|
| `bench_rmsnorm` | Category-local build of `general/bench_rmsnorm.cu` |
| `bench_shared_expert` | Router gate GEMM, shared expert gate GEMV, and `sigmoid(gate) * shared + routed` fusion |
| `w4a16/trtllm/moe_w4a16_standalone/` | TensorRT-LLM MoE W4A16 grouped GEMM |
| `w4a16/trtllm/auxiliary/` | TensorRT-LLM routing, expand, activation, finalize helpers |
| `w4a16/machete/` | SM90 per-expert Machete MoE prefill benchmark |
| `w4a16/vllm/marlin/` | vLLM Marlin MoE W4A16 GEMM |
| `w4a16/vllm/auxiliary/` | vLLM topk, align, activation, finalize helpers |

## Build

```bash
make -C moe_ffn

# Or use the repo-level build script for bench_all pieces and related experiments:
./compile.sh build moe-ffn moe-trtllm moe-trtllm-auxiliary moe-machete moe-vllm
```

`./compile.sh build moe` is a smaller alias for the category-local Makefile
targets plus the TensorRT-LLM MoE GEMM and Machete MoE prefill CMake targets.
It does not build every vLLM decode auxiliary used by the default benchmark
suite.

## Decode Backend Switch

`bench_all.sh` uses vLLM Marlin for routed MoE decode by default. To run the
TensorRT-LLM decode routed MoE pipeline instead:

```bash
./bench_all.sh --decode-moe-backend=trtllm --case moe_gate_up_decode_trtllm

DECODE_MOE_BACKEND=trtllm ./bench_all.sh decode_trtllm
```

The TensorRT-LLM MoE GEMM cache includes the decode keys:

```text
fp16,8,1,2048,3072,128|...
fp16,8,1,3072,1024,128|...
```

## Shared Expert

Qwen3-Next style MoE also has a shared expert path in addition to the routed
experts:

```text
router_gate:        (tokens, 3072) x (3072, 256) -> (tokens, 256)
shared_expert_gate: (tokens, 3072) x (3072, 1) -> (tokens, 1)
shared_fusion:      routed + sigmoid(shared_expert_gate) * shared_expert
```

The formula matches the TensorRT-LLM/vLLM Qwen3-Next model path. The router
gate and shared expert gate projections are modeled with `general/bench_cublas_gemm`.
The fusion is isolated as one standalone CUDA kernel so it can be profiled
separately.

```bash
general/bench_cublas_gemm --m=3823 --n=256 --k=3072 --dtype fp16 --bench 0 1
general/bench_cublas_gemm --m=3823 --n=1 --k=3072 --dtype fp16 --bench 0 1
moe_ffn/bench_shared_expert --op=sigmoid_mul_add --tokens=3823 --hidden=3072 --dtype fp16 --bench 0 1
```

For the full MoE-FFN execution order and `bench_all.sh` labels, see
`../OPERATOR_COVERAGE.md`.
