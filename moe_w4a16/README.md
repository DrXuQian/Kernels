# MoE W4A16 kernels

Standalone MoE kernels are split by upstream implementation. The vLLM and
TensorRT-LLM paths use different MoE GEMM implementations and different
auxiliary kernels, so they live in separate subtrees.

## Layout

```
moe_w4a16/
├── vllm/
│   ├── marlin/       # vLLM Marlin MoE W4A16 GEMM
│   ├── auxiliary/    # vLLM topk, align, silu_and_mul, sum
│   └── bench_python/ # older PyTorch extension benchmarks from vLLM
└── trtllm/
    └── moe_w4a16_standalone/ # TensorRT-LLM MoE grouped W4A16 GEMM
```

## vLLM Pipeline

```
topk_gating -> moe_align -> Marlin GEMM (gate_up) -> silu_and_mul -> Marlin GEMM (down) -> moe_sum
[auxiliary]    [auxiliary]   [marlin]                [auxiliary]      [marlin]             [auxiliary]
```

Build the vLLM standalone CUDA pieces:

```bash
make -C moe_w4a16/vllm/marlin
make -C moe_w4a16/vllm/auxiliary
```

Example commands:

```bash
moe_w4a16/vllm/marlin/bench_marlin_moe 1 64 8 2048 5632
moe_w4a16/vllm/auxiliary/bench_topk_gating 1 64 8
moe_w4a16/vllm/auxiliary/bench_moe_align 1 64 8 16
moe_w4a16/vllm/auxiliary/bench_silu_and_mul 1 8 5632
moe_w4a16/vllm/auxiliary/bench_moe_sum 1 8 5632
```

## TensorRT-LLM Pipeline

`trtllm/moe_w4a16_standalone` contains the TensorRT-LLM grouped MoE W4A16 GEMM
extraction. It includes its own TRT-LLM-style tactic cache and does not share
the vLLM Marlin/auxiliary kernel path.

```bash
cmake -S moe_w4a16/trtllm/moe_w4a16_standalone \
  -B moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCUTLASS_DIR=$PWD/third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release \
  --target test_moe_w4a16_gemm -j$(nproc)
```
