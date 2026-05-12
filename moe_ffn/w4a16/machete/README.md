# Machete MoE W4A16 Prefill

This directory contains a MoE-prefill benchmark built from the standalone vLLM
Machete W4A16 extraction in `general/w4a16_gemm/machete_standalone`.

The implementation is intentionally simple: it assumes routed activations are
already grouped by expert and launches one SM90 Machete GEMM per active expert on
the same stream. Weight prepack is modeled as offline: the benchmark initializes
synthetic prepacked Machete-layout weights directly and does not time prepack or
file I/O.

This is not a fused/grouped single-kernel MoE GEMM like the TensorRT-LLM path.
It isolates whether the per-expert SM90 Machete kernel is fast enough on the
Qwen3.5-122B-A10B prefill expert shapes to beat the checked-in TRT-LLM grouped
GEMM baseline.

## Build

```bash
./compile.sh build moe-machete
```

Equivalent direct CMake build:

```bash
cmake -S moe_ffn/w4a16/machete \
  -B moe_ffn/w4a16/machete/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release

cmake --build moe_ffn/w4a16/machete/build_cmake_release \
  --target bench_machete_moe -j$(nproc)
```

## Qwen3.5-122B-A10B Prefill GEMM Shapes

Gate/up projection:

```bash
moe_ffn/w4a16/machete/build_cmake_release/bench_machete_moe \
  --experts=8 --m_per_expert=3823 --n=2048 --k=3072 \
  --group_size=128 --warmup=20 --iters=100 --no_checksum
```

Down projection:

```bash
moe_ffn/w4a16/machete/build_cmake_release/bench_machete_moe \
  --experts=8 --m_per_expert=3823 --n=3072 --k=1024 \
  --group_size=128 --warmup=20 --iters=100 --no_checksum
```

Both commands launch `experts` GEMM kernels per benchmark iteration.

## H800 Check

Measured with CUDA events on H800, FP16 activations, INT4 weights, group size
128, `warmup=20`, `iters=100`.

| Shape | Machete MoE | TRT-LLM grouped MoE | Speedup |
|---|---:|---:|---:|
| gate/up `experts=8,m=3823,n=2048,k=3072` | 817.072 us | 1240.5 us | 1.52x |
| down `experts=8,m=3823,n=3072,k=1024` | 518.570 us | 682.7 us | 1.32x |

TRT-LLM baseline commands used the checked-in
`moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache`.
