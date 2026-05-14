# Machete MoE W4A16 Prefill

This directory contains a MoE-prefill benchmark built from the standalone vLLM
Machete W4A16 extraction in `general/w4a16_gemm/machete_standalone`.

The benchmark assumes routed activations are already grouped by expert and uses
CUTLASS `GemmUniversal` batch dimension `L=experts` to launch one grouped
Machete GEMM kernel for all active experts. Weight prepack is modeled as
offline: the benchmark initializes synthetic prepacked Machete-layout weights
directly and does not time prepack or file I/O.

`--sequential` is kept only as a comparison mode. This benchmark is a standalone
comparison path; the root `bench_all.sh` default uses the TensorRT-LLM MoE GEMM.

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
  --group_size=128 --tactic=moe_ffn/w4a16/machete/machete_moe_tactics_h800.cache \
  --warmup=20 --iters=100 --no_checksum
```

Down projection:

```bash
moe_ffn/w4a16/machete/build_cmake_release/bench_machete_moe \
  --experts=8 --m_per_expert=3823 --n=3072 --k=1024 \
  --group_size=128 --tactic=moe_ffn/w4a16/machete/machete_moe_tactics_h800.cache \
  --warmup=20 --iters=100 --no_checksum
```

Both commands launch one grouped/batched GEMM kernel per benchmark iteration.
Use `--sequential` to force the older one-kernel-per-expert comparison mode.

## Correctness Smoke

`--verify` runs a lightweight zero-input invariant check before timing: it sets
all activations to zero, pre-fills the output with a sentinel, launches the
grouped MoE GEMM once, and checks every output element is zero within `1e-3`. It then
restores the benchmark input and continues with the requested timing loop.

```bash
moe_ffn/w4a16/machete/build_cmake_release/bench_machete_moe \
  --experts=8 --m_per_expert=3823 --n=2048 --k=3072 \
  --group_size=128 --warmup=1 --iters=1 --verify --no_checksum
```

`--verify_reference` runs a stronger sampled CPU reference check. In this mode
the benchmark generates raw GPTQ u4b8 col-major weights, runs the Machete GPU
prepack once before timing, launches the grouped MoE GEMM, and compares sampled GPU
outputs against a CPU dequantize-and-accumulate reference computed from the raw
weights, scales, and activations. The reference check is intentionally sampled so
it remains usable for the full Qwen prefill shape.

```bash
moe_ffn/w4a16/machete/build_cmake_release/bench_machete_moe \
  --experts=8 --m_per_expert=3823 --n=2048 --k=3072 \
  --group_size=128 --warmup=1 --iters=1 --verify_reference \
  --verify_samples=4096 --no_checksum
```

## H800 Check

Measured with CUDA events on H800, FP16 activations, INT4 weights, group size
128, `warmup=20`, `iters=100`.

| Shape | Machete grouped MoE | TRT-LLM grouped MoE | Speedup |
|---|---:|---:|---:|
| gate/up `experts=8,m=3823,n=2048,k=3072` | 604.848 us | 1240.5 us | 2.05x |
| down `experts=8,m=3823,n=3072,k=1024` | 358.692 us | 682.7 us | 1.90x |

TRT-LLM baseline commands used the checked-in
`moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache`.
Machete commands use `machete_moe_tactics_h800.cache`, which exact-matches the
Qwen3.5-122B-A10B prefill shapes to a Machete schedule.
