Standalone MoE W4A16 GEMM (FP16/BF16 x INT4) for SM80 path
===========================================================

This directory contains a minimal, standalone build of TensorRT-LLM's MoE
grouped GEMM path for weight-only W4A16. It supports:

- FP16 activations x INT4 weights -> FP16 output
- BF16 activations x INT4 weights -> BF16 output
- groupwise INT4 scales with group size 128 or K
- CUTLASS SM80 grouped-GEMM path, including Hopper fallback to the SM80 kernels

The standalone target does not link against TensorRT, TensorRT-LLM runtime,
Python, or Torch. CUTLASS is still required.

What I extracted
----------------
- MoE W4A16 instantiations:
  `moe_gemm_kernels_fp16_uint4.cu` and `moe_gemm_kernels_bf16_uint4.cu`
- MoE dispatch headers and SM80 launchers used by `MoeGemmRunner`
- CUTLASS grouped-GEMM extension headers needed by the MoE path
- lightweight TRT-LLM common headers already used by the local standalone
  extractions
- Standalone CLI test and benchmark harness (`test_moe_w4a16_gemm.cu`)

What is NOT included (limitations)
----------------------------------
- routing, token permutation, top-k weight fusion, or finalize fusion
- fused activation variants beyond `ActivationType::Identity`
- FP8/FP4/NVFP4 or INT8 weight paths
- SM90 TMA MoE kernels for W4A16
- Python/Torch bindings or TensorRT-LLM runtime integration

The CLI expects input rows to already be grouped by expert.

Config selection (aligned with TRT-LLM-style profiling)
-------------------------------------------------------
The benchmark can run one explicit SM80 config or profile all supported SM80
MoE W4A16 configs for the given shape.

- Default: run `--tile_enum`/`--stages` directly.
- `--sweep_configs`: time every supported SM80 candidate and print each result.
- `--tactic=<file>`: load a cached best config for the exact shape. On a miss,
  profile the supported configs and append the best result to the cache.

The tactic serialization after `|` matches `fpA_intB_standalone`:

```
cuda=0,tma=0,tile80=<enum>,stages=<n>,splitk=1,sk_style=0
```

The MoE cache key includes dtype and expert layout:

```
dtype,experts,m_per_expert,n,k,group_size|...
```

Input/weight layout assumptions
-------------------------------
- A: FP16 or BF16 row-major, shape `[sum(tokens_per_expert), K]`.
- `total_tokens_including_expert`: device int64 array of length `num_experts`;
  entry `i` is the cumulative row count through expert `i`.
- B: packed INT4 expert weights, shape `[num_experts, K, N]` in the layout
  expected by TRT-LLM's MoE grouped GEMM.
- Scales: FP16 or BF16, shape `[num_experts, K / group_size, N]`.
- Zeros: not used by this smoke test (`zeros=nullptr`), so this is scale-only
  groupwise quantization.

Build
-----
From the repo root:

```
cmake -S moe_w4a16/trtllm/moe_w4a16_standalone \
  -B moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCUTLASS_DIR=$PWD/third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release \
  --target test_moe_w4a16_gemm -j$(nproc)
```

The output binary is
`moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm`.
Use `-DCMAKE_CUDA_ARCHITECTURES=80` for an Ampere build. The MoE W4A16
extraction uses the SM80 grouped GEMM fallback on Hopper, so `90` is sufficient.

Run
---
FP16 sanity check:

```
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=4 --m_per_expert=16 --n=128 --k=128 --verify
```

BF16 sanity check:

```
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=bf16 --experts=4 --m_per_expert=16 --n=128 --k=128 --verify
```

List configs:

```
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --list_configs
```

Sweep configs for one shape:

```
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=1 --n=1024 --k=3072 \
  --group_size=128 --warmup=100 --iters=1000 --sweep_configs
```

Use the H800 tactic cache:

```
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=1 --n=1024 --k=3072 \
  --group_size=128 --warmup=100 --iters=1000 \
  --tactic=moe_w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache
```

Qwen MoE shapes in `tactics_h800.cache`
---------------------------------------
The checked-in cache contains FP16 and BF16 tactics for these Qwen MoE GEMMs:

- gate/up prefill: `experts=8, m_per_expert=3823, n=3072, k=2048`
- down prefill: `experts=8, m_per_expert=3823, n=1024, k=3072`
- gate/up decode: `experts=8, m_per_expert=1, n=3072, k=2048`
- down decode: `experts=8, m_per_expert=1, n=1024, k=3072`

Interpreting `--list_configs` output (`tile_enum/stages/split_k`)
-----------------------------------------------------------------
`--list_configs` prints CUTLASS candidates in the form:

- `tile_enum=<int> stages=<int> split_k=<int> sm=<int>`

Meaning:

- `tile_enum`: the integer value of
  `cutlass_extensions::CutlassTileConfig` (see
  `moe_w4a16/trtllm/moe_w4a16_standalone/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h`).
- `stages`: CUTLASS mainloop pipeline stages.
- `split_k`: serial split-K factor. This extraction only keeps `split_k=1`
  candidates for MoE W4A16.

Notes
-----
- K must be divisible by `group_size`.
- `group_size` must be 128 or K.
- K must be a multiple of 64 and N must be a multiple of 128 for the SM80
  MoE W4A16 configs used here.
- Decode shapes with very small `m_per_expert` are dominated by launch,
  grouped-kernel overhead, and M-dimension padding; prefill shapes are the main
  throughput path.
