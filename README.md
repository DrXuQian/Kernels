# Kernels

Standalone CUDA kernel benchmarks for Qwen3.5-122B-A10B inference profiling.

The docs use the same module order everywhere:

1. Flash-Attn
2. Linear-Attn
3. MoE-FFN
4. Sampling

Shared kernels such as RMSNorm, cuBLAS GEMM, and W4A16 GEMM live under
`general/`, but they are documented under the module that uses them.

## Documentation

| Document | Purpose |
|---|---|
| [BUILD_AND_BENCH.md](BUILD_AND_BENCH.md) | Per-module compile targets, benchmark commands, profiling modes, and tactic-cache notes. |
| [OPERATOR_COVERAGE.md](OPERATOR_COVERAGE.md) | Per-module operator list, shapes, implementation source, and `bench_all.sh` labels. |
| [bench_122B.md](bench_122B.md) | Per-module benchmark results. Missing PPU data is marked `not tested`. |
| [studies/ppu_finalize_moe_routing/README.md](studies/ppu_finalize_moe_routing/README.md) | MoE-FFN finalize routing study; not part of default compile or benchmark flows. |

CUTLASS is kept as a git submodule under `third_party/cutlass`. Its upstream
Markdown files are vendor documentation and are intentionally not indexed here.

## Layout

| Path | Contents |
|---|---|
| `compile.sh` | Repo-level build script. |
| `bench_all.sh` | Qwen3.5-122B-A10B benchmark runner. |
| `bench_attention_inference.sh` | Python/JIT attention benchmark runner. |
| `flash_attn/` | Flash-Attn local binaries and Python FlashAttention runner. |
| `linear_attn/` | Linear-Attn kernels and FlashInfer GDN prefill standalone. |
| `moe_ffn/` | MoE-FFN kernels from TRT-LLM and vLLM extractions. |
| `sampling/` | Sampling benchmarks using vendored FlashInfer headers. |
| `general/` | Shared kernels used by the modules above. |
| `studies/` | Isolated experiments outside the default benchmark suite. |
| `helpers/` | Tactic and benchmark analysis utilities. |
| `third_party/cutlass/` | CUTLASS submodule. |

## Quick Start

```bash
git submodule update --init third_party/cutlass
./compile.sh env
./compile.sh build all
./bench_all.sh --list
./bench_all.sh --case flash_attn_decode_full_attn
```

Use [BUILD_AND_BENCH.md](BUILD_AND_BENCH.md) for module-specific commands.
