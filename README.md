# Kernels

Standalone CUDA kernel benchmarks for Qwen3.5-122B-A10B inference profiling.

The repository is organized by model block type, with shared kernels under
`general/` and one repo-level build and benchmark entry point.

## Documentation

| Document | Purpose |
|---|---|
| [BUILD_AND_BENCH.md](BUILD_AND_BENCH.md) | Build targets, environment checks, `bench_all.sh` usage, single-case commands, and profiling notes. |
| [OPERATOR_COVERAGE.md](OPERATOR_COVERAGE.md) | Operator list, Qwen3.5-122B-A10B shapes, implementation source, and `bench_all.sh` case labels. |
| [bench_122B.md](bench_122B.md) | Reference benchmark results and performance notes. |
| [studies/ppu_finalize_moe_routing/README.md](studies/ppu_finalize_moe_routing/README.md) | Isolated `finalizeMoeRoutingKernel` source-level optimization study. |
| [sampling/third_party/flashinfer/README.md](sampling/third_party/flashinfer/README.md) | Provenance for the vendored FlashInfer sampling headers. |

CUTLASS is kept as a git submodule under `third_party/cutlass`. Its upstream
Markdown files are vendor documentation and are intentionally not indexed here.

## Layout

| Path | Contents |
|---|---|
| `compile.sh` | Repo-level build script. |
| `bench_all.sh` | Qwen3.5-122B-A10B kernel benchmark runner. |
| `bench_attention_inference.sh` | Python attention benchmark runner. |
| `general/` | Shared kernels: cuBLAS GEMM, RMSNorm, W4A16 GEMM standalones. |
| `linear_attn/` | Linear-attention kernels and FlashInfer GDN prefill standalone. |
| `flash_attn/` | Flash-attention category-local binaries and Python FlashAttention runner. |
| `moe_ffn/` | MoE/FFN kernels from TRT-LLM and vLLM extractions. |
| `sampling/` | Sampling benchmarks using vendored FlashInfer headers. |
| `studies/` | Isolated experiments that are not part of default build or benchmark flows. |
| `helpers/` | Utility scripts for tactic and benchmark analysis. |
| `third_party/cutlass/` | CUTLASS submodule. |

## Quick Start

```bash
git submodule update --init third_party/cutlass
./compile.sh env
./compile.sh build all
./bench_all.sh --list
./bench_all.sh --case sampling_lm_head_gemm
```

See [BUILD_AND_BENCH.md](BUILD_AND_BENCH.md) for target-specific build commands,
runtime directory handling, single-case benchmark usage, and profiling modes.
