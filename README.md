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

CUTLASS is kept as a git submodule under `third_party/cutlass`. Its upstream
Markdown files are vendor documentation and are intentionally not indexed here.

## Dense Technical Notes

These module-local notes are kept because they contain extraction details,
tactic/config information, or performance analysis that is not duplicated in
the top-level command reference.

| Document | Extra information |
|---|---|
| [general/README.md](general/README.md) | Shared standalone kernels and direct commands for RMSNorm, LayerNorm, and cuBLAS GEMM. |
| [flash_attn/README.md](flash_attn/README.md) | Full-attention covered pieces, missing MRoPE/output-gate standalone caveats. |
| [linear_attn/README.md](linear_attn/README.md) | DeltaNet CUDA/Triton kernel mapping and Triton JIT workflow. |
| [moe_ffn/README.md](moe_ffn/README.md) | MoE shared-expert path and category-local build entry. |
| [moe_ffn/w4a16/README.md](moe_ffn/w4a16/README.md) | TRT-LLM prefill and vLLM decode MoE W4A16 pipeline commands. |
| [moe_ffn/w4a16/TRTLLM_VS_VLLM_BREAKDOWN.md](moe_ffn/w4a16/TRTLLM_VS_VLLM_BREAKDOWN.md) | TRT-LLM prefill vs vLLM decode component breakdown and caveats. |
| [sampling/README.md](sampling/README.md) | Decode sampling stages and vendored FlashInfer-header dependency boundary. |
| [general/w4a16_gemm/fpA_intB_standalone/README.md](general/w4a16_gemm/fpA_intB_standalone/README.md) | TensorRT-LLM fpA_intB standalone extraction scope, config selection, and tactic usage. |
| [general/w4a16_gemm/machete_standalone/README.md](general/w4a16_gemm/machete_standalone/README.md) | vLLM Machete extraction, CUTLASS55 backend, schedules, and tactic cache flow. |
| [general/w4a16_gemm/cutlass55_standalone/README.md](general/w4a16_gemm/cutlass55_standalone/README.md) | CUTLASS example 55 standalone wrapper, single-kernel mode, and setup-skip behavior. |
| [general/w4a16_gemm/machete_standalone/CUTLASS55_TACTICS_H800.md](general/w4a16_gemm/machete_standalone/CUTLASS55_TACTICS_H800.md) | H800 CUTLASS55 tactic-cache measurements. |
| [general/w4a16_gemm/machete_standalone/INSTRUCTION_COUNT_ANALYSIS.md](general/w4a16_gemm/machete_standalone/INSTRUCTION_COUNT_ANALYSIS.md) | PPU instruction-count comparison for CUTLASS55 vs Machete. |
| [studies/ppu_finalize_moe_routing/README.md](studies/ppu_finalize_moe_routing/README.md) | Isolated MoE finalize routing optimization study; not part of default compile or benchmark flows. |

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
