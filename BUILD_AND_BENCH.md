# Build And Benchmark

This is the repo-level command reference. It is organized by module:

1. Flash-Attn
2. Linear-Attn
3. Dense-FFN
4. MoE-FFN
5. Sampling

All commands are run from the repo root unless noted otherwise.

## Common Setup

```bash
git submodule update --init third_party/cutlass
./compile.sh env
./compile.sh list
./bench_all.sh --list
```

`compile.sh` does not hardcode SDK paths. Pass them explicitly when needed:

```bash
./compile.sh env \
  --cuda-root <CUDA_ROOT> \
  --ppu-root <COMPANION_SDK_ROOT> \
  --cutlass-dir third_party/cutlass \
  -v
```

Common build options:

| Option | Meaning | Default |
|---|---|---|
| `--cuda-root DIR` | CUDA-compatible toolkit root. Also read from `CUDA_ROOT`. | unset |
| `--ppu-root DIR` | Optional companion SDK root. Also read from `PPU_ROOT`. | unset |
| `--cutlass-dir DIR` | CUTLASS checkout. | `third_party/cutlass` |
| `--arch ARCH` | Main CUDA architecture. | `sm_90a` |
| `--linear-arch ARCH` | Arch for `general` and `linear_attn` Makefile targets. | same as `--arch` |
| `--marlin-arch ARCH` | Arch for legacy Marlin standalone. | `sm_80` |
| `--build-type TYPE` | CMake build type. | `Release` |
| `--build-dir-name NAME` | CMake build directory basename. | `build_cmake_release` |
| `-j N`, `--jobs N` | Parallel build jobs. | `nproc` |
| `--ppu-elf-version VER` | Required ELF version for post-build checks. | `1.7` |
| `--no-elf-check` | Disable post-build ELF checks. | enabled |
| `--dry-run` | Print build commands without executing them. | disabled |

Build everything used by the default benchmark suite:

```bash
./compile.sh build all
```

Model wrapper scripts:

```bash
./bench_all.sh --list                                  # Qwen3.5-122B-A10B default
./run_all_qwen3.5-122B-A10B_GPTQ.sh --list             # dense cuBLAS baseline variant
./bench_Qwen3.5-122B-A10B-GPTQ_TP2.sh --list           # Qwen3.5-122B-A10B-GPTQ TP=2
./bench_Qwen3.5_27B.sh --list                          # Qwen3.5-27B dense model
./bench_MiniMax-M2.7_TP1.sh --list                     # MiniMax-M2.7 TP=1
./bench_MiniMax-M2.7_TP2.sh --list                     # MiniMax-M2.7 TP=2
./bench_MiniMax-M2.7_TP4.sh --list                     # MiniMax-M2.7 TP=4
```

Benchmark runner common options:

| Option | Meaning |
|---|---|
| `--case LABEL` | Run cases matching `LABEL`. |
| `--kernel LABEL`, `--only LABEL` | Aliases for `--case`. |
| positional `LABEL` | Same as `--case LABEL`. |
| `--resume-from LABEL` | Skip cases before `LABEL`, then continue. |
| `--run-dir DIR` | Run every benchmark with `DIR` as current working directory. |
| `--perf-model-dir DIR` | Alias for `--run-dir`. |
| `--ncu-cycles` | Run selected cases under Nsight Compute and summarize cycles. |
| `--ncu-bandwidth` | Run selected cases under Nsight Compute with DRAM byte/throughput metrics and summarize bandwidth utilization. |
| `--nsys-latency` | Run selected cases under Nsight Systems and summarize CUDA kernel duration. Use this when local NCU counters are unavailable. |
| `--ncu-launch-skip N` | Forward `--launch-skip N` to Nsight Compute. |
| `--ncu-launch-count N` | Forward `--launch-count N` to Nsight Compute. |

Useful log/runtime variables:

| Variable | Meaning | Default |
|---|---|---|
| `BENCH_RUN_ID` | Stable run id used in the default output directory. | timestamp |
| `OUT_DIR` | Explicit benchmark log directory. | `.bench_logs/bench_<id>` |
| `RUN_DIR` | Benchmark working directory. | `PERF_MODEL_DIR` or repo root |
| `PERFRAWLOG_POSTPROCESS` | Set `0` to skip perfrawlog post-processing. | `1` |
| `BENCH_DEDUPE` | Set `0` to rerun duplicate commands/shapes. | `1` |
| `PERF_STATISTICS_GHZ` | Clock used for perfstatistics latency summary. | `1.5` |
| `NCU_PEAK_GBPS` | Peak DRAM GB/s used by `--ncu-bandwidth` when NCU peak-percent metric is unavailable. | `3350` |

## Benchmark Policy

- Reuse the Qwen3.5 kernels for models whose operator semantics and shapes match.
- For MiniMax or other non-Qwen gaps, use the best available vLLM or TensorRT-LLM CUDA implementation before adding a local kernel. Do not silently replace a missing quantized kernel with a different dtype/quantization path.
- MiniMax-M2.7 uses FP8 E4M3 block-wise weight quantization. Its quantized GEMM/MoE path must not be labeled or benchmarked as W4A16. Non-quantized modules such as router gate and `lm_head` stay dense.
- Any configuration found by search must be serialized to a tactic/cache file. Final benchmark scripts should load cache entries and should not hide search inside timing.
- `nsys` latency means CUDA kernel duration. Do not use CPU wall time or launch overhead for kernel latency.
- Run profiling cases serially. Do not run independent performance benchmarks concurrently.
- Bandwidth conclusions should come from `--ncu-bandwidth` or equivalent NCU DRAM byte/throughput metrics, not from inferred wall-clock timing alone.
- If local `ncu` is unavailable or lacks permission, use `nsys` as a kernel-duration fallback only. Mark bandwidth utilization as unverified until NCU DRAM counters are available.

Qwen3.5-27B wrapper status:

- Dense GEMM/GEMV payloads use fp16.
- Linear-attention conv1d and fused RMSNorm gate run with `LINEAR_ATTN_DTYPE=fp16`.
- `linear_decode_gdn` still uses the existing float recurrent-state CUDA kernel, and `linear_prefill_flashinfer_gdn` is currently the extracted FlashInfer bf16-only prefill path. Do not treat those two as completed fp16 extractions until a matching fp16 implementation is added or selected from upstream.

MiniMax-M2.7 wrapper status:

- MiniMax-M2.7 published weights are FP8 E4M3 block-wise with
  `weight_block_size=[128,128]`; this is not MXFP8.
- Dense quantized projections now use the standalone vLLM/CUTLASS SM90
  block-FP8 path (`fp8_block_*_cutlass`).
- Routed MoE FP8 bodies default to `MOE_GEMM_BACKEND=fp8_block_dense`, which
  uses the repo-owned CUTLASS block-FP8 dense GEMM on expanded expert tokens.
  This keeps the benchmark standalone, but it is a transition path rather than
  the final grouped/fused MoE implementation.
- `MOE_GEMM_BACKEND=flashinfer_fp8` runs the FlashInfer SM90 CUTLASS fused MoE
  reference with a checked-in tactic cache. This path validates the expected
  upstream kernel choice, but it depends on the installed FlashInfer package and
  is not final standalone coverage.
- Final MiniMax MoE work item: extract or reimplement the SM90 grouped/fused
  block-FP8 MoE source into a repo-owned C++/CUDA binary, then regenerate cache
  and bandwidth numbers.

## Flash-Attn

Build:

```bash
./compile.sh build flash_attn w4a16-machete w4a16-fpa linear_attn
```

Run all Flash-Attn cases:

```bash
./bench_all.sh --case flash_attn
./bench_all.sh --case w4a16_prefill_full_attn,w4a16_decode_full_attn
```

Run selected single cases:

```bash
./bench_all.sh --case flash_attn_prefill_full_attn
./bench_all.sh --case flash_attn_decode_full_attn
./bench_all.sh --case w4a16_prefill_full_attn_q_proj_gate_cutlass55
./bench_all.sh --case w4a16_decode_full_attn_o_proj_fpA_intB
```

The core FlashAttention cases are Python-backed and require the active Python
environment to provide the attention package used by `flash_attn/bench_flash_attn.py`.

## Linear-Attn

Build:

```bash
./compile.sh build general linear_attn flashinfer-gdn w4a16-machete w4a16-fpa
```

Run all Linear-Attn cases:

```bash
./bench_all.sh --case linear_attn
./bench_all.sh --case linear_decode
./bench_all.sh --case linear_prefill
./bench_all.sh --case w4a16_prefill_linear_attn,w4a16_decode_linear_attn
```

Run selected single cases:

```bash
./bench_all.sh --case linear_prefill_flashinfer_gdn
./bench_all.sh --case linear_decode_gdn
./bench_all.sh --case linear_attn_prefill_fused_rms_norm_gate
./bench_all.sh --case linear_attn_decode_fused_rms_norm_gate
./bench_all.sh --case w4a16_prefill_linear_attn_in_proj_qkv_cutlass55
./bench_all.sh --case w4a16_decode_linear_attn_out_proj_fpA_intB
```

Direct GDN commands when bypassing `bench_all.sh`:

```bash
linear_attn/bench_gated_delta_net 1 64 128 1 --bench 0 1
linear_attn/bench_gdn_prefill 3823 16 64 128 1 --bench 0 1
linear_attn/bench_fused_rms_norm_gate 64 128 --bench 0 1
linear_attn/bench_fused_rms_norm_gate $((3823 * 64)) 128 --bench 0 1
```

## Dense-FFN

Dense FFN cases are enabled by model wrappers such as `bench_Qwen3.5_27B.sh`.
They use dense cuBLAS GEMM for gate/up and down projections plus the same
standalone gated activation used by the MoE auxiliary path.

Run all dense FFN cases:

```bash
./bench_Qwen3.5_27B.sh --case dense_ffn
```

Run selected single cases:

```bash
./bench_Qwen3.5_27B.sh --case dense_ffn_prefill_gate_up_cublas
./bench_Qwen3.5_27B.sh --case dense_ffn_prefill_gated_activation
./bench_Qwen3.5_27B.sh --case dense_ffn_prefill_down_cublas
```

## MoE-FFN

Build:

```bash
./compile.sh build moe-ffn moe-trtllm moe-trtllm-auxiliary moe-vllm w4a16-machete w4a16-fpa
```

Run all MoE-FFN cases:

```bash
./bench_all.sh --case moe
./bench_all.sh --case decode_vllm
./bench_all.sh --case prefill_trtllm
./bench_all.sh --case consistent_expert
```

Run selected single cases:

```bash
./bench_all.sh --case moe_gate_up_prefill_trtllm
./bench_all.sh --case moe_down_prefill_trtllm
./bench_all.sh --case moe_gate_up_decode_vllm
./bench_all.sh --case moe_finalize_prefill_trtllm
./bench_all.sh --case moe_shared_expert_activation_prefill_trtllm
./bench_all.sh --case moe_shared_expert_activation_decode_trtllm
```

Direct MoE commands when bypassing `bench_all.sh`:

```bash
moe_ffn/w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 \
  --n=2048 --k=3072 --group_size=128 \
  --tactic=moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1

moe_ffn/w4a16/vllm/marlin/bench_marlin_moe \
  1 256 8 3072 2048 --balanced --no-topk-weights --bench 0 1

moe_ffn/w4a16/trtllm/auxiliary/bench_shared_expert_activation \
  3823 1024 fp16 --bench 0 1

moe_ffn/w4a16/trtllm/auxiliary/bench_shared_expert_activation \
  1 1024 fp16 --bench 0 1
```

The isolated MoE finalize study is separate from default builds:

```bash
cd studies/ppu_finalize_moe_routing
make
./bench_finalize_moe_routing_study 3823 8 1024 fp16 --mode optimized --bench 0 1
```

## Sampling

Build:

```bash
./compile.sh build general sampling
```

Run all Sampling cases:

```bash
./bench_all.sh --case sampling
```

Run selected single cases:

```bash
./bench_all.sh --case sampling_lm_head_gemm
./bench_all.sh --case sampling_topk_mask_logits
./bench_all.sh --case sampling_softmax
./bench_all.sh --case sampling_top_p
```

Direct Sampling commands when bypassing `bench_all.sh`:

```bash
general/bench_cublas_gemm \
  --m=1 --n=248320 --k=3072 --dtype fp16 --out-dtype fp32 --bench 0 1

sampling/bench_sampling \
  --op=top_p --hidden=3072 --vocab=248320 --top-k=50 --top-p=0.9 --bench 0 1
```

## Profiling

H800 nsys single-case capture:

```bash
./bench_MiniMax-M2.7_TP1.sh --nsys-latency --case fp8_block_prefill_full_attn_q_proj_gate_cutlass

RUN_ID=case_$(date +%Y%m%d_%H%M%S)
BENCH_RUN_ID="$RUN_ID" \
PERFRAWLOG_POSTPROCESS=0 \
nsys profile \
  --force-overwrite=true \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --output=".bench_profiles/$RUN_ID" \
  ./bench_all.sh --case flash_attn_decode_full_attn

nsys stats ".bench_profiles/$RUN_ID.nsys-rep" \
  --report cuda_gpu_trace \
  --format csv \
  --output ".bench_profiles/${RUN_ID}_trace"
```

`--nsys-latency` writes `.nsys-rep` files under `<OUT_DIR>/nsys/`, a
case-level kernel-duration table at `<OUT_DIR>/nsys_latency_summary.md`, and a
model-level latency report at `<OUT_DIR>/model_latency_nsys/model_latency_summary.md`.
It intentionally does not print bandwidth utilization; use `--ncu-bandwidth`
on a machine with NCU performance-counter permission for DRAM metrics.

H800 Nsight Compute cycles:

```bash
./bench_all.sh --ncu-cycles --case sampling_lm_head_gemm
./bench_all.sh --ncu-cycles --ncu-launch-skip 1 --ncu-launch-count 1 \
  --case sampling_topk_mask_logits
```

`--ncu-cycles` writes the raw per-case CSV files under `<OUT_DIR>/ncu/`, a
case-level aggregate at `<OUT_DIR>/ncu_cycles_summary.md`, and a model-level
latency report with SVG charts at
`<OUT_DIR>/model_latency_ncu/model_latency_summary.md`.

perfrawlog post-processing:

```bash
RUN_DIR=<RUNTIME_WORKDIR> ./bench_all.sh --case moe_gate_up_prefill_trtllm
python helpers/summarize_perfstatistics.py <OUT_DIR>/perfstatistics --ghz 1.5
python helpers/summarize_perfstatistics.py <OUT_DIR>/perfstatistics --ghz 1.5 \
  --bench-out-dir <OUT_DIR> \
  --model-summary-dir <OUT_DIR>/model_latency_perfstatistics
python helpers/summarize_ncu_cycles.py <OUT_DIR>/ncu --detail \
  --bench-out-dir <OUT_DIR> \
  --model-summary-dir <OUT_DIR>/model_latency_ncu
```

The model-level reports split latency into prefill and decode, then into
Flash-Attn, Linear-Attn, MoE-FFN, and Sampling. They also report total covered
model latency and generate phase/module/operator pie and bar charts. When
`BENCH_DEDUPE=1` skips an identical logical case, the model summary expands the
skipped case by reusing the measured source-case latency so the model total is
not undercounted.

## Tactic Cache Checks

Some W4A16 cases require a cache entry. `bench_all.sh` checks these before
launching the executable.

Machete CUTLASS55 prefill cache key:

```bash
grep -F "3823,12288,3072,128,fp16|" \
  general/w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache
```

fpA_intB decode cache key:

```bash
grep -F "1,12288,3072,128|" \
  general/w4a16_gemm/fpA_intB_standalone/tactics_h800.cache
```
