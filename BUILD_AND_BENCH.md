# Build And Benchmark

This is the repo-level command reference. It is organized by module:

1. Flash-Attn
2. Linear-Attn
3. MoE-FFN
4. Sampling

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
  --n=3072 --k=2048 --group_size=128 \
  --tactic=moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1

moe_ffn/w4a16/vllm/marlin/bench_marlin_moe \
  1 256 8 2048 3072 --balanced --no-topk-weights --bench 0 1

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

H800 Nsight Compute cycles:

```bash
./bench_all.sh --ncu-cycles --case sampling_lm_head_gemm
./bench_all.sh --ncu-cycles --ncu-launch-skip 1 --ncu-launch-count 1 \
  --case sampling_topk_mask_logits
```

perfrawlog post-processing:

```bash
RUN_DIR=<RUNTIME_WORKDIR> ./bench_all.sh --case moe_gate_up_prefill_trtllm
python helpers/summarize_perfstatistics.py <OUT_DIR>/perfstatistics --ghz 1.5
```

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
