# Build And Benchmark Entry Points

This note documents the repo-level build script and benchmark runner. It is
intended as the quick reference for compiling one standalone kernel family and
running one benchmark case.

## Build Script

Use `compile.sh` from the repo root.

```bash
./compile.sh list
./compile.sh env
./compile.sh build
```

The default command is `build`. With no target, the script builds `all`.

```bash
./compile.sh
./compile.sh build all
```

## Build Environment Options

The script does not hardcode SDK paths. Pass them explicitly or export them in
the shell:

```bash
./compile.sh env \
  --cuda-root <CUDA_ROOT> \
  --ppu-root <COMPANION_SDK_ROOT>
```

Common options:

| Option | Meaning | Default |
|---|---|---|
| `--cuda-root DIR` | CUDA-compatible toolkit root. Also read from `CUDA_ROOT`. | unset |
| `--ppu-root DIR` | Optional companion SDK root. Also read from `PPU_ROOT`. | unset |
| `--cutlass-dir DIR` | CUTLASS checkout. | `third_party/cutlass` |
| `--arch ARCH` | Main CUDA architecture. | `sm_90a` |
| `--linear-arch ARCH` | Arch for `general` and `linear_attn` Makefile targets. | same as `--arch` |
| `--marlin-arch ARCH` | Arch for legacy standalone Marlin. | `sm_80` |
| `--build-type TYPE` | CMake build type. | `Release` |
| `--build-dir-name NAME` | CMake build directory basename. | `build_cmake_release` |
| `-j N`, `--jobs N` | Parallel build jobs. | `nproc` |
| `--ppu-elf-version VER` | Required ELF version for post-build checks. | `1.7` |
| `--no-elf-check` | Disable post-build ELF checks. | enabled |
| `--dry-run` | Print build commands without executing them. | disabled |
| `--no-env-check` | Skip environment validation. | disabled |
| `-v`, `--verbose` | Print extra environment details. | disabled |

Recommended environment check:

```bash
./compile.sh env \
  --cuda-root <CUDA_ROOT> \
  --ppu-root <COMPANION_SDK_ROOT> \
  --cutlass-dir third_party/cutlass \
  -v
```

This prints the selected `nvcc`, visible duplicate `nvcc` entries when verbose,
arch settings, and the CUTLASS path.

## Build Commands

Build a specific target:

```bash
./compile.sh build w4a16-fpa
./compile.sh build w4a16-machete
./compile.sh build moe-trtllm
./compile.sh build linear_attn
./compile.sh build flash_attn
./compile.sh build sampling
```

Configure without building, for CMake targets:

```bash
./compile.sh configure w4a16-fpa
./compile.sh configure moe-trtllm
```

Clean and rebuild one target:

```bash
./compile.sh clean w4a16-fpa
./compile.sh rebuild w4a16-fpa
```

Build several targets:

```bash
./compile.sh build linear_attn flash_attn sampling w4a16-machete w4a16-fpa moe-trtllm moe-ffn
```

Build with explicit SDK and arch:

```bash
./compile.sh build w4a16-fpa \
  --cuda-root <CUDA_ROOT> \
  --ppu-root <COMPANION_SDK_ROOT> \
  --arch sm_90a \
  --cutlass-dir third_party/cutlass
```

Dry-run the same build to inspect commands:

```bash
./compile.sh build w4a16-fpa --dry-run
```

## Build Targets

| Target | Builds |
|---|---|
| `general` | `general/` |
| `linear_attn` | `linear_attn/` |
| `flash_attn` | `flash_attn/` category-local shared attention ops |
| `sampling` | `sampling/` decode sampling stages |
| `flashinfer-gdn` | `linear_attn/src/flashinfer_gdn/` |
| `moe-ffn` | `moe_ffn/` category-local shared FFN ops + CUDA MoE pieces |
| `moe-vllm-marlin` | `moe_ffn/w4a16/vllm/marlin/` |
| `moe-vllm-auxiliary` | `moe_ffn/w4a16/vllm/auxiliary/` |
| `moe-vllm` | `moe-vllm-marlin` + `moe-vllm-auxiliary` |
| `moe-trtllm` | `moe_ffn/w4a16/trtllm/moe_w4a16_standalone/` |
| `moe-trtllm-auxiliary` | `moe_ffn/w4a16/trtllm/auxiliary/` |
| `moe` | `moe-ffn` + `moe-trtllm` |
| `w4a16-marlin` | `general/w4a16_gemm/marlin_standalone/` |
| `w4a16-fpa` | `general/w4a16_gemm/fpA_intB_standalone/` |
| `w4a16-machete` | `general/w4a16_gemm/machete_standalone/` |
| `w4a16-cutlass55` | `general/w4a16_gemm/cutlass55_standalone/` |
| `w4a16-cublas` | `general/w4a16_gemm/cublas_bf16_bench.cu` |
| `w4a16` | all W4A16 targets |
| `all` | every target above |

Useful aliases accepted by `compile.sh`:

| Alias | Expands to |
|---|---|
| `fpa`, `fpA_intB`, `fpaintb` | `w4a16-fpa` |
| `machete` | `w4a16-machete` |
| `cutlass55` | `w4a16-cutlass55` |
| `marlin` | `w4a16-marlin` |
| `trtllm-moe`, `moe_w4a16_standalone` | `moe-trtllm` |
| `trtllm-aux`, `trtllm-auxiliary` | `moe-trtllm-auxiliary` |
| `linear`, `linear-attention` | `linear_attn` |
| `flash`, `flash-attn`, `flash_attention` | `flash_attn` |
| `sample`, `sampling` | `sampling` |
| `flashinfer`, `gdn`, `flashinfer_gdn` | `flashinfer-gdn` |

## Benchmark Runner

Use `bench_all.sh` from the repo root.

```bash
./bench_all.sh --list
./bench_all.sh
```

With no case filter, it runs the full Qwen3.5-122B-A10B standalone kernel
suite. Every case uses single-run settings (`--bench 0 1` or
`--warmup=0 --iters=1`) in the underlying benchmark command.

## Benchmark Runner Options

| Option | Meaning |
|---|---|
| `--list` | List all benchmark case labels. |
| `--case LABEL` | Run cases matching `LABEL`. |
| `--kernel LABEL` | Alias for `--case`. |
| `--only LABEL` | Alias for `--case`. |
| positional `LABEL` | Same as `--case LABEL`. |
| `--resume-from LABEL` | Skip cases before `LABEL`, then continue. |
| `--run-dir DIR` | Run every benchmark with `DIR` as current working directory. |
| `--perf-model-dir DIR` | Alias for `--run-dir`. |
| `--ncu-cycles` | Run selected cases under Nsight Compute and summarize cycles. |
| `--ncu-launch-skip N` | Forward `--launch-skip N` to Nsight Compute. |
| `--ncu-launch-count N` | Forward `--launch-count N` to Nsight Compute. |

Case matching accepts exact labels, sanitized labels, or substrings.

Run one exact case:

```bash
./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

Run all decode MoE vLLM cases:

```bash
./bench_all.sh decode_vllm
```

Run multiple filters:

```bash
./bench_all.sh w4a16_prefill_linear_attn w4a16_decode_linear_attn
./bench_all.sh --case w4a16_prefill_linear_attn,w4a16_decode_linear_attn
```

Resume from a known label:

```bash
./bench_all.sh --resume-from w4a16_prefill_linear_attn_out_proj_cutlass55
```

Run from a separate runtime directory:

```bash
./bench_all.sh --run-dir <RUNTIME_WORKDIR> --case moe_gate_up_prefill_trtllm
```

Collect H800 cycles with Nsight Compute:

```bash
./bench_all.sh --ncu-cycles --case sampling_lm_head_gemm

# If setup kernels are captured before the target kernel, rerun the selected
# case with explicit launch skip/count.
./bench_all.sh --ncu-cycles --ncu-launch-skip 1 --ncu-launch-count 1 \
  --case sampling_topk_mask_logits
```

The per-case Nsight Compute CSV logs are written to
`.bench_logs/bench_<id>/ncu/`. The final table is written to
`.bench_logs/bench_<id>/ncu_cycles_summary.md`. This mode disables
`perfrawlog` post-processing for that run. The machine must allow access to
NVIDIA performance counters; otherwise `ncu` exits with `ERR_NVGPUCTRPERM`.

## Benchmark Output Options

By default logs are written to `.bench_logs/bench_<timestamp>/`, and the same
benchmark output is also printed to the terminal.

Useful environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `BENCH_RUN_ID` | Stable run id used in the default output directory. | timestamp |
| `OUT_DIR` | Explicit benchmark log directory. | `.bench_logs/bench_<id>` |
| `RUN_DIR` | Benchmark working directory. | `PERF_MODEL_DIR` or repo root |
| `PERF_MODEL_DIR` | Alias/default source for `RUN_DIR`. | unset |
| `PERFRAWLOG_POSTPROCESS` | Set `0` to skip perfrawlog post-processing. | `1` |
| `PERFRAWLOG_CLEAR` | Set `0` to keep an existing `perfrawlog` before each case. | `1` |
| `PERFRAWLOG_PATH` | Override perfrawlog path. | `$RUN_DIR/perfrawlog` |
| `PERF_STATISTICS_DIR` | Root for per-case perfstatistics reports. | `$OUT_DIR/perfstatistics` |
| `PERF_STATISTICS_MP` | `perf_statistics_gen --mp` value. | `16` |
| `PERF_STATISTICS_GHZ` | Clock used for latency summary. | `1.5` |
| `PERF_STATISTICS_SUMMARY` | Set `0` to skip the final summary table. | `1` |
| `NCU_METRICS` | Metrics used by `--ncu-cycles`. | `sm__cycles_elapsed.avg,sm__cycles_elapsed.max,gpu__time_duration.sum` |
| `NCU_LAUNCH_SKIP` | Optional Nsight Compute launch skip. | unset |
| `NCU_LAUNCH_COUNT` | Optional Nsight Compute launch count. | unset |

Example with a stable output directory:

```bash
BENCH_RUN_ID=w4a16_decode_qkv \
OUT_DIR=$PWD/.bench_logs/w4a16_decode_qkv \
./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

Skip perfrawlog processing:

```bash
PERFRAWLOG_POSTPROCESS=0 ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

## perfrawlog Post-Processing

After each selected case, if `$RUN_DIR/perfrawlog` exists, `bench_all.sh` runs:

```bash
python -m perf_model.perf_statistics_gen \
  --report_dir_path <OUT_DIR>/perfstatistics/<case> \
  --mp 16 \
  . perfrawlog
```

Each case gets its own report directory. The final summary is generated by:

```bash
python helpers/summarize_perfstatistics.py <OUT_DIR>/perfstatistics --ghz 1.5
```

The summary prints `compute_cycles` and latency at the configured clock.

## Compile Target To Benchmark Case Map

Build the target on the left before running matching cases on the right.

| Compile target | Benchmark cases |
|---|---|
| `linear_attn` | `linear_attn_*`, `linear_decode_*`, `linear_prefill_*` |
| `flash_attn` | `flash_attn_*_rmsnorm` |
| `sampling` | `sampling_*` |
| `w4a16-machete` | `w4a16_prefill_*_cutlass55` |
| `w4a16-fpa` | `w4a16_decode_*_fpA_intB` |
| `moe-ffn` | `moe_ffn_*_rmsnorm`, `moe_shared_expert_*`, plus vLLM/TRT-LLM auxiliary Makefile cases |
| `moe-trtllm` | `moe_gate_up_prefill_trtllm`, `moe_down_prefill_trtllm` |
| `moe-trtllm-auxiliary` | `moe_*_prefill_trtllm` auxiliary cases |
| `moe-vllm-marlin` | `moe_gate_up_decode_vllm`, `moe_down_decode_vllm` |
| `moe-vllm-auxiliary` | `moe_*_decode_vllm` auxiliary cases |

To build everything needed by `bench_all.sh`:

```bash
./compile.sh build linear_attn flash_attn sampling w4a16-machete w4a16-fpa moe-trtllm moe-ffn
```

## Direct Benchmark Commands

Use direct commands when the exact shape or tactic needs to be controlled
outside `bench_all.sh`.

W4A16 prefill, Machete CUTLASS55 backend:

```bash
general/w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --backend=cutlass55 \
  --cutlass55_tactic=general/w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache \
  --m=3823 --n=12288 --k=3072 --group_size=128 \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --profile_gemm_only --no_checksum \
  --warmup=0 --iters=1
```

W4A16 decode, fpA_intB backend:

```bash
general/w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=12288 --k=3072 --group_size=128 \
  --tactic=general/w4a16_gemm/fpA_intB_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
```

TRT-LLM MoE prefill GEMM:

```bash
moe_ffn/w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 \
  --n=3072 --k=2048 --group_size=128 \
  --tactic=moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
```

vLLM Marlin MoE decode GEMM:

```bash
moe_ffn/w4a16/vllm/marlin/bench_marlin_moe \
  1 64 8 2048 3072 \
  --balanced --no-topk-weights --bench 0 1
```

Linear attention decode GDN:

```bash
linear_attn/bench_gated_delta_net 1 64 128 1 --bench 0 1
```

TRT-LLM MoE finalize auxiliary:

```bash
moe_ffn/w4a16/trtllm/auxiliary/bench_finalize_moe_routing \
  3823 8 1024 fp16 --bench 0 1
```

vLLM MoE top-k gating auxiliary:

```bash
moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating \
  1 64 8 --bench 0 1
```

## Tactic Cache Checks

Some W4A16 cases require a tactic cache entry. `bench_all.sh` checks this before
launching the executable.

Machete CUTLASS55 prefill cache key format:

```bash
grep -F "3823,12288,3072,128,fp16|" \
  general/w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache
```

fpA_intB decode cache key format:

```bash
grep -F "1,12288,3072,128|" \
  general/w4a16_gemm/fpA_intB_standalone/tactics_h800.cache
```
