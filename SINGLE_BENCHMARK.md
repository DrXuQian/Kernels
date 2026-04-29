# Running One Benchmark Case

This file lists the supported ways to run a single benchmark from this repo.

## Recommended Entry

Use `bench_all.sh` from the repo root. It knows the Qwen3.5-122B-A10B shapes, tactic cache paths, benchmark log paths, and per-case perfstatistics output paths.

List all case labels:

```bash
./bench_all.sh --list
```

Run one exact case:

```bash
./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

Equivalent aliases:

```bash
./bench_all.sh --kernel w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
./bench_all.sh --only w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

Positional arguments are also accepted:

```bash
./bench_all.sh w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

Case matching accepts substrings. This runs every case whose label contains `w4a16_decode_linear_attn`:

```bash
./bench_all.sh w4a16_decode_linear_attn
```

Run several filters in one command:

```bash
./bench_all.sh w4a16_prefill_linear_attn w4a16_decode_linear_attn
./bench_all.sh --case w4a16_prefill_linear_attn,w4a16_decode_linear_attn
```

## Logs And Run IDs

By default, each selected case writes a log under `.bench_logs/bench_<timestamp>/`.

Set a stable run id:

```bash
BENCH_RUN_ID=my_single_case ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

Set an explicit output directory:

```bash
OUT_DIR=$PWD/.bench_logs/my_single_case ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

The script prints benchmark output to the terminal and writes the same output to the per-case log.

## H800 / nsys Mode

For GPU kernel-time measurement on H800, skip perfrawlog post-processing and wrap one case with `nsys`:

```bash
RUN_ID=w4a16_decode_linear_attn_in_proj_qkv_fpA_intB_$(date +%Y%m%d_%H%M%S)

BENCH_RUN_ID="$RUN_ID" \
OUT_DIR="$PWD/.bench_logs/$RUN_ID" \
PERFRAWLOG_POSTPROCESS=0 \
nsys profile \
  --force-overwrite=true \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --output="$PWD/.bench_profiles/$RUN_ID" \
  ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

Export the ordered CUDA trace:

```bash
nsys stats ".bench_profiles/$RUN_ID.nsys-rep" \
  --report cuda_gpu_trace \
  --format csv \
  --output ".bench_profiles/${RUN_ID}_trace"
```

Export the aggregated kernel summary:

```bash
nsys stats ".bench_profiles/$RUN_ID.nsys-rep" \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output ".bench_profiles/${RUN_ID}_kern_sum"
```

## perfrawlog Mode

When the runtime requires a separate perf-model working directory, run the benchmark with that directory as `RUN_DIR` while keeping the executable path in this repo:

```bash
RUN_DIR=<PERF_MODEL_DIR> ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

Equivalent:

```bash
PERF_MODEL_DIR=<PERF_MODEL_DIR> ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

When `RUN_DIR/perfrawlog` exists, `bench_all.sh` runs:

```bash
python -m perf_model.perf_statistics_gen \
  --report_dir_path <out>/perfstatistics/<case> \
  --mp 16 \
  . perfrawlog
```

It then prints a summary table from the generated per-case `perfstatistics.log`.

Useful knobs:

```bash
PERFRAWLOG_POSTPROCESS=0 ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
PERF_STATISTICS_MP=16 ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
PERF_STATISTICS_GHZ=1.5 ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
PERF_STATISTICS_SUMMARY=0 ./bench_all.sh --case w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
```

## Case Labels

Every label below can be passed to `--case`, `--kernel`, `--only`, or as a positional argument.

| Group | Case |
|---|---|
| linear decode | `linear_decode_conv1d_update` |
| linear decode | `linear_decode_gdn` |
| linear prefill | `linear_prefill_conv1d_fwd` |
| linear prefill | `linear_prefill_flashinfer_gdn` |
| dense W4A16 linear attention | `w4a16_prefill_linear_attn_in_proj_qkv_cutlass55` |
| dense W4A16 linear attention | `w4a16_prefill_linear_attn_in_proj_z_cutlass55` |
| dense W4A16 linear attention | `w4a16_prefill_linear_attn_out_proj_cutlass55` |
| dense W4A16 linear attention | `w4a16_decode_linear_attn_in_proj_qkv_fpA_intB` |
| dense W4A16 linear attention | `w4a16_decode_linear_attn_in_proj_z_fpA_intB` |
| dense W4A16 linear attention | `w4a16_decode_linear_attn_out_proj_fpA_intB` |
| dense W4A16 full attention | `w4a16_prefill_full_attn_q_proj_gate_cutlass55` |
| dense W4A16 full attention | `w4a16_prefill_full_attn_k_proj_cutlass55` |
| dense W4A16 full attention | `w4a16_prefill_full_attn_v_proj_cutlass55` |
| dense W4A16 full attention | `w4a16_prefill_full_attn_o_proj_cutlass55` |
| dense W4A16 full attention | `w4a16_decode_full_attn_q_proj_gate_fpA_intB` |
| dense W4A16 full attention | `w4a16_decode_full_attn_k_proj_fpA_intB` |
| dense W4A16 full attention | `w4a16_decode_full_attn_v_proj_fpA_intB` |
| dense W4A16 full attention | `w4a16_decode_full_attn_o_proj_fpA_intB` |
| dense W4A16 consistent expert | `w4a16_prefill_consistent_expert_up_cutlass55` |
| dense W4A16 consistent expert | `w4a16_prefill_consistent_expert_down_cutlass55` |
| dense W4A16 consistent expert | `w4a16_decode_consistent_expert_up_fpA_intB` |
| dense W4A16 consistent expert | `w4a16_decode_consistent_expert_down_fpA_intB` |
| MoE prefill TRT-LLM | `moe_routing_prefill_trtllm` |
| MoE prefill TRT-LLM | `moe_expert_map_prefill_trtllm` |
| MoE prefill TRT-LLM | `moe_expand_prefill_trtllm` |
| MoE prefill TRT-LLM | `moe_gate_up_prefill_trtllm` |
| MoE prefill TRT-LLM | `moe_gated_prefill_trtllm` |
| MoE prefill TRT-LLM | `moe_down_prefill_trtllm` |
| MoE prefill TRT-LLM | `moe_finalize_prefill_trtllm` |
| MoE decode vLLM | `moe_routing_decode_vllm` |
| MoE decode vLLM | `moe_align_decode_vllm` |
| MoE decode vLLM | `moe_gate_up_decode_vllm` |
| MoE decode vLLM | `moe_gated_decode_vllm` |
| MoE decode vLLM | `moe_down_decode_vllm` |
| MoE decode vLLM | `moe_finalize_decode_vllm` |

## Direct Executable Mode

Use direct executable mode when you want full control over arguments or want to bypass `bench_all.sh`.

Dense W4A16 prefill through Machete CUTLASS55:

```bash
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --backend=cutlass55 \
  --cutlass55_tactic=w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache \
  --m=3823 --n=12288 --k=3072 --group_size=128 \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --profile_gemm_only --no_checksum \
  --warmup=0 --iters=1
```

Dense W4A16 decode through fpA_intB:

```bash
w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=12288 --k=3072 --group_size=128 \
  --tactic=w4a16_gemm/fpA_intB_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
```

TRT-LLM MoE prefill GEMM:

```bash
moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm \
  --dtype=fp16 --experts=8 --m_per_expert=3823 \
  --n=3072 --k=2048 --group_size=128 \
  --tactic=moe_w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache \
  --warmup=0 --iters=1
```

vLLM Marlin MoE decode GEMM:

```bash
moe_w4a16/vllm/marlin/bench_marlin_moe \
  1 64 8 2048 3072 \
  --balanced --no-topk-weights --bench 0 1
```

Linear-attention standalone kernel:

```bash
linear_attention/bench_gated_delta_net 1 64 128 1 --bench 0 1
```

Auxiliary MoE kernel:

```bash
moe_w4a16/vllm/auxiliary/bench_topk_gating 1 64 8 --bench 0 1
```

## Tactic Cache Requirement

The dense W4A16 `bench_all.sh` cases require the corresponding tactic-cache entry to exist before the benchmark is launched. Missing entries fail early instead of falling back to a default config.

Check a CUTLASS55 tactic entry:

```bash
grep -F "3823,12288,3072,128,fp16|" \
  w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache
```

Check an fpA_intB tactic entry:

```bash
grep -F "1,12288,3072,128|" \
  w4a16_gemm/fpA_intB_standalone/tactics_h800.cache
```

Search and save one missing CUTLASS55 tactic:

```bash
w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm \
  --backend=cutlass55 \
  --m=3823 --n=12288 --k=3072 --group_size=128 \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --profile_gemm_only --no_checksum \
  --search_cutlass55_configs \
  --save_cutlass55_tactic=w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache \
  --warmup=20 --iters=100
```
