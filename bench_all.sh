#!/usr/bin/env bash
# Qwen3.5-122B-A10B standalone kernel benchmark suite.
#
# This script runs each standalone benchmark with no warmup and exactly one
# measured iteration. Benchmark stdout/stderr is written to per-case logs.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-${PERF_MODEL_DIR:-$ROOT_DIR}}"

BENCH_RUN_ID="${BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/.bench_logs/bench_$BENCH_RUN_ID}"

PREFILL_TOKENS=3823
DECODE_TOKENS=1
LINEAR_DIM=12288
CONV_WIDTH=4
LINEAR_Q_HEADS=16
LINEAR_V_HEADS=64
LINEAR_HEAD_DIM=128

MOE_EXPERTS=8
MOE_ROUTER_EXPERTS=64
MOE_TOPK=8
MOE_GROUP=128
MOE_GATE_N=3072
MOE_GATE_K=2048
MOE_DOWN_N=1024
MOE_DOWN_K=3072

W4A16_M=4096
W4A16_N=4096
W4A16_K=4096
W4A16_GROUP=128

repo_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$path"
  fi
}

MACHETE_BIN="$(repo_path "w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm")"
FPA_BIN="$(repo_path "w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm")"
MOE_TRTLLM_BIN="$(repo_path "moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm")"
MOE_TRTLLM_AUX_DIR="$(repo_path "moe_w4a16/trtllm/auxiliary")"
MOE_VLLM_MARLIN_BIN="$(repo_path "moe_w4a16/vllm/marlin/bench_marlin_moe")"
MOE_VLLM_AUX_DIR="$(repo_path "moe_w4a16/vllm/auxiliary")"

MACHETE_TACTIC="$(repo_path "w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache")"
FPA_TACTIC="$(repo_path "w4a16_gemm/fpA_intB_standalone/tactics_h800.cache")"
MOE_TRTLLM_TACTIC="$(repo_path "moe_w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache")"

FAILED=0
LIST_CASES=0
MATCHED_CASES=0
RAN_CASES=0
CASE_FILTERS=()

usage() {
  cat <<'EOF'
Usage:
  ./bench_all.sh                         # run all benchmark cases
  ./bench_all.sh --list                  # list available case labels
  ./bench_all.sh --case LABEL            # run one case
  ./bench_all.sh --kernel LABEL          # alias for --case
  ./bench_all.sh --run-dir DIR           # run every benchmark with DIR as cwd
  ./bench_all.sh LABEL [LABEL ...]       # run selected cases

Case matching accepts exact labels or substrings. Examples:
  ./bench_all.sh w4a16_decode_fpA_intB
  ./bench_all.sh --case moe_gate_up_decode_vllm
  ./bench_all.sh decode_vllm

After each selected case, if RUN_DIR/perfrawlog exists, the script runs:
  python -m perf_model.perf_statistics_gen --report_dir_path <out>/perfstatistics/<case> --mp 16 . perfrawlog
Then it prints a compute_cycles/latency summary across per-case reports.

Environment variables:
  RUN_DIR                 Benchmark working directory. Default: PERF_MODEL_DIR or repo root.
  PERF_MODEL_DIR          Alias/default source for RUN_DIR.
  OUT_DIR                  Log output directory. Default: .bench_logs/bench_<timestamp>
  BENCH_RUN_ID             Timestamp/name used when OUT_DIR is not set.
  PERF_STATISTICS_DIR      Root directory for per-case perf statistics reports.
  PERF_STATISTICS_MP       perf_statistics_gen --mp value. Default: 16
  PERF_STATISTICS_GHZ      Clock used for latency summary. Default: 1.5.
  PERF_STATISTICS_SUMMARY  Set to 0 to skip the final perfstatistics table.
  PERFRAWLOG_CLEAR         Set to 0 to keep an existing perfrawlog before each case.
  PERFRAWLOG_POSTPROCESS   Set to 0 to skip perfrawlog post-processing.
EOF
}

add_case_filter() {
  local value="$1"
  local part
  local old_ifs="$IFS"
  IFS=','
  for part in $value; do
    if [[ -n "$part" ]]; then
      CASE_FILTERS+=("$part")
    fi
  done
  IFS="$old_ifs"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case|--kernel|--only)
      add_case_filter "${2:?missing value for $1}"
      shift 2
      ;;
    --case=*|--kernel=*|--only=*)
      add_case_filter "${1#*=}"
      shift
      ;;
    --list)
      LIST_CASES=1
      shift
      ;;
    --run-dir|--perf-model-dir)
      RUN_DIR="${2:?missing value for $1}"
      shift 2
      ;;
    --run-dir=*|--perf-model-dir=*)
      RUN_DIR="${1#*=}"
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        add_case_filter "$1"
        shift
      done
      ;;
    -*)
      echo "[bench_all][error] unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      add_case_filter "$1"
      shift
      ;;
  esac
done

require_bin() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "[bench_all][error] missing executable: $path" >&2
    echo "[bench_all][hint] build required targets first, for example:" >&2
    echo "  ./compile.sh build linear_attention w4a16-machete w4a16-fpa moe-trtllm moe-trtllm-auxiliary moe-vllm" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[bench_all][error] missing file: $path" >&2
    exit 1
  fi
}

safe_name() {
  echo "$1" | tr ' /:' '___' | tr -cd '[:alnum:]_.-'
}

case_selected() {
  local label="$1"
  local safe_label filter

  if [[ ${#CASE_FILTERS[@]} -eq 0 ]]; then
    return 0
  fi

  safe_label="$(safe_name "$label")"
  for filter in "${CASE_FILTERS[@]}"; do
    if [[ "$label" == "$filter" || "$safe_label" == "$filter" || "$label" == *"$filter"* ]]; then
      return 0
    fi
  done

  return 1
}

selection_name() {
  local name="all"
  if [[ ${#CASE_FILTERS[@]} -gt 0 ]]; then
    name="$(IFS=_; echo "${CASE_FILTERS[*]}")"
  fi
  safe_name "$name"
}

perfstatistics_base_dir() {
  printf '%s\n' "${PERF_STATISTICS_DIR:-$OUT_DIR/perfstatistics}"
}

perfstatistics_report_dir() {
  local label="$1"
  printf '%s/%s\n' "$(perfstatistics_base_dir)" "$(safe_name "$label")"
}

clear_perfrawlog_for_case() {
  if [[ "${PERFRAWLOG_POSTPROCESS:-1}" == 0 || "${PERFRAWLOG_CLEAR:-1}" == 0 ]]; then
    return
  fi

  local perfrawlog_path="${PERFRAWLOG_PATH:-$RUN_DIR/perfrawlog}"
  if [[ "$perfrawlog_path" == "$RUN_DIR/perfrawlog" && -e "$perfrawlog_path" ]]; then
    rm -rf "$perfrawlog_path"
  fi
}

run_case() {
  local label="$1"
  shift 1

  if [[ "$LIST_CASES" == 1 ]]; then
    echo "$label"
    return
  fi

  if ! case_selected "$label"; then
    return
  fi

  MATCHED_CASES=$((MATCHED_CASES + 1))

  local required_files=()
  while [[ $# -gt 0 && "$1" == "--require-file" ]]; do
    required_files+=("${2:?missing file after --require-file}")
    shift 2
  done

  local -a cmd=("$@")
  if [[ "${cmd[0]}" != /* ]]; then
    cmd[0]="$(repo_path "${cmd[0]}")"
  fi

  require_bin "${cmd[0]}"
  local required_file
  for required_file in "${required_files[@]}"; do
    require_file "$required_file"
  done

  RAN_CASES=$((RAN_CASES + 1))

  local safe
  safe="$(safe_name "$label")"
  local log="$OUT_DIR/$safe.log"

  echo
  echo "=== $label ==="
  printf '[bench_all] command:'
  printf ' cd %q &&' "$RUN_DIR"
  printf ' %q' "${cmd[@]}"
  echo
  echo "[bench_all] log: $log"

  {
    echo "label: $label"
    echo "run_dir: $RUN_DIR"
    printf 'command:'
    printf ' cd %q &&' "$RUN_DIR"
    printf ' %q' "${cmd[@]}"
    echo
    echo "started_at: $(date -Is)"
    echo "---- output ----"
  } >"$log"

  clear_perfrawlog_for_case

  set +e
  (cd "$RUN_DIR" && "${cmd[@]}") 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" == 0 ]]; then
    echo "finished_at: $(date -Is)" >>"$log"
    if run_perfrawlog_postprocess "$label" "${cmd[@]}"; then
      printf '[bench_all] %-34s ok\n' "$label"
    else
      local post_status=$?
      printf '[bench_all] %-34s failed perfstatistics_status=%s\n' "$label" "$post_status" >&2
      FAILED=1
    fi
  else
    echo "failed_at: $(date -Is)" >>"$log"
    echo "exit_status: $status" >>"$log"
    printf '[bench_all] %-34s failed exit_status=%s\n' "$label" "$status" >&2
    echo "[bench_all][error] last lines from $log:" >&2
    tail -80 "$log" >&2 || true
    FAILED=1
  fi
}

run_perfrawlog_postprocess() {
  local label="$1"
  shift 1
  local -a cmd=("$@")

  if [[ "${PERFRAWLOG_POSTPROCESS:-1}" == 0 ]]; then
    echo "[bench_all] PERFRAWLOG_POSTPROCESS=0, skip perfrawlog post-processing."
    return
  fi

  local perfrawlog_path="${PERFRAWLOG_PATH:-$RUN_DIR/perfrawlog}"
  if [[ ! -e "$perfrawlog_path" ]]; then
    echo "[bench_all] no perfrawlog found under run dir, skip perf statistics generation."
    return
  fi

  local report_dir log perfrawlog_arg
  report_dir="$(perfstatistics_report_dir "$label")"
  mkdir -p "$report_dir"
  log="$report_dir/perf_statistics_gen.log"
  perfrawlog_arg="$perfrawlog_path"
  if [[ "$perfrawlog_path" == "$RUN_DIR/perfrawlog" ]]; then
    perfrawlog_arg="perfrawlog"
  fi

  echo
  echo "=== perfrawlog statistics ==="
  echo "[bench_all] run_dir: $RUN_DIR"
  echo "[bench_all] perfrawlog: $perfrawlog_path"
  echo "[bench_all] report_dir: $report_dir"
  echo "[bench_all] log: $log"

  {
    echo "label: $label"
    echo "executable: ${cmd[0]:-}"
    echo "run_dir: $RUN_DIR"
    printf 'command:'
    printf ' %q' "${cmd[@]}"
    echo
    echo "perfrawlog: $perfrawlog_path"
    echo "report_dir: $report_dir"
  } >"$report_dir/bench_metadata.txt"

  {
    echo "report_dir: $report_dir"
    echo "mp: ${PERF_STATISTICS_MP:-16}"
    echo "run_dir: $RUN_DIR"
    echo "perfrawlog: $perfrawlog_path"
    echo "started_at: $(date -Is)"
    echo "command: cd $RUN_DIR && python -m perf_model.perf_statistics_gen --report_dir_path $report_dir --mp ${PERF_STATISTICS_MP:-16} . $perfrawlog_arg"
    echo "---- output ----"
  } >"$log"

  set +e
  (cd "$RUN_DIR" && python -m perf_model.perf_statistics_gen \
    --report_dir_path "$report_dir" \
    --mp "${PERF_STATISTICS_MP:-16}" \
    . "$perfrawlog_arg") 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

summarize_perfstatistics() {
  if [[ "${PERFRAWLOG_POSTPROCESS:-1}" == 0 || "${PERF_STATISTICS_SUMMARY:-1}" == 0 ]]; then
    return
  fi

  local report_base summary_log
  report_base="$(perfstatistics_base_dir)"
  if [[ ! -d "$report_base" ]]; then
    echo "[bench_all] no per-case perfstatistics directory found, skip summary."
    return
  fi

  summary_log="$OUT_DIR/perfstatistics_summary.txt"
  echo
  echo "=== perfstatistics summary ==="
  set +e
  python "$ROOT_DIR/helpers/summarize_perfstatistics.py" \
    "$report_base" \
    --ghz "${PERF_STATISTICS_GHZ:-1.5}" 2>&1 | tee "$summary_log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" != 0 ]]; then
    echo "[bench_all][warn] perfstatistics summary did not find any compute_cycles."
    return
  fi
  echo "[bench_all] perfstatistics summary: $summary_log"
}

if [[ "$LIST_CASES" != 1 ]]; then
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "[bench_all][error] run dir does not exist: $RUN_DIR" >&2
    exit 1
  fi
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$ROOT_DIR/$OUT_DIR"
  fi
  if [[ -n "${PERF_STATISTICS_DIR:-}" && "$PERF_STATISTICS_DIR" != /* ]]; then
    PERF_STATISTICS_DIR="$ROOT_DIR/$PERF_STATISTICS_DIR"
  fi
  mkdir -p "$OUT_DIR"
fi

if [[ "$LIST_CASES" == 1 ]]; then
  echo "Available benchmark cases:"
else
  echo "============================================================"
  echo "Qwen3.5-122B-A10B standalone kernel benchmark suite"
  echo "repo: $ROOT_DIR"
  echo "run dir: $RUN_DIR"
  echo "logs: $OUT_DIR"
  echo "prefill tokens: $PREFILL_TOKENS"
  echo "decode tokens:  $DECODE_TOKENS"
  echo "moe prefill:    TensorRT-LLM components"
  echo "moe decode:     vLLM components"
  if [[ ${#CASE_FILTERS[@]} -gt 0 ]]; then
    echo "case filters:   ${CASE_FILTERS[*]}"
  fi
  echo "============================================================"
fi

run_case "linear_decode_conv1d_update" \
  linear_attention/bench_conv1d_update "$LINEAR_DIM" "$CONV_WIDTH" "$DECODE_TOKENS" --bench 0 1

run_case "linear_decode_gdn" \
  linear_attention/bench_gated_delta_net "$DECODE_TOKENS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --bench 0 1

run_case "linear_prefill_conv1d_fwd" \
  linear_attention/bench_conv1d_fwd "$PREFILL_TOKENS" "$LINEAR_DIM" "$CONV_WIDTH" 1 --bench 0 1

run_case "linear_prefill_flashinfer_gdn" \
  linear_attention/bench_gdn_prefill "$PREFILL_TOKENS" "$LINEAR_Q_HEADS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --bench 0 1

run_case "w4a16_prefill_cutlass55" \
  --require-file "$MACHETE_TACTIC" \
  "$MACHETE_BIN" \
  --backend=cutlass55 \
  --cutlass55_tactic="$MACHETE_TACTIC" \
  --m="$W4A16_M" --n="$W4A16_N" --k="$W4A16_K" --group_size="$W4A16_GROUP" \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --profile_gemm_only --no_checksum \
  --warmup=0 --iters=1

run_case "w4a16_decode_fpA_intB" \
  --require-file "$FPA_TACTIC" \
  "$FPA_BIN" \
  --m="$DECODE_TOKENS" --n="$W4A16_N" --k="$W4A16_K" --group_size="$W4A16_GROUP" \
  --tactic="$FPA_TACTIC" \
  --warmup=0 --iters=1

run_case "moe_routing_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_custom_moe_routing" "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" fp16 \
  --bench 0 1

run_case "moe_expert_map_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_expert_map" "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" auto \
  --bench 0 1

run_case "moe_expand_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_expand_input_rows" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_GATE_K" fp16 \
  --bench 0 1

run_case "moe_gate_up_prefill_trtllm" \
  --require-file "$MOE_TRTLLM_TACTIC" \
  "$MOE_TRTLLM_BIN" \
  --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$PREFILL_TOKENS" \
  --n="$MOE_GATE_N" --k="$MOE_GATE_K" --group_size="$MOE_GROUP" \
  --tactic="$MOE_TRTLLM_TACTIC" \
  --warmup=0 --iters=1

run_case "moe_gated_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_gated_activation" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_GATE_N" fp16 \
  --bench 0 1

run_case "moe_down_prefill_trtllm" \
  --require-file "$MOE_TRTLLM_TACTIC" \
  "$MOE_TRTLLM_BIN" \
  --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$PREFILL_TOKENS" \
  --n="$MOE_DOWN_N" --k="$MOE_DOWN_K" --group_size="$MOE_GROUP" \
  --tactic="$MOE_TRTLLM_TACTIC" \
  --warmup=0 --iters=1

run_case "moe_finalize_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_finalize_moe_routing" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_DOWN_N" fp16 \
  --bench 0 1

run_case "moe_routing_decode_vllm" \
  "$MOE_VLLM_AUX_DIR/bench_topk_gating" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" \
  --bench 0 1

run_case "moe_align_decode_vllm" \
  "$MOE_VLLM_AUX_DIR/bench_moe_align" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" 16 \
  --bench 0 1

run_case "moe_gate_up_decode_vllm" \
  "$MOE_VLLM_MARLIN_BIN" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" "$MOE_GATE_K" "$MOE_GATE_N" \
  --balanced --no-topk-weights --bench 0 1

run_case "moe_gated_decode_vllm" \
  "$MOE_VLLM_AUX_DIR/bench_silu_and_mul" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_GATE_N" \
  --bench 0 1

run_case "moe_down_decode_vllm" \
  "$MOE_VLLM_MARLIN_BIN" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" "$MOE_DOWN_K" "$MOE_DOWN_N" \
  --balanced --bench 0 1

run_case "moe_finalize_decode_vllm" \
  "$MOE_VLLM_AUX_DIR/bench_moe_sum" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_DOWN_N" \
  --bench 0 1

if [[ "$LIST_CASES" == 1 ]]; then
  exit 0
fi

if [[ ${#CASE_FILTERS[@]} -gt 0 && "$MATCHED_CASES" == 0 ]]; then
  echo "[bench_all][error] no benchmark case matched: ${CASE_FILTERS[*]}" >&2
  echo "[bench_all][hint] run ./bench_all.sh --list to see available labels." >&2
  exit 1
fi

echo
echo "============================================================"
echo "benchmark logs are under: $OUT_DIR"
echo "ran cases: $RAN_CASES"
if [[ "$FAILED" == 0 ]]; then
  echo "All selected cases completed successfully."
else
  echo "Some selected cases failed."
fi
echo "============================================================"

summarize_perfstatistics

exit "$FAILED"
