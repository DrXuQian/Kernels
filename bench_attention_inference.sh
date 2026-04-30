#!/usr/bin/env bash
# Qwen3.5-122B-A10B Python/JIT attention inference benchmark suite.
#
# This complements bench_all.sh with kernels that are normally launched through
# Python runtimes: vLLM/FLA Triton linear attention and FlashAttention
# inference paths.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-${PERF_MODEL_DIR:-$ROOT_DIR}}"

BENCH_RUN_ID="${BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/.bench_logs/attention_$BENCH_RUN_ID}"

PYTHON="${PYTHON:-python3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT_DIR/.triton_cache}"

PREFILL_TOKENS="${PREFILL_TOKENS:-3823}"
DECODE_TOKENS="${DECODE_TOKENS:-1}"
CTX_LEN="${CTX_LEN:-3823}"

LINEAR_Q_HEADS="${LINEAR_Q_HEADS:-16}"
LINEAR_V_HEADS="${LINEAR_V_HEADS:-64}"
LINEAR_HEAD_DIM="${LINEAR_HEAD_DIM:-128}"

FULL_ATTN_HEADS="${FULL_ATTN_HEADS:-32}"
FULL_ATTN_KV_HEADS="${FULL_ATTN_KV_HEADS:-2}"
FULL_ATTN_HEAD_DIM="${FULL_ATTN_HEAD_DIM:-256}"

TRITON_DTYPE="${TRITON_DTYPE:-bf16}"
FLASH_DTYPE="${FLASH_DTYPE:-fp16}"
BENCH_WARMUP="${BENCH_WARMUP:-0}"
BENCH_ITERS="${BENCH_ITERS:-1}"
FLASHINFER_BACKEND="${FLASHINFER_BACKEND:-}"

FAILED=0
LIST_CASES=0
MATCHED_CASES=0
RAN_CASES=0
CASE_FILTERS=()
RESUME_FROM=""
RESUME_SEEN=1
RESUME_FOUND=0

repo_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$path"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./bench_attention_inference.sh                    # run all attention inference cases
  ./bench_attention_inference.sh --list             # list available case labels
  ./bench_attention_inference.sh --case LABEL       # run one case
  ./bench_attention_inference.sh LABEL [LABEL ...]  # run selected cases
  ./bench_attention_inference.sh --resume-from LABEL
  ./bench_attention_inference.sh --run-dir DIR

Case matching accepts exact labels or substrings.

Cases:
  linear_triton_decode_gdn
  linear_triton_prefill_gdn
  linear_triton_prefill_gdn_core_only
  flashinfer_decode_full_attn
  flashinfer_prefill_full_attn
  flash_attn_decode_full_attn
  flash_attn_prefill_full_attn

Environment variables:
  PYTHON                  Python executable. Default: python3
  RUN_DIR                 Benchmark working directory. Default: PERF_MODEL_DIR or repo root.
  OUT_DIR                 Log output directory. Default: .bench_logs/attention_<timestamp>
  TRITON_CACHE_DIR        Triton JIT cache. Default: .triton_cache
  PREFILL_TOKENS          Default: 3823
  CTX_LEN                 FlashAttention decode KV cache length. Default: 3823
  BENCH_WARMUP            Default: 0
  BENCH_ITERS             Default: 1
  TRITON_DTYPE            fp16 or bf16. Default: bf16
  FLASH_DTYPE             fp16 or bf16. Default: fp16
  FLASHINFER_BACKEND      Optional FlashInfer prefill backend, e.g. fa2/fa3/auto.
  PERFRAWLOG_POSTPROCESS  Set to 0 to skip perfrawlog post-processing.
  PERF_STATISTICS_SUMMARY Set to 0 to skip final perfstatistics summary.

For a clean nsys capture of Python kernels that use cudaProfilerStart/Stop:
  nsys profile --trace=cuda --capture-range=cudaProfilerApi \
    --capture-range-end=stop ./bench_attention_inference.sh --case flashinfer_decode_full_attn

For Triton, run once to populate TRITON_CACHE_DIR before profiling if you want
to avoid first-run JIT compile overhead in host-side traces.
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
    --resume|--resume-from)
      RESUME_FROM="${2:?missing value for $1}"
      shift 2
      ;;
    --resume=*|--resume-from=*)
      RESUME_FROM="${1#*=}"
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
      echo "[bench_attention][error] unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      add_case_filter "$1"
      shift
      ;;
  esac
done

if [[ -n "$RESUME_FROM" ]]; then
  RESUME_SEEN=0
fi

require_command_or_file() {
  local path="$1"
  if [[ "$path" == */* ]]; then
    if [[ ! -e "$path" ]]; then
      echo "[bench_attention][error] missing command/file: $path" >&2
      exit 1
    fi
  elif ! command -v "$path" >/dev/null 2>&1; then
    echo "[bench_attention][error] command not found: $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[bench_attention][error] missing file: $path" >&2
    exit 1
  fi
}

safe_name() {
  echo "$1" | tr ' /:' '___' | tr -cd '[:alnum:]_.-'
}

label_matches_filter() {
  local label="$1"
  local filter="$2"
  local safe_label
  safe_label="$(safe_name "$label")"

  [[ "$label" == "$filter" || "$safe_label" == "$filter" || "$label" == *"$filter"* ]]
}

label_matches_exact() {
  local label="$1"
  local query="$2"
  local safe_label safe_query
  safe_label="$(safe_name "$label")"
  safe_query="$(safe_name "$query")"

  [[ "$label" == "$query" || "$safe_label" == "$query" || "$safe_label" == "$safe_query" ]]
}

case_selected() {
  local label="$1"
  local filter

  if [[ ${#CASE_FILTERS[@]} -eq 0 ]]; then
    return 0
  fi

  for filter in "${CASE_FILTERS[@]}"; do
    if label_matches_filter "$label" "$filter"; then
      return 0
    fi
  done

  return 1
}

case_after_resume_point() {
  local label="$1"

  if [[ -z "$RESUME_FROM" || "$RESUME_SEEN" == 1 ]]; then
    return 0
  fi

  if label_matches_exact "$label" "$RESUME_FROM"; then
    RESUME_SEEN=1
    RESUME_FOUND=1
    return 0
  fi

  return 1
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

run_perfrawlog_postprocess() {
  local label="$1"
  shift 1
  local -a cmd=("$@")

  if [[ "${PERFRAWLOG_POSTPROCESS:-1}" == 0 ]]; then
    echo "[bench_attention] PERFRAWLOG_POSTPROCESS=0, skip perfrawlog post-processing."
    return
  fi

  local perfrawlog_path="${PERFRAWLOG_PATH:-$RUN_DIR/perfrawlog}"
  if [[ ! -e "$perfrawlog_path" ]]; then
    echo "[bench_attention] no perfrawlog found under run dir, skip perf statistics generation."
    return
  fi

  local report_dir log_dir log perfrawlog_arg
  report_dir="$(perfstatistics_report_dir "$label")"
  log_dir="$OUT_DIR/perfstatistics_logs"
  mkdir -p "$log_dir" "$(dirname "$report_dir")"
  log="$log_dir/$(safe_name "$label").log"
  perfrawlog_arg="$perfrawlog_path"
  if [[ "$perfrawlog_path" == "$RUN_DIR/perfrawlog" ]]; then
    perfrawlog_arg="perfrawlog"
  fi

  echo
  echo "=== perfrawlog statistics ==="
  echo "[bench_attention] run_dir: $RUN_DIR"
  echo "[bench_attention] perfrawlog: $perfrawlog_path"
  echo "[bench_attention] report_dir: $report_dir"
  echo "[bench_attention] log: $log"

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

  mkdir -p "$report_dir"
  {
    echo "label: $label"
    echo "executable: ${cmd[0]:-}"
    echo "run_dir: $RUN_DIR"
    printf 'command:'
    printf ' %q' "${cmd[@]}"
    echo
    echo "perfrawlog: $perfrawlog_path"
    echo "report_dir: $report_dir"
    echo "perf_statistics_log: $log"
  } >"$report_dir/bench_metadata.txt"

  return "$status"
}

run_case() {
  local label="$1"
  shift 1

  if [[ "$LIST_CASES" == 1 ]]; then
    echo "$label"
    return
  fi

  if ! case_after_resume_point "$label"; then
    return
  fi

  if ! case_selected "$label"; then
    return
  fi

  MATCHED_CASES=$((MATCHED_CASES + 1))

  local required_files=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --require-file)
        required_files+=("${2:?missing file after --require-file}")
        shift 2
        ;;
      *)
        break
        ;;
    esac
  done

  local -a cmd=("$@")
  require_command_or_file "${cmd[0]}"
  local required_file
  for required_file in "${required_files[@]}"; do
    require_file "$required_file"
  done

  RAN_CASES=$((RAN_CASES + 1))

  local safe log
  safe="$(safe_name "$label")"
  log="$OUT_DIR/$safe.log"

  echo
  echo "=== $label ==="
  printf '[bench_attention] command:'
  printf ' cd %q &&' "$RUN_DIR"
  printf ' %q' "${cmd[@]}"
  echo
  echo "[bench_attention] log: $log"

  {
    echo "label: $label"
    echo "run_dir: $RUN_DIR"
    echo "triton_cache_dir: $TRITON_CACHE_DIR"
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
      printf '[bench_attention] %-38s ok\n' "$label"
    else
      local post_status=$?
      printf '[bench_attention] %-38s failed perfstatistics_status=%s\n' "$label" "$post_status" >&2
      FAILED=1
    fi
  else
    echo "failed_at: $(date -Is)" >>"$log"
    echo "exit_status: $status" >>"$log"
    printf '[bench_attention] %-38s failed exit_status=%s\n' "$label" "$status" >&2
    echo "[bench_attention][error] last lines from $log:" >&2
    tail -80 "$log" >&2 || true
    FAILED=1
  fi
}

summarize_perfstatistics() {
  if [[ "${PERFRAWLOG_POSTPROCESS:-1}" == 0 || "${PERF_STATISTICS_SUMMARY:-1}" == 0 ]]; then
    return
  fi

  local report_base summary_log
  report_base="$(perfstatistics_base_dir)"
  if [[ ! -d "$report_base" ]]; then
    echo "[bench_attention] no per-case perfstatistics directory found, skip summary."
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
    echo "[bench_attention][warn] perfstatistics summary did not find any compute_cycles."
    return
  fi
  echo "[bench_attention] perfstatistics summary: $summary_log"
}

run_python_case() {
  local label="$1"
  local script="$2"
  shift 2
  local script_path
  script_path="$(repo_path "$script")"

  run_case "$label" \
    --require-file "$script_path" \
    "$PYTHON" "$script_path" "$@"
}

if [[ "$LIST_CASES" != 1 ]]; then
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "[bench_attention][error] run dir does not exist: $RUN_DIR" >&2
    exit 1
  fi
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$ROOT_DIR/$OUT_DIR"
  fi
  if [[ -n "${PERF_STATISTICS_DIR:-}" && "$PERF_STATISTICS_DIR" != /* ]]; then
    PERF_STATISTICS_DIR="$ROOT_DIR/$PERF_STATISTICS_DIR"
  fi
  mkdir -p "$OUT_DIR" "$TRITON_CACHE_DIR"
fi

if [[ "$LIST_CASES" == 1 ]]; then
  echo "Available attention inference benchmark cases:"
else
  echo "============================================================"
  echo "Qwen3.5-122B-A10B attention inference benchmark suite"
  echo "repo: $ROOT_DIR"
  echo "run dir: $RUN_DIR"
  echo "logs: $OUT_DIR"
  echo "python: $PYTHON"
  echo "triton cache: $TRITON_CACHE_DIR"
  echo "prefill tokens: $PREFILL_TOKENS"
  echo "decode tokens:  $DECODE_TOKENS"
  echo "ctx len:        $CTX_LEN"
  echo "bench:          warmup=$BENCH_WARMUP iters=$BENCH_ITERS"
  if [[ ${#CASE_FILTERS[@]} -gt 0 ]]; then
    echo "case filters:   ${CASE_FILTERS[*]}"
  fi
  if [[ -n "$RESUME_FROM" ]]; then
    echo "resume from:    $RESUME_FROM"
  fi
  echo "============================================================"
fi

run_python_case "linear_triton_decode_gdn" \
  "linear_attention/src/bench_vllm_triton_gdn_decode.py" \
  "$DECODE_TOKENS" "$LINEAR_Q_HEADS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" \
  --dtype "$TRITON_DTYPE" --bench "$BENCH_WARMUP" "$BENCH_ITERS"

run_python_case "linear_triton_prefill_gdn" \
  "linear_attention/src/bench_vllm_triton_gdn_prefill.py" \
  "$PREFILL_TOKENS" "$LINEAR_Q_HEADS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 \
  --dtype "$TRITON_DTYPE" --bench "$BENCH_WARMUP" "$BENCH_ITERS"

run_python_case "linear_triton_prefill_gdn_core_only" \
  "linear_attention/src/bench_vllm_triton_gdn_prefill.py" \
  "$PREFILL_TOKENS" "$LINEAR_Q_HEADS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 \
  --dtype "$TRITON_DTYPE" --core-only --bench "$BENCH_WARMUP" "$BENCH_ITERS"

run_python_case "flashinfer_decode_full_attn" \
  "flash_attn/bench_flash_infer.py" \
  decode "$CTX_LEN" "$FULL_ATTN_HEADS" "$FULL_ATTN_KV_HEADS" "$FULL_ATTN_HEAD_DIM" \
  --dtype "$FLASH_DTYPE" --use-tensor-cores --bench "$BENCH_WARMUP" "$BENCH_ITERS"

flashinfer_prefill_extra=()
if [[ -n "$FLASHINFER_BACKEND" ]]; then
  flashinfer_prefill_extra=(--backend "$FLASHINFER_BACKEND")
fi

run_python_case "flashinfer_prefill_full_attn" \
  "flash_attn/bench_flash_infer.py" \
  prefill "$PREFILL_TOKENS" "$FULL_ATTN_HEADS" "$FULL_ATTN_KV_HEADS" "$FULL_ATTN_HEAD_DIM" \
  --dtype "$FLASH_DTYPE" "${flashinfer_prefill_extra[@]}" --bench "$BENCH_WARMUP" "$BENCH_ITERS"

run_python_case "flash_attn_decode_full_attn" \
  "flash_attn/bench_flash_attn.py" \
  decode "$CTX_LEN" "$FULL_ATTN_HEADS" "$FULL_ATTN_KV_HEADS" "$FULL_ATTN_HEAD_DIM" \
  --bench "$BENCH_WARMUP" "$BENCH_ITERS"

run_python_case "flash_attn_prefill_full_attn" \
  "flash_attn/bench_flash_attn.py" \
  prefill "$PREFILL_TOKENS" "$FULL_ATTN_HEADS" "$FULL_ATTN_KV_HEADS" "$FULL_ATTN_HEAD_DIM" \
  --bench "$BENCH_WARMUP" "$BENCH_ITERS"

if [[ "$LIST_CASES" == 1 ]]; then
  exit 0
fi

if [[ -n "$RESUME_FROM" && "$RESUME_FOUND" == 0 ]]; then
  echo "[bench_attention][error] resume label was not found: $RESUME_FROM" >&2
  echo "[bench_attention][hint] run ./bench_attention_inference.sh --list to see available labels." >&2
  exit 1
fi

if [[ ${#CASE_FILTERS[@]} -gt 0 && "$MATCHED_CASES" == 0 ]]; then
  echo "[bench_attention][error] no benchmark case matched: ${CASE_FILTERS[*]}" >&2
  echo "[bench_attention][hint] run ./bench_attention_inference.sh --list to see available labels." >&2
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
