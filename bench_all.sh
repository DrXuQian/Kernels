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
PYTHON_BIN="${PYTHON:-$(command -v python3 || true)}"
ATTN_BENCH_WARMUP="${ATTN_BENCH_WARMUP:-0}"
ATTN_BENCH_ITERS="${ATTN_BENCH_ITERS:-1}"

PREFILL_TOKENS=3823
DECODE_TOKENS=1
CTX_LEN="${CTX_LEN:-3823}"
LINEAR_DIM=12288
HIDDEN_DIM=3072
CONV_WIDTH=4
LINEAR_Q_HEADS=16
LINEAR_V_HEADS=64
LINEAR_HEAD_DIM=128
LINEAR_SMALL_PROJ_N=64

MOE_EXPERTS=8
MOE_ROUTER_EXPERTS=256
MOE_TOPK=8
MOE_GROUP=128
MOE_GATE_N=3072
MOE_GATE_K=2048
MOE_DOWN_N=1024
MOE_DOWN_K=3072
MOE_SHARED_HIDDEN=3072

W4A16_GROUP=128
W4A16_LINEAR_QKV_N=12288
W4A16_LINEAR_QKV_K=3072
W4A16_LINEAR_Z_N=8192
W4A16_LINEAR_Z_K=3072
W4A16_LINEAR_OUT_N=3072
W4A16_LINEAR_OUT_K=8192
W4A16_FULL_ATTN_Q_PROJ_GATE_N=16384
W4A16_FULL_ATTN_Q_PROJ_GATE_K=3072
W4A16_FULL_ATTN_K_PROJ_N=512
W4A16_FULL_ATTN_K_PROJ_K=3072
W4A16_FULL_ATTN_V_PROJ_N=512
W4A16_FULL_ATTN_V_PROJ_K=3072
W4A16_FULL_ATTN_O_PROJ_N=3072
W4A16_FULL_ATTN_O_PROJ_K=8192
FULL_ATTN_Q_HEADS=32
FULL_ATTN_KV_HEADS=2
FULL_ATTN_HEAD_DIM=256
W4A16_CONSISTENT_EXPERT_UP_N=3072
W4A16_CONSISTENT_EXPERT_UP_K=2048
W4A16_CONSISTENT_EXPERT_DOWN_N=1024
W4A16_CONSISTENT_EXPERT_DOWN_K=3072

SAMPLING_VOCAB=248320
SAMPLING_TOPK=50
SAMPLING_TOPP=0.9

repo_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$path"
  fi
}

MACHETE_BIN="$(repo_path "general/w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm")"
FPA_BIN="$(repo_path "general/w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm")"
CUBLAS_GEMM_BIN="$(repo_path "general/bench_cublas_gemm")"
MOE_TRTLLM_BIN="$(repo_path "moe_ffn/w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm")"
MOE_TRTLLM_AUX_DIR="$(repo_path "moe_ffn/w4a16/trtllm/auxiliary")"
MOE_VLLM_MARLIN_BIN="$(repo_path "moe_ffn/w4a16/vllm/marlin/bench_marlin_moe")"
MOE_VLLM_AUX_DIR="$(repo_path "moe_ffn/w4a16/vllm/auxiliary")"
LINEAR_RMSNORM_BIN="$(repo_path "linear_attn/bench_rmsnorm")"
LINEAR_OPS_BIN="$(repo_path "linear_attn/bench_linear_ops")"
FLASH_RMSNORM_BIN="$(repo_path "flash_attn/bench_rmsnorm")"
FLASH_ATTN_SCRIPT="$(repo_path "flash_attn/bench_flash_attn.py")"
MOE_RMSNORM_BIN="$(repo_path "moe_ffn/bench_rmsnorm")"
MOE_SHARED_EXPERT_BIN="$(repo_path "moe_ffn/bench_shared_expert")"
SAMPLING_BIN="$(repo_path "sampling/bench_sampling")"

MACHETE_TACTIC="$(repo_path "general/w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache")"
FPA_TACTIC="$(repo_path "general/w4a16_gemm/fpA_intB_standalone/tactics_h800.cache")"
MOE_TRTLLM_TACTIC="$(repo_path "moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache")"

FAILED=0
LIST_CASES=0
MATCHED_CASES=0
RAN_CASES=0
CASE_FILTERS=()
RESUME_FROM=""
RESUME_SEEN=1
RESUME_FOUND=0
NCU_CYCLES=0
NCU_METRICS="${NCU_METRICS:-sm__cycles_elapsed.avg,sm__cycles_elapsed.max,gpu__time_duration.sum}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-}"

usage() {
  cat <<'EOF'
Usage:
  ./bench_all.sh                         # run all benchmark cases
  ./bench_all.sh --list                  # list available case labels
  ./bench_all.sh --case LABEL            # run one case
  ./bench_all.sh --kernel LABEL          # alias for --case
  ./bench_all.sh --resume-from LABEL     # skip cases before LABEL, then continue
  ./bench_all.sh --run-dir DIR           # run every benchmark with DIR as cwd
  ./bench_all.sh --ncu-cycles            # run selected cases under Nsight Compute
  ./bench_all.sh LABEL [LABEL ...]       # run selected cases

Case matching accepts exact labels or substrings. Examples:
  ./bench_all.sh w4a16_decode_linear_attn_in_proj_qkv_fpA_intB
  ./bench_all.sh --case moe_gate_up_decode_vllm
  ./bench_all.sh decode_vllm
  ./bench_all.sh --resume-from w4a16_prefill_linear_attn_out_proj_cutlass55

Resume matching accepts an exact label or its sanitized form from --list.

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
  PYTHON                   Python executable for Python attention cases. Default: python3 in PATH.
  ATTN_BENCH_WARMUP        Warmup iterations for Python full-attention cases. Default: 0.
  ATTN_BENCH_ITERS         Timed iterations for Python full-attention cases. Default: 1.
  NCU_METRICS              Nsight Compute metrics for --ncu-cycles.
                           Default: sm__cycles_elapsed.avg,sm__cycles_elapsed.max,gpu__time_duration.sum
  NCU_LAUNCH_SKIP          Optional Nsight Compute --launch-skip value.
  NCU_LAUNCH_COUNT         Optional Nsight Compute --launch-count value.
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
    --ncu-cycles|--ncu)
      NCU_CYCLES=1
      PERFRAWLOG_POSTPROCESS=0
      shift
      ;;
    --ncu-launch-skip)
      NCU_LAUNCH_SKIP="${2:?missing value for $1}"
      shift 2
      ;;
    --ncu-launch-skip=*)
      NCU_LAUNCH_SKIP="${1#*=}"
      shift
      ;;
    --ncu-launch-count)
      NCU_LAUNCH_COUNT="${2:?missing value for $1}"
      shift 2
      ;;
    --ncu-launch-count=*)
      NCU_LAUNCH_COUNT="${1#*=}"
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

if [[ -n "$RESUME_FROM" ]]; then
  RESUME_SEEN=0
fi

require_bin() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "[bench_all][error] missing executable: $path" >&2
    echo "[bench_all][hint] build required targets first, for example:" >&2
    echo "  ./compile.sh build linear_attn flash_attn sampling moe-ffn moe-trtllm w4a16-machete w4a16-fpa" >&2
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

require_tactic_entry() {
  local path="$1"
  local key="$2"
  if [[ ! -f "$path" ]]; then
    echo "[bench_all][error] missing tactic cache: $path" >&2
    exit 1
  fi
  if ! grep -qF "$key" "$path"; then
    echo "[bench_all][error] tactic cache missing shape: $key" >&2
    echo "[bench_all][error] cache file: $path" >&2
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

  if ! case_after_resume_point "$label"; then
    return
  fi

  if ! case_selected "$label"; then
    return
  fi

  MATCHED_CASES=$((MATCHED_CASES + 1))

  local required_files=()
  local required_tactic_files=()
  local required_tactic_keys=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --require-file)
        required_files+=("${2:?missing file after --require-file}")
        shift 2
        ;;
      --require-tactic-entry)
        required_tactic_files+=("${2:?missing tactic file after --require-tactic-entry}")
        required_tactic_keys+=("${3:?missing tactic key after --require-tactic-entry}")
        shift 3
        ;;
      *)
        break
        ;;
    esac
  done

  local -a cmd=("$@")
  if [[ "${cmd[0]}" != /* ]]; then
    cmd[0]="$(repo_path "${cmd[0]}")"
  fi

  require_bin "${cmd[0]}"
  if [[ "$NCU_CYCLES" == 1 ]] && ! command -v ncu >/dev/null 2>&1; then
    echo "[bench_all][error] --ncu-cycles requested but ncu is not in PATH" >&2
    exit 1
  fi
  local required_file
  for required_file in "${required_files[@]}"; do
    require_file "$required_file"
  done
  local i
  for i in "${!required_tactic_files[@]}"; do
    require_tactic_entry "${required_tactic_files[$i]}" "${required_tactic_keys[$i]}"
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

  local status
  set +e
  if [[ "$NCU_CYCLES" == 1 ]]; then
    local ncu_dir ncu_log
    local -a ncu_cmd
    ncu_dir="$OUT_DIR/ncu"
    ncu_log="$ncu_dir/$safe.csv"
    mkdir -p "$ncu_dir"
    ncu_cmd=(ncu --target-processes all --kernel-name-base demangled --page raw --csv --metrics "$NCU_METRICS")
    if [[ -n "$NCU_LAUNCH_SKIP" ]]; then
      ncu_cmd+=(--launch-skip "$NCU_LAUNCH_SKIP")
    fi
    if [[ -n "$NCU_LAUNCH_COUNT" ]]; then
      ncu_cmd+=(--launch-count "$NCU_LAUNCH_COUNT")
    fi

    echo "[bench_all] ncu log: $ncu_log"
    printf '[bench_all] ncu command:'
    printf ' %q' "${ncu_cmd[@]}" "${cmd[@]}"
    echo
    {
      echo "ncu_log: $ncu_log"
      printf 'ncu_command:'
      printf ' %q' "${ncu_cmd[@]}" "${cmd[@]}"
      echo
    } >>"$log"

    (cd "$RUN_DIR" && "${ncu_cmd[@]}" "${cmd[@]}") 2>&1 | tee -a "$log" | tee "$ncu_log"
    status=${PIPESTATUS[0]}
  else
    (cd "$RUN_DIR" && "${cmd[@]}") 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
  fi
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
  echo "[bench_all] run_dir: $RUN_DIR"
  echo "[bench_all] perfrawlog: $perfrawlog_path"
  echo "[bench_all] report_dir: $report_dir"
  echo "[bench_all] log: $log"

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

summarize_ncu_cycles() {
  if [[ "$NCU_CYCLES" != 1 ]]; then
    return
  fi

  local ncu_dir summary_log
  ncu_dir="$OUT_DIR/ncu"
  if [[ ! -d "$ncu_dir" ]]; then
    echo "[bench_all] no Nsight Compute output directory found, skip ncu summary."
    return
  fi

  summary_log="$OUT_DIR/ncu_cycles_summary.md"
  echo
  echo "=== Nsight Compute cycles summary ==="
  set +e
  python "$ROOT_DIR/helpers/summarize_ncu_cycles.py" "$ncu_dir" --detail 2>&1 | tee "$summary_log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" != 0 ]]; then
    echo "[bench_all][warn] ncu cycles summary did not find any metric rows."
    return
  fi
  echo "[bench_all] ncu cycles summary: $summary_log"
}

run_w4a16_prefill_cutlass55_case() {
  local label="$1"
  local m="$2"
  local n="$3"
  local k="$4"

  run_case "$label" \
    --require-file "$MACHETE_TACTIC" \
    --require-tactic-entry "$MACHETE_TACTIC" "$m,$n,$k,$W4A16_GROUP,fp16|" \
    "$MACHETE_BIN" \
    --backend=cutlass55 \
    --cutlass55_tactic="$MACHETE_TACTIC" \
    --m="$m" --n="$n" --k="$k" --group_size="$W4A16_GROUP" \
    --act=fp16 --quant=cutlass_s4 \
    --offline_prepack --profile_gemm_only --no_checksum \
    --warmup=0 --iters=1
}

run_w4a16_decode_fpa_case() {
  local label="$1"
  local m="$2"
  local n="$3"
  local k="$4"

  run_case "$label" \
    --require-file "$FPA_TACTIC" \
    --require-tactic-entry "$FPA_TACTIC" "$m,$n,$k,$W4A16_GROUP|" \
    "$FPA_BIN" \
    --m="$m" --n="$n" --k="$k" --group_size="$W4A16_GROUP" \
    --tactic="$FPA_TACTIC" \
    --warmup=0 --iters=1
}

run_rmsnorm_case() {
  local label="$1"
  local bin="$2"
  local tokens="$3"

  run_rmsnorm_shape_case "$label" "$bin" "$tokens" "$HIDDEN_DIM"
}

run_rmsnorm_shape_case() {
  local label="$1"
  local bin="$2"
  local batch="$3"
  local embed="$4"

  run_case "$label" \
    "$bin" \
    --batch "$batch" --embed "$embed" --dtype fp16 --no-check \
    --bench 0 1
}

run_flash_attn_core_case() {
  local label="$1"
  local mode="$2"
  local seq_len="$3"

  run_case "$label" \
    --require-file "$FLASH_ATTN_SCRIPT" \
    "$PYTHON_BIN" "$FLASH_ATTN_SCRIPT" \
    "$mode" "$seq_len" "$FULL_ATTN_Q_HEADS" "$FULL_ATTN_KV_HEADS" "$FULL_ATTN_HEAD_DIM" \
    --bench "$ATTN_BENCH_WARMUP" "$ATTN_BENCH_ITERS"
}

run_sampling_case() {
  local label="$1"
  local op="$2"

  run_case "$label" \
    "$SAMPLING_BIN" \
    --op="$op" --hidden="$HIDDEN_DIM" --vocab="$SAMPLING_VOCAB" \
    --top-k="$SAMPLING_TOPK" --top-p="$SAMPLING_TOPP" \
    --bench 0 1
}

run_cublas_gemm_case() {
  local label="$1"
  local m="$2"
  local n="$3"
  local k="$4"
  local out_dtype="${5:-fp16}"

  run_case "$label" \
    "$CUBLAS_GEMM_BIN" \
    --m="$m" --n="$n" --k="$k" --dtype fp16 --out-dtype "$out_dtype" \
    --bench 0 1
}

run_moe_shared_expert_case() {
  local label="$1"
  local op="$2"
  local tokens="$3"
  local out_dim="${4:-1}"

  run_case "$label" \
    "$MOE_SHARED_EXPERT_BIN" \
    --op="$op" --tokens="$tokens" --hidden="$MOE_SHARED_HIDDEN" --out-dim="$out_dim" --dtype fp16 \
    --bench 0 1
}

run_linear_dense_case() {
  local label="$1"
  local op="$2"
  local tokens="$3"
  local out_dim="$4"

  run_cublas_gemm_case "$label" "$tokens" "$out_dim" "$HIDDEN_DIM" fp16
}

run_residual_add_case() {
  local label="$1"
  local tokens="$2"

  run_case "$label" \
    "$LINEAR_OPS_BIN" \
    --op=residual_add --tokens="$tokens" --hidden="$HIDDEN_DIM" --dtype fp16 \
    --bench 0 1
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
  echo "ctx len:        $CTX_LEN"
  echo "moe prefill:    TensorRT-LLM components"
  echo "moe decode:     vLLM components"
  if [[ ${#CASE_FILTERS[@]} -gt 0 ]]; then
    echo "case filters:   ${CASE_FILTERS[*]}"
  fi
  if [[ -n "$RESUME_FROM" ]]; then
    echo "resume from:    $RESUME_FROM"
  fi
  if [[ "$NCU_CYCLES" == 1 ]]; then
    echo "ncu cycles:     enabled"
    echo "ncu metrics:    $NCU_METRICS"
    if [[ -n "$NCU_LAUNCH_SKIP" ]]; then
      echo "ncu skip:       $NCU_LAUNCH_SKIP"
    fi
    if [[ -n "$NCU_LAUNCH_COUNT" ]]; then
      echo "ncu count:      $NCU_LAUNCH_COUNT"
    fi
  fi
  echo "============================================================"
fi

run_rmsnorm_case "linear_attn_decode_rmsnorm" "$LINEAR_RMSNORM_BIN" "$DECODE_TOKENS"
run_rmsnorm_case "linear_attn_prefill_rmsnorm" "$LINEAR_RMSNORM_BIN" "$PREFILL_TOKENS"

run_linear_dense_case "linear_attn_decode_in_proj_a_cublas" "in_proj_a" "$DECODE_TOKENS" "$LINEAR_SMALL_PROJ_N"
run_linear_dense_case "linear_attn_decode_in_proj_b_cublas" "in_proj_b" "$DECODE_TOKENS" "$LINEAR_SMALL_PROJ_N"

run_linear_dense_case "linear_attn_prefill_in_proj_a_cublas" "in_proj_a" "$PREFILL_TOKENS" "$LINEAR_SMALL_PROJ_N"
run_linear_dense_case "linear_attn_prefill_in_proj_b_cublas" "in_proj_b" "$PREFILL_TOKENS" "$LINEAR_SMALL_PROJ_N"

run_case "linear_decode_conv1d_update" \
  linear_attn/bench_conv1d_update "$LINEAR_DIM" "$CONV_WIDTH" "$DECODE_TOKENS" --bench 0 1

run_case "linear_decode_gdn" \
  linear_attn/bench_gated_delta_net "$DECODE_TOKENS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --bench 0 1

run_case "linear_prefill_conv1d_fwd" \
  linear_attn/bench_conv1d_fwd "$PREFILL_TOKENS" "$LINEAR_DIM" "$CONV_WIDTH" 1 --bench 0 1

run_case "linear_prefill_flashinfer_gdn" \
  linear_attn/bench_gdn_prefill "$PREFILL_TOKENS" "$LINEAR_Q_HEADS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --bench 0 1

run_residual_add_case "linear_attn_decode_residual_add" "$DECODE_TOKENS"
run_residual_add_case "linear_attn_prefill_residual_add" "$PREFILL_TOKENS"

run_rmsnorm_case "flash_attn_decode_rmsnorm" "$FLASH_RMSNORM_BIN" "$DECODE_TOKENS"
run_rmsnorm_case "flash_attn_prefill_rmsnorm" "$FLASH_RMSNORM_BIN" "$PREFILL_TOKENS"

run_residual_add_case "flash_attn_decode_residual_add" "$DECODE_TOKENS"
run_residual_add_case "flash_attn_prefill_residual_add" "$PREFILL_TOKENS"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_linear_attn_in_proj_qkv_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_LINEAR_QKV_N" "$W4A16_LINEAR_QKV_K"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_linear_attn_in_proj_z_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_LINEAR_Z_N" "$W4A16_LINEAR_Z_K"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_linear_attn_out_proj_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_LINEAR_OUT_N" "$W4A16_LINEAR_OUT_K"

run_w4a16_decode_fpa_case "w4a16_decode_linear_attn_in_proj_qkv_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_LINEAR_QKV_N" "$W4A16_LINEAR_QKV_K"

run_w4a16_decode_fpa_case "w4a16_decode_linear_attn_in_proj_z_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_LINEAR_Z_N" "$W4A16_LINEAR_Z_K"

run_w4a16_decode_fpa_case "w4a16_decode_linear_attn_out_proj_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_LINEAR_OUT_N" "$W4A16_LINEAR_OUT_K"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_full_attn_q_proj_gate_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_Q_PROJ_GATE_N" "$W4A16_FULL_ATTN_Q_PROJ_GATE_K"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_full_attn_k_proj_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_K_PROJ_N" "$W4A16_FULL_ATTN_K_PROJ_K"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_full_attn_v_proj_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_V_PROJ_N" "$W4A16_FULL_ATTN_V_PROJ_K"

run_rmsnorm_shape_case "flash_attn_prefill_q_norm" \
  "$FLASH_RMSNORM_BIN" "$((PREFILL_TOKENS * FULL_ATTN_Q_HEADS))" "$FULL_ATTN_HEAD_DIM"

run_rmsnorm_shape_case "flash_attn_prefill_k_norm" \
  "$FLASH_RMSNORM_BIN" "$((PREFILL_TOKENS * FULL_ATTN_KV_HEADS))" "$FULL_ATTN_HEAD_DIM"

run_flash_attn_core_case "flash_attn_prefill_full_attn" \
  prefill "$PREFILL_TOKENS"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_full_attn_o_proj_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_O_PROJ_N" "$W4A16_FULL_ATTN_O_PROJ_K"

run_w4a16_decode_fpa_case "w4a16_decode_full_attn_q_proj_gate_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_FULL_ATTN_Q_PROJ_GATE_N" "$W4A16_FULL_ATTN_Q_PROJ_GATE_K"

run_w4a16_decode_fpa_case "w4a16_decode_full_attn_k_proj_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_FULL_ATTN_K_PROJ_N" "$W4A16_FULL_ATTN_K_PROJ_K"

run_w4a16_decode_fpa_case "w4a16_decode_full_attn_v_proj_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_FULL_ATTN_V_PROJ_N" "$W4A16_FULL_ATTN_V_PROJ_K"

run_rmsnorm_shape_case "flash_attn_decode_q_norm" \
  "$FLASH_RMSNORM_BIN" "$((DECODE_TOKENS * FULL_ATTN_Q_HEADS))" "$FULL_ATTN_HEAD_DIM"

run_rmsnorm_shape_case "flash_attn_decode_k_norm" \
  "$FLASH_RMSNORM_BIN" "$((DECODE_TOKENS * FULL_ATTN_KV_HEADS))" "$FULL_ATTN_HEAD_DIM"

run_flash_attn_core_case "flash_attn_decode_full_attn" \
  decode "$CTX_LEN"

run_w4a16_decode_fpa_case "w4a16_decode_full_attn_o_proj_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_FULL_ATTN_O_PROJ_N" "$W4A16_FULL_ATTN_O_PROJ_K"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_consistent_expert_up_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_CONSISTENT_EXPERT_UP_N" "$W4A16_CONSISTENT_EXPERT_UP_K"

run_w4a16_prefill_cutlass55_case "w4a16_prefill_consistent_expert_down_cutlass55" \
  "$PREFILL_TOKENS" "$W4A16_CONSISTENT_EXPERT_DOWN_N" "$W4A16_CONSISTENT_EXPERT_DOWN_K"

run_cublas_gemm_case "moe_router_gate_prefill_cublas" \
  "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$HIDDEN_DIM" fp16

run_cublas_gemm_case "moe_shared_expert_gate_prefill_cublas" \
  "$PREFILL_TOKENS" 1 "$HIDDEN_DIM" fp16

run_moe_shared_expert_case "moe_shared_expert_fusion_prefill" "sigmoid_mul_add" "$PREFILL_TOKENS"

run_w4a16_decode_fpa_case "w4a16_decode_consistent_expert_up_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_CONSISTENT_EXPERT_UP_N" "$W4A16_CONSISTENT_EXPERT_UP_K"

run_w4a16_decode_fpa_case "w4a16_decode_consistent_expert_down_fpA_intB" \
  "$DECODE_TOKENS" "$W4A16_CONSISTENT_EXPERT_DOWN_N" "$W4A16_CONSISTENT_EXPERT_DOWN_K"

run_cublas_gemm_case "moe_router_gate_decode_cublas" \
  "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$HIDDEN_DIM" fp16

run_cublas_gemm_case "moe_shared_expert_gate_decode_cublas" \
  "$DECODE_TOKENS" 1 "$HIDDEN_DIM" fp16

run_moe_shared_expert_case "moe_shared_expert_fusion_decode" "sigmoid_mul_add" "$DECODE_TOKENS"

run_rmsnorm_case "moe_ffn_decode_rmsnorm" "$MOE_RMSNORM_BIN" "$DECODE_TOKENS"
run_rmsnorm_case "moe_ffn_prefill_rmsnorm" "$MOE_RMSNORM_BIN" "$PREFILL_TOKENS"

run_residual_add_case "moe_ffn_decode_residual_add" "$DECODE_TOKENS"
run_residual_add_case "moe_ffn_prefill_residual_add" "$PREFILL_TOKENS"

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

run_cublas_gemm_case "sampling_lm_head_gemm" \
  "$DECODE_TOKENS" "$SAMPLING_VOCAB" "$HIDDEN_DIM" fp32
run_sampling_case "sampling_topk_mask_logits" "topk_mask"
run_sampling_case "sampling_softmax" "softmax"
run_sampling_case "sampling_top_p" "top_p"

if [[ "$LIST_CASES" == 1 ]]; then
  exit 0
fi

if [[ -n "$RESUME_FROM" && "$RESUME_FOUND" == 0 ]]; then
  echo "[bench_all][error] resume label was not found: $RESUME_FROM" >&2
  echo "[bench_all][hint] run ./bench_all.sh --list to see available labels." >&2
  exit 1
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
summarize_ncu_cycles

exit "$FAILED"
