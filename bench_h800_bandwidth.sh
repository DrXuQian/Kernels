#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS="qwen27,minimax-tp1,minimax-tp2,minimax-tp4"
PHASE="decode"
BACKEND="auto"
RUN_PREFLIGHT=1
CONTINUE_ON_ERROR=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./bench_h800_bandwidth.sh [options] [-- extra wrapper args]

Runs model benchmark wrappers serially for H800 bandwidth-utilization
collection. By default it tries NCU DRAM counters first and falls back to nsys
kernel duration plus theoretical benchmark bytes when counters are unavailable.

Options:
  --models LIST        Comma-separated models. Default:
                       qwen27,minimax-tp1,minimax-tp2,minimax-tp4
                       Valid names: qwen27, qwen122-tp2, minimax,
                       minimax-tp1, minimax-tp2, minimax-tp4, all
  --phase PHASE        decode, prefill, or all. Default: decode
  --case LABEL         Forward a more specific case filter instead of PHASE.
  --backend BACKEND    auto, ncu, or nsys. Default: auto
  --skip-preflight     Skip helpers/ncu_bandwidth_preflight.sh.
  --continue-on-error  Continue remaining model wrappers after a failure.
  -h, --help           Show this help.

Examples:
  ./bench_h800_bandwidth.sh
  ./bench_h800_bandwidth.sh --models qwen27 --phase decode
  ./bench_h800_bandwidth.sh --models minimax --phase all
  ./bench_h800_bandwidth.sh --models minimax-tp4 --case moe_gate_up_decode_fp8_trtllm

Environment:
  BENCH_RUN_ID_PREFIX  Prefix for per-wrapper BENCH_RUN_ID. Default: h800_bw
  OUT_DIR              Do not set globally for this script unless running one
                       wrapper only; each wrapper normally gets its own OUT_DIR.
  BANDWIDTH_OUT_ROOT   Parent directory for per-wrapper output dirs.
                       Default: .bench_logs
  NCU_PEAK_GBPS        Peak DRAM GB/s for bandwidth summary. Default inherited
                       by run_all wrapper, currently 3350.
EOF
}

add_models() {
  local value="$1"
  value="${value// /}"
  case "$value" in
    all)
      MODELS="qwen27,qwen122-tp2,minimax-tp1,minimax-tp2,minimax-tp4"
      ;;
    minimax)
      MODELS="minimax-tp1,minimax-tp2,minimax-tp4"
      ;;
    *)
      MODELS="$value"
      ;;
  esac
}

CASE_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      add_models "${2:?missing value for --models}"
      shift 2
      ;;
    --models=*)
      add_models "${1#*=}"
      shift
      ;;
    --phase)
      PHASE="${2:?missing value for --phase}"
      shift 2
      ;;
    --phase=*)
      PHASE="${1#*=}"
      shift
      ;;
    --case|--kernel|--only)
      CASE_FILTER="${2:?missing value for $1}"
      shift 2
      ;;
    --case=*|--kernel=*|--only=*)
      CASE_FILTER="${1#*=}"
      shift
      ;;
    --backend)
      BACKEND="${2:?missing value for --backend}"
      shift 2
      ;;
    --backend=*)
      BACKEND="${1#*=}"
      shift
      ;;
    --skip-preflight)
      RUN_PREFLIGHT=0
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "[h800-bandwidth][error] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PHASE" in
  decode|prefill|all) ;;
  *)
    echo "[h800-bandwidth][error] --phase must be decode, prefill, or all; got: $PHASE" >&2
    exit 2
    ;;
esac

case "$BACKEND" in
  auto|ncu|nsys) ;;
  *)
    echo "[h800-bandwidth][error] --backend must be auto, ncu, or nsys; got: $BACKEND" >&2
    exit 2
    ;;
esac

wrapper_for_model() {
  case "$1" in
    qwen27) echo "$ROOT_DIR/bench_Qwen3.5_27B.sh" ;;
    qwen122-tp2) echo "$ROOT_DIR/bench_Qwen3.5-122B-A10B-GPTQ_TP2.sh" ;;
    minimax-tp1) echo "$ROOT_DIR/bench_MiniMax-M2.7_TP1.sh" ;;
    minimax-tp2) echo "$ROOT_DIR/bench_MiniMax-M2.7_TP2.sh" ;;
    minimax-tp4) echo "$ROOT_DIR/bench_MiniMax-M2.7_TP4.sh" ;;
    *)
      echo "[h800-bandwidth][error] unknown model: $1" >&2
      echo "[h800-bandwidth][hint] valid: qwen27, qwen122-tp2, minimax, minimax-tp1, minimax-tp2, minimax-tp4, all" >&2
      return 2
      ;;
  esac
}

sanitize() {
  echo "$1" | tr -c 'A-Za-z0-9_' '_'
}

if [[ "$RUN_PREFLIGHT" == 1 && "$BACKEND" != "nsys" ]]; then
  echo "[h800-bandwidth] running NCU bandwidth preflight"
  set +e
  "$ROOT_DIR/helpers/ncu_bandwidth_preflight.sh"
  preflight_status=$?
  set -e
  if [[ "$preflight_status" != 0 ]]; then
    if [[ "$BACKEND" == "ncu" ]]; then
      exit "$preflight_status"
    fi
    echo "[h800-bandwidth][warn] NCU bandwidth preflight failed; falling back to nsys effective bandwidth."
    BACKEND="nsys"
  elif [[ "$BACKEND" == "auto" ]]; then
    BACKEND="ncu"
  fi
fi
if [[ "$BACKEND" == "auto" ]]; then
  BACKEND="ncu"
fi

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
RUN_ID_PREFIX="${BENCH_RUN_ID_PREFIX:-h800_bw}"
OUT_ROOT="${BANDWIDTH_OUT_ROOT:-$ROOT_DIR/.bench_logs}"
FAILED=0

echo "[h800-bandwidth] models: ${MODEL_LIST[*]}"
echo "[h800-bandwidth] phase:  ${CASE_FILTER:-$PHASE}"
echo "[h800-bandwidth] backend: $BACKEND"

for model in "${MODEL_LIST[@]}"; do
  [[ -n "$model" ]] || continue
  wrapper="$(wrapper_for_model "$model")" || exit $?
  if [[ ! -x "$wrapper" ]]; then
    echo "[h800-bandwidth][error] wrapper is missing or not executable: $wrapper" >&2
    exit 2
  fi

  case_arg="${CASE_FILTER:-$PHASE}"
  run_id="${RUN_ID_PREFIX}_$(sanitize "$model")_$(sanitize "$case_arg")"
  out_dir="$OUT_ROOT/bench_$run_id"
  if [[ "$BACKEND" == "ncu" ]]; then
    profiler_arg="--ncu-bandwidth"
  else
    profiler_arg="--nsys-latency"
  fi

  echo
  echo "============================================================"
  echo "[h800-bandwidth] model: $model"
  echo "[h800-bandwidth] wrapper: $wrapper"
  echo "[h800-bandwidth] case: $case_arg"
  echo "[h800-bandwidth] backend: $BACKEND"
  echo "[h800-bandwidth] OUT_DIR=$out_dir"
  echo "[h800-bandwidth] BENCH_RUN_ID=$run_id"
  echo "============================================================"

  set +e
  BENCH_RUN_ID="$run_id" OUT_DIR="$out_dir" "$wrapper" "$profiler_arg" --case "$case_arg" "${EXTRA_ARGS[@]}"
  status=$?
  set -e

  if [[ "$status" != 0 ]]; then
    echo "[h800-bandwidth][error] $model failed with status $status" >&2
    FAILED=1
    if [[ "$CONTINUE_ON_ERROR" != 1 ]]; then
      exit "$status"
    fi
  elif [[ "$BACKEND" == "nsys" ]]; then
    summary="$out_dir/nsys_effective_bandwidth_summary.md"
    echo
    echo "=== Nsight Systems effective bandwidth summary ==="
    set +e
    python "$ROOT_DIR/helpers/summarize_nsys_effective_bandwidth.py" "$out_dir" \
      --peak-gbps "${NCU_PEAK_GBPS:-3350}" 2>&1 | tee "$summary"
    summary_status=${PIPESTATUS[0]}
    set -e
    if [[ "$summary_status" != 0 ]]; then
      echo "[h800-bandwidth][warn] failed to summarize nsys effective bandwidth for $model" >&2
      FAILED=1
      if [[ "$CONTINUE_ON_ERROR" != 1 ]]; then
        exit "$summary_status"
      fi
    else
      echo "[h800-bandwidth] nsys effective bandwidth summary: $summary"
    fi
  fi
done

if [[ "$FAILED" != 0 ]]; then
  exit 1
fi

echo
echo "[h800-bandwidth] all selected bandwidth runs completed."
