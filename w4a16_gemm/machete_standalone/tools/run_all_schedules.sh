#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MACHETE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

BIN="$MACHETE_DIR/build_cmake_release/test_machete_gemm"
M=4096
N=4096
K=4096
GROUP=128
ACT=fp16
QUANT=gptq_u4b8
WARMUP=10
ITERS=100
OUT=""
OFFLINE_PREPACK=0
OFFLINE_PREPACK_PATH=""

usage() {
  cat <<'USAGE'
Usage: machete_standalone/tools/run_all_schedules.sh [options]

Options:
  --bin PATH        test_machete_gemm binary
  --m N            M dimension
  --n N            N dimension
  --k N            K dimension
  --group N        group size
  --act fp16|bf16
  --quant gptq_u4b8|awq_u4
  --warmup N
  --iters N
  --offline-prepack [PATH]   skip runtime prepack; optional PATH loads an offline prepack file
  --out PATH       tee output to PATH
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    --act) ACT="$2"; shift 2 ;;
    --quant) QUANT="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --offline-prepack|--offline_prepack)
      OFFLINE_PREPACK=1
      if [[ $# -ge 2 && "$2" != --* ]]; then
        OFFLINE_PREPACK_PATH="$2"
        shift 2
      else
        shift
      fi
      ;;
    --out) OUT="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN" >&2
  exit 1
fi

run() {
  extra_args=()
  if [[ "$OFFLINE_PREPACK" -eq 1 ]]; then
    if [[ -n "$OFFLINE_PREPACK_PATH" ]]; then
      extra_args+=(--offline_prepack="$OFFLINE_PREPACK_PATH")
    else
      extra_args+=(--offline_prepack)
    fi
  fi
  "$BIN" --list_schedules | awk '/^[[:space:]]+[0-9]+:/ {print $2}' | while read -r schedule; do
    echo "===== $schedule ====="
    "$BIN" \
      --m="$M" --n="$N" --k="$K" --group_size="$GROUP" \
      --act="$ACT" --quant="$QUANT" \
      --warmup="$WARMUP" --iters="$ITERS" \
      --schedule="$schedule" \
      "${extra_args[@]}"
  done
}

if [[ -n "$OUT" ]]; then
  run 2>&1 | tee "$OUT"
else
  run
fi
