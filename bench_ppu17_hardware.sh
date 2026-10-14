#!/usr/bin/env bash
# Physical PPU1.7 profiling. Build the selected binaries/extensions for that SDK first.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=qwen27,minimax-tp1,minimax-tp2,minimax-tp4,sol
PHASE=decode
CASE_FILTER=""
DRY_RUN=0
CONTINUE_ON_ERROR=0
EXTRA_ARGS=()
PYTHON_BIN="${PYTHON:-python3}"
export LINEAR_GDN_MODE=device
export ACTLIZE_LA_BACKEND="${ACTLIZE_LA_BACKEND:-ppu17}"
export PERFRAWLOG_POSTPROCESS=0

usage() {
  cat <<'EOF'
Usage: bash bench_ppu17_hardware.sh [options] [-- extra model-wrapper args]

  --models LIST        Comma-separated: qwen27, qwen122, qwen122-tp2,
                       minimax-tp1, minimax-tp2, minimax-tp4, sol, minimax, all.
                       Default: qwen27,minimax-tp1,minimax-tp2,minimax-tp4,sol.
  --phase PHASE        decode (default), prefill, all. SOL is phase-independent.
  --case LABEL         Model case filter, or read_heavy_bw/copy_bw for SOL.
  --dry-run            Print commands without ACU, device access, or output files.
  --continue-on-error  Continue remaining models after a failed wrapper.
  -h, --help           Show help.

Environment:
  ACU                  ACU executable (default: acu in PATH).
  PYTHON               Python with matplotlib for model charts (default: python3).
  PPU17_OUT_ROOT       Parent output directory (default: .bench_logs).
  BENCH_RUN_ID_PREFIX  Run prefix (default: ppu17_acu_<UTC timestamp>).
  OUT_DIR              Exact output directory; only valid with one selected model.
  ACU_METRICS          Default: ppu__time_duration.sum; retain it in overrides.
  ACU_CACHE_CONTROL    all (default) or none. The SDK may still flush L1/L2 for none.
  ACU_MODEL_SUMMARY    Set to 0 for text-only model summaries without matplotlib.
  PPU_DEVICE_LABEL, PPU_SM_COUNT, PPU_CLOCK_GHZ
                       Optional actual hardware metadata. Unknown unless supplied;
                       neither changes hardware settings nor rescales duration.
  SOL_ROWS, SOL_COLS    Read test dimensions (default: 12288, 12288).
  SOL_MIB              Copy buffer size in MiB (default: 288).

GDN prefill needs a native PPU1.7 extension and a matching fixed configuration:
  ACTLIZE_LA_SM90_CONFIGURATION=control
  GDN_QSA_SM90_EXTENSION=/path/to/native/extension.so
The installed CUDA SM90 auto bundle is not a native PPU1.7 backend.

Examples:
  bash bench_ppu17_hardware.sh --models qwen27 --phase decode
  bash bench_ppu17_hardware.sh --models sol
  bash bench_ppu17_hardware.sh --models all --phase all --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models|--phase|--case)
      key="$1"; value="${2:?missing value for $1}"; shift 2
      case "$key" in --models) MODELS="$value" ;; --phase) PHASE="$value" ;; --case) CASE_FILTER="$value" ;; esac
      ;;
    --models=*) MODELS="${1#*=}"; shift ;;
    --phase=*) PHASE="${1#*=}"; shift ;;
    --case=*) CASE_FILTER="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "[ppu17-acu] unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in decode|prefill|all) ;; *) echo "invalid phase: $PHASE" >&2; exit 2 ;; esac

declare -a MODEL_LIST=()
declare -A SEEN=()
IFS=',' read -r -a requested <<< "${MODELS// /}"
for item in "${requested[@]}"; do
  case "$item" in
    all) expanded=(qwen27 qwen122 qwen122-tp2 minimax-tp1 minimax-tp2 minimax-tp4 sol) ;;
    minimax) expanded=(minimax-tp1 minimax-tp2 minimax-tp4) ;;
    qwen27|qwen122|qwen122-tp2|minimax-tp1|minimax-tp2|minimax-tp4|sol) expanded=("$item") ;;
    *) echo "[ppu17-acu] unknown model: $item" >&2; exit 2 ;;
  esac
  for model in "${expanded[@]}"; do
    if [[ -z "${SEEN[$model]:-}" ]]; then
      MODEL_LIST+=("$model"); SEEN[$model]=1
    fi
  done
done
if [[ ${#MODEL_LIST[@]} == 0 ]]; then
  echo "[ppu17-acu] select at least one model or sol" >&2; exit 2
fi
if [[ -n "${OUT_DIR:-}" && ${#MODEL_LIST[@]} != 1 ]]; then
  echo "[ppu17-acu] OUT_DIR requires one model; use PPU17_OUT_ROOT for a suite" >&2; exit 2
fi

run_id="${BENCH_RUN_ID_PREFIX:-ppu17_acu_$(date -u +%Y%m%d_%H%M%S)}"
out_root="${PPU17_OUT_ROOT:-$ROOT_DIR/.bench_logs}"
failed=0

run_command() {
  printf '[ppu17-acu]'; printf ' %q' "$@"; printf '\n'
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

run_sol() {
  local out="$1" label kernel read_bytes write_bytes
  local rows="${SOL_ROWS:-12288}" cols="${SOL_COLS:-12288}" mib="${SOL_MIB:-288}"
  # Bound arithmetic and reject zero/negative/non-decimal dimensions before launch.
  for value in "$rows" "$cols" "$mib"; do
    if [[ ! "$value" =~ ^[1-9][0-9]{0,7}$ ]]; then
      echo "[ppu17-acu] invalid SOL size: $value" >&2; return 2
    fi
  done
  if (( cols % 8 != 0 )); then echo "SOL_COLS must be a multiple of 8" >&2; return 2; fi
  local selected=0 status=0
  local -a cmd
  for label in read_heavy_bw copy_bw; do
    [[ -z "$CASE_FILTER" || "$label" == *"$CASE_FILTER"* ]] || continue
    selected=$((selected + 1))
    if [[ "$label" == read_heavy_bw ]]; then
      kernel=read_heavy_kernel; read_bytes=$((rows * cols * 2)); write_bytes=$((rows * 4))
      cmd=("$ROOT_DIR/sol/read_heavy_bw" "$rows" "$cols" --single-launch)
    else
      kernel=copy_kernel; read_bytes=$((mib * 1048576)); write_bytes=$read_bytes
      cmd=("$ROOT_DIR/sol/copy_bw" "$mib" --single-launch)
    fi
    run_command "$PYTHON_BIN" "$ROOT_DIR/helpers/profile_acu.py" \
      --output-prefix "$out/acu/$label" --include-kernel "$kernel" --expect-kernels 1 \
      --read-bytes "$read_bytes" --write-bytes "$write_bytes" -- "${cmd[@]}" || status=1
  done
  if [[ "$selected" == 0 ]]; then echo "no SOL case matched: $CASE_FILTER" >&2; return 2; fi
  if [[ "$DRY_RUN" != 1 ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/helpers/summarize_acu_kernels.py" "$out/acu" --detail --no-model \
      | tee "$out/acu_latency_summary.md" || status=1
  fi
  return "$status"
}

for model in "${MODEL_LIST[@]}"; do
  out="${OUT_DIR:-$out_root/${run_id}_${model}}"
  [[ "$out" == /* ]] || out="$ROOT_DIR/$out"
  if [[ "$model" == sol ]]; then
    if run_sol "$out"; then status=0; else status=$?; fi
  else
    case "$model" in
      qwen27) wrapper=bench_Qwen3.5_27B.sh ;;
      qwen122) wrapper=bench_Qwen3.5-122B-A10B-GPTQ.sh ;;
      qwen122-tp2) wrapper=bench_Qwen3.5-122B-A10B-GPTQ_TP2.sh ;;
      minimax-tp1) wrapper=bench_MiniMax-M2.7_TP1.sh ;;
      minimax-tp2) wrapper=bench_MiniMax-M2.7_TP2.sh ;;
      minimax-tp4) wrapper=bench_MiniMax-M2.7_TP4.sh ;;
    esac
    filters=()
    if [[ -n "$CASE_FILTER" ]]; then filters=(--case "$CASE_FILTER")
    elif [[ "$PHASE" != all ]]; then filters=(--case "$PHASE"); fi
    if run_command env OUT_DIR="$out" BENCH_RUN_ID="${run_id}_${model}" PYTHON="$PYTHON_BIN" \
      bash "$ROOT_DIR/$wrapper" --acu-latency "${filters[@]}" "${EXTRA_ARGS[@]}"; then
      status=0
    else status=$?; fi
  fi
  if [[ "$status" != 0 ]]; then
    failed=1
    [[ "$CONTINUE_ON_ERROR" == 1 ]] || exit "$status"
  fi
done
exit "$failed"
