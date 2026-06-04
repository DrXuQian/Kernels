#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCU_BIN="${NCU_BIN:-ncu}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/.bench_logs/ncu_preflight_$(date +%Y%m%d_%H%M%S)}"
NCU_PEAK_GBPS="${NCU_PEAK_GBPS:-3350}"
METRICS="${NCU_BANDWIDTH_METRICS:-sm__cycles_elapsed.avg,sm__cycles_elapsed.max,gpu__time_duration.avg,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed}"
BIN="${NCU_PREFLIGHT_BIN:-$ROOT_DIR/general/bench_cuda_core_gemv}"

usage() {
  cat <<EOF
Usage:
  helpers/ncu_bandwidth_preflight.sh

Environment:
  NCU_BIN                 Nsight Compute executable. Default: ncu
  OUT_DIR                 Output directory. Default: .bench_logs/ncu_preflight_<timestamp>
  NCU_PEAK_GBPS           Peak DRAM GB/s for summary utilization. Default: 3350
  NCU_BANDWIDTH_METRICS   Metrics passed to ncu.
  NCU_PREFLIGHT_BIN       Binary to profile. Default: general/bench_cuda_core_gemv

This checks whether the current machine can collect NCU DRAM bandwidth counters.
It exits nonzero on ERR_NVGPUCTRPERM so bandwidth results are not confused with
latency-only nsys runs.
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if ! command -v "$NCU_BIN" >/dev/null 2>&1; then
  echo "[ncu-preflight][error] ncu not found: $NCU_BIN" >&2
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  echo "[ncu-preflight][error] preflight binary is missing or not executable: $BIN" >&2
  echo "[ncu-preflight][hint] build it with: ./compile.sh build general" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/ncu_preflight.csv"

echo "[ncu-preflight] gpu:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
echo "[ncu-preflight] ncu: $("$NCU_BIN" --version | head -1)"
echo "[ncu-preflight] out: $OUT_DIR"
echo "[ncu-preflight] metrics: $METRICS"
echo "[ncu-preflight] command: $NCU_BIN --target-processes all --kernel-name-base demangled --page raw --csv --metrics $METRICS $BIN --m 1 --n 4096 --k 4096 --dtype fp16 --bench 0 1"

set +e
"$NCU_BIN" \
  --target-processes all \
  --kernel-name-base demangled \
  --page raw \
  --csv \
  --metrics "$METRICS" \
  "$BIN" --m 1 --n 4096 --k 4096 --dtype fp16 --bench 0 1 \
  >"$CSV" 2>&1
status=$?
set -e

cat "$CSV"

if [[ "$status" != 0 ]]; then
  if grep -q "ERR_NVGPUCTRPERM" "$CSV"; then
    echo "[ncu-preflight][error] NCU performance counters are blocked by ERR_NVGPUCTRPERM." >&2
    echo "[ncu-preflight][hint] Enable NVIDIA performance counters on the host/container, then rerun this script." >&2
  else
    echo "[ncu-preflight][error] ncu failed with status $status." >&2
  fi
  exit "$status"
fi

python "$ROOT_DIR/helpers/summarize_ncu_bandwidth.py" "$OUT_DIR" \
  --peak-gbps "$NCU_PEAK_GBPS" \
  --detail

echo "[ncu-preflight] ok: NCU bandwidth counters are available."
