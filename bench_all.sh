#!/usr/bin/env bash
# Qwen3.5-122B-A10B standalone kernel profile suite.
#
# This script uses Nsight Systems GPU-kernel summaries and runs each benchmark
# with a single timed kernel launch: no warmup and exactly one iteration.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

NSYS="${NSYS:-nsys}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/.bench_profiles/nsys_$(date +%Y%m%d_%H%M%S)}"

PREFILL_TOKENS=3823
DECODE_TOKENS=1
LINEAR_DIM=12288
CONV_WIDTH=4
LINEAR_Q_HEADS=16
LINEAR_V_HEADS=64
LINEAR_HEAD_DIM=128

MOE_EXPERTS=8
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

MACHETE_BIN="w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm"
FPA_BIN="w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm"
MOE_TRTLLM_BIN="moe_w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm"

MACHETE_TACTIC="w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache"
FPA_TACTIC="w4a16_gemm/fpA_intB_standalone/tactics_h800.cache"
MOE_TRTLLM_TACTIC="moe_w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache"

FAILED=0

require_bin() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "[bench_all][error] missing executable: $path" >&2
    echo "[bench_all][hint] build required targets first, for example:" >&2
    echo "  ./compile.sh build linear_attention w4a16-machete w4a16-fpa moe-trtllm" >&2
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

profile_case() {
  local label="$1"
  local capture_mode="$2"
  local expected_instances="$3"
  shift 3

  local safe
  safe="$(safe_name "$label")"
  local out="$OUT_DIR/$safe"
  local log="$out.log"
  local stats="$out.cuda_gpu_kern_sum.csv"

  local -a nsys_args=(
    profile
    --force-overwrite=true
    --trace=cuda
    --sample=none
    --cpuctxsw=none
    --output="$out"
  )

  if [[ "$capture_mode" == "cudaProfilerApi" ]]; then
    nsys_args+=(--capture-range=cudaProfilerApi --capture-range-end=stop)
  fi

  echo
  echo "=== $label ==="
  printf '[bench_all] command:'
  printf ' %q' "$@"
  echo

  "$NSYS" "${nsys_args[@]}" "$@" >"$log" 2>&1
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true "$out.nsys-rep" >"$stats"

  local instances total_ns avg_us
  instances="$(awk -F, '/^[0-9]/ {s += $3} END {print s + 0}' "$stats")"
  total_ns="$(awk -F, '/^[0-9]/ {s += $2} END {printf "%.0f", s + 0}' "$stats")"
  if [[ "$instances" -gt 0 ]]; then
    avg_us="$(awk -v total="$total_ns" -v inst="$instances" 'BEGIN {printf "%.3f", total / inst / 1000.0}')"
  else
    avg_us="nan"
  fi

  awk -F, '/^[0-9]/ {printf "  kernels=%s avg_us=%s name=%s\n", $3, $4 / 1000.0, $9}' "$stats"
  printf '[bench_all] %-34s instances=%s expected=%s avg_us=%s\n' "$label" "$instances" "$expected_instances" "$avg_us"

  if [[ "$instances" != "$expected_instances" ]]; then
    echo "[bench_all][warn] $label captured $instances GPU kernel launches, expected $expected_instances" >&2
    FAILED=1
  fi
}

command -v "$NSYS" >/dev/null 2>&1 || {
  echo "[bench_all][error] nsys not found. Set NSYS=/path/to/nsys." >&2
  exit 1
}

mkdir -p "$OUT_DIR"

require_bin linear_attention/bench_conv1d_update
require_bin linear_attention/bench_conv1d_fwd
require_bin linear_attention/bench_gated_delta_net
require_bin linear_attention/bench_gdn_prefill
require_bin "$MACHETE_BIN"
require_bin "$FPA_BIN"
require_bin "$MOE_TRTLLM_BIN"
require_file "$MACHETE_TACTIC"
require_file "$FPA_TACTIC"
require_file "$MOE_TRTLLM_TACTIC"

echo "============================================================"
echo "Qwen3.5-122B-A10B standalone kernel nsys suite"
echo "profiles: $OUT_DIR"
echo "prefill tokens: $PREFILL_TOKENS"
echo "decode tokens:  $DECODE_TOKENS"
echo "============================================================"

profile_case "linear_decode_conv1d_update" all 1 \
  linear_attention/bench_conv1d_update "$LINEAR_DIM" "$CONV_WIDTH" "$DECODE_TOKENS" --bench 0 1

profile_case "linear_decode_gdn" all 1 \
  linear_attention/bench_gated_delta_net "$DECODE_TOKENS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --bench 0 1

profile_case "linear_prefill_conv1d_fwd" all 1 \
  linear_attention/bench_conv1d_fwd "$PREFILL_TOKENS" "$LINEAR_DIM" "$CONV_WIDTH" 1 --bench 0 1

profile_case "linear_prefill_flashinfer_gdn" all 1 \
  linear_attention/bench_gdn_prefill "$PREFILL_TOKENS" "$LINEAR_Q_HEADS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --bench 0 1

profile_case "w4a16_prefill_cutlass55" cudaProfilerApi 1 \
  "$MACHETE_BIN" \
  --backend=cutlass55 \
  --cutlass55_tactic="$MACHETE_TACTIC" \
  --m="$W4A16_M" --n="$W4A16_N" --k="$W4A16_K" --group_size="$W4A16_GROUP" \
  --act=fp16 --quant=cutlass_s4 \
  --offline_prepack --profile_gemm_only --no_checksum \
  --warmup=0 --iters=1

profile_case "w4a16_decode_fpA_intB" all 1 \
  "$FPA_BIN" \
  --m="$DECODE_TOKENS" --n="$W4A16_N" --k="$W4A16_K" --group_size="$W4A16_GROUP" \
  --tactic="$FPA_TACTIC" \
  --warmup=0 --iters=1

profile_case "moe_gate_up_prefill_trtllm" all 1 \
  "$MOE_TRTLLM_BIN" \
  --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$PREFILL_TOKENS" \
  --n="$MOE_GATE_N" --k="$MOE_GATE_K" --group_size="$MOE_GROUP" \
  --tactic="$MOE_TRTLLM_TACTIC" \
  --warmup=0 --iters=1

profile_case "moe_down_prefill_trtllm" all 1 \
  "$MOE_TRTLLM_BIN" \
  --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$PREFILL_TOKENS" \
  --n="$MOE_DOWN_N" --k="$MOE_DOWN_K" --group_size="$MOE_GROUP" \
  --tactic="$MOE_TRTLLM_TACTIC" \
  --warmup=0 --iters=1

profile_case "moe_gate_up_decode_trtllm" all 1 \
  "$MOE_TRTLLM_BIN" \
  --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$DECODE_TOKENS" \
  --n="$MOE_GATE_N" --k="$MOE_GATE_K" --group_size="$MOE_GROUP" \
  --tactic="$MOE_TRTLLM_TACTIC" \
  --warmup=0 --iters=1

profile_case "moe_down_decode_trtllm" all 1 \
  "$MOE_TRTLLM_BIN" \
  --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$DECODE_TOKENS" \
  --n="$MOE_DOWN_N" --k="$MOE_DOWN_K" --group_size="$MOE_GROUP" \
  --tactic="$MOE_TRTLLM_TACTIC" \
  --warmup=0 --iters=1

echo
echo "============================================================"
echo "nsys profiles and CSV summaries are under: $OUT_DIR"
if [[ "$FAILED" == 0 ]]; then
  echo "All captured cases have exactly one GPU kernel launch."
else
  echo "Some cases did not capture exactly one GPU kernel launch."
fi
echo "============================================================"

exit "$FAILED"
