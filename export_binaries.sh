#!/usr/bin/env bash
# Export compiled standalone benchmark binaries into a self-contained folder.
#
# Usage:
#   ./export_binaries.sh /path/to/export_dir
#   ./export_binaries.sh --clean /path/to/export_dir
#   ./export_binaries.sh --dry-run /path/to/export_dir
#
# The export directory receives:
#   - compiled benchmark executables, preserving repo-relative paths
#   - tactic/cache files needed by W4A16 kernels
#   - Python benchmark entry points for FlashAttention
#   - run_exported_benchmarks.sh, a small runner documenting which network
#     structure each executable/case corresponds to

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DEST="$ROOT_DIR/exported_binaries/qwen35_122b_a10b"

DEST_DIR=""
CLEAN=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  $0 [--clean] [--dry-run] [DEST_DIR]

Options:
  --clean      remove DEST_DIR before copying
  --dry-run    print what would be copied without writing files
  -h, --help   show this help

Default DEST_DIR:
  $DEFAULT_DEST

Examples:
  $0 /tmp/qwen35_kernel_bins
  $0 --clean /tmp/qwen35_kernel_bins
  cd /tmp/qwen35_kernel_bins && ./run_exported_benchmarks.sh --list
  cd /tmp/qwen35_kernel_bins && ./run_exported_benchmarks.sh --case moe_gate_up_prefill_trtllm
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$DEST_DIR" ]]; then
        echo "[export][error] multiple destination directories provided: $DEST_DIR and $1" >&2
        exit 1
      fi
      DEST_DIR="$1"
      shift
      ;;
  esac
done

DEST_DIR="${DEST_DIR:-$DEFAULT_DEST}"
if [[ "$DEST_DIR" != /* ]]; then
  DEST_DIR="$ROOT_DIR/$DEST_DIR"
fi

info() { echo "[export] $*"; }
warn() { echo "[export][warn] $*" >&2; }

copy_item() {
  local mode="$1"
  local rel="$2"
  local note="$3"
  local src="$ROOT_DIR/$rel"
  local dst="$DEST_DIR/$rel"

  if [[ ! -e "$src" ]]; then
    warn "missing: $rel ($note)"
    return 0
  fi

  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[export][dry-run] %-4s %s\n' "$mode" "$rel"
    return 0
  fi

  mkdir -p "$(dirname "$dst")"
  case "$mode" in
    exe|script)
      install -m 755 "$src" "$dst"
      ;;
    file)
      install -m 644 "$src" "$dst"
      ;;
    *)
      echo "[export][error] unknown copy mode: $mode" >&2
      exit 1
      ;;
  esac
  printf '%s\t%s\t%s\n' "$mode" "$rel" "$note" >> "$MANIFEST"
}

write_runner() {
  local runner="$DEST_DIR/run_exported_benchmarks.sh"
  if [[ "$DRY_RUN" == 1 ]]; then
    info "[dry-run] would write run_exported_benchmarks.sh"
    return 0
  fi

  cat > "$runner" <<'RUNNER'
#!/usr/bin/env bash
# Runner for exported Qwen3.5-122B-A10B standalone kernel binaries.
# The --list output is the authoritative mapping from case to network structure.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR}"
PYTHON_BIN="${PYTHON:-$(command -v python3 || true)}"
LIST_CASES=0
CASE_FILTERS=()
MODULE_FILTERS=()
DECODE_MOE_BACKEND="${DECODE_MOE_BACKEND:-vllm}"
DECODE_DENSE_BACKEND="${DECODE_DENSE_BACKEND:-cuda_core}"

PREFILL_TOKENS=3823
DECODE_TOKENS=1
CTX_LEN="${CTX_LEN:-3823}"
HIDDEN_DIM=3072
LINEAR_DIM=12288
CONV_WIDTH=4
LINEAR_Q_HEADS=16
LINEAR_V_HEADS=64
LINEAR_HEAD_DIM=128
LINEAR_SMALL_PROJ_N=64
FULL_ATTN_Q_HEADS=32
FULL_ATTN_KV_HEADS=2
FULL_ATTN_HEAD_DIM=256

MOE_EXPERTS=8
MOE_ROUTER_EXPERTS=256
MOE_TOPK=8
MOE_GROUP=128
MOE_INTERMEDIATE=1024
MOE_GATE_N=2048
MOE_GATE_K=3072
MOE_DOWN_N=3072
MOE_DOWN_K=1024
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
W4A16_CONSISTENT_EXPERT_UP_N=2048
W4A16_CONSISTENT_EXPERT_UP_K=3072
W4A16_CONSISTENT_EXPERT_DOWN_N=3072
W4A16_CONSISTENT_EXPERT_DOWN_K=1024

SAMPLING_VOCAB=248320
SAMPLING_TOPK=50
SAMPLING_TOPP=0.9

path() {
  printf '%s/%s\n' "$ROOT_DIR" "$1"
}

MACHETE_BIN="$(path "general/w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm")"
FPA_BIN="$(path "general/w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm")"
CUBLAS_GEMM_BIN="$(path "general/bench_cublas_gemm")"
CUDA_CORE_GEMV_BIN="$(path "general/bench_cuda_core_gemv")"
MOE_TRTLLM_BIN="$(path "moe_ffn/w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm")"
MOE_TRTLLM_AUX_DIR="$(path "moe_ffn/w4a16/trtllm/auxiliary")"
MOE_VLLM_MARLIN_BIN="$(path "moe_ffn/w4a16/vllm/marlin/bench_marlin_moe")"
MOE_VLLM_AUX_DIR="$(path "moe_ffn/w4a16/vllm/auxiliary")"
LINEAR_RMSNORM_BIN="$(path "linear_attn/bench_rmsnorm")"
LINEAR_OPS_BIN="$(path "linear_attn/bench_linear_ops")"
LINEAR_FUSED_RMS_GATE_BIN="$(path "linear_attn/bench_fused_rms_norm_gate")"
FLASH_RMSNORM_BIN="$(path "flash_attn/bench_rmsnorm")"
FLASH_ATTN_SCRIPT="$(path "flash_attn/bench_flash_attn.py")"
MOE_RMSNORM_BIN="$(path "moe_ffn/bench_rmsnorm")"
MOE_SHARED_EXPERT_BIN="$(path "moe_ffn/bench_shared_expert")"
SAMPLING_BIN="$(path "sampling/bench_sampling")"

MACHETE_TACTIC="$(path "general/w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache")"
FPA_TACTIC="$(path "general/w4a16_gemm/fpA_intB_standalone/tactics_h800.cache")"
MOE_TRTLLM_TACTIC="$(path "moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache")"

usage() {
  cat <<'EOF'
Usage:
  ./run_exported_benchmarks.sh --list
  ./run_exported_benchmarks.sh [--case LABEL ...] [--module MODULE ...] [--run-dir DIR]

Options:
  --list                 list cases and their network-structure mapping
  --case LABEL           run only one case; can be repeated
  --module MODULE        run only one module: linear_attn, flash_attn, moe_ffn, sampling
  --run-dir DIR          run commands from DIR; useful when profiling must happen from a fixed directory
  --decode-moe BACKEND   vllm or trtllm, default: vllm
  --decode-dense BACKEND cuda_core or cublas, default: cuda_core

Every command uses warmup=0 and iters=1 where the executable supports it.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_CASES=1
      shift
      ;;
    --case|--kernel)
      CASE_FILTERS+=("$2")
      shift 2
      ;;
    --module)
      MODULE_FILTERS+=("$2")
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --decode-moe)
      DECODE_MOE_BACKEND="$2"
      shift 2
      ;;
    --decode-dense)
      DECODE_DENSE_BACKEND="$2"
      shift 2
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "[runner][error] unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

matches_filter() {
  local value="$1"
  shift
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  local item
  for item in "$@"; do
    [[ "$value" == "$item" ]] && return 0
  done
  return 1
}

require_exec() {
  local file="$1"
  if [[ ! -x "$file" ]]; then
    echo "[runner][error] missing executable: $file" >&2
    echo "[runner][hint] rebuild it in the source repo, then rerun export_binaries.sh." >&2
    exit 1
  fi
}

run_case() {
  local label="$1"
  local module="$2"
  local phase="$3"
  local network_op="$4"
  shift 4

  matches_filter "$label" "${CASE_FILTERS[@]}" || return 0
  matches_filter "$module" "${MODULE_FILTERS[@]}" || return 0

  if [[ "$LIST_CASES" == 1 ]]; then
    printf '%-48s %-14s %-8s %s\n' "$label" "$module" "$phase" "$network_op"
    return 0
  fi

  require_exec "$1"
  echo
  echo "[runner] case:    $label"
  echo "[runner] module:  $module"
  echo "[runner] phase:   $phase"
  echo "[runner] op:      $network_op"
  printf '[runner] command:'
  printf ' %q' "$@"
  printf '\n'
  (cd "$RUN_DIR" && "$@")
}

run_w4a16_prefill_cutlass55_case() {
  local label="$1"; local module="$2"; local op="$3"; local m="$4"; local n="$5"; local k="$6"
  run_case "$label" "$module" prefill "$op; W4A16 Cutlass55 GEMM m=$m n=$n k=$k group=128" \
    "$MACHETE_BIN" --backend=cutlass55 --cutlass55_tactic="$MACHETE_TACTIC" \
    --m="$m" --n="$n" --k="$k" --group_size="$W4A16_GROUP" \
    --act=fp16 --quant=cutlass_s4 --offline_prepack --profile_gemm_only --no_checksum \
    --warmup=0 --iters=1
}

run_w4a16_decode_fpa_case() {
  local label="$1"; local module="$2"; local op="$3"; local m="$4"; local n="$5"; local k="$6"
  run_case "$label" "$module" decode "$op; W4A16 fpA_intB GEMM m=$m n=$n k=$k group=128" \
    "$FPA_BIN" --m="$m" --n="$n" --k="$k" --group_size="$W4A16_GROUP" \
    --tactic="$FPA_TACTIC" --warmup=0 --iters=1
}

run_rmsnorm_case() {
  local label="$1"; local module="$2"; local phase="$3"; local op="$4"; local bin="$5"; local batch="$6"; local embed="$7"
  run_case "$label" "$module" "$phase" "$op; RMSNorm batch=$batch embed=$embed" \
    "$bin" --batch "$batch" --embed "$embed" --dtype fp16 --no-check --bench 0 1
}

run_residual_add_case() {
  local label="$1"; local module="$2"; local phase="$3"; local tokens="$4"
  run_case "$label" "$module" "$phase" "Residual add tokens=$tokens hidden=3072" \
    "$LINEAR_OPS_BIN" --op=residual_add --tokens="$tokens" --hidden="$HIDDEN_DIM" --dtype fp16 --bench 0 1
}

run_dense_decode_case() {
  local cublas_label="$1"; local cuda_label="$2"; local module="$3"; local op="$4"; local m="$5"; local n="$6"; local k="$7"; local out="${8:-fp16}"
  if [[ "$DECODE_DENSE_BACKEND" == "cublas" ]]; then
    run_case "$cublas_label" "$module" decode "$op; dense cuBLAS GEMM m=$m n=$n k=$k out=$out" \
      "$CUBLAS_GEMM_BIN" --m="$m" --n="$n" --k="$k" --dtype fp16 --out-dtype "$out" --bench 0 1
  else
    run_case "$cuda_label" "$module" decode "$op; dense CUDA-core GEMV m=$m n=$n k=$k out=$out" \
      "$CUDA_CORE_GEMV_BIN" --m="$m" --n="$n" --k="$k" --dtype fp16 --out-dtype "$out" --bench 0 1
  fi
}

run_dense_prefill_case() {
  local label="$1"; local module="$2"; local op="$3"; local m="$4"; local n="$5"; local k="$6"; local out="${7:-fp16}"
  run_case "$label" "$module" prefill "$op; dense cuBLAS GEMM m=$m n=$n k=$k out=$out" \
    "$CUBLAS_GEMM_BIN" --m="$m" --n="$n" --k="$k" --dtype fp16 --out-dtype "$out" --bench 0 1
}

run_flash_attn_core_case() {
  local label="$1"; local phase="$2"; local seq="$3"
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "[runner][error] python3 not found; cannot run $label" >&2
    exit 1
  fi
  run_case "$label" flash_attn "$phase" "FlashAttention core seq=$seq q_heads=32 kv_heads=2 head_dim=256" \
    "$PYTHON_BIN" "$FLASH_ATTN_SCRIPT" "$phase" "$seq" "$FULL_ATTN_Q_HEADS" "$FULL_ATTN_KV_HEADS" "$FULL_ATTN_HEAD_DIM" --bench 0 1
}

run_moe_trtllm_gemm_case() {
  local label="$1"; local phase="$2"; local op="$3"; local m_per_expert="$4"; local n="$5"; local k="$6"
  if [[ "$m_per_expert" == 1 ]]; then
    run_case "$label" moe_ffn "$phase" "$op; TRT-LLM MoE W4A16 CUDA-core GEMV experts=8 m_per_expert=$m_per_expert n=$n k=$k" \
      "$MOE_TRTLLM_BIN" --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$m_per_expert" \
      --n="$n" --k="$k" --group_size="$MOE_GROUP" --cuda_core --warmup=0 --iters=1
  else
    run_case "$label" moe_ffn "$phase" "$op; TRT-LLM MoE W4A16 grouped GEMM experts=8 m_per_expert=$m_per_expert n=$n k=$k" \
      "$MOE_TRTLLM_BIN" --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$m_per_expert" \
      --n="$n" --k="$k" --group_size="$MOE_GROUP" --tactic="$MOE_TRTLLM_TACTIC" --warmup=0 --iters=1
  fi
}

run_sampling_case() {
  local label="$1"; local op="$2"; local desc="$3"
  run_case "$label" sampling decode "$desc" \
    "$SAMPLING_BIN" --op="$op" --hidden="$HIDDEN_DIM" --vocab="$SAMPLING_VOCAB" \
    --top-k="$SAMPLING_TOPK" --top-p="$SAMPLING_TOPP" --bench 0 1
}

run_sampling_lm_head_case() {
  run_case "sampling_lm_head_gemm" sampling decode "LM head projection m=1 n=248320 k=3072 out=fp32" \
    "$CUBLAS_GEMM_BIN" --m="$DECODE_TOKENS" --n="$SAMPLING_VOCAB" --k="$HIDDEN_DIM" \
    --dtype fp16 --out-dtype fp32 --bench 0 1
}

if [[ "$LIST_CASES" == 1 ]]; then
  printf '%-48s %-14s %-8s %s\n' "case" "module" "phase" "network structure / operator"
  printf '%-48s %-14s %-8s %s\n' "----" "------" "-----" "----------------------------"
fi

# Linear-attention block.
run_rmsnorm_case "linear_attn_decode_rmsnorm" linear_attn decode "Linear-attention block input norm" "$LINEAR_RMSNORM_BIN" "$DECODE_TOKENS" "$HIDDEN_DIM"
run_rmsnorm_case "linear_attn_prefill_rmsnorm" linear_attn prefill "Linear-attention block input norm" "$LINEAR_RMSNORM_BIN" "$PREFILL_TOKENS" "$HIDDEN_DIM"
run_dense_decode_case "linear_attn_decode_in_proj_a_cublas" "linear_attn_decode_in_proj_a_cuda_core" linear_attn "Gated delta in_proj_a" "$DECODE_TOKENS" "$LINEAR_SMALL_PROJ_N" "$HIDDEN_DIM"
run_dense_decode_case "linear_attn_decode_in_proj_b_cublas" "linear_attn_decode_in_proj_b_cuda_core" linear_attn "Gated delta in_proj_b" "$DECODE_TOKENS" "$LINEAR_SMALL_PROJ_N" "$HIDDEN_DIM"
run_dense_prefill_case "linear_attn_prefill_in_proj_a_cublas" linear_attn "Gated delta in_proj_a" "$PREFILL_TOKENS" "$LINEAR_SMALL_PROJ_N" "$HIDDEN_DIM"
run_dense_prefill_case "linear_attn_prefill_in_proj_b_cublas" linear_attn "Gated delta in_proj_b" "$PREFILL_TOKENS" "$LINEAR_SMALL_PROJ_N" "$HIDDEN_DIM"
run_case "linear_decode_conv1d_update" linear_attn decode "Linear-attention short conv state update" "$(path "linear_attn/bench_conv1d_update")" "$LINEAR_DIM" "$CONV_WIDTH" "$DECODE_TOKENS" --bench 0 1
run_case "linear_decode_gdn" linear_attn decode "Gated delta net recurrent decode" "$(path "linear_attn/bench_gated_delta_net")" "$DECODE_TOKENS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --bench 0 1
run_case "linear_prefill_conv1d_fwd" linear_attn prefill "Linear-attention short conv prefill" "$(path "linear_attn/bench_conv1d_fwd")" "$PREFILL_TOKENS" "$LINEAR_DIM" "$CONV_WIDTH" 1 --bench 0 1
run_case "linear_prefill_flashinfer_gdn" linear_attn prefill "FlashInfer gated delta net prefill" "$(path "linear_attn/bench_gdn_prefill")" "$PREFILL_TOKENS" "$LINEAR_Q_HEADS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --bench 0 1
run_case "linear_attn_decode_fused_rms_norm_gate" linear_attn decode "Fused RMSNorm gate for linear-attention output" "$LINEAR_FUSED_RMS_GATE_BIN" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" --bench 0 1
run_case "linear_attn_prefill_fused_rms_norm_gate" linear_attn prefill "Fused RMSNorm gate for linear-attention output" "$LINEAR_FUSED_RMS_GATE_BIN" "$((PREFILL_TOKENS * LINEAR_V_HEADS))" "$LINEAR_HEAD_DIM" --bench 0 1
run_residual_add_case "linear_attn_decode_residual_add" linear_attn decode "$DECODE_TOKENS"
run_residual_add_case "linear_attn_prefill_residual_add" linear_attn prefill "$PREFILL_TOKENS"
run_w4a16_prefill_cutlass55_case "w4a16_prefill_linear_attn_in_proj_qkv_cutlass55" linear_attn "Linear-attention W4A16 qkv projection" "$PREFILL_TOKENS" "$W4A16_LINEAR_QKV_N" "$W4A16_LINEAR_QKV_K"
run_w4a16_prefill_cutlass55_case "w4a16_prefill_linear_attn_in_proj_z_cutlass55" linear_attn "Linear-attention W4A16 z projection" "$PREFILL_TOKENS" "$W4A16_LINEAR_Z_N" "$W4A16_LINEAR_Z_K"
run_w4a16_prefill_cutlass55_case "w4a16_prefill_linear_attn_out_proj_cutlass55" linear_attn "Linear-attention W4A16 output projection" "$PREFILL_TOKENS" "$W4A16_LINEAR_OUT_N" "$W4A16_LINEAR_OUT_K"
run_w4a16_decode_fpa_case "w4a16_decode_linear_attn_in_proj_qkv_fpA_intB" linear_attn "Linear-attention W4A16 qkv projection" "$DECODE_TOKENS" "$W4A16_LINEAR_QKV_N" "$W4A16_LINEAR_QKV_K"
run_w4a16_decode_fpa_case "w4a16_decode_linear_attn_in_proj_z_fpA_intB" linear_attn "Linear-attention W4A16 z projection" "$DECODE_TOKENS" "$W4A16_LINEAR_Z_N" "$W4A16_LINEAR_Z_K"
run_w4a16_decode_fpa_case "w4a16_decode_linear_attn_out_proj_fpA_intB" linear_attn "Linear-attention W4A16 output projection" "$DECODE_TOKENS" "$W4A16_LINEAR_OUT_N" "$W4A16_LINEAR_OUT_K"

# Full-attention block.
run_rmsnorm_case "flash_attn_decode_rmsnorm" flash_attn decode "Full-attention block input norm" "$FLASH_RMSNORM_BIN" "$DECODE_TOKENS" "$HIDDEN_DIM"
run_rmsnorm_case "flash_attn_prefill_rmsnorm" flash_attn prefill "Full-attention block input norm" "$FLASH_RMSNORM_BIN" "$PREFILL_TOKENS" "$HIDDEN_DIM"
run_residual_add_case "flash_attn_decode_residual_add" flash_attn decode "$DECODE_TOKENS"
run_residual_add_case "flash_attn_prefill_residual_add" flash_attn prefill "$PREFILL_TOKENS"
run_w4a16_prefill_cutlass55_case "w4a16_prefill_full_attn_q_proj_gate_cutlass55" flash_attn "Full-attention W4A16 q_proj + output gate projection" "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_Q_PROJ_GATE_N" "$W4A16_FULL_ATTN_Q_PROJ_GATE_K"
run_w4a16_prefill_cutlass55_case "w4a16_prefill_full_attn_k_proj_cutlass55" flash_attn "Full-attention W4A16 k projection" "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_K_PROJ_N" "$W4A16_FULL_ATTN_K_PROJ_K"
run_w4a16_prefill_cutlass55_case "w4a16_prefill_full_attn_v_proj_cutlass55" flash_attn "Full-attention W4A16 v projection" "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_V_PROJ_N" "$W4A16_FULL_ATTN_V_PROJ_K"
run_rmsnorm_case "flash_attn_prefill_q_norm" flash_attn prefill "Full-attention q RMSNorm" "$FLASH_RMSNORM_BIN" "$((PREFILL_TOKENS * FULL_ATTN_Q_HEADS))" "$FULL_ATTN_HEAD_DIM"
run_rmsnorm_case "flash_attn_prefill_k_norm" flash_attn prefill "Full-attention k RMSNorm" "$FLASH_RMSNORM_BIN" "$((PREFILL_TOKENS * FULL_ATTN_KV_HEADS))" "$FULL_ATTN_HEAD_DIM"
run_flash_attn_core_case "flash_attn_prefill_full_attn" prefill "$PREFILL_TOKENS"
run_w4a16_prefill_cutlass55_case "w4a16_prefill_full_attn_o_proj_cutlass55" flash_attn "Full-attention W4A16 output projection" "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_O_PROJ_N" "$W4A16_FULL_ATTN_O_PROJ_K"
run_w4a16_decode_fpa_case "w4a16_decode_full_attn_q_proj_gate_fpA_intB" flash_attn "Full-attention W4A16 q_proj + output gate projection" "$DECODE_TOKENS" "$W4A16_FULL_ATTN_Q_PROJ_GATE_N" "$W4A16_FULL_ATTN_Q_PROJ_GATE_K"
run_w4a16_decode_fpa_case "w4a16_decode_full_attn_k_proj_fpA_intB" flash_attn "Full-attention W4A16 k projection" "$DECODE_TOKENS" "$W4A16_FULL_ATTN_K_PROJ_N" "$W4A16_FULL_ATTN_K_PROJ_K"
run_w4a16_decode_fpa_case "w4a16_decode_full_attn_v_proj_fpA_intB" flash_attn "Full-attention W4A16 v projection" "$DECODE_TOKENS" "$W4A16_FULL_ATTN_V_PROJ_N" "$W4A16_FULL_ATTN_V_PROJ_K"
run_rmsnorm_case "flash_attn_decode_q_norm" flash_attn decode "Full-attention q RMSNorm" "$FLASH_RMSNORM_BIN" "$((DECODE_TOKENS * FULL_ATTN_Q_HEADS))" "$FULL_ATTN_HEAD_DIM"
run_rmsnorm_case "flash_attn_decode_k_norm" flash_attn decode "Full-attention k RMSNorm" "$FLASH_RMSNORM_BIN" "$((DECODE_TOKENS * FULL_ATTN_KV_HEADS))" "$FULL_ATTN_HEAD_DIM"
run_flash_attn_core_case "flash_attn_decode_full_attn" decode "$CTX_LEN"
run_w4a16_decode_fpa_case "w4a16_decode_full_attn_o_proj_fpA_intB" flash_attn "Full-attention W4A16 output projection" "$DECODE_TOKENS" "$W4A16_FULL_ATTN_O_PROJ_N" "$W4A16_FULL_ATTN_O_PROJ_K"

# MoE-FFN block.
run_rmsnorm_case "moe_ffn_decode_rmsnorm" moe_ffn decode "MoE-FFN block input norm" "$MOE_RMSNORM_BIN" "$DECODE_TOKENS" "$HIDDEN_DIM"
run_rmsnorm_case "moe_ffn_prefill_rmsnorm" moe_ffn prefill "MoE-FFN block input norm" "$MOE_RMSNORM_BIN" "$PREFILL_TOKENS" "$HIDDEN_DIM"
run_residual_add_case "moe_ffn_decode_residual_add" moe_ffn decode "$DECODE_TOKENS"
run_residual_add_case "moe_ffn_prefill_residual_add" moe_ffn prefill "$PREFILL_TOKENS"
run_w4a16_prefill_cutlass55_case "w4a16_prefill_consistent_expert_up_cutlass55" moe_ffn "Consistent/shared expert W4A16 up projection" "$PREFILL_TOKENS" "$W4A16_CONSISTENT_EXPERT_UP_N" "$W4A16_CONSISTENT_EXPERT_UP_K"
run_case "moe_shared_expert_activation_prefill_trtllm" moe_ffn prefill "Shared expert activation" "$MOE_TRTLLM_AUX_DIR/bench_shared_expert_activation" "$PREFILL_TOKENS" "$MOE_INTERMEDIATE" fp16 --bench 0 1
run_w4a16_prefill_cutlass55_case "w4a16_prefill_consistent_expert_down_cutlass55" moe_ffn "Consistent/shared expert W4A16 down projection" "$PREFILL_TOKENS" "$W4A16_CONSISTENT_EXPERT_DOWN_N" "$W4A16_CONSISTENT_EXPERT_DOWN_K"
run_dense_prefill_case "moe_router_gate_prefill_cublas" moe_ffn "Router gate logits" "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$HIDDEN_DIM"
run_dense_prefill_case "moe_shared_expert_gate_prefill_cublas" moe_ffn "Shared expert scalar gate" "$PREFILL_TOKENS" 1 "$HIDDEN_DIM"
run_case "moe_shared_expert_fusion_prefill" moe_ffn prefill "Shared expert sigmoid/mul/add fusion" "$MOE_SHARED_EXPERT_BIN" --op=sigmoid_mul_add --tokens="$PREFILL_TOKENS" --hidden="$MOE_SHARED_HIDDEN" --out-dim=1 --dtype fp16 --bench 0 1
run_w4a16_decode_fpa_case "w4a16_decode_consistent_expert_up_fpA_intB" moe_ffn "Consistent/shared expert W4A16 up projection" "$DECODE_TOKENS" "$W4A16_CONSISTENT_EXPERT_UP_N" "$W4A16_CONSISTENT_EXPERT_UP_K"
run_case "moe_shared_expert_activation_decode_trtllm" moe_ffn decode "Shared expert activation" "$MOE_TRTLLM_AUX_DIR/bench_shared_expert_activation" "$DECODE_TOKENS" "$MOE_INTERMEDIATE" fp16 --bench 0 1
run_w4a16_decode_fpa_case "w4a16_decode_consistent_expert_down_fpA_intB" moe_ffn "Consistent/shared expert W4A16 down projection" "$DECODE_TOKENS" "$W4A16_CONSISTENT_EXPERT_DOWN_N" "$W4A16_CONSISTENT_EXPERT_DOWN_K"
run_dense_decode_case "moe_router_gate_decode_cublas" "moe_router_gate_decode_cuda_core" moe_ffn "Router gate logits" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$HIDDEN_DIM"
run_dense_decode_case "moe_shared_expert_gate_decode_cublas" "moe_shared_expert_gate_decode_cuda_core" moe_ffn "Shared expert scalar gate" "$DECODE_TOKENS" 1 "$HIDDEN_DIM"
run_case "moe_shared_expert_fusion_decode" moe_ffn decode "Shared expert sigmoid/mul/add fusion" "$MOE_SHARED_EXPERT_BIN" --op=sigmoid_mul_add --tokens="$DECODE_TOKENS" --hidden="$MOE_SHARED_HIDDEN" --out-dim=1 --dtype fp16 --bench 0 1
run_case "moe_routing_prefill_trtllm" moe_ffn prefill "TRT-LLM custom MoE routing/topk" "$MOE_TRTLLM_AUX_DIR/bench_custom_moe_routing" "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" fp16 --bench 0 1
run_case "moe_expert_map_prefill_trtllm" moe_ffn prefill "TRT-LLM expert map metadata" "$MOE_TRTLLM_AUX_DIR/bench_expert_map" "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" auto --bench 0 1
run_case "moe_expand_prefill_trtllm" moe_ffn prefill "TRT-LLM expand input rows for top-k experts" "$MOE_TRTLLM_AUX_DIR/bench_expand_input_rows" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_GATE_K" fp16 --bench 0 1
run_moe_trtllm_gemm_case "moe_gate_up_prefill_trtllm" prefill "TRT-LLM gate/up expert GEMM" "$PREFILL_TOKENS" "$MOE_GATE_N" "$MOE_GATE_K"
run_case "moe_gated_prefill_trtllm" moe_ffn prefill "TRT-LLM SiLU and multiply" "$MOE_TRTLLM_AUX_DIR/bench_gated_activation" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_INTERMEDIATE" fp16 --bench 0 1
run_moe_trtllm_gemm_case "moe_down_prefill_trtllm" prefill "TRT-LLM down expert GEMM" "$PREFILL_TOKENS" "$MOE_DOWN_N" "$MOE_DOWN_K"
run_case "moe_finalize_prefill_trtllm" moe_ffn prefill "TRT-LLM finalize routing / reduce top-k experts" "$MOE_TRTLLM_AUX_DIR/bench_finalize_moe_routing" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_DOWN_N" fp16 --bench 0 1

if [[ "$DECODE_MOE_BACKEND" == "trtllm" ]]; then
  run_case "moe_routing_decode_trtllm" moe_ffn decode "TRT-LLM custom MoE routing/topk" "$MOE_TRTLLM_AUX_DIR/bench_custom_moe_routing" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" fp16 --bench 0 1
  run_case "moe_expert_map_decode_trtllm" moe_ffn decode "TRT-LLM expert map metadata" "$MOE_TRTLLM_AUX_DIR/bench_expert_map" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" auto --bench 0 1
  run_case "moe_expand_decode_trtllm" moe_ffn decode "TRT-LLM expand input rows for top-k experts" "$MOE_TRTLLM_AUX_DIR/bench_expand_input_rows" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_GATE_K" fp16 --bench 0 1
  run_moe_trtllm_gemm_case "moe_gate_up_decode_trtllm" decode "TRT-LLM gate/up expert GEMV" "$DECODE_TOKENS" "$MOE_GATE_N" "$MOE_GATE_K"
  run_case "moe_gated_decode_trtllm" moe_ffn decode "TRT-LLM SiLU and multiply" "$MOE_TRTLLM_AUX_DIR/bench_gated_activation" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_INTERMEDIATE" fp16 --bench 0 1
  run_moe_trtllm_gemm_case "moe_down_decode_trtllm" decode "TRT-LLM down expert GEMV" "$DECODE_TOKENS" "$MOE_DOWN_N" "$MOE_DOWN_K"
  run_case "moe_finalize_decode_trtllm" moe_ffn decode "TRT-LLM finalize routing / reduce top-k experts" "$MOE_TRTLLM_AUX_DIR/bench_finalize_moe_routing" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_DOWN_N" fp16 --bench 0 1
else
  run_case "moe_routing_decode_vllm" moe_ffn decode "vLLM top-k gating" "$MOE_VLLM_AUX_DIR/bench_topk_gating" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" --bench 0 1
  run_case "moe_align_decode_vllm" moe_ffn decode "vLLM top-k/expert metadata alignment" "$MOE_VLLM_AUX_DIR/bench_moe_align" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" 16 --bench 0 1
  run_case "moe_gate_up_decode_vllm" moe_ffn decode "vLLM Marlin gate/up expert GEMM" "$MOE_VLLM_MARLIN_BIN" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" "$MOE_GATE_K" "$MOE_GATE_N" --balanced --no-topk-weights --bench 0 1
  run_case "moe_gated_decode_vllm" moe_ffn decode "vLLM SiLU and multiply" "$MOE_VLLM_AUX_DIR/bench_silu_and_mul" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_INTERMEDIATE" --bench 0 1
  run_case "moe_down_decode_vllm" moe_ffn decode "vLLM Marlin down expert GEMM" "$MOE_VLLM_MARLIN_BIN" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" "$MOE_DOWN_K" "$MOE_DOWN_N" --balanced --bench 0 1
  run_case "moe_finalize_decode_vllm" moe_ffn decode "vLLM reduce top-k experts" "$MOE_VLLM_AUX_DIR/bench_moe_sum" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_DOWN_N" --bench 0 1
fi

# Sampling block.
run_sampling_lm_head_case
run_sampling_case "sampling_topk_mask_logits" topk_mask "TopKMaskLogits over vocab=248320"
run_sampling_case "sampling_softmax" softmax "Softmax over vocab=248320"
run_sampling_case "sampling_top_p" top_p "Top-p sampling from probabilities"
RUNNER

  chmod +x "$runner"
}

EXECUTABLES=(
  "general/bench_cublas_gemm|General dense FP16/BF16 GEMM and LM head"
  "general/bench_cuda_core_gemv|General dense decode GEMV CUDA-core baseline"
  "general/bench_layernorm|General layernorm standalone"
  "general/bench_rmsnorm|General RMSNorm standalone"
  "general/w4a16_gemm/cutlass55_standalone/build_cmake_release/cutlass55_fp16_gemm|CUTLASS example-55 style W4A16 FP16 GEMM"
  "general/w4a16_gemm/cutlass55_standalone/build_cmake_release/cutlass55_bf16_gemm|CUTLASS example-55 style W4A16 BF16 GEMM"
  "general/w4a16_gemm/fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm|W4A16 decode fpA_intB GEMM"
  "general/w4a16_gemm/machete_standalone/build_cmake_release/test_machete_gemm|W4A16 prefill Machete/Cutlass55 GEMM"
  "general/w4a16_gemm/marlin_standalone/marlin_standalone|W4A16 Marlin standalone"
  "linear_attn/bench_rmsnorm|Linear-attention RMSNorm"
  "linear_attn/bench_linear_ops|Linear-attention residual/elementwise ops"
  "linear_attn/bench_fused_rms_norm_gate|Linear-attention fused RMSNorm gate"
  "linear_attn/bench_conv1d_update|Linear-attention decode conv1d update"
  "linear_attn/bench_gated_delta_net|Linear-attention decode gated delta net"
  "linear_attn/bench_conv1d_fwd|Linear-attention prefill conv1d"
  "linear_attn/bench_gdn_prefill|Linear-attention FlashInfer GDN prefill"
  "flash_attn/bench_rmsnorm|Flash-attention RMSNorm/qk norm"
  "moe_ffn/bench_rmsnorm|MoE-FFN RMSNorm"
  "moe_ffn/bench_shared_expert|MoE-FFN shared expert fusion"
  "moe_ffn/w4a16/machete/build_cmake_release/bench_machete_moe|Experimental grouped Machete MoE GEMM"
  "moe_ffn/w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm|TRT-LLM MoE W4A16 GEMM"
  "moe_ffn/w4a16/trtllm/auxiliary/bench_custom_moe_routing|TRT-LLM MoE routing"
  "moe_ffn/w4a16/trtllm/auxiliary/bench_expert_map|TRT-LLM MoE expert map"
  "moe_ffn/w4a16/trtllm/auxiliary/bench_expand_input_rows|TRT-LLM MoE expand rows"
  "moe_ffn/w4a16/trtllm/auxiliary/bench_gated_activation|TRT-LLM MoE gated activation"
  "moe_ffn/w4a16/trtllm/auxiliary/bench_finalize_moe_routing|TRT-LLM MoE finalize routing"
  "moe_ffn/w4a16/trtllm/auxiliary/bench_shared_expert_activation|TRT-LLM shared expert activation"
  "moe_ffn/w4a16/trtllm/auxiliary/bench_moe_align|TRT-LLM MoE alignment utility"
  "moe_ffn/w4a16/vllm/auxiliary/bench_topk_gating|vLLM MoE decode top-k gating"
  "moe_ffn/w4a16/vllm/auxiliary/bench_moe_align|vLLM MoE decode alignment"
  "moe_ffn/w4a16/vllm/auxiliary/bench_silu_and_mul|vLLM MoE decode SiLU/mul"
  "moe_ffn/w4a16/vllm/auxiliary/bench_moe_sum|vLLM MoE decode top-k reduction"
  "moe_ffn/w4a16/vllm/marlin/bench_marlin_moe|vLLM MoE decode Marlin GEMM"
  "sampling/bench_sampling|FlashInfer-derived sampling kernels"
)

FILES=(
  "general/w4a16_gemm/machete_standalone/cutlass55_tactics_h800.cache|Machete/Cutlass55 W4A16 tactic cache"
  "general/w4a16_gemm/fpA_intB_standalone/tactics_h800.cache|fpA_intB W4A16 tactic cache"
  "moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache|TRT-LLM MoE W4A16 tactic cache"
  "moe_ffn/w4a16/machete/machete_moe_tactics_h800.cache|Experimental Machete MoE tactic cache"
  "flash_attn/bench_flash_attn.py|FlashAttention Python benchmark entry point"
  "flash_attn/bench_flash_infer.py|FlashInfer Python benchmark entry point"
)

if [[ "$CLEAN" == 1 && -e "$DEST_DIR" ]]; then
  info "cleaning $DEST_DIR"
  if [[ "$DRY_RUN" == 0 ]]; then
    rm -rf "$DEST_DIR"
  fi
fi

if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$DEST_DIR"
  MANIFEST="$DEST_DIR/MANIFEST.tsv"
  printf 'type\tpath\tnote\n' > "$MANIFEST"
else
  MANIFEST=/dev/null
fi

info "source: $ROOT_DIR"
info "dest:   $DEST_DIR"

for item in "${EXECUTABLES[@]}"; do
  rel="${item%%|*}"
  note="${item#*|}"
  copy_item exe "$rel" "$note"
done

for item in "${FILES[@]}"; do
  rel="${item%%|*}"
  note="${item#*|}"
  copy_item file "$rel" "$note"
done

write_runner

if [[ "$DRY_RUN" == 0 ]]; then
  info "wrote manifest: $MANIFEST"
  info "wrote runner:   $DEST_DIR/run_exported_benchmarks.sh"
  info "try: cd \"$DEST_DIR\" && ./run_exported_benchmarks.sh --list"
fi
