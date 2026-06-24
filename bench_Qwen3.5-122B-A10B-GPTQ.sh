#!/usr/bin/env bash
# Qwen3.5-122B-A10B (GPTQ) standalone kernel benchmark suite -- cuBLAS GEMM variant.
#
# This is a variant of bench_all.sh in which the W4A16 (GPTQ) linear
# projections normally served by the Machete/CUTLASS55 prefill GEMM and the
# TensorRT-LLM fpA_intB decode GEMV kernels are instead benchmarked as dense
# cuBLAS FP16 GEMMs at the same M/N/K shapes. Every other kernel is identical
# to bench_all.sh.
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
QUANTIZED_GEMM_DTYPE="${QUANTIZED_GEMM_DTYPE:-fp16}"
QUANTIZED_GEMM_LABEL_KIND="${QUANTIZED_GEMM_LABEL_KIND:-w4a16}"

detect_peak_gbps() {
  local name
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  case "$name" in
    *"H800 PCIe"*|*"H100 PCIe"*)
      echo 2000
      ;;
    *)
      echo 3350
      ;;
  esac
}

MODEL_NAME="${MODEL_NAME:-Qwen3.5-122B-A10B-GPTQ}"
PREFILL_TOKENS="${PREFILL_TOKENS:-3823}"
DECODE_TOKENS="${DECODE_TOKENS:-1}"
CTX_LEN="${CTX_LEN:-3823}"
LINEAR_DIM="${LINEAR_DIM:-12288}"
HIDDEN_DIM="${HIDDEN_DIM:-3072}"
CONV_WIDTH="${CONV_WIDTH:-4}"
LINEAR_Q_HEADS="${LINEAR_Q_HEADS:-16}"
LINEAR_V_HEADS="${LINEAR_V_HEADS:-64}"
LINEAR_HEAD_DIM="${LINEAR_HEAD_DIM:-128}"
LINEAR_SMALL_PROJ_N="${LINEAR_SMALL_PROJ_N:-64}"
LINEAR_ATTN_DTYPE="${LINEAR_ATTN_DTYPE:-bf16}"

MOE_EXPERTS="${MOE_EXPERTS:-8}"
MOE_ROUTER_EXPERTS="${MOE_ROUTER_EXPERTS:-256}"
MOE_TOPK="${MOE_TOPK:-8}"
MOE_GROUP="${MOE_GROUP:-128}"
MOE_INTERMEDIATE="${MOE_INTERMEDIATE:-1024}"
MOE_GATE_N="${MOE_GATE_N:-2048}"
MOE_GATE_K="${MOE_GATE_K:-3072}"
MOE_DOWN_N="${MOE_DOWN_N:-3072}"
MOE_DOWN_K="${MOE_DOWN_K:-1024}"
MOE_SHARED_HIDDEN="${MOE_SHARED_HIDDEN:-3072}"

W4A16_GROUP="${W4A16_GROUP:-128}"
W4A16_LINEAR_QKV_N="${W4A16_LINEAR_QKV_N:-12288}"
W4A16_LINEAR_QKV_K="${W4A16_LINEAR_QKV_K:-3072}"
W4A16_LINEAR_Z_N="${W4A16_LINEAR_Z_N:-8192}"
W4A16_LINEAR_Z_K="${W4A16_LINEAR_Z_K:-3072}"
W4A16_LINEAR_OUT_N="${W4A16_LINEAR_OUT_N:-3072}"
W4A16_LINEAR_OUT_K="${W4A16_LINEAR_OUT_K:-8192}"
W4A16_FULL_ATTN_Q_PROJ_GATE_N="${W4A16_FULL_ATTN_Q_PROJ_GATE_N:-16384}"
W4A16_FULL_ATTN_Q_PROJ_GATE_K="${W4A16_FULL_ATTN_Q_PROJ_GATE_K:-3072}"
W4A16_FULL_ATTN_K_PROJ_N="${W4A16_FULL_ATTN_K_PROJ_N:-512}"
W4A16_FULL_ATTN_K_PROJ_K="${W4A16_FULL_ATTN_K_PROJ_K:-3072}"
W4A16_FULL_ATTN_V_PROJ_N="${W4A16_FULL_ATTN_V_PROJ_N:-512}"
W4A16_FULL_ATTN_V_PROJ_K="${W4A16_FULL_ATTN_V_PROJ_K:-3072}"
W4A16_FULL_ATTN_O_PROJ_N="${W4A16_FULL_ATTN_O_PROJ_N:-3072}"
W4A16_FULL_ATTN_O_PROJ_K="${W4A16_FULL_ATTN_O_PROJ_K:-8192}"
FULL_ATTN_Q_HEADS="${FULL_ATTN_Q_HEADS:-32}"
FULL_ATTN_KV_HEADS="${FULL_ATTN_KV_HEADS:-2}"
FULL_ATTN_HEAD_DIM="${FULL_ATTN_HEAD_DIM:-256}"
W4A16_CONSISTENT_EXPERT_UP_N="${W4A16_CONSISTENT_EXPERT_UP_N:-2048}"
W4A16_CONSISTENT_EXPERT_UP_K="${W4A16_CONSISTENT_EXPERT_UP_K:-3072}"
W4A16_CONSISTENT_EXPERT_DOWN_N="${W4A16_CONSISTENT_EXPERT_DOWN_N:-3072}"
W4A16_CONSISTENT_EXPERT_DOWN_K="${W4A16_CONSISTENT_EXPERT_DOWN_K:-1024}"

DENSE_FFN_INTERMEDIATE="${DENSE_FFN_INTERMEDIATE:-17408}"
DENSE_FFN_GATE_N="${DENSE_FFN_GATE_N:-$((2 * DENSE_FFN_INTERMEDIATE))}"
DENSE_FFN_GATE_K="${DENSE_FFN_GATE_K:-$HIDDEN_DIM}"
DENSE_FFN_DOWN_N="${DENSE_FFN_DOWN_N:-$HIDDEN_DIM}"
DENSE_FFN_DOWN_K="${DENSE_FFN_DOWN_K:-$DENSE_FFN_INTERMEDIATE}"

SAMPLING_VOCAB="${SAMPLING_VOCAB:-248320}"
SAMPLING_TOPK="${SAMPLING_TOPK:-50}"
SAMPLING_TOPP="${SAMPLING_TOPP:-0.9}"

ENABLE_LINEAR_ATTN="${ENABLE_LINEAR_ATTN:-1}"
ENABLE_FULL_ATTN="${ENABLE_FULL_ATTN:-1}"
ENABLE_MOE_FFN="${ENABLE_MOE_FFN:-1}"
ENABLE_DENSE_FFN="${ENABLE_DENSE_FFN:-0}"
ENABLE_SHARED_EXPERT="${ENABLE_SHARED_EXPERT:-1}"
ENABLE_SAMPLING="${ENABLE_SAMPLING:-1}"
USE_W4A16_MOE_SHARED_EXPERT="${USE_W4A16_MOE_SHARED_EXPERT:-1}"

MODEL_LAYERS="${MODEL_LAYERS:-48}"
MODEL_FULL_ATTN_LAYERS="${MODEL_FULL_ATTN_LAYERS:-12}"
MODEL_LINEAR_ATTN_LAYERS="${MODEL_LINEAR_ATTN_LAYERS:-36}"
MODEL_DENSE_FFN_LAYERS="${MODEL_DENSE_FFN_LAYERS:-0}"
MODEL_MOE_FFN_LAYERS="${MODEL_MOE_FFN_LAYERS:-48}"
MODEL_SAMPLING_PREFILL_COUNT="${MODEL_SAMPLING_PREFILL_COUNT:-1}"
MODEL_SAMPLING_DECODE_COUNT="${MODEL_SAMPLING_DECODE_COUNT:-1}"

MODEL_SUMMARY_ARGS=(
  --model-layers "$MODEL_LAYERS"
  --full-attn-layers "$MODEL_FULL_ATTN_LAYERS"
  --linear-attn-layers "$MODEL_LINEAR_ATTN_LAYERS"
  --dense-ffn-layers "$MODEL_DENSE_FFN_LAYERS"
  --moe-ffn-layers "$MODEL_MOE_FFN_LAYERS"
  --sampling-prefill-count "$MODEL_SAMPLING_PREFILL_COUNT"
  --sampling-decode-count "$MODEL_SAMPLING_DECODE_COUNT"
)

repo_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$path"
  fi
}

CUBLAS_GEMM_BIN="$(repo_path "general/bench_cublas_gemm")"
CUDA_CORE_GEMV_BIN="$(repo_path "general/bench_cuda_core_gemv")"
BLOCK_FP8_GEMM_BIN="$(repo_path "general/bench_cutlass_block_fp8_gemm")"
LM_HEAD_GEMV_BIN="$(repo_path "studies/lm_head_gemv_bw/bench_lm_head_gemv")"
FLASHINFER_FP8_MOE_SCRIPT="$(repo_path "moe_ffn/fp8/flashinfer_cutlass/bench_flashinfer_cutlass_fp8_moe.py")"
MOE_FP8_TRTLLM_BIN="$(repo_path "moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release/bench_moe_fp8_blockscale_gemm")"
MOE_TRTLLM_BIN="$(repo_path "moe_ffn/w4a16/trtllm/moe_w4a16_standalone/build_cmake_release/test_moe_w4a16_gemm")"
MOE_TRTLLM_AUX_DIR="$(repo_path "moe_ffn/w4a16/trtllm/auxiliary")"
MOE_VLLM_MARLIN_BIN="$(repo_path "moe_ffn/w4a16/vllm/marlin/bench_marlin_moe")"
MOE_VLLM_AUX_DIR="$(repo_path "moe_ffn/w4a16/vllm/auxiliary")"
LINEAR_RMSNORM_BIN="$(repo_path "linear_attn/bench_rmsnorm")"
LINEAR_OPS_BIN="$(repo_path "linear_attn/bench_linear_ops")"
LINEAR_FUSED_RMS_GATE_BIN="$(repo_path "linear_attn/bench_fused_rms_norm_gate")"
FLASH_RMSNORM_BIN="$(repo_path "flash_attn/bench_rmsnorm")"
FLASH_ATTN_SCRIPT="$(repo_path "flash_attn/bench_flash_attn.py")"
MOE_RMSNORM_BIN="$(repo_path "moe_ffn/bench_rmsnorm")"
MOE_SHARED_EXPERT_BIN="$(repo_path "moe_ffn/bench_shared_expert")"
SAMPLING_BIN="$(repo_path "sampling/bench_sampling")"

MOE_TRTLLM_TACTIC="$(repo_path "moe_ffn/w4a16/trtllm/moe_w4a16_standalone/tactics_h800.cache")"
FLASHINFER_FP8_MOE_TACTIC="${FLASHINFER_FP8_MOE_TACTIC:-$(repo_path "moe_ffn/fp8/flashinfer_cutlass/tactics_h800_minimax.json")}"
MOE_FP8_TRTLLM_TACTIC="${MOE_FP8_TRTLLM_TACTIC:-$(repo_path "moe_ffn/fp8/trtllm_cutlass_standalone/tactics_h800_minimax.cache")}"

FAILED=0
LIST_CASES=0
LIST_HEADER_PRINTED=0
LIST_MISSING_BINS=0
LIST_TOTAL_CASES=0
MATCHED_CASES=0
RAN_CASES=0
SKIPPED_CASES=0
CASE_FILTERS=()
RESUME_FROM=""
RESUME_SEEN=1
RESUME_FOUND=0
NCU_CYCLES=0
NCU_BANDWIDTH=0
NSYS_LATENCY=0
if [[ -n "${NCU_METRICS+x}" ]]; then
  NCU_METRICS_ENV_SET=1
else
  NCU_METRICS_ENV_SET=0
fi
NCU_METRICS="${NCU_METRICS:-sm__cycles_elapsed.avg,sm__cycles_elapsed.max,gpu__time_duration.sum}"
NCU_BANDWIDTH_METRICS="${NCU_BANDWIDTH_METRICS:-sm__cycles_elapsed.avg,sm__cycles_elapsed.max,gpu__time_duration.avg,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed}"
NCU_PEAK_GBPS="${NCU_PEAK_GBPS:-$(detect_peak_gbps)}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-}"
BENCH_DEDUPE="${BENCH_DEDUPE:-1}"
DECODE_CUBLAS_BACKEND="${DECODE_CUBLAS_BACKEND:-cuda_core}"
DECODE_MOE_BACKEND="${DECODE_MOE_BACKEND:-vllm}"
MOE_GEMM_BACKEND="${MOE_GEMM_BACKEND:-trtllm}"
LM_HEAD_GEMV_OP="${LM_HEAD_GEMV_OP:-ptx_tma_ws}"
LM_HEAD_GEMV_K_UNROLL="${LM_HEAD_GEMV_K_UNROLL:-8}"

declare -A DEDUPE_LABEL_BY_KEY=()
declare -A DEDUPE_LOG_BY_KEY=()

usage() {
  cat <<'EOF'
Usage:
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh                         # run all benchmark cases
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --list                  # list available case labels
                                                             # plus executable existence
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --case LABEL            # run one case
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --kernel LABEL          # alias for --case
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --resume-from LABEL     # skip cases before LABEL, then continue
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --run-dir DIR           # run every benchmark with DIR as cwd
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --ncu-cycles            # run selected cases under Nsight Compute
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --ncu-bandwidth         # ncu cycles + DRAM bandwidth metrics
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --nsys-latency          # nsys kernel-duration fallback
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh LABEL [LABEL ...]       # run selected cases

Case matching accepts exact labels or substrings. Examples:
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh w4a16_decode_linear_attn_in_proj_qkv_cublas
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --case moe_gate_up_decode_vllm
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh decode_vllm
  ./bench_Qwen3.5-122B-A10B-GPTQ.sh --resume-from w4a16_prefill_linear_attn_out_proj_cublas

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
  BENCH_DEDUPE             Set to 0 to rerun duplicate benchmark commands/shapes.
                           Default: 1. Duplicate case logs point at the first
                           case that measured the same key.
  QUANTIZED_GEMM_DTYPE     Quantized projection dtype family. Default: fp16.
                           MiniMax wrappers set fp8 and use the repo-owned
                           CUTLASS block-FP8 dense GEMM path.
  QUANTIZED_GEMM_LABEL_KIND Label family for dense replacement projections.
                           Default: w4a16. Dense model wrappers set dense.
  MOE_GEMM_BACKEND         trtllm, cublas, fp8_trtllm, fp8_block_dense, or flashinfer_fp8
                           for routed MoE bodies. Default: trtllm. cublas is
                           a dense baseline path. fp8_trtllm uses the
                           standalone TRT-LLM/CUTLASS block-FP8 grouped MoE
                           GEMM. fp8_block_dense uses the
                           repo-owned CUTLASS block-FP8 dense GEMM on expanded
                           tokens as a standalone MiniMax path.
                           flashinfer_fp8 uses FlashInfer CUTLASS block-FP8
                           fused MoE as a reference, not final standalone.
  DECODE_CUBLAS_BACKEND    cublas or cuda_core for non-lm-head decode dense GEMM. Default: cuda_core.
  DECODE_MOE_BACKEND       vllm or trtllm for decode routed MoE pipeline. Default: vllm.
  LM_HEAD_GEMV_OP          Local lm_head GEMV op. Default: ptx_tma_ws.
  LM_HEAD_GEMV_K_UNROLL    Local lm_head GEMV --k-unroll. Default: 8.
  PYTHON                   Python executable for Python attention cases. Default: python3 in PATH.
  ATTN_BENCH_WARMUP        Warmup iterations for Python full-attention cases. Default: 0.
  ATTN_BENCH_ITERS         Timed iterations for Python full-attention cases. Default: 1.
  MODEL_FULL_ATTN_LAYERS   Model summary full-attention multiplier. Default: 12.
  MODEL_LINEAR_ATTN_LAYERS Model summary linear-attention multiplier. Default: 36.
  MODEL_DENSE_FFN_LAYERS   Model summary dense-FFN multiplier. Default: 0.
  MODEL_MOE_FFN_LAYERS     Model summary MoE-FFN multiplier. Default: 48.
  MODEL_SAMPLING_PREFILL_COUNT Model summary prefill sampling count. Default: 1.
  MODEL_SAMPLING_DECODE_COUNT  Model summary decode sampling count. Default: 1.
  NCU_METRICS              Nsight Compute metrics for --ncu-cycles.
                           Default: sm__cycles_elapsed.avg,sm__cycles_elapsed.max,gpu__time_duration.sum
  NCU_BANDWIDTH_METRICS    Nsight Compute metrics for --ncu-bandwidth.
  NCU_PEAK_GBPS            Peak DRAM GB/s used for computed bandwidth utilization. Default: 3350.
  NCU_LAUNCH_SKIP          Optional Nsight Compute --launch-skip value.
  NCU_LAUNCH_COUNT         Optional Nsight Compute --launch-count value.

Profiler note:
  Use --nsys-latency for Nsight Systems kernel duration when NCU counters are
  unavailable. Use bench_h800_bandwidth.sh for the default H800 bandwidth flow:
  it falls back to nsys effective bandwidth when NCU DRAM counters are blocked.
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
    --ncu-bandwidth)
      NCU_CYCLES=1
      NCU_BANDWIDTH=1
      PERFRAWLOG_POSTPROCESS=0
      if [[ "$NCU_METRICS_ENV_SET" == 0 ]]; then
        NCU_METRICS="$NCU_BANDWIDTH_METRICS"
      fi
      shift
      ;;
    --nsys-latency|--nsys)
      NSYS_LATENCY=1
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
    --decode-moe-backend)
      DECODE_MOE_BACKEND="${2:?missing value for $1}"
      shift 2
      ;;
    --decode-moe-backend=*)
      DECODE_MOE_BACKEND="${1#*=}"
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
      echo "[bench][error] unknown option: $1" >&2
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

case "$DECODE_CUBLAS_BACKEND" in
  cublas|cuda_core)
    ;;
  *)
    echo "[bench][error] DECODE_CUBLAS_BACKEND must be cublas or cuda_core, got: $DECODE_CUBLAS_BACKEND" >&2
    exit 1
    ;;
esac

case "$DECODE_MOE_BACKEND" in
  vllm|trtllm)
    ;;
  *)
    echo "[bench][error] DECODE_MOE_BACKEND must be vllm or trtllm, got: $DECODE_MOE_BACKEND" >&2
    exit 1
    ;;
esac

case "$MOE_GEMM_BACKEND" in
  trtllm|cublas|fp8_trtllm|fp8_block_dense|flashinfer_fp8)
    ;;
  *)
    echo "[bench][error] MOE_GEMM_BACKEND must be trtllm, cublas, fp8_trtllm, fp8_block_dense, or flashinfer_fp8, got: $MOE_GEMM_BACKEND" >&2
    exit 1
    ;;
esac

require_bin() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "[bench][error] missing executable: $path" >&2
    echo "[bench][hint] build required targets first, for example:" >&2
    echo "  ./compile.sh build linear_attn flash_attn sampling moe-ffn moe-trtllm w4a16-machete w4a16-fpa" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[bench][error] missing file: $path" >&2
    exit 1
  fi
}

require_tactic_entry() {
  local path="$1"
  local key="$2"
  if [[ ! -f "$path" ]]; then
    echo "[bench][error] missing tactic cache: $path" >&2
    exit 1
  fi
  if ! grep -qF "$key" "$path"; then
    echo "[bench][error] tactic cache missing shape: $key" >&2
    echo "[bench][error] cache file: $path" >&2
    exit 1
  fi
}

safe_name() {
  echo "$1" | tr ' /:' '___' | tr -cd '[:alnum:]_.-'
}

quote_command() {
  local arg
  printf 'cd %q &&' "$RUN_DIR"
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
}

command_dedupe_key() {
  printf 'cmd:'
  quote_command "$@"
}

label_matches_filter() {
  local label="$1"
  local filter="$2"
  local safe_label
  safe_label="$(safe_name "$label")"

  if [[ "$filter" == "decode" ]]; then
    [[ "$label" == *decode* || "$label" == sampling_* ]]
    return
  fi
  if [[ "$filter" == "prefill" ]]; then
    [[ "$label" == *prefill* ]]
    return
  fi

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

print_list_header() {
  if [[ "$LIST_HEADER_PRINTED" == 0 ]]; then
    printf '%-8s  %-48s  %s\n' "binary" "case" "executable"
    printf '%-8s  %-48s  %s\n' "------" "----" "----------"
    LIST_HEADER_PRINTED=1
  fi
}

list_case_binary() {
  local label="$1"
  local binary="$2"
  local status="missing"

  if [[ -n "$binary" && -x "$binary" ]]; then
    status="ok"
  else
    LIST_MISSING_BINS=$((LIST_MISSING_BINS + 1))
  fi
  LIST_TOTAL_CASES=$((LIST_TOTAL_CASES + 1))

  print_list_header
  printf '%-8s  %-48s  %s\n' "$status" "$label" "${binary:-<empty>}"
}

run_case() {
  local label="$1"
  shift 1
  local dedupe_key=""

  local required_files=()
  local required_tactic_files=()
  local required_tactic_keys=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dedupe-key)
        dedupe_key="${2:?missing key after --dedupe-key}"
        shift 2
        ;;
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

  if ! case_after_resume_point "$label"; then
    return
  fi

  if ! case_selected "$label"; then
    return
  fi

  MATCHED_CASES=$((MATCHED_CASES + 1))

  if [[ "$LIST_CASES" == 1 ]]; then
    list_case_binary "$label" "${cmd[0]:-}"
    return
  fi

  require_bin "${cmd[0]}"
  if [[ "$NCU_CYCLES" == 1 && "$NSYS_LATENCY" == 1 ]]; then
    echo "[bench][error] choose only one profiler: --ncu-cycles/--ncu-bandwidth or --nsys-latency" >&2
    exit 1
  fi
  if [[ "$NCU_CYCLES" == 1 ]] && ! command -v ncu >/dev/null 2>&1; then
    echo "[bench][error] --ncu-cycles requested but ncu is not in PATH" >&2
    exit 1
  fi
  if [[ "$NSYS_LATENCY" == 1 ]] && ! command -v nsys >/dev/null 2>&1; then
    echo "[bench][error] --nsys-latency requested but nsys is not in PATH" >&2
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

  local safe
  safe="$(safe_name "$label")"
  local log="$OUT_DIR/$safe.log"

  if [[ -z "$dedupe_key" ]]; then
    dedupe_key="$(command_dedupe_key "${cmd[@]}")"
  fi

  if [[ "$BENCH_DEDUPE" != 0 && -n "${DEDUPE_LABEL_BY_KEY[$dedupe_key]:-}" ]]; then
    local first_label first_log
    first_label="${DEDUPE_LABEL_BY_KEY[$dedupe_key]}"
    first_log="${DEDUPE_LOG_BY_KEY[$dedupe_key]}"
    SKIPPED_CASES=$((SKIPPED_CASES + 1))

    echo
    echo "=== $label ==="
    echo "[bench] skipped duplicate of: $first_label"
    echo "[bench] first log: $first_log"
    echo "[bench] log: $log"

    {
      echo "label: $label"
      echo "run_dir: $RUN_DIR"
      echo "status: skipped_duplicate"
      echo "duplicate_of: $first_label"
      echo "dedupe_duplicate_of: $first_label"
      echo "duplicate_log: $first_log"
      echo "dedupe_key: $dedupe_key"
      printf 'command:'
      printf ' cd %q &&' "$RUN_DIR"
      printf ' %q' "${cmd[@]}"
      echo
      echo "started_at: $(date -Is)"
      echo "finished_at: $(date -Is)"
    } >"$log"
    printf '[bench] %-34s skipped duplicate\n' "$label"
    return
  fi

  RAN_CASES=$((RAN_CASES + 1))

  echo
  echo "=== $label ==="
  printf '[bench] command:'
  printf ' cd %q &&' "$RUN_DIR"
  printf ' %q' "${cmd[@]}"
  echo
  echo "[bench] log: $log"

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

    echo "[bench] ncu log: $ncu_log"
    printf '[bench] ncu command:'
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
  elif [[ "$NSYS_LATENCY" == 1 ]]; then
    local nsys_dir nsys_base nsys_report
    local -a nsys_cmd
    nsys_dir="$OUT_DIR/nsys"
    nsys_base="$nsys_dir/$safe"
    nsys_report="$nsys_base.nsys-rep"
    mkdir -p "$nsys_dir"
    nsys_cmd=(nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=none --output "$nsys_base")

    echo "[bench] nsys report: $nsys_report"
    printf '[bench] nsys command:'
    printf ' %q' "${nsys_cmd[@]}" "${cmd[@]}"
    echo
    {
      echo "nsys_report: $nsys_report"
      printf 'nsys_command:'
      printf ' %q' "${nsys_cmd[@]}" "${cmd[@]}"
      echo
    } >>"$log"

    (cd "$RUN_DIR" && "${nsys_cmd[@]}" "${cmd[@]}") 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
  else
    (cd "$RUN_DIR" && "${cmd[@]}") 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
  fi
  set -e

  if [[ "$status" == 0 ]]; then
    echo "finished_at: $(date -Is)" >>"$log"
    if run_perfrawlog_postprocess "$label" "${cmd[@]}"; then
      if [[ "$BENCH_DEDUPE" != 0 ]]; then
        DEDUPE_LABEL_BY_KEY["$dedupe_key"]="$label"
        DEDUPE_LOG_BY_KEY["$dedupe_key"]="$log"
      fi
      printf '[bench] %-34s ok\n' "$label"
    else
      local post_status=$?
      printf '[bench] %-34s failed perfstatistics_status=%s\n' "$label" "$post_status" >&2
      FAILED=1
    fi
  else
    echo "failed_at: $(date -Is)" >>"$log"
    echo "exit_status: $status" >>"$log"
    printf '[bench] %-34s failed exit_status=%s\n' "$label" "$status" >&2
    echo "[bench][error] last lines from $log:" >&2
    tail -80 "$log" >&2 || true
    FAILED=1
  fi
}

run_perfrawlog_postprocess() {
  local label="$1"
  shift 1
  local -a cmd=("$@")

  if [[ "${PERFRAWLOG_POSTPROCESS:-1}" == 0 ]]; then
    echo "[bench] PERFRAWLOG_POSTPROCESS=0, skip perfrawlog post-processing."
    return
  fi

  local perfrawlog_path="${PERFRAWLOG_PATH:-$RUN_DIR/perfrawlog}"
  if [[ ! -e "$perfrawlog_path" ]]; then
    echo "[bench] no perfrawlog found under run dir, skip perf statistics generation."
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
  echo "[bench] run_dir: $RUN_DIR"
  echo "[bench] perfrawlog: $perfrawlog_path"
  echo "[bench] report_dir: $report_dir"
  echo "[bench] log: $log"

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

  local report_base summary_log model_summary_dir
  report_base="$(perfstatistics_base_dir)"
  if [[ ! -d "$report_base" ]]; then
    echo "[bench] no per-case perfstatistics directory found, skip summary."
    return
  fi

  summary_log="$OUT_DIR/perfstatistics_summary.txt"
  model_summary_dir="$OUT_DIR/model_latency_perfstatistics"
  echo
  echo "=== perfstatistics summary ==="
  set +e
  python "$ROOT_DIR/helpers/summarize_perfstatistics.py" \
    "$report_base" \
    --ghz "${PERF_STATISTICS_GHZ:-1.5}" \
    --bench-out-dir "$OUT_DIR" \
    --model-summary-dir "$model_summary_dir" \
    "${MODEL_SUMMARY_ARGS[@]}" 2>&1 | tee "$summary_log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" != 0 ]]; then
    echo "[bench][warn] perfstatistics summary did not find any compute_cycles."
    return
  fi
  echo "[bench] perfstatistics summary: $summary_log"
  echo "[bench] perfstatistics model latency summary: $model_summary_dir/model_latency_summary.md"
}

summarize_ncu_cycles() {
  if [[ "$NCU_CYCLES" != 1 ]]; then
    return
  fi

  local ncu_dir summary_log model_summary_dir bandwidth_log
  ncu_dir="$OUT_DIR/ncu"
  if [[ ! -d "$ncu_dir" ]]; then
    echo "[bench] no Nsight Compute output directory found, skip ncu summary."
    return
  fi

  summary_log="$OUT_DIR/ncu_cycles_summary.md"
  model_summary_dir="$OUT_DIR/model_latency_ncu"
  echo
  echo "=== Nsight Compute cycles summary ==="
  set +e
  python "$ROOT_DIR/helpers/summarize_ncu_cycles.py" "$ncu_dir" \
    --detail \
    --ghz "${PERF_STATISTICS_GHZ:-1.5}" \
    --bench-out-dir "$OUT_DIR" \
    --model-summary-dir "$model_summary_dir" \
    "${MODEL_SUMMARY_ARGS[@]}" 2>&1 | tee "$summary_log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" != 0 ]]; then
    echo "[bench][warn] ncu cycles summary did not find any metric rows."
    return
  fi
  echo "[bench] ncu cycles summary: $summary_log"
  echo "[bench] ncu model latency summary: $model_summary_dir/model_latency_summary.md"

  if [[ "$NCU_BANDWIDTH" == 1 ]]; then
    bandwidth_log="$OUT_DIR/ncu_bandwidth_summary.md"
    echo
    echo "=== Nsight Compute bandwidth summary ==="
    set +e
    python "$ROOT_DIR/helpers/summarize_ncu_bandwidth.py" "$ncu_dir" \
      --peak-gbps "$NCU_PEAK_GBPS" \
      --detail 2>&1 | tee "$bandwidth_log"
    local bandwidth_status=${PIPESTATUS[0]}
    set -e
    if [[ "$bandwidth_status" != 0 ]]; then
      echo "[bench][warn] ncu bandwidth summary did not find any DRAM metric rows."
      return
    fi
    echo "[bench] ncu bandwidth summary: $bandwidth_log"
  fi
}

summarize_nsys_latency() {
  if [[ "$NSYS_LATENCY" != 1 ]]; then
    return
  fi

  local nsys_dir summary_log model_summary_dir
  nsys_dir="$OUT_DIR/nsys"
  if [[ ! -d "$nsys_dir" ]]; then
    echo "[bench] no Nsight Systems output directory found, skip nsys summary."
    return
  fi

  summary_log="$OUT_DIR/nsys_latency_summary.md"
  model_summary_dir="$OUT_DIR/model_latency_nsys"
  echo
  echo "=== Nsight Systems latency summary ==="
  set +e
  python "$ROOT_DIR/helpers/summarize_nsys_kernels.py" "$nsys_dir" \
    --detail \
    --bench-out-dir "$OUT_DIR" \
    --model-summary-dir "$model_summary_dir" \
    "${MODEL_SUMMARY_ARGS[@]}" 2>&1 | tee "$summary_log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" != 0 ]]; then
    echo "[bench][warn] nsys latency summary did not find any CUDA kernel rows."
    return
  fi
  echo "[bench] nsys latency summary: $summary_log"
  echo "[bench] nsys model latency summary: $model_summary_dir/model_latency_summary.md"
  echo "[bench][note] nsys records kernel duration. Use bench_h800_bandwidth.sh for nsys-derived effective bandwidth, or --ncu-bandwidth for DRAM-counter bandwidth."
}

run_w4a16_prefill_gemm_cublas_case() {
  local label="$1"
  local m="$2"
  local n="$3"
  local k="$4"

  if [[ "$QUANTIZED_GEMM_DTYPE" == fp8* ]]; then
    run_block_fp8_gemm_case "$(quantized_block_fp8_label "$label")" "$m" "$n" "$k"
    return
  fi

  # GPTQ W4A16 prefill projection benchmarked as a dense cuBLAS GEMM
  # instead of the Machete/CUTLASS55 W4A16 kernel.
  run_cublas_gemm_case "$(quantized_cublas_label "$label")" "$m" "$n" "$k" fp16 "$QUANTIZED_GEMM_DTYPE"
}

run_w4a16_decode_gemv_cublas_case() {
  local label="$1"
  local m="$2"
  local n="$3"
  local k="$4"

  if [[ "$QUANTIZED_GEMM_DTYPE" == fp8* ]]; then
    run_block_fp8_gemm_case "$(quantized_block_fp8_label "$label")" "$m" "$n" "$k"
    return
  fi

  # GPTQ W4A16 decode projection benchmarked as a dense cuBLAS GEMM
  # instead of the TensorRT-LLM fpA_intB W4A16 GEMV kernel.
  run_cublas_gemm_case "$(quantized_cublas_label "$label")" "$m" "$n" "$k" fp16 "$QUANTIZED_GEMM_DTYPE"
}

quantized_block_fp8_label() {
  local label="$1"
  label="${label/w4a16_/fp8_block_}"
  label="${label/_cublas/_cutlass}"
  label="${label/_trtllm/_cutlass}"
  label="${label/_vllm/_cutlass}"
  if [[ "$label" != fp8_block_* ]]; then
    label="fp8_block_${label}"
  fi
  printf '%s\n' "$label"
}

quantized_cublas_label() {
  local label="$1"
  if [[ "$QUANTIZED_GEMM_DTYPE" == fp8* ]]; then
    label="${label/w4a16_/fp8_}"
    label="${label/_trtllm/_cublas}"
    label="${label/_vllm/_cublas}"
    if [[ "$label" != fp8_* ]]; then
      label="fp8_${label}"
    fi
    label="${label/_cublas/_cublas_baseline}"
  elif [[ "$QUANTIZED_GEMM_LABEL_KIND" == "dense" ]]; then
    label="${label/w4a16_/dense_}"
  fi
  printf '%s\n' "$label"
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
    --dedupe-key "rmsnorm:$batch,$embed,fp16" \
    "$bin" \
    --batch "$batch" --embed "$embed" --dtype fp16 --no-check \
    --bench 0 1
}

run_flash_attn_core_case() {
  local label="$1"
  local mode="$2"
  local seq_len="$3"

  # Prefill models chunked prefill: seq_len new query tokens attend to a
  # CTX_LEN-long context. Decode already passes CTX_LEN as its seq_len.
  local ctx_args=()
  local ctx_key=""
  if [[ "$mode" == "prefill" ]]; then
    ctx_args=(--ctx "$CTX_LEN")
    ctx_key=",ctx=$CTX_LEN"
  fi

  run_case "$label" \
    --require-file "$FLASH_ATTN_SCRIPT" \
    --dedupe-key "flash-attn-core:$mode,$seq_len$ctx_key,$FULL_ATTN_Q_HEADS,$FULL_ATTN_KV_HEADS,$FULL_ATTN_HEAD_DIM" \
    "$PYTHON_BIN" "$FLASH_ATTN_SCRIPT" \
    "$mode" "$seq_len" "$FULL_ATTN_Q_HEADS" "$FULL_ATTN_KV_HEADS" "$FULL_ATTN_HEAD_DIM" \
    "${ctx_args[@]}" \
    --bench "$ATTN_BENCH_WARMUP" "$ATTN_BENCH_ITERS"
}

run_sampling_case() {
  local label="$1"
  local op="$2"

  run_case "$label" \
    --dedupe-key "sampling:$op,$HIDDEN_DIM,$SAMPLING_VOCAB,$SAMPLING_TOPK,$SAMPLING_TOPP" \
    "$SAMPLING_BIN" \
    --op="$op" --hidden="$HIDDEN_DIM" --vocab="$SAMPLING_VOCAB" \
    --top-k="$SAMPLING_TOPK" --top-p="$SAMPLING_TOPP" \
    --bench 0 1
}

run_lm_head_gemv_tma_case() {
  run_case "sampling_lm_head_gemv_tma" \
    --dedupe-key "lm-head-gemv:$LM_HEAD_GEMV_OP,$SAMPLING_VOCAB,$HIDDEN_DIM,$LM_HEAD_GEMV_K_UNROLL" \
    "$LM_HEAD_GEMV_BIN" \
    --op="$LM_HEAD_GEMV_OP" --n "$SAMPLING_VOCAB" --k "$HIDDEN_DIM" \
    --k-unroll="$LM_HEAD_GEMV_K_UNROLL" \
    --warmup=0 --iters=1 --no-verify
}

run_cublas_gemm_case() {
  local label="$1"
  local m="$2"
  local n="$3"
  local k="$4"
  local out_dtype="${5:-fp16}"
  local input_dtype="${6:-fp16}"

  run_case "$label" \
    --dedupe-key "cublas-gemm:$m,$n,$k,$input_dtype,$out_dtype" \
    "$CUBLAS_GEMM_BIN" \
    --m="$m" --n="$n" --k="$k" --dtype "$input_dtype" --out-dtype "$out_dtype" \
    --bench 0 1
}

run_block_fp8_gemm_case() {
  local label="$1"
  local m="$2"
  local n="$3"
  local k="$4"
  local out_dtype="${5:-fp16}"

  run_case "$label" \
    --dedupe-key "block-fp8-gemm:$m,$n,$k,$out_dtype" \
    "$BLOCK_FP8_GEMM_BIN" \
    --m="$m" --n="$n" --k="$k" --out-dtype "$out_dtype" \
    --bench 0 1
}

run_cuda_core_gemv_case() {
  local label="$1"
  local m="$2"
  local n="$3"
  local k="$4"
  local out_dtype="${5:-fp16}"

  run_case "$label" \
    --dedupe-key "cuda-core-gemv:$m,$n,$k,fp16,$out_dtype" \
    "$CUDA_CORE_GEMV_BIN" \
    --m="$m" --n="$n" --k="$k" --dtype fp16 --out-dtype "$out_dtype" \
    --bench 0 1
}

run_decode_dense_gemm_case() {
  local cublas_label="$1"
  local cuda_core_label="$2"
  local m="$3"
  local n="$4"
  local k="$5"
  local out_dtype="${6:-fp16}"

  if [[ "$DECODE_CUBLAS_BACKEND" == "cuda_core" ]]; then
    run_cuda_core_gemv_case "$cuda_core_label" "$m" "$n" "$k" "$out_dtype"
  else
    run_cublas_gemm_case "$cublas_label" "$m" "$n" "$k" "$out_dtype"
  fi
}

run_moe_shared_expert_case() {
  local label="$1"
  local op="$2"
  local tokens="$3"
  local out_dim="${4:-1}"

  run_case "$label" \
    --dedupe-key "moe-shared-expert:$op,$tokens,$MOE_SHARED_HIDDEN,$out_dim,fp16" \
    "$MOE_SHARED_EXPERT_BIN" \
    --op="$op" --tokens="$tokens" --hidden="$MOE_SHARED_HIDDEN" --out-dim="$out_dim" --dtype fp16 \
    --bench 0 1
}

run_moe_shared_expert_activation_case() {
  local label="$1"
  local tokens="$2"

  run_case "$label" \
    --dedupe-key "trtllm-shared-expert-activation:$tokens,$MOE_INTERMEDIATE,fp16" \
    "$MOE_TRTLLM_AUX_DIR/bench_shared_expert_activation" "$tokens" "$MOE_INTERMEDIATE" fp16 \
    --bench 0 1
}

run_moe_trtllm_gemm_case() {
  local label="$1"
  local m_per_expert="$2"
  local n="$3"
  local k="$4"

  if [[ "$MOE_GEMM_BACKEND" == "cublas" ]]; then
    local total_m=$((m_per_expert * MOE_EXPERTS))
    local cublas_label="${label/_trtllm/_cublas}"
    cublas_label="$(quantized_cublas_label "$cublas_label")"
    run_cublas_gemm_case "$cublas_label" "$total_m" "$n" "$k" fp16 "$QUANTIZED_GEMM_DTYPE"
    return
  fi

  if [[ "$MOE_GEMM_BACKEND" == "fp8_block_dense" ]]; then
    local total_m=$((m_per_expert * MOE_EXPERTS))
    local fp8_label="${label/_trtllm/_fp8_block_dense}"
    run_block_fp8_gemm_case "$fp8_label" "$total_m" "$n" "$k" bf16
    return
  fi

  if [[ "$MOE_GEMM_BACKEND" == "fp8_trtllm" ]]; then
    local fp8_label="${label/_trtllm/_fp8_trtllm}"
    run_case "$fp8_label" \
      --dedupe-key "trtllm-fp8-moe-gemm:$MOE_EXPERTS,$m_per_expert,$n,$k" \
      --require-tactic-entry "$MOE_FP8_TRTLLM_TACTIC" "$MOE_EXPERTS,$m_per_expert,$n,$k,1x128,128x128|" \
      "$MOE_FP8_TRTLLM_BIN" \
      --experts="$MOE_EXPERTS" --m_per_expert="$m_per_expert" --n="$n" --k="$k" \
      --tactic="$MOE_FP8_TRTLLM_TACTIC" \
      --bench 0 1
    return
  fi

  if [[ "$m_per_expert" == "1" ]]; then
    run_case "$label" \
      "$MOE_TRTLLM_BIN" \
      --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$m_per_expert" \
      --n="$n" --k="$k" --group_size="$MOE_GROUP" \
      --cuda_core \
      --warmup=0 --iters=1
  else
    run_case "$label" \
      --require-file "$MOE_TRTLLM_TACTIC" \
      --require-tactic-entry "$MOE_TRTLLM_TACTIC" "fp16,$MOE_EXPERTS,$m_per_expert,$n,$k,$MOE_GROUP|" \
      "$MOE_TRTLLM_BIN" \
      --dtype=fp16 --experts="$MOE_EXPERTS" --m_per_expert="$m_per_expert" \
      --n="$n" --k="$k" --group_size="$MOE_GROUP" \
      --tactic="$MOE_TRTLLM_TACTIC" \
      --warmup=0 --iters=1
  fi
}

run_flashinfer_fp8_moe_case() {
  local label="$1"
  local tokens="$2"

  require_file "$FLASHINFER_FP8_MOE_SCRIPT"
  run_case "$label" \
    --dedupe-key "flashinfer-fp8-moe:$tokens,$MOE_EXPERTS,$MOE_TOPK,$HIDDEN_DIM,$MOE_INTERMEDIATE,bf16,$FLASHINFER_FP8_MOE_TACTIC" \
    --require-file "$FLASHINFER_FP8_MOE_TACTIC" \
    "$PYTHON_BIN" "$FLASHINFER_FP8_MOE_SCRIPT" \
    --tokens "$tokens" \
    --hidden "$HIDDEN_DIM" \
    --intermediate "$MOE_INTERMEDIATE" \
    --experts "$MOE_EXPERTS" \
    --topk "$MOE_TOPK" \
    --tactic-cache "$FLASHINFER_FP8_MOE_TACTIC" \
    --warmup 0 \
    --iters 1 \
    --skip-check
}

run_linear_dense_case() {
  local label="$1"
  local op="$2"
  local tokens="$3"
  local out_dim="$4"

  run_cublas_gemm_case "$label" "$tokens" "$out_dim" "$HIDDEN_DIM" fp16
}

run_dense_ffn_cases() {
  local phase="$1"
  local tokens="$2"

  run_cublas_gemm_case "dense_ffn_${phase}_gate_up_cublas" \
    "$tokens" "$DENSE_FFN_GATE_N" "$DENSE_FFN_GATE_K" fp16

  run_case "dense_ffn_${phase}_gated_activation" \
    --dedupe-key "dense-ffn-gated:$tokens,$DENSE_FFN_INTERMEDIATE,fp16" \
    "$MOE_TRTLLM_AUX_DIR/bench_gated_activation" "$tokens" 1 "$DENSE_FFN_INTERMEDIATE" fp16 \
    --bench 0 1

  run_cublas_gemm_case "dense_ffn_${phase}_down_cublas" \
    "$tokens" "$DENSE_FFN_DOWN_N" "$DENSE_FFN_DOWN_K" fp16
}

run_residual_add_case() {
  local label="$1"
  local tokens="$2"

  run_case "$label" \
    --dedupe-key "residual-add:$tokens,$HIDDEN_DIM,fp16" \
    "$LINEAR_OPS_BIN" \
    --op=residual_add --tokens="$tokens" --hidden="$HIDDEN_DIM" --dtype fp16 \
    --bench 0 1
}

run_linear_fused_rms_gate_case() {
  local label="$1"
  local rows="$2"

  run_case "$label" \
    --dedupe-key "linear-fused-rms-gate:$rows,$LINEAR_HEAD_DIM,$LINEAR_ATTN_DTYPE" \
    "$LINEAR_FUSED_RMS_GATE_BIN" "$rows" "$LINEAR_HEAD_DIM" --dtype "$LINEAR_ATTN_DTYPE" --bench 0 1
}

if [[ "$LIST_CASES" != 1 ]]; then
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "[bench][error] run dir does not exist: $RUN_DIR" >&2
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
  echo "$MODEL_NAME standalone kernel benchmark suite"
  echo "repo: $ROOT_DIR"
  echo "run dir: $RUN_DIR"
  echo "logs: $OUT_DIR"
  echo "prefill tokens: $PREFILL_TOKENS"
  echo "decode tokens:  $DECODE_TOKENS"
  echo "ctx len:        $CTX_LEN"
  echo "moe prefill:    TensorRT-LLM components"
  echo "moe decode:     $DECODE_MOE_BACKEND components"
  echo "moe gemm:       $MOE_GEMM_BACKEND"
  echo "decode dense:   $DECODE_CUBLAS_BACKEND"
  echo "lm head:        $LM_HEAD_GEMV_OP k_unroll=$LM_HEAD_GEMV_K_UNROLL"
  if [[ "$QUANTIZED_GEMM_DTYPE" == fp8* ]]; then
    echo "quant gemm:     CUTLASS block-FP8 dense projection; moe gemm backend=$MOE_GEMM_BACKEND"
  else
    echo "quant gemm:     cuBLAS dense $QUANTIZED_GEMM_DTYPE GEMM baseline"
  fi
  echo "enabled:        linear_attn=$ENABLE_LINEAR_ATTN full_attn=$ENABLE_FULL_ATTN moe_ffn=$ENABLE_MOE_FFN dense_ffn=$ENABLE_DENSE_FFN shared_expert=$ENABLE_SHARED_EXPERT sampling=$ENABLE_SAMPLING"
  echo "model repeats:  full_attn=$MODEL_FULL_ATTN_LAYERS linear_attn=$MODEL_LINEAR_ATTN_LAYERS dense_ffn=$MODEL_DENSE_FFN_LAYERS moe_ffn=$MODEL_MOE_FFN_LAYERS sampling_prefill=$MODEL_SAMPLING_PREFILL_COUNT sampling_decode=$MODEL_SAMPLING_DECODE_COUNT"
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
  if [[ "$NSYS_LATENCY" == 1 ]]; then
    echo "nsys latency:   enabled"
    echo "bandwidth:      nsys duration only here; bench_h800_bandwidth.sh derives effective bandwidth from shapes"
  fi
  if [[ "$BENCH_DEDUPE" != 0 ]]; then
    echo "dedupe:         enabled"
  else
    echo "dedupe:         disabled"
  fi
  echo "============================================================"
fi

if [[ "$ENABLE_LINEAR_ATTN" == 1 ]]; then
run_rmsnorm_case "linear_attn_decode_rmsnorm" "$LINEAR_RMSNORM_BIN" "$DECODE_TOKENS"
run_rmsnorm_case "linear_attn_prefill_rmsnorm" "$LINEAR_RMSNORM_BIN" "$PREFILL_TOKENS"

run_decode_dense_gemm_case "linear_attn_decode_in_proj_a_cublas" "linear_attn_decode_in_proj_a_cuda_core" \
  "$DECODE_TOKENS" "$LINEAR_SMALL_PROJ_N" "$HIDDEN_DIM" fp16
run_decode_dense_gemm_case "linear_attn_decode_in_proj_b_cublas" "linear_attn_decode_in_proj_b_cuda_core" \
  "$DECODE_TOKENS" "$LINEAR_SMALL_PROJ_N" "$HIDDEN_DIM" fp16

run_linear_dense_case "linear_attn_prefill_in_proj_a_cublas" "in_proj_a" "$PREFILL_TOKENS" "$LINEAR_SMALL_PROJ_N"
run_linear_dense_case "linear_attn_prefill_in_proj_b_cublas" "in_proj_b" "$PREFILL_TOKENS" "$LINEAR_SMALL_PROJ_N"

run_case "linear_decode_conv1d_update" \
  linear_attn/bench_conv1d_update "$LINEAR_DIM" "$CONV_WIDTH" "$DECODE_TOKENS" --dtype "$LINEAR_ATTN_DTYPE" --bench 0 1

run_case "linear_decode_gdn" \
  linear_attn/bench_gated_delta_net "$DECODE_TOKENS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --dtype "$LINEAR_ATTN_DTYPE" --bench 0 1

run_case "linear_prefill_conv1d_fwd" \
  linear_attn/bench_conv1d_fwd "$PREFILL_TOKENS" "$LINEAR_DIM" "$CONV_WIDTH" 1 --dtype "$LINEAR_ATTN_DTYPE" --bench 0 1

run_case "linear_prefill_flashinfer_gdn" \
  linear_attn/bench_gdn_prefill "$PREFILL_TOKENS" "$LINEAR_Q_HEADS" "$LINEAR_V_HEADS" "$LINEAR_HEAD_DIM" 1 --dtype "$LINEAR_ATTN_DTYPE" --bench 0 1

run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_linear_attn_in_proj_qkv_cublas" \
  "$PREFILL_TOKENS" "$W4A16_LINEAR_QKV_N" "$W4A16_LINEAR_QKV_K"

run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_linear_attn_in_proj_z_cublas" \
  "$PREFILL_TOKENS" "$W4A16_LINEAR_Z_N" "$W4A16_LINEAR_Z_K"

run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_linear_attn_out_proj_cublas" \
  "$PREFILL_TOKENS" "$W4A16_LINEAR_OUT_N" "$W4A16_LINEAR_OUT_K"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_linear_attn_in_proj_qkv_cublas" \
  "$DECODE_TOKENS" "$W4A16_LINEAR_QKV_N" "$W4A16_LINEAR_QKV_K"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_linear_attn_in_proj_z_cublas" \
  "$DECODE_TOKENS" "$W4A16_LINEAR_Z_N" "$W4A16_LINEAR_Z_K"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_linear_attn_out_proj_cublas" \
  "$DECODE_TOKENS" "$W4A16_LINEAR_OUT_N" "$W4A16_LINEAR_OUT_K"

run_linear_fused_rms_gate_case "linear_attn_decode_fused_rms_norm_gate" "$LINEAR_V_HEADS"
run_linear_fused_rms_gate_case "linear_attn_prefill_fused_rms_norm_gate" "$((PREFILL_TOKENS * LINEAR_V_HEADS))"

run_residual_add_case "linear_attn_decode_residual_add" "$DECODE_TOKENS"
run_residual_add_case "linear_attn_prefill_residual_add" "$PREFILL_TOKENS"
fi

if [[ "$ENABLE_FULL_ATTN" == 1 ]]; then
run_rmsnorm_case "flash_attn_decode_rmsnorm" "$FLASH_RMSNORM_BIN" "$DECODE_TOKENS"
run_rmsnorm_case "flash_attn_prefill_rmsnorm" "$FLASH_RMSNORM_BIN" "$PREFILL_TOKENS"

run_residual_add_case "flash_attn_decode_residual_add" "$DECODE_TOKENS"
run_residual_add_case "flash_attn_prefill_residual_add" "$PREFILL_TOKENS"

run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_full_attn_q_proj_gate_cublas" \
  "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_Q_PROJ_GATE_N" "$W4A16_FULL_ATTN_Q_PROJ_GATE_K"

run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_full_attn_k_proj_cublas" \
  "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_K_PROJ_N" "$W4A16_FULL_ATTN_K_PROJ_K"

run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_full_attn_v_proj_cublas" \
  "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_V_PROJ_N" "$W4A16_FULL_ATTN_V_PROJ_K"

run_rmsnorm_shape_case "flash_attn_prefill_q_norm" \
  "$FLASH_RMSNORM_BIN" "$((PREFILL_TOKENS * FULL_ATTN_Q_HEADS))" "$FULL_ATTN_HEAD_DIM"

run_rmsnorm_shape_case "flash_attn_prefill_k_norm" \
  "$FLASH_RMSNORM_BIN" "$((PREFILL_TOKENS * FULL_ATTN_KV_HEADS))" "$FULL_ATTN_HEAD_DIM"

run_flash_attn_core_case "flash_attn_prefill_full_attn" \
  prefill "$PREFILL_TOKENS"

run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_full_attn_o_proj_cublas" \
  "$PREFILL_TOKENS" "$W4A16_FULL_ATTN_O_PROJ_N" "$W4A16_FULL_ATTN_O_PROJ_K"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_full_attn_q_proj_gate_cublas" \
  "$DECODE_TOKENS" "$W4A16_FULL_ATTN_Q_PROJ_GATE_N" "$W4A16_FULL_ATTN_Q_PROJ_GATE_K"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_full_attn_k_proj_cublas" \
  "$DECODE_TOKENS" "$W4A16_FULL_ATTN_K_PROJ_N" "$W4A16_FULL_ATTN_K_PROJ_K"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_full_attn_v_proj_cublas" \
  "$DECODE_TOKENS" "$W4A16_FULL_ATTN_V_PROJ_N" "$W4A16_FULL_ATTN_V_PROJ_K"

run_rmsnorm_shape_case "flash_attn_decode_q_norm" \
  "$FLASH_RMSNORM_BIN" "$((DECODE_TOKENS * FULL_ATTN_Q_HEADS))" "$FULL_ATTN_HEAD_DIM"

run_rmsnorm_shape_case "flash_attn_decode_k_norm" \
  "$FLASH_RMSNORM_BIN" "$((DECODE_TOKENS * FULL_ATTN_KV_HEADS))" "$FULL_ATTN_HEAD_DIM"

run_flash_attn_core_case "flash_attn_decode_full_attn" \
  decode "$CTX_LEN"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_full_attn_o_proj_cublas" \
  "$DECODE_TOKENS" "$W4A16_FULL_ATTN_O_PROJ_N" "$W4A16_FULL_ATTN_O_PROJ_K"
fi

if [[ "$ENABLE_SHARED_EXPERT" == 1 ]]; then
run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_consistent_expert_up_cublas" \
  "$PREFILL_TOKENS" "$W4A16_CONSISTENT_EXPERT_UP_N" "$W4A16_CONSISTENT_EXPERT_UP_K"

run_moe_shared_expert_activation_case "moe_shared_expert_activation_prefill_trtllm" "$PREFILL_TOKENS"

run_w4a16_prefill_gemm_cublas_case "w4a16_prefill_consistent_expert_down_cublas" \
  "$PREFILL_TOKENS" "$W4A16_CONSISTENT_EXPERT_DOWN_N" "$W4A16_CONSISTENT_EXPERT_DOWN_K"

run_cublas_gemm_case "moe_shared_expert_gate_prefill_cublas" \
  "$PREFILL_TOKENS" 1 "$HIDDEN_DIM" fp16

run_moe_shared_expert_case "moe_shared_expert_fusion_prefill" "sigmoid_mul_add" "$PREFILL_TOKENS"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_consistent_expert_up_cublas" \
  "$DECODE_TOKENS" "$W4A16_CONSISTENT_EXPERT_UP_N" "$W4A16_CONSISTENT_EXPERT_UP_K"

run_moe_shared_expert_activation_case "moe_shared_expert_activation_decode_trtllm" "$DECODE_TOKENS"

run_w4a16_decode_gemv_cublas_case "w4a16_decode_consistent_expert_down_cublas" \
  "$DECODE_TOKENS" "$W4A16_CONSISTENT_EXPERT_DOWN_N" "$W4A16_CONSISTENT_EXPERT_DOWN_K"

run_decode_dense_gemm_case "moe_shared_expert_gate_decode_cublas" "moe_shared_expert_gate_decode_cuda_core" \
  "$DECODE_TOKENS" 1 "$HIDDEN_DIM" fp16

run_moe_shared_expert_case "moe_shared_expert_fusion_decode" "sigmoid_mul_add" "$DECODE_TOKENS"
fi

if [[ "$ENABLE_DENSE_FFN" == 1 ]]; then
run_rmsnorm_case "dense_ffn_decode_rmsnorm" "$MOE_RMSNORM_BIN" "$DECODE_TOKENS"
run_rmsnorm_case "dense_ffn_prefill_rmsnorm" "$MOE_RMSNORM_BIN" "$PREFILL_TOKENS"

run_dense_ffn_cases "decode" "$DECODE_TOKENS"
run_dense_ffn_cases "prefill" "$PREFILL_TOKENS"

run_residual_add_case "dense_ffn_decode_residual_add" "$DECODE_TOKENS"
run_residual_add_case "dense_ffn_prefill_residual_add" "$PREFILL_TOKENS"
fi

if [[ "$ENABLE_MOE_FFN" == 1 ]]; then
run_rmsnorm_case "moe_ffn_decode_rmsnorm" "$MOE_RMSNORM_BIN" "$DECODE_TOKENS"
run_rmsnorm_case "moe_ffn_prefill_rmsnorm" "$MOE_RMSNORM_BIN" "$PREFILL_TOKENS"

run_residual_add_case "moe_ffn_decode_residual_add" "$DECODE_TOKENS"
run_residual_add_case "moe_ffn_prefill_residual_add" "$PREFILL_TOKENS"

run_cublas_gemm_case "moe_router_gate_prefill_cublas" \
  "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$HIDDEN_DIM" fp16

run_decode_dense_gemm_case "moe_router_gate_decode_cublas" "moe_router_gate_decode_cuda_core" \
  "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$HIDDEN_DIM" fp16

run_case "moe_routing_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_custom_moe_routing" "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" fp16 \
  --bench 0 1

if [[ "$MOE_GEMM_BACKEND" == "flashinfer_fp8" ]]; then
run_flashinfer_fp8_moe_case "moe_fused_prefill_flashinfer_cutlass_fp8" "$PREFILL_TOKENS"
else
run_case "moe_expert_map_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_expert_map" "$PREFILL_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" auto \
  --bench 0 1

run_case "moe_expand_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_expand_input_rows" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_GATE_K" fp16 \
  --bench 0 1

run_moe_trtllm_gemm_case "moe_gate_up_prefill_trtllm" "$PREFILL_TOKENS" "$MOE_GATE_N" "$MOE_GATE_K"

run_case "moe_gated_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_gated_activation" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_INTERMEDIATE" fp16 \
  --bench 0 1

run_moe_trtllm_gemm_case "moe_down_prefill_trtllm" "$PREFILL_TOKENS" "$MOE_DOWN_N" "$MOE_DOWN_K"

run_case "moe_finalize_prefill_trtllm" \
  "$MOE_TRTLLM_AUX_DIR/bench_finalize_moe_routing" "$PREFILL_TOKENS" "$MOE_TOPK" "$MOE_DOWN_N" fp16 \
  --bench 0 1
fi

if [[ "$MOE_GEMM_BACKEND" == "flashinfer_fp8" ]]; then
  run_case "moe_routing_decode_trtllm" \
    "$MOE_TRTLLM_AUX_DIR/bench_custom_moe_routing" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" fp16 \
    --bench 0 1

  run_flashinfer_fp8_moe_case "moe_fused_decode_flashinfer_cutlass_fp8" "$DECODE_TOKENS"
elif [[ "$DECODE_MOE_BACKEND" == "trtllm" ]]; then
  run_case "moe_routing_decode_trtllm" \
    "$MOE_TRTLLM_AUX_DIR/bench_custom_moe_routing" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" fp16 \
    --bench 0 1

  run_case "moe_expert_map_decode_trtllm" \
    "$MOE_TRTLLM_AUX_DIR/bench_expert_map" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" auto \
    --bench 0 1

  run_case "moe_expand_decode_trtllm" \
    "$MOE_TRTLLM_AUX_DIR/bench_expand_input_rows" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_GATE_K" fp16 \
    --bench 0 1

  run_moe_trtllm_gemm_case "moe_gate_up_decode_trtllm" \
    "$DECODE_TOKENS" "$MOE_GATE_N" "$MOE_GATE_K"

  run_case "moe_gated_decode_trtllm" \
    "$MOE_TRTLLM_AUX_DIR/bench_gated_activation" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_INTERMEDIATE" fp16 \
    --bench 0 1

  run_moe_trtllm_gemm_case "moe_down_decode_trtllm" \
    "$DECODE_TOKENS" "$MOE_DOWN_N" "$MOE_DOWN_K"

  run_case "moe_finalize_decode_trtllm" \
    "$MOE_TRTLLM_AUX_DIR/bench_finalize_moe_routing" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_DOWN_N" fp16 \
    --bench 0 1
else
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
    "$MOE_VLLM_AUX_DIR/bench_silu_and_mul" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_INTERMEDIATE" \
    --bench 0 1

  run_case "moe_down_decode_vllm" \
    "$MOE_VLLM_MARLIN_BIN" "$DECODE_TOKENS" "$MOE_ROUTER_EXPERTS" "$MOE_TOPK" "$MOE_DOWN_K" "$MOE_DOWN_N" \
    --balanced --bench 0 1

  run_case "moe_finalize_decode_vllm" \
    "$MOE_VLLM_AUX_DIR/bench_moe_sum" "$DECODE_TOKENS" "$MOE_TOPK" "$MOE_DOWN_N" \
    --bench 0 1
fi
fi

if [[ "$ENABLE_SAMPLING" == 1 ]]; then
run_lm_head_gemv_tma_case
run_sampling_case "sampling_topk_mask_logits" "topk_mask"
run_sampling_case "sampling_softmax" "softmax"
run_sampling_case "sampling_top_p" "top_p"
fi

if [[ "$LIST_CASES" == 1 ]]; then
  if [[ ${#CASE_FILTERS[@]} -gt 0 && "$MATCHED_CASES" == 0 ]]; then
    echo "[bench][error] no benchmark case matched: ${CASE_FILTERS[*]}" >&2
    exit 1
  fi
  if [[ "$LIST_TOTAL_CASES" -gt 0 ]]; then
    echo
    echo "[bench] listed cases: $LIST_TOTAL_CASES"
    echo "[bench] missing binaries: $LIST_MISSING_BINS"
  fi
  if [[ "$LIST_MISSING_BINS" -gt 0 ]]; then
    exit 1
  fi
  exit 0
fi

if [[ -n "$RESUME_FROM" && "$RESUME_FOUND" == 0 ]]; then
  echo "[bench][error] resume label was not found: $RESUME_FROM" >&2
  echo "[bench][hint] run ./bench_Qwen3.5-122B-A10B-GPTQ.sh --list to see available labels." >&2
  exit 1
fi

if [[ ${#CASE_FILTERS[@]} -gt 0 && "$MATCHED_CASES" == 0 ]]; then
  echo "[bench][error] no benchmark case matched: ${CASE_FILTERS[*]}" >&2
  echo "[bench][hint] run ./bench_Qwen3.5-122B-A10B-GPTQ.sh --list to see available labels." >&2
  exit 1
fi

echo
echo "============================================================"
echo "benchmark logs are under: $OUT_DIR"
echo "ran cases: $RAN_CASES"
echo "skipped duplicates: $SKIPPED_CASES"
if [[ "$FAILED" == 0 ]]; then
  echo "All selected cases completed successfully."
else
  echo "Some selected cases failed."
fi
echo "============================================================"

summarize_perfstatistics
summarize_ncu_cycles
summarize_nsys_latency

exit "$FAILED"
