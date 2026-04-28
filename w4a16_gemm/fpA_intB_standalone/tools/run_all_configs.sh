#!/usr/bin/env bash
set -u

BIN=""
M=1
N=2048
K=2048
GROUP=128
WARMUP=10
ITERS=100
OUT=""
SKIP_CUDA=0

usage() {
  cat <<USAGE
Usage: $0 --bin <test_fpA_intB_gemm> [--m N] [--n N] [--k N] [--group N] [--warmup N] [--iters N] [--out FILE] [--skip-cuda]

Example:
  $0 --bin fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --m 1 --n 2048 --k 2048 --group 128 --out results.txt
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2;;
    --m) M="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --group|--group_size) GROUP="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --skip-cuda) SKIP_CUDA=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
 done

if [[ -z "$BIN" ]]; then
  echo "--bin is required" >&2
  usage
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN" >&2
  exit 1
fi

if [[ -n "$OUT" ]]; then
  : > "$OUT"
fi

map_tile_enum() {
  case "$1" in
    3) echo "16x128x64";;
    4) echo "32x128x64";;
    7) echo "64x128x64";;
    11) echo "128x128x64";;
    15) echo "16x256x64";;
    *) return 1;;
  esac
}

decode_shape_tuple() {
  local value="$1"
  echo "$((value / 1000000))x$(((value % 1000000) / 1000))x$((value % 1000))"
}

config_line_to_arg() {
  local cfg="$1"

  case "$cfg" in
    cuda|cuda_kernel|cuda=1)
      echo "cuda"
      return 0
      ;;
    sm80:*|sm90:*|tma:*)
      echo "$cfg"
      return 0
      ;;
  esac

  if [[ "$cfg" =~ tile_enum=([0-9]+)[[:space:]]+stages=([0-9]+)[[:space:]]+split_k=([0-9]+) ]]; then
    local tile_enum="${BASH_REMATCH[1]}"
    local stages="${BASH_REMATCH[2]}"
    local split_k="${BASH_REMATCH[3]}"
    local tile_shape
    tile_shape=$(map_tile_enum "$tile_enum") || return 1
    echo "sm80:${tile_shape}:${stages}:${split_k}"
    return 0
  fi

  if [[ "$cfg" =~ cuda=0,tma=1,sm=90,tile90=([0-9]+),ml=0,el=0,cl=([0-9]+) ]]; then
    echo "sm90:$(decode_shape_tuple "${BASH_REMATCH[1]}"):$(decode_shape_tuple "${BASH_REMATCH[2]}")"
    return 0
  fi

  if [[ "$cfg" =~ cuda=0,tma=0,tile80=([0-9]+),stages=([0-9]+),splitk=([0-9]+) ]]; then
    local tile_shape
    tile_shape=$(map_tile_enum "${BASH_REMATCH[1]}") || return 1
    echo "sm80:${tile_shape}:${BASH_REMATCH[2]}:${BASH_REMATCH[3]}"
    return 0
  fi

  return 1
}

configs=$($BIN --list_configs | sed -n 's/^ *[0-9]\+: //p')

fail=0
idx=0
while IFS= read -r cfg; do
  if [[ -z "$cfg" ]]; then
    continue
  fi

  if [[ "$SKIP_CUDA" -eq 1 && ( "$cfg" == "cuda" || "$cfg" == "cuda_kernel" || "$cfg" == "cuda=1" ) ]]; then
    continue
  fi

  if config_arg=$(config_line_to_arg "$cfg"); then
    :
  else
    echo "Unrecognized config line: $cfg" >&2
    fail=$((fail+1))
    continue
  fi

  cmd=("$BIN" "--m=$M" "--n=$N" "--k=$K" "--group_size=$GROUP" "--warmup=$WARMUP" "--iters=$ITERS" "--config=$config_arg")
  echo "[${idx}] ${cmd[*]}"
  if [[ -n "$OUT" ]]; then
    {
      echo "[${idx}] ${cmd[*]}"
      "${cmd[@]}"
      echo
    } >> "$OUT" 2>&1 || fail=$((fail+1))
  else
    "${cmd[@]}" || fail=$((fail+1))
  fi

  idx=$((idx+1))
 done <<< "$configs"

if [[ $fail -ne 0 ]]; then
  echo "Done with $fail failures." >&2
  exit 1
fi

echo "Done."
