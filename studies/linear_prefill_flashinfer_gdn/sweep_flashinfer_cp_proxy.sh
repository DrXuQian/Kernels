#!/usr/bin/env bash
set -euo pipefail

# Sweep a correctness-relaxed CP proxy by splitting one total sequence into
# multiple independent cu_seqlens entries. This measures the performance upper
# bound of increasing launch parallelism with the current local CUDA/FlashInfer
# GDN kernel. It is not mathematically equivalent to one recurrent sequence.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEQLEN="${SEQLEN:-3823}"
H_QK="${H_QK:-16}"
H_V="${H_V:-64}"
HEAD_DIM="${HEAD_DIM:-128}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
SEGMENTS="${SEGMENTS:-1 2 4 8 16 32}"

if [[ ! -x ./bench_gdn_tile_study_single_tu ]]; then
  make single_tu -j"$(nproc)"
fi

echo "segments,seqlen,h_qk,h_v,head_dim,median_ms"
for seg in $SEGMENTS; do
  out="$(./bench_gdn_tile_study_single_tu "$SEQLEN" "$H_QK" "$H_V" "$HEAD_DIM" "$seg" --bench "$WARMUP" "$ITERS")"
  median="$(printf '%s\n' "$out" | sed -n 's/.*median=\([0-9.]*\) ms.*/\1/p')"
  echo "$seg,$SEQLEN,$H_QK,$H_V,$HEAD_DIM,$median"
done
