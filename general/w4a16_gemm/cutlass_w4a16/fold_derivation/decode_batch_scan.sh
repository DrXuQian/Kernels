#!/usr/bin/env bash
# THE BAND BETWEEN batch=1 AND batch=1024, which was never measured. Only the two endpoints existed -- decode at 24.8% of the
# memory roof, and 128 rows/expert at 56.5% MFU -- and the whole crossover region in between was empty. That region is where
# llama.cpp switches kernels (MMVQ up to batch 8, MMQ above, thresholds tuned PER FORMAT) and it is where real serving lives.
#
# No code change is needed. mode 0 gives all L experts `Rows` rows each; with top-k = 8 out of L = 64 that is batch = 8*Rows.
# mode 3 is the batch=1 point (8 active experts x 1 row).
#
#   BIN=<path to test_lowbit_moe_bench> ./decode_batch_scan.sh [N] [K] [gs]
set -Eeuo pipefail
: "${BIN:?set BIN to the test_lowbit_moe_bench binary}"
N="${1:-2048}"; K="${2:-2048}"; GS="${3:-32}"
echo "### batch=1  (mode 3: top-k 8 of 64, one row each)"
"$BIN" 64 8 "$N" "$K" "$GS" 3 2>&1 | sed -n '/roofline/,$p'
for R in 1 2 4 8 16 32 64 128; do
  echo
  echo "### batch=$((8*R))  (mode 0: 64 experts x $R rows)"
  "$BIN" 64 "$R" "$N" "$K" "$GS" 0 2>&1 | sed -n '/roofline/,$p'
done
