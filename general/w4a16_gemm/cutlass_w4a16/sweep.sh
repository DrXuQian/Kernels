#!/usr/bin/env bash
# Autotune the actlize W4A16 tile, then run the winner once.
#
# TILE_M/TILE_N/WARP_M/WARP_N/STAGES are COMPILE-TIME (they parameterize the cutlass type stack), so there is
# no in-process tile switch -- the sweep is build-time, exactly how cutlass/vLLM autotune. For each candidate
# it builds via ./build.sh, runs a short timing (the "warmup"), parses the [CUTLASS ...] TFLOP/s line, keeps
# the best, then rebuilds that one config and runs it once at full iterations.
#
# Shape/mode are fixed for the whole sweep (env-overridable): M N K G MODE ITERS_WARM ITERS_FINAL.
# Add/trim candidates in CONFIGS below (fields: TILE_M TILE_N WARP_M WARP_N STAGES).
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTLIZE="$(cd "$HERE/../../../third_party/actlize" && pwd)"
BUILD="$ACTLIZE/build_w4a16_compare"

M="${M:-2048}"; N="${N:-4096}"; K="${K:-4096}"; G="${G:-128}"; MODE="${MODE:-1}"
ITERS_WARM="${ITERS_WARM:-30}"; ITERS_FINAL="${ITERS_FINAL:-200}"
RUN_ARGS="--m=$M --n=$N --k=$K --g=$G --mode=$MODE"

# Candidate tiles. WARP must divide TILE and be a multiple of the 16x16 MMA atom. Not all will compile/run on
# the mixed-input AIU mainloop; failures are skipped. Stock 32x32 is kept as the baseline.
CONFIGS=(
  "32 32 16 16 3"     # stock baseline (~27% MFU)
  "64 64 32 32 3"     # 59%
  "64 64 32 32 4"     # 61% -- winner of the first sweep; refine around it
  "64 64 32 32 5"
  "64 64 32 32 6"
  "64 128 32 64 4"
  "128 64 64 32 4"
  "32 64 16 32 4"
  "64 32 32 16 4"
  "128 128 64 64 3"   # 49% -- bigger tiles undersubscribe 72 CU at these shapes
  "128 256 64 64 3"
  "256 128 64 64 3"
)

LOGDIR="$(mktemp -d)"
echo "[sweep] build logs in $LOGDIR"

best_tflops="0"; best_cfg=""; best_line=""
printf '%-22s %-10s %s\n' "CONFIG (TILE/WARP/ST)" "TFLOP/s" "status"
printf '%.0s-' {1..60}; echo
for cfg in "${CONFIGS[@]}"; do
  read -r TILE_M TILE_N WARP_M WARP_N STAGES <<<"$cfg"
  label="${TILE_M}x${TILE_N}/${WARP_M}x${WARP_N}/s${STAGES}"
  if ! TILE_M=$TILE_M TILE_N=$TILE_N WARP_M=$WARP_M WARP_N=$WARP_N STAGES=$STAGES \
        "$HERE/build.sh" >"$LOGDIR/build_${label//\//_}.log" 2>&1; then
    printf '%-22s %-10s %s\n' "$label" "-" "build FAILED (see $LOGDIR)"; continue
  fi
  BIN="$(find "$BUILD" -name bench_cutlass_w4a16 -type f -perm -u+x | head -1)"
  out="$("$BIN" $RUN_ARGS --iterations=$ITERS_WARM 2>&1 || true)"
  if ! grep -q "Disposition: *Passed" <<<"$out"; then
    printf '%-22s %-10s %s\n' "$label" "-" "run FAILED / not Passed"; continue
  fi
  tf="$(grep -oE '[0-9.]+ TFLOP/s' <<<"$out" | grep -oE '[0-9.]+' | tail -1)"
  [ -z "$tf" ] && tf="0"
  printf '%-22s %-10s %s\n' "$label" "$tf" "ok"
  if awk "BEGIN{exit !($tf > $best_tflops)}"; then
    best_tflops="$tf"; best_cfg="$cfg"; best_line="$label"
  fi
done

echo
if [ -z "$best_cfg" ]; then
  echo "[sweep] no config built and passed -- nothing to run." >&2
  exit 1
fi
# Persist the winner as a tactic, machete-style: an exact-match cache keyed by shape, plus a human-readable
# markdown table. Key = "m,n,k,g|config=<label>,tflops=<x>". Load the winning tile from the cache with the
# same key later, or just read TACTICS.md.
CACHE="${OUT_CACHE:-$HERE/tactics_ppu001.cache}"
MD="${OUT_MD:-$HERE/TACTICS_PPU001.md}"
key="${M},${N},${K},${G}"
tmpc="$(mktemp)"; [ -f "$CACHE" ] && grep -v "^${key}|" "$CACHE" > "$tmpc" || true
echo "${key}|config=${best_line},tflops=${best_tflops}" >> "$tmpc"
sort -o "$CACHE" "$tmpc"; rm -f "$tmpc"
{
  echo "# CUTLASS W4A16 (actlize) tactics on ppu001 -- best tile per shape (mode=1, gs from key)"
  echo
  echo "| M | N | K | G | best config (TILE/WARP/ST) | TFLOP/s |"
  echo "|---|---|---|---|---|---|"
  while IFS='|' read -r k rest; do
    IFS=',' read -r m n kk g <<<"$k"
    cfg="$(sed -E 's/.*config=([^,]*),.*/\1/' <<<"$rest")"
    tf="$(sed -E 's/.*tflops=([0-9.]*).*/\1/' <<<"$rest")"
    echo "| $m | $n | $kk | $g | $cfg | $tf |"
  done < "$CACHE"
} > "$MD"
echo "[sweep] wrote tactic cache $CACHE and table $MD"

echo "[sweep] best = $best_line at $best_tflops TFLOP/s. Rebuilding and running once at $ITERS_FINAL iters."
read -r TILE_M TILE_N WARP_M WARP_N STAGES <<<"$best_cfg"
TILE_M=$TILE_M TILE_N=$TILE_N WARP_M=$WARP_M WARP_N=$WARP_N STAGES=$STAGES "$HERE/build.sh" >/dev/null 2>&1
BIN="$(find "$BUILD" -name bench_cutlass_w4a16 -type f -perm -u+x | head -1)"
echo "=========================== WINNER: $best_line ==========================="
"$BIN" $RUN_ARGS --iterations=$ITERS_FINAL
