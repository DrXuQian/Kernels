#!/usr/bin/env bash
# Capture ONE kernel launch of ONE sweep row under acu.
#
# Fixing the invocation here because it lived only in shell history: the repo recorded acu's OUTPUT fields (Regs,
# Theoretical/Achieved occupancy, Block Limit Registers, Block Limit Shared Mem, Grid, Waves Per CU) and the .acurep
# artifacts, but never the command, so every profiling round started by asking for it again.
#
#   A=<acu binary> BIN=<test_lowbit_moe_bench> ./acu_capture.sh "i4 16x32:32 w16x32 s4" [outname] [-- bench args...]
#
# MOE_ACU=1 makes the harness issue EXACTLY ONE launch with no warmup (time_it(f,0)); MOE_ONLY skips the offline pack for
# every other shape, so the process is short and acu attributes everything to the row you asked for. The tag must be copied
# BYTE FOR BYTE out of the VERDICT line -- it is a substring match, spaces included, and a typo silently profiles nothing.
set -Eeuo pipefail
: "${A:?set A to the acu binary}"
: "${BIN:?set BIN to the test_lowbit_moe_bench binary}"
TAG="${1:?pass the row tag, e.g. \"i4 16x32:32 w16x32 s4\"}"
OUT="${2:-$(echo "$TAG" | tr -c 'A-Za-z0-9' '_')}"
shift; [ $# -gt 0 ] && shift || true
ARGS=("${@:-}")
[ "${#ARGS[@]}" -eq 1 ] && [ -z "${ARGS[0]}" ] && ARGS=(64 8 2048 2048 32 3)   # default: decode batch=1, top-k 8 of 64

# The report must be able to name its own subject. Attributing a capture to the wrong config has cost this work rounds, so
# print the tag and let the harness echo it back from inside the process.
echo "[acu_capture] tag='$TAG'  args='${ARGS[*]}'  ->  ${OUT}.report"
export MOE_ONLY="$TAG"
export MOE_ACU=1
"$A" -o "${OUT}.report" --set full -f "$BIN" "${ARGS[@]}"
echo "[acu_capture] wrote ${OUT}.report -- confirm the run printed '[acu] ONE launch: $TAG' above."
