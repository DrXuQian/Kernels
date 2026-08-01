#!/usr/bin/env bash
# ONE RUN FOR EVERY OPEN QUESTION. Six binaries, built once, kept, then measured on one pinned configuration.
#
# WHY A SCRIPT AND NOT A COMMAND BLOCK. Every trap this task has hit is a step that is easy to drop by hand:
#   * build.sh `rm -rf`s the SAME output directory, so a binary must be copied out BEFORE the next build starts;
#     forget it and you compare a binary with itself, which is indistinguishable from "the change does nothing"
#   * a macro that never reached the device compile produces the same "no change"; build.sh prints
#     "PPU_DEFS verified on <target>'s compile command" and this script FAILS if that line is missing
#   * SPLITK_ONLY matches the whole tag, so "16x128:256" keeps every warp shape, every Stages and every slice count --
#     ~228 rows, one cold launch each. SPLITK_CFG + SPLITK_S pin exactly one
#   * same-config run-to-run spread is ~13%, so every number here comes from ONE run of ONE binary set
#
#   ./run_batch.sh build     build all six and copy them out (slow: ~6 compiles)
#   ./run_batch.sh check     correctness gates -- must pass before any timing is meaningful
#   ./run_batch.sh perf      the pinned rows
#   ./run_batch.sh           all three, in that order
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EX="${EX:-$(cd "$HERE/../../../third_party/actlize" && pwd)/build_w4a16_compare/examples/99_kernels_w4a16_compare}"
OUT="${OUT:-$HOME/ab}"
BAND="${BAND:-64 8 2048 2048 32 3}"                 # L=64, top-k 8, N=K=2048, gs=32, decode
CFG="${CFG:-16x128:256 w16x16 s2}"                  # the pinned row; SPLITK_S below pins the slice count
mkdir -p "$OUT"

# name : defines.  SK_QUANT=2 is FinegrainedScaleZero, what ships.
VARIANTS=(
  "base:SK_QUANT=2"
  "swz:SK_QUANT=2 PPU_SCALE_SWIZZLE=1"
  "bdqnop:SK_QUANT=2 PPU_B_DEQUANT_NOP=1"
  "pack:SK_QUANT=2 PPU_PACKED_SCALE=1"
  "packnop:SK_QUANT=2 PPU_PACKED_SCALE=1 PPU_PACKED_SCALE_NOP=1"
  "packsplit:SK_QUANT=2 PPU_PACKED_SCALE=1 PPU_PACKED_SPLIT_GROUPS=1"
)

build_one() {  # $1 name  $2 defines  $3 target
  # SEPARATE STATEMENTS. `local a=$1 log=...$a...` fails under `set -u` in bash 5.1: local declares every name first
  # and only then assigns, so the self-reference is unbound. bash -n does not catch it -- only running does, which is
  # why this script is now dry-run against a stubbed build.sh before it ships.
  local name="$1" defs="$2" tgt="$3"
  local log="$OUT/build_$name.log"
  printf '  %-10s %s\n' "$name" "$defs"
  ( cd "$HERE" && PPU_DEFS="$defs" TARGET="$tgt" ./build.sh ) >"$log" 2>&1
  if ! grep -q "PPU_DEFS verified on $tgt" "$log"; then
    # A missing verification line means the define never reached the DEVICE compile, and every number from this
    # binary would silently be the default build's.
    echo "    FAILED: no 'PPU_DEFS verified' line in $log -- the macro did not reach the device compile"; return 1
  fi
  [ -x "$EX/$tgt" ] || { echo "    FAILED: $EX/$tgt not built (see $log)"; return 1; }
  cp "$EX/$tgt" "$OUT/${tgt}__$name"                 # BEFORE the next build deletes it
}

do_build() {
  echo "== build =="
  local ok=1
  for v in "${VARIANTS[@]}"; do build_one "${v%%:*}" "${v#*:}" test_moe_splitk_bench || ok=0; done
  # the correctness gate needs its own target, packed on, plus the two variants that change the packed path
  build_one gate_pack  "PPU_PACKED_SCALE=1"                            test_q4k_packed_gemm || ok=0
  build_one gate_split "PPU_PACKED_SCALE=1 PPU_PACKED_SPLIT_GROUPS=1"  test_q4k_packed_gemm || ok=0
  build_one gate_swz   "PPU_PACKED_SCALE=1 PPU_SCALE_SWIZZLE=1"        test_q4k_packed_gemm || ok=0
  build_one gate_swz_only "PPU_SCALE_SWIZZLE=1"                        test_moe_grouped_verify || ok=0
  echo "== distinct-binary check =="
  # Two identical binaries mean an A/B that compares something with itself.
  local dup
  dup=$(md5sum "$OUT"/test_moe_splitk_bench__* | awk '{print $1}' | sort | uniq -d)
  if [ -n "$dup" ]; then echo "  !!! two splitk binaries are IDENTICAL -- the A/B is invalid"; md5sum "$OUT"/test_moe_splitk_bench__*; ok=0
  else echo "  all six splitk binaries differ"; fi
  return $((1-ok))
}

do_check() {
  echo
  echo "== correctness (nothing below matters until these pass) =="
  # rowC is the only row where Scale_TileK == 8, i.e. the only one on the packed path. It has been INTERMITTENT --
  # bad=128, then 724, then MATCH with no semantic source change -- and the cause is still unknown, so it is run five
  # times. A single pass proves nothing about a flaky failure.
  echo "-- packed, five runs (rowC has been intermittent; the cause is NOT known)"
  for i in 1 2 3 4 5; do
    printf '   run %d: ' "$i"
    "$OUT/test_q4k_packed_gemm__gate_pack" "$HERE/real_weight/q4k_packed.bin" 2>&1 | grep -E "rowC|== (PASS|FAIL)" | tr '\n' ' '; echo
  done
  for g in split swz; do
    echo "-- packed + $g"
    "$OUT/test_q4k_packed_gemm__gate_$g" "$HERE/real_weight/q4k_packed.bin" 2>&1 | grep -E "rowA|rowB|rowC|== (PASS|FAIL)"
  done
  # The swizzle changes ADDRESSES, not values. Any mismatch here means a view of the scale buffer that does not carry
  # it -- i.e. the kernel writes at one address and reads at another.
  echo "-- swizzle alone, on the grouped verifier (addresses change, numbers must not)"
  "$OUT/test_moe_grouped_verify__gate_swz_only" 8 1 2>&1 | tail -6
}

do_perf() {
  echo
  echo "== perf: one pinned row, $CFG S=1, band '$BAND' =="
  for v in "${VARIANTS[@]}"; do
    local n="${v%%:*}"
    printf -- '-- %s\n' "$n"
    SPLITK_CFG="$CFG" SPLITK_S=1 "$OUT/test_moe_splitk_bench__$n" $BAND 2>&1 \
      | grep -E "^  i4|SPLITK_CFG|launches refused" | sed 's/^/   /'
  done
  cat <<'EOT'

== how to read it ==
   base                     the shipped path
   swz     - base           the scale read's bank conflicts, 4-way -> 1-way (l98). Attacks the 6.6% that
                            PPU_SCALE_PREFETCH could not: SK_QUANT=0 prices the whole reload at 7.3% and
                            prefetch, which removes only the WAITING, recovered 0.7%
   base    - bdqnop         what the BASELINE int4->fp16 dequant costs. Upper bound: the ablation also drops
                            most of the scale/zero LOADS, since only one fragment element stays live
   pack    - base           the native-format tax as it stands (+12.9% at last measurement)
   pack    - packnop        the packed decode's ARITHMETIC alone
   packnop - base           its transport, its explicit stores, and the barrier slot they sit in
   packsplit - pack         eight warps decoding four groups each instead of four decoding eight. Better means the
                            publication barrier's critical path costs; no change means aggregate issue demand does

   bdqnop and packnop produce DELIBERATELY WRONG numbers -- read their time, never their MATCH.
EOT
}

case "${1:-all}" in
  build) do_build ;;
  check) do_check ;;
  perf)  do_perf ;;
  all)   do_build && do_check && do_perf ;;
  *)     echo "usage: $0 [build|check|perf|all]"; exit 2 ;;
esac
