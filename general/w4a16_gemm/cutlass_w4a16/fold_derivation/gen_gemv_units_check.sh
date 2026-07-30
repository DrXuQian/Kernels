#!/usr/bin/env bash
# Run the REAL GEMV unit generator (a slice of CMakeLists.txt, via cmake -P) and check what it produced.
#
# WHY THE REAL GENERATOR AND NOT A RE-IMPLEMENTATION. The MoE generator once shipped a row with a ';' inside
# it, which CMake split into two -- and the malformed iterations still summed to the RIGHT unit count, so the
# configure log said "128 generated units" while five of them were garbage. A gate that re-derives the count
# its own way cannot catch that; a gate that runs the generator and compares against the count DERIVED FROM
# THE AXIS LISTS can.
#
#   ./gen_gemv_units_check.sh          check
#   BAD=1 ./gen_gemv_units_check.sh    negative control (inject a ';' into a row and require failure)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CML="$HERE/../CMakeLists.txt"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# The slice: the group table through the generated-inc write, stopping before the executable that consumes it.
awk '/^set\(_GEMV_GROUPS/{p=1} /^ppu_w4a16_executable\(/{if(p)exit} p' "$CML" > "$OUT/slice.cmake"
[ -s "$OUT/slice.cmake" ] || { echo "  [FAIL] gen_gemv_units_check: empty slice -- the anchors moved"; exit 1; }

if [ "${BAD:-0}" = 1 ]; then
  # A ';' inside a tier entry: CMake splits the row, so the fields no longer line up.
  sed -i 's/"16,128,8,4"/"16,128;8,4"/' "$OUT/slice.cmake"
fi

# cmake -P has no CMAKE_CURRENT_BINARY_DIR, so give it one.
{ echo "set(CMAKE_CURRENT_BINARY_DIR \"$OUT\")"; cat "$OUT/slice.cmake"; } > "$OUT/run.cmake"
log="$OUT/log"
if ! cmake -P "$OUT/run.cmake" > "$log" 2>&1; then
  if [ "${BAD:-0}" = 1 ]; then
    echo "  [ok]   gen_gemv_units_check (negative control): the generator REJECTED the malformed row"
    grep -m1 -i "fields, expected" "$log" | sed 's/^/           /'
    exit 0
  fi
  echo "  [FAIL] gen_gemv_units_check: the generator errored"; sed 's/^/           /' "$log"; exit 1
fi
if [ "${BAD:-0}" = 1 ]; then
  echo "  [FAIL] gen_gemv_units_check (negative control): a malformed row was ACCEPTED"
  sed 's/^/           /' "$log"; exit 1
fi

# EXPECTATIONS DERIVED FROM THE AXIS LISTS, not written down. Sum over groups of that group's tier length.
exp=0
groups=0
while IFS= read -r row; do
  tier="${row##*|}"; tier="${tier%\"}"
  n=$(awk -v t="$tier" '$0 ~ "^set\\(_GEMV_TIER_" t "([[:space:]]|$)" {f=1} f{c+=gsub(/"[0-9]+,[0-9]+,[0-9]+,[0-9]+"/,"&")} f&&/\)[[:space:]]*$/{print c; exit}' "$OUT/slice.cmake")
  exp=$((exp + n)); groups=$((groups + 1))
done < <(awk '/^set\(_GEMV_GROUPS/{f=1;next} f&&/^\)/{exit} f&&/\|/{gsub(/^[[:space:]]*"/,"");print}' "$OUT/slice.cmake")

got=$(ls "$OUT/gemv_units"/*.cu 2>/dev/null | wc -l)
inc="$OUT/gemv_units/gemv_perf_units.inc"
[ -f "$inc" ] || { echo "  [FAIL] gen_gemv_units_check: no gemv_perf_units.inc produced"; exit 1; }
inc_n=$(grep -oE '#define GEMV_UNIT_COUNT [0-9]+' "$inc" | grep -oE '[0-9]+')
inc_g=$(grep -oE '#define GEMV_GROUP_COUNT [0-9]+' "$inc" | grep -oE '[0-9]+')
decls=$(grep -c '^void gemv_unit_' "$inc")
calls=$(grep -cE '^    gemv_unit_.*\(sh, _bf, b\[[0-9]+\]\);' "$inc")

fail=0
[ "$got" = "$exp" ]    || { echo "  [FAIL] units on disk $got != derived $exp"; fail=1; }
[ "$inc_n" = "$exp" ]  || { echo "  [FAIL] GEMV_UNIT_COUNT $inc_n != derived $exp"; fail=1; }
[ "$inc_g" = "$groups" ] || { echo "  [FAIL] GEMV_GROUP_COUNT $inc_g != $groups"; fail=1; }
[ "$decls" = "$exp" ]  || { echo "  [FAIL] $decls declarations != derived $exp"; fail=1; }
[ "$calls" = "$exp" ]  || { echo "  [FAIL] $calls calls != derived $exp"; fail=1; }
# Every unit on disk must be both declared and called, by name -- a count match alone would miss a duplicate.
for f in "$OUT/gemv_units"/*.cu; do
  fn=$(basename "$f" .cu)
  grep -q "^void $fn(" "$inc" || { echo "  [FAIL] $fn generated but not declared"; fail=1; }
  grep -q "    $fn(sh, _bf," "$inc" || { echo "  [FAIL] $fn generated but not called"; fail=1; }
done
[ "$fail" = 0 ] && echo "  [ok]   gen_gemv_units_check: $got units == derived from the axis lists, $groups groups, all declared and called"
exit "$fail"
