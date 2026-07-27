#!/usr/bin/env bash
# Enumerate, MECHANICALLY, every B-operand instantiation the sources can produce, and flag the ones the delivery
# predicate rejects. Run this before pushing anything the box will compile.
#
# WHY IT EXISTS. sweep_shapes.cpp section 1 was a HAND-WRITTEN list of "the configs the tree uses". It missed
# CORR_DISPATCH(64,64,64) at fbits==1 -- int1, TK=64, WN=32, which over-delivers -- because a dispatch ladder
# instantiates every (TM,TN,TK) triple regardless of the runtime choice, and I had enumerated intent instead of
# reality. The box build found it, and it did not have to: fold_traits.hpp is plain C++ and the ladder is right
# there in the source. Reading code and listing what you believe it instantiates is not a check.
#
# Two kinds of site, treated differently:
#   * dispatch MACROS may select several element widths, so every sub-byte width is crossed in. The warp shape
#     comes from which macro it is -- CORR/FOLD dispatch at 32x32, BITPACK at 32x64, which is its whole point. A
#     rejected combination is acceptable IF the macro gates on fold::deliverable with `if constexpr`.
#   * EXPLICIT filter_and_run<...> instantiations name their width and nothing gates them, so a rejected one is a
#     hard failure -- the box build will not compile.
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
FLAG=$(mktemp); GEN=$(mktemp --suffix=.cpp); BIN=$(mktemp)
gated=$(grep -l 'if constexpr (fold::deliverable' "$SRC"/*.cu 2>/dev/null | wc -l)

emit_warp() {  # tm tn wm wn kind -- the warp tiling must divide the block tile
  printf '#include "%s/fold_traits.hpp"\n#include <cstdio>\nint main(){printf("%%d",fold::warp_shape_ok<%s,%s,%s,%s>?1:0);}\n' \
    "$SRC" "$1" "$2" "$3" "$4" > "$GEN"
  local ok=0
  if g++ -std=c++17 "$GEN" -o "$BIN" 2>/dev/null; then ok=$("$BIN"); fi
  if [ "$ok" = 1 ]; then
    printf "  %-16s TM=%-4s TN=%-4s w%sx%-4s OK\n" "$5" "$1" "$2" "$3" "$4"
  else
    printf "  %-16s TM=%-4s TN=%-4s w%sx%-4s rejected -> HARD FAILURE (warpOnM/N would be 0)\n" "$5" "$1" "$2" "$3" "$4"
    echo hardbad >> "$FLAG"
  fi
}

emit() {  # bits tn tk wm wn kind
  printf '#include "%s/fold_traits.hpp"\n#include <cstdio>\nint main(){printf("%%d",fold::deliverable<%s,%s,%s,%s,%s>?1:0);}\n' \
    "$SRC" "$1" "$2" "$3" "$4" "$5" > "$GEN"
  local ok=0
  if g++ -std=c++17 "$GEN" -o "$BIN" 2>/dev/null; then ok=$("$BIN"); fi
  local verdict
  if [ "$ok" = 1 ]; then verdict="OK"
  else
    case "$6" in
      macro*) verdict="rejected -> must be gated"; echo gatedbad >> "$FLAG" ;;
      *)      verdict="rejected -> HARD FAILURE";  echo hardbad  >> "$FLAG" ;;
    esac
  fi
  printf "  %-16s int%-2s TN=%-4s TK=%-4s w%sx%-4s %s\n" "$6" "$1" "$2" "$3" "$4" "$5" "$verdict"
}

echo "B-operand instantiations found in $SRC:"
grep -rhoE '\b(CORR|BITPACK|FOLD)_DISPATCH\(\s*[0-9]+\s*,\s*[0-9]+\s*,\s*[0-9]+\s*\)' "$SRC"/*.cu \
  | tr -d ' ' | sed -E 's/^(CORR|BITPACK|FOLD)_DISPATCH\(([0-9]+),([0-9]+),([0-9]+)\)/\1 \3 \4/' | sort -u > "$FLAG.m"
while read -r MAC TN TK; do
  [ -n "${MAC:-}" ] || continue
  if [ "$MAC" = BITPACK ]; then WN=64; else WN=32; fi
  for B in 1 2; do emit "$B" "$TN" "$TK" 32 "$WN" "macro:$MAC"; done
done < "$FLAG.m"

# Indirect instantiations: a sweep may wrap filter_and_run in its own template, so the literal grep above cannot
# see the configs. run_cfg<QM, TM, TN, TK, WM, WN, ST> is that wrapper in test_int1_sweep.cu -- a real blind spot the
# first version of this script had, found while adding that file.
grep -rhoE 'run_cfg<[^>]*>' "$SRC"/*.cu | tr -d ' ' \
  | sed -E 's/run_cfg<[A-Za-z:_]+,([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+)>/\2 \3 \4 \5 uint1_t/' \
  | grep -E '^[0-9]' | sort -u > "$FLAG.w"
while read -r TN TK WM WN E; do
  [ -n "${E:-}" ] || continue
  emit 1 "$TN" "$TK" "$WM" "$WN" "indirect:run_cfg"
done < "$FLAG.w"

# The warp-shape constraint. The front-end gate CANNOT see this class: the stub environment's own errors abort
# template instantiation before a config is reached, so TM=16 with WM=32 reached the box reporting "parses clean".
# It is pure arithmetic, so checking it needs no instantiation at all.
grep -rhoE 'run_cfg<[^>]*>' "$SRC"/*.cu | tr -d ' ' \
  | sed -E 's/run_cfg<[A-Za-z:_]+,([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+)>/\1 \2 \4 \5/' \
  | grep -E '^[0-9]' | sort -u > "$FLAG.s"
while read -r TM TN WM WN; do
  [ -n "${WN:-}" ] || continue
  emit_warp "$TM" "$TN" "$WM" "$WN" "warp:run_cfg"
done < "$FLAG.s"

grep -rhoE 'filter_and_run<[^;]{0,200}' "$SRC"/*.cu | tr -d ' ' \
  | grep -oE '<[A-Za-z:_]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[A-Za-z_:0-9]+' \
  | sed -E 's/<[A-Za-z:_]+,[0-9]+,([0-9]+),([0-9]+),([0-9]+),([0-9]+),[0-9]+,(.*)/\1 \2 \3 \4 \5/' | sort -u > "$FLAG.e"
while read -r TN TK WM WN E; do
  [ -n "${E:-}" ] || continue
  case "$E" in *uint1*) B=1;; *uint2*) B=2;; *int4*) B=4;; *) continue;; esac
  emit "$B" "$TN" "$TK" "$WM" "$WN" explicit
done < "$FLAG.e"

echo
rc=0
if grep -q hardbad "$FLAG" 2>/dev/null; then
  echo "FAIL: an explicit instantiation is rejected -- the box build will NOT compile."; rc=1
elif grep -q gatedbad "$FLAG" 2>/dev/null; then
  if [ "$gated" -gt 0 ]; then
    echo "OK: rejected combinations exist, and the macro gates on fold::deliverable ($gated file(s))."
  else
    echo "FAIL: rejected macro combinations with NO 'if constexpr (fold::deliverable' gate anywhere."; rc=1
  fi
else
  echo "OK: every enumerated instantiation is deliverable."
fi
rm -f "$FLAG" "$FLAG.m" "$FLAG.e" "$FLAG.w" "$FLAG.s" "$GEN" "$BIN"
exit $rc
