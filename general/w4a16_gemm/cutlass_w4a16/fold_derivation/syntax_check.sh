#!/usr/bin/env bash
# Run nvcc's FRONT END over a source locally, so a typo or a bad template instantiation does not need a round trip to
# ppu001 to be found. `nvcc -cuda` stops after the front end and inline PPU asm is an opaque string at that stage, so
# the file parses without an assembler for the target. -D__HGGCCC__ is required or CUTLASS_DEVICE degrades to host
# `inline` and every __syncthreads lands in host code.
#
# WHY BASELINE-DIFF AND NOT PATTERN FILTERING. The actlize headers emit a fixed set of complaints the real hgcc does
# not -- missing PPU intrinsics, host/device qualifiers, types the stubs do not model. Two earlier designs both failed:
#   * filtering by FILE ("only count errors attributed to the source") is blind to template instantiation failures,
#     because those report their `error:` against the library header and name the source only in the
#     "note: ... requested here" chain. That is how run_cfg<...,16,128,256,32,32,2> -- TM=16 with WM=32, so
#     warpOnM = 0 and the collective builder returns `int` -- reached the box while this script said "parses clean".
#   * filtering by PATTERN needs a list so loose it hides real errors, since the noise is large and generic
#     ("type name is not allowed", "expected a type specifier").
# So: record the noise ONCE per file into a baseline, and fail only on error signatures that are NEW.
#
#   ./syntax_check.sh --baseline <files...>   record/refresh the accepted noise
#   ./syntax_check.sh <files...>              fail on anything not in the baseline
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
STUB="$(cd "$(dirname "$0")/stub_inc" && pwd)"
ACT="$(cd "$(dirname "$0")/../../../../third_party/actlize" && pwd)"
BLDIR="$(cd "$(dirname "$0")" && pwd)/syntax_baseline"
mkdir -p "$BLDIR"
RECORD=0
if [ "${1:-}" = "--baseline" ]; then RECORD=1; shift; fi
# EXTRA_DEFS lets a FLAG-ON variant get its own baseline, so a build that only breaks with a macro set is caught
# locally instead of on the box. Two box round trips were burned on errors this would have shown:
#   EXTRA_DEFS=-DPPU_B_CHUNK=1 ./syntax_check.sh --baseline <file>   then   EXTRA_DEFS=... ./syntax_check.sh <file>
EXTRA_DEFS="${EXTRA_DEFS:-}"
# NOTE the baseline file is deliberately NOT keyed on EXTRA_DEFS: a flag-on run is diffed against the flag-OFF
# baseline, so anything that appears only with the macro set shows up as NEW. Keying it would have let me baseline my
# own bugs.
FILES=${*:-"$SRC/test_fold_int2.cu"}
rc=0
for f in $FILES; do
  base=$(basename "$f")
  # signature = file + the MESSAGE, with the LINE NUMBER STRIPPED. Line numbers made the gate false-positive on
  # every edit that shifted the noise (adding 12 lines to test_fold_int2 reported 5 "NEW ERRORS" that were the same
  # 5 known ones), and a gate that cries wolf on every edit is a gate that stops being read -- which is how a real
  # error gets through. Dropping the line number costs the ability to distinguish two identical messages at
  # different lines; the count guard below covers that.
  sig=$(nvcc -std=c++17 -D__HGGCCC__ $EXTRA_DEFS -I"$STUB" -I"$ACT/include" -I"$ACT/tools/util/include" -I"$SRC" \
        -cuda -o /dev/null -x cu "$f" -Wno-deprecated-gpu-targets 2>&1 \
        | grep ": error" | sed -E 's#^.*/([^/]+)#\1#; s#\(([0-9]+)\)#()#' | sort | uniq -c \
        | sed -E 's/^ +//' | sort)
  bl="$BLDIR/$base.txt"
  if [ "$RECORD" = 1 ]; then
    printf '%s\n' "$sig" > "$bl"
    echo "$base: baseline recorded ($(printf '%s\n' "$sig" | grep -c . ) accepted noise lines)"
    continue
  fi
  if [ ! -f "$bl" ]; then echo "$base: NO BASELINE -- run --baseline once, then review it"; rc=1; continue; fi
  new=$(comm -13 "$bl" <(printf '%s\n' "$sig"))
  if [ -n "$new" ]; then
    echo "$base: NEW ERRORS (not in baseline)"; printf '%s\n' "$new" | head -12; rc=1
  else
    echo "$base: clean ($(grep -c . "$bl") known-noise lines, 0 new)"
  fi
  # THE BLIND SPOT, STATED OUT LOUD. The stubs make ppu_mma_builder.inl fail ("expected a type specifier"), so the
  # CollectiveMma it would have produced is never instantiated locally -- and every error DOWNSTREAM of it is
  # invisible here. A folded 2-plane gB2 cut with the unfolded TileShape sailed through as "clean" and then failed on
  # the box with cute/algorithm/copy.hpp's `size<1>(src) == size<1>(dst)`. "clean" means "no NEW front-end error",
  # NOT "the collective type-checks". Layout-consistency questions belong in an l4x harness that builds the types
  # directly; this gate cannot answer them.
  if grep -q "ppu_mma_builder.inl" "$bl" 2>/dev/null; then
    echo "  NOTE: the builder fails under the stubs, so nothing downstream of CollectiveMma was instantiated."
    echo "        Mainloop layout/partition mismatches CANNOT be caught here -- use a fold_derivation harness."
  fi
done
exit $rc
