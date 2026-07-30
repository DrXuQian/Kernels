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
# GENERATED INCLUDES. A target whose sweep is generated (test_gemv_perf, test_moe_splitk_bench) includes a .inc
# that only exists in the build tree, so without this the front end stops at that include and the file is not
# checked at all. Point it at a directory holding the generated .inc:
#   GEN_INC=/path/to/gemv_units ./syntax_check.sh ../test_gemv_perf.cu
GEN_INC="${GEN_INC:-}"
_gen_flag=""
[ -n "$GEN_INC" ] && _gen_flag="-I$GEN_INC"
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
  # THE FLAGS MATTER MORE THAN THE SCRIPT. Without them the front end never instantiates the collective, so every
  # template-DEPENDENT error in the mainloop is invisible and only parse-time typos are caught:
  #   -arch=sm_80              without a real arch, __hfma2 is undeclared -> an error -> EDG stops instantiating
  #   --expt-relaxed-constexpr nvcc-only restriction on constexpr host fns in device code; clang/hgcc allow it
  #   -DPPU_FORCE_INSTANTIATE  odr-uses device_kernel<GemmKernel>, which pulls in the whole mainloop
  # Verified: with these, a static_assert planted in the 2-plane mainloop fires 32 times; without them, 0.
  #
  # THREE ERROR FORMS, READ OFF THE ACTUAL OUTPUT. The pattern was `: error`, so a source that could not even find its
# headers was reported CLEAN -- which happened to test_lowbit_moe_bench.cu the moment it started including a GENERATED
# file that is not on this gate's -I. The forms are:
#   path(line): error: ...                     EDG, the normal one
#   path(line): catastrophic error: ...        EDG, cannot open source file
#   path:line:col: fatal error: ...            the PREPROCESSOR, gcc-style -- this is the one a missing include gives
# I fixed this once by adding `catastrophic error` from memory and it still passed, because the actual line was the
# third form. All three share `: <kind>: `, so that is what is matched.
#
# Two signatures are dropped as ENVIRONMENTAL, not filtered by pattern-guessing: cute::_ and cute::product report
  # "undefined in device code" because CUTE_INLINE_CONSTANT resolves to `static constexpr` here while the box takes
  # the `static const __device__` branch. They cannot mask a real error -- a real one has a different message.
  sig=$(nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__ -DPPU_FORCE_INSTANTIATE=1 $EXTRA_DEFS $_gen_flag \
        -I"$STUB" -I"$ACT/include" -I"$ACT/tools/util/include" -I"$SRC" \
        -cuda -o /dev/null -x cu "$f" -Wno-deprecated-gpu-targets 2>&1 \
        | grep -E ": (error|fatal error|catastrophic error):" | grep -v 'identifier "cute::_" is undefined in device code' \
                         | grep -v 'identifier "cute::product" is undefined in device code' \
        | sed -E 's#^.*/([^/]+)#\1#; s#\(([0-9]+)\)#()#' | sort | uniq -c \
        | sed -E 's/^ +//' | sort)
  bl="$BLDIR/$base.txt"
  # A FATAL ERROR MEANS NOTHING WAS CHECKED, so it must never become "accepted noise". The preprocessor stops at
  # the first one -- a missing generated .inc gives exactly this -- and baselining it turns the check into a
  # no-op that reports success forever. This happened: `--baseline` on a target whose sweep is generated recorded
  # `fatal error: moe_splitk_units.inc: No such file or directory` as one accepted line, so the file was never
  # parsed and the script still said "baseline recorded". Fixing the include (GEN_INC) is the answer, not
  # accepting the line.
  if printf '%s\n' "$sig" | grep -q ": fatal error:\|: catastrophic error:"; then
    echo "$base: REFUSING to proceed -- a fatal/catastrophic error means the front end STOPPED and the file was"
    echo "        not checked. Fix the include (see GEN_INC) rather than baselining it:"
    printf '%s\n' "$sig" | grep ": fatal error:\|: catastrophic error:" | head -4 | sed 's/^/          /'
    rc=1; continue
  fi
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
done
exit $rc
