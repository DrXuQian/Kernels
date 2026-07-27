#!/usr/bin/env bash
# Run nvcc's FRONT END over the harness sources locally, so a typo does not need a round trip to ppu001 to be found.
#
# WHY. `return bad == 0 ? 0 : 1;` in a block above where `bad` is declared shipped to the box and cost a build
# cycle. It is an undeclared-identifier error in host code -- the most local kind of error there is.
#
# HOW. `nvcc -cuda` runs the front end only; inline PPU asm is an opaque string at this stage, so it parses.
# CUTLASS_DEVICE degrades to host `inline` unless __HGGCCC__ is defined, which would put __syncthreads in host
# code, so define it. The actlize headers then emit a fixed set of host/device-qualifier complaints that the real
# hgcc does not -- those are library noise and are ignored. Only errors attributed to the SOURCE FILES count, and
# one stub artefact is filtered: CUTLASS_PPU_CHECK's std::cerr << becomes ambiguous against the stub runtime.
#
# This is a SYNTAX gate, not a build. It cannot tell you the kernel is correct; it tells you the file parses.
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
STUB="$(cd "$(dirname "$0")/stub_inc" && pwd)"
ACT="$(cd "$(dirname "$0")/../../../../third_party/actlize" && pwd)"
FILES=${*:-"$SRC/test_fold_int2.cu"}
rc=0
for f in $FILES; do
  base=$(basename "$f")
  out=$(nvcc -std=c++17 -D__HGGCCC__ -I"$STUB" -I"$ACT/include" -I"$ACT/tools/util/include" -I"$SRC" \
        -cuda -o /dev/null -x cu "$f" -Wno-deprecated-gpu-targets 2>&1 \
        | grep -E "(^|/)${base}\([0-9]+\): error" \
        | grep -v 'more than one operator "<<" matches')   # stub artefact, see header
  if [ -n "$out" ]; then echo "$base: REAL ERRORS"; echo "$out" | head -10; rc=1
  else echo "$base: parses clean (library-header noise and the << stub artefact ignored)"; fi
done
exit $rc
