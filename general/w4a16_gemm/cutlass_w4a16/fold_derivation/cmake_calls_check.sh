#!/usr/bin/env bash
# Every PROJECT-LOCAL cmake helper called in CMakeLists.txt must be defined in CMakeLists.txt.
#
# WHY THIS EXISTS. Three target registrations were written against `cutlass_w4a16_example`, a name that does
# not exist -- the real helper is `ppu_w4a16_executable`. Nothing local catches that: it is not a C++ error, the
# syntax check never reads CMakeLists.txt, and cmake only complains at CONFIGURE time, which on this project
# means a full pull-and-build on the box before the mistake is visible. The name was written down from memory
# instead of read off the file, which is this codebase's most expensive recurring failure.
#
# The check is deliberately narrow: only calls whose name looks like a project helper (contains w4a16, ppu_,
# example or executable) are required to be defined here. Builtins and cutlass's own helpers are out of scope,
# because an allowlist of those would go stale and start hiding real errors.
set -u
CM="$(cd "$(dirname "$0")/.." && pwd)/CMakeLists.txt"
[ -f "$CM" ] || { echo "no CMakeLists.txt at $CM"; exit 1; }

defined=$(grep -oE '^[[:space:]]*(function|macro)\([[:space:]]*[A-Za-z0-9_]+' "$CM" \
          | grep -oE '[A-Za-z0-9_]+$' | sort -u)
# calls at column 0 (a nested call is inside a helper we already trust)
called=$(grep -oE '^[A-Za-z0-9_]+\(' "$CM" | tr -d '(' | sort -u)

bad=""
for c in $called; do
  case "$c" in
    *w4a16*|ppu_*|*example*|*executable*) ;;
    *) continue ;;
  esac
  echo "$defined" | grep -qx "$c" || bad="$bad $c"
done

# The check must be able to FAIL, or it proves nothing: confirm it sees the real helper as defined.
echo "$defined" | grep -qx "ppu_w4a16_executable" \
  || { echo "  [FAIL] cmake_calls_check: ppu_w4a16_executable is not defined -- the check itself is looking in the wrong place"; exit 1; }

if [ -n "$bad" ]; then
  echo "  [FAIL] CMakeLists.txt calls project helpers that are not defined in it:$bad"
  echo "         defined here:" $defined
  exit 1
fi
echo "  [ok]   cmake_calls_check: every project helper called is defined ($(echo $defined | wc -w) helper(s))"
