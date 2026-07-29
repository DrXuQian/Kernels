#!/usr/bin/env bash
# Gate for the MoE sweep unit GENERATOR. It SLICES ../CMakeLists.txt at run time rather than holding a copy, because the
# copy went stale within an hour: the committed .cmake still had the old dispatcher shape and reported OK while the real
# generator had moved on to per-format Best slots. A gate that can disagree with the thing it gates is not a gate.
#
# WHY IT MUST BE CMAKE. The bug it was written for could only be caught by CMake: `;` IS the list separator, so a WarpN
# list written "32;64" inside a row does not stay inside it -- foreach(IN LISTS) splits the row and the stray fragment has
# one field. A python mirror of the same table cannot see this (python's split(';') has no such semantics); the mirror
# validated the enumeration and passed while the real configure died with six "list index out of range" errors per row.
# The generator also printed "128 generated units" while broken, because 8 malformed iterations x 16 equals the 5 correct
# formats x 16 -- so the count that exists to catch a shrunken enumeration agreed with the right answer for the wrong
# reason.
#
#   ./gen_moe_units_check.sh                 assert the generator is sane (expectations are DERIVED from the axis lists)
#   BAD=1 ./gen_moe_units_check.sh           negative control: put a ';' back, must FATAL
#
# The negative control gets its OWN output directory. The script starts with REMOVE_RECURSE, so running the broken variant
# into the shared dir wipes the good tree and dies partway, and the wreckage reads exactly like a real failure -- I
# diagnosed "q6 is missing WarpN=64" from the corpse of my own negative control once.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CML="$HERE/../CMakeLists.txt"
OUT="$HERE/.moe_units_check${BAD:+_bad}"
GEN="$OUT/gen.cmake"
mkdir -p "$OUT"
{
  echo "set(CMAKE_CURRENT_BINARY_DIR \"$OUT\")"
  echo 'file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/moe_units")'
  # the slice: from the format table to just before the executable that consumes the generated sources
  awk '/^set\(_MOE_FORMATS/{p=1} /^ppu_w4a16_executable\(/{if(p)exit} p' "$CML" \
    | { if [ -n "${BAD:-}" ]; then sed 's/4|2|32,64/4|2|32;64/'; else cat; fi; }
  cat <<'ASSERT'
# DERIVE every expectation from the same lists the slice defines -- writing 128 in here went stale the moment TileN gained
# a third value, and a gate whose expected value is hand-maintained fails for the wrong reason.
list(LENGTH _MOE_TM_LIST _ntm)
list(LENGTH _MOE_TN_LIST _ntn)
list(LENGTH _MOE_WM_LIST _nwm)
list(LENGTH _MOE_FORMATS  _nfmt)
set(_exp 0)
foreach(_row IN LISTS _MOE_FORMATS)
  string(REPLACE "|" ";" _f "${_row}")
  list(GET _f 6 _w)
  string(REPLACE "," ";" _w "${_w}")
  list(LENGTH _w _nwn)
  math(EXPR _exp "${_exp} + ${_nwn} * ${_ntn} * ${_nwm} * ${_ntm}")
endforeach()
file(GLOB _gen "${_MOE_GEN_DIR}/*.cu")
list(LENGTH _gen _ngen)
message(STATUS "generated .cu files on disk: ${_ngen} (derived expectation ${_exp}, generator said ${_MOE_UNIT_N})")
# two independent comparisons: vs the derived product (catches a loop that skipped rows) and vs the generator's own list
# length (catches a file that failed to write).
if(NOT _ngen EQUAL _exp)
  message(FATAL_ERROR "expected ${_exp} generated units from the axis lists, got ${_ngen} on disk")
endif()
if(NOT _ngen EQUAL _MOE_UNIT_N)
  message(FATAL_ERROR "generator listed ${_MOE_UNIT_N} sources but ${_ngen} are on disk")
endif()
# DERIVED, like the total. This said 8 (4 TileM x 2 WarpM) and went stale the moment TileM gained 16 and WarpM gained 16 --
# the same hand-maintained-expectation failure the total count already had, one check further down.
math(EXPR _per_slice "${_ntm} * ${_nwm}")
foreach(_pat q6_tn64_wn32 q6_tn64_wn64 i4_tn128_wn64 i2_tn64_wn32)
  file(GLOB _hit "${_MOE_GEN_DIR}/moe_unit_${_pat}_*.cu")
  list(LENGTH _hit _nh)
  if(NOT _nh EQUAL _per_slice)
    message(FATAL_ERROR "moe_unit_${_pat}_* : expected ${_per_slice} (TileM x WarpM), got ${_nh}")
  endif()
endforeach()
file(GLOB _stray "${_MOE_GEN_DIR}/moe_unit_64_*.cu")
list(LENGTH _stray _nstray)
if(NOT _nstray EQUAL 0)
  message(FATAL_ERROR "the semicolon bug is back: ${_nstray} units named after a stray WarpN field")
endif()
# the dispatcher must declare and call every unit, and name one Best slot per FORMAT
file(READ "${_MOE_GEN_DIR}/moe_bench_units.inc" _d)
string(REGEX MATCHALL "\nvoid moe_unit_" _dc "${_d}")
list(LENGTH _dc _ndc)
string(REGEX MATCHALL "\n  moe_unit_" _cc "${_d}")
list(LENGTH _cc _ncc)
if(NOT _ndc EQUAL _exp OR NOT _ncc EQUAL _exp)
  message(FATAL_ERROR "dispatcher has ${_ndc} declarations and ${_ncc} calls, expected ${_exp} of each")
endif()
foreach(_need "MOE_UNIT_COUNT ${_exp}" "MOE_FMT_COUNT ${_nfmt}" "moe_fmt_names")
  string(FIND "${_d}" "${_need}" _f)
  if(_f EQUAL -1)
    message(FATAL_ERROR "dispatcher is missing '${_need}' -- main will not compile against it")
  endif()
endforeach()
message(STATUS "OK: ${_exp} units, both WarpN for q6/i2/i4, no stray-format units, dispatcher ${_exp}/${_exp} with ${_nfmt} format slots")
ASSERT
} > "$GEN"
cmake -P "$GEN"
