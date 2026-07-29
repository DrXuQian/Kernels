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
#   ./gen_moe_units_check.sh                 assert the generator is sane
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
file(GLOB _gen "${_MOE_GEN_DIR}/*.cu")
list(LENGTH _gen _ngen)
message(STATUS "generated .cu files on disk: ${_ngen}")
if(NOT _ngen EQUAL 128)
  message(FATAL_ERROR "expected 128 generated units, got ${_ngen}")
endif()
foreach(_pat q6_tn64_wn32 q6_tn64_wn64 i4_tn128_wn64 i2_tn64_wn32)
  file(GLOB _hit "${_MOE_GEN_DIR}/moe_unit_${_pat}_*.cu")
  list(LENGTH _hit _nh)
  if(NOT _nh EQUAL 8)
    message(FATAL_ERROR "moe_unit_${_pat}_* : expected 8 (TileM x WarpM), got ${_nh}")
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
if(NOT _ndc EQUAL 128 OR NOT _ncc EQUAL 128)
  message(FATAL_ERROR "dispatcher has ${_ndc} declarations and ${_ncc} calls, expected 128 of each")
endif()
foreach(_need "MOE_UNIT_COUNT 128" "MOE_FMT_COUNT 5" "moe_fmt_names")
  string(FIND "${_d}" "${_need}" _f)
  if(_f EQUAL -1)
    message(FATAL_ERROR "dispatcher is missing '${_need}' -- main will not compile against it")
  endif()
endforeach()
message(STATUS "OK: 128 units, both WarpN for q6/i2/i4, no stray-format units, dispatcher 128/128 with 5 format slots")
ASSERT
} > "$GEN"
cmake -P "$GEN"
