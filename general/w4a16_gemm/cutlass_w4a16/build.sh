#!/usr/bin/env bash
# Build the actlize (PPU cutlass3) W4A16 gs=128 comparison bench.
#
# WHY THIS IS NOT IN THE MARLIN Makefile. actlize does not build with the bare `nvcc` (no -arch) that the
# marlin kernels use. Its toolchain is the PPU SDK: hgcc as the device compiler (-arch=ppu_10), g++ as host,
# linking libhggc_wrapper / libhggcrt1 / libhggc, with -DSWITCH_TO_HGGCRT -DCUTLASS_USE_PACKED_TUPLE=1. All of
# that is set up by third_party/actlize/cmake/PPUToolchain.cmake, so we drive actlize's own cmake.
#
# WHY THE OVERLAY. actlize's examples are an explicit foreach() list in examples/CMakeLists.txt that
# add_subdirectory's each one; there is no out-of-tree example hook. The least-invasive way to build our .cu
# through the *proven* example machinery is to drop it in as a new example dir and append it to that list.
# We do it as untracked files + a restorable one-line edit, so the submodule's tracked content is unchanged
# (the script restores examples/CMakeLists.txt at the end).
#
# Prereq: PPU_SDK=<path with bin/hgcc> (or PPU_HOME). ppu001 == ppu0010 == ACOMPUTE 10000.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTLIZE="$(cd "$HERE/../../../third_party/actlize" && pwd)"
EX_NAME="99_kernels_w4a16_compare"
EX_DIR="$ACTLIZE/examples/$EX_NAME"
EX_LIST="$ACTLIZE/examples/CMakeLists.txt"
ARCH="${PPU_ARCHS:-ppu0010}"
# Default to this box's SDK location; override with PPU_SDK=<path> (or PPU_HOME) if it moves.
PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-/sim/eec/shared/junfu.qx/PPU_SDK}}"

if [ ! -x "$PPU_SDK_ROOT/bin/hgcc" ]; then
  echo "ERROR: hgcc not found at $PPU_SDK_ROOT/bin/hgcc. Set PPU_SDK=<path> and re-run." >&2
  exit 1
fi
export PATH="$PPU_SDK_ROOT/bin:$PATH"

cleanup() {
  # Restore ONLY the example-list registration + remove the overlay. The actlize W4A16 changes now live as real
  # commits on the DrXuQian/actlize fork (submodule branch ppu-w4a16-dev), NOT as build-time patches, so we must
  # NOT `git checkout --` the include/ files here (that would wipe uncommitted collective WIP during iteration).
  git -C "$ACTLIZE" checkout -- examples/CMakeLists.txt 2>/dev/null || true
  rm -rf "$EX_DIR"
}
trap cleanup EXIT
echo "[build.sh] CUTLASS_PPU_ARCHS=$ARCH"

# NOTE: the former MoE/gs32 *.patch files are now baked into the submodule (fork ppu-w4a16-dev). No patch step.

# CHEAP LOCAL CHECKS FIRST, because the expensive failures here are configure-time and box-only. Three target
# registrations once named a helper that does not exist; nothing caught it until cmake ran on the box, which
# costs a full pull-and-build to discover a typo.
if [ -x "$HERE/fold_derivation/cmake_calls_check.sh" ]; then
  "$HERE/fold_derivation/cmake_calls_check.sh" || exit 1
fi
# NVIDIA-only spellings in box-built sources. The syntax check cannot see these: local nvcc has cuda_fp16.h
# whether or not -D__HGGCCC__ is passed, so a wrong-platform include parses clean and only hgcc disagrees.
if [ -x "$HERE/fold_derivation/ppu_portability_check.py" ]; then
  "$HERE/fold_derivation/ppu_portability_check.py" || exit 1
fi
# The unit generators, run against the axis lists rather than a written-down count. A malformed row once
# produced the RIGHT unit count out of the wrong iterations, so the configure log looked correct.
if [ -x "$HERE/fold_derivation/gen_gemv_units_check.sh" ]; then
  "$HERE/fold_derivation/gen_gemv_units_check.sh" || exit 1
fi

# --- overlay our example into the actlize example tree ---
mkdir -p "$EX_DIR"
# nullglob so patterns that match nothing (e.g. no *.cpp right now) vanish instead of aborting under set -e.
# *.inc is in the list because the MoE sweep's generated units all #include moe_bench_unit.inc, and this glob is an
# EXTENSION WHITELIST: leaving it out did not fail here, it failed 100+ lines into hgcc as
# `fatal error: moe_bench_unit.inc: No such file or directory` repeated once per generated unit.
shopt -s nullglob
_overlay_files=("$HERE"/*.cu "$HERE"/*.cpp "$HERE"/*.cuh "$HERE"/*.hpp "$HERE"/*.h "$HERE"/*.inc "$HERE/CMakeLists.txt")
shopt -u nullglob
cp "${_overlay_files[@]}" "$EX_DIR/"

# SOURCE SUBDIRECTORIES that must be overlaid whole. This list is separate from the extension whitelist above
# because most tracked subdirectories (fold_derivation/, real_weight/, low_bit/) are local harnesses that must
# NOT reach the box -- but a library split across a subdirectory has to. The completeness check below covers
# these paths instead of skipping them; before this list existed it skipped ALL subdirectories, which meant a
# library moved into one would be dropped silently by the very check written to catch dropped source.
_overlay_dirs=(gemv_lowbit)
for _d in "${_overlay_dirs[@]}"; do
  [ -d "$HERE/$_d" ] || continue
  mkdir -p "$EX_DIR/$_d"
  shopt -s nullglob
  _sub=("$HERE/$_d"/*.cu "$HERE/$_d"/*.cpp "$HERE/$_d"/*.cuh "$HERE/$_d"/*.hpp "$HERE/$_d"/*.h "$HERE/$_d"/*.inc)
  shopt -u nullglob
  [ ${#_sub[@]} -gt 0 ] && cp "${_sub[@]}" "$EX_DIR/$_d/"
done

# ...and then ASSERT THE WHITELIST IS COMPLETE. The criterion is GIT TRACKING, and the two earlier attempts show why it
# has to be something this specific:
#
#   * scanning the #includes of the overlaid files could not catch the bug it was written for -- the file that includes
#     moe_bench_unit.inc is a GENERATED unit, created later by cmake in the build tree, so it is not in the scan set at
#     overlay time. It passed on the real tree while the real build was broken.
#   * "every regular file here must be copied unless its extension is ignored" then failed the build on this box for
#     *.acurep -- acu reports, an untracked artifact of previous runs that does not exist in a fresh checkout, so the
#     ignore list could not have been written correctly from a clean tree.
#
# Tracked-ness is the right criterion because the box builds from COMMITTED state: what the build can possibly need is
# exactly what is committed. Artifacts (acu reports, logs, binaries, dumped weights) are untracked and skip themselves, and
# a newly added .def or .tpp is tracked the moment it is `git add`ed, which is also the moment it could reach the box.
_ignored='sh|md|py|log|bin|json|patch|pyc|txt'
_missing=""
if _tracked=$(git -C "$HERE" ls-files . 2>/dev/null) && [ -n "$_tracked" ]; then
  while IFS= read -r _b; do
    # Subdirectories are skipped UNLESS they are in _overlay_dirs, in which case their files are checked like
    # any other -- see the comment on that list.
    case "$_b" in
      */*) _dir="${_b%%/*}"; _keep=0
           for _od in "${_overlay_dirs[@]}"; do [ "$_dir" = "$_od" ] && _keep=1; done
           [ "$_keep" = 1 ] || continue ;;
    esac
    echo "$_b" | grep -qE "\.($_ignored)\$" && continue
    [ -f "$EX_DIR/$_b" ] || _missing="$_missing $_b"
  done <<< "$_tracked"
else
  echo "  NOTE: not a git checkout, skipping the overlay completeness check"
fi
if [ -n "$_missing" ]; then
  echo "  ERROR: the overlay is an extension whitelist and it dropped tracked source:$_missing"
  echo "         add the extension to _overlay_files above, or to _ignored if it is genuinely not needed to build."
  exit 1
fi

# AND PROVE THE OVERLAY IS THE CHECKOUT, not a survivor of an earlier run. cleanup() rm -rf's $EX_DIR on exit, so a stale
# copy should be impossible -- but a build failed with a macro expansion from the PRE-fix header while the checkout had the
# fixed one, and from outside the box there was no way to tell "built an older commit" from "overlay was stale". cmp is
# cheap and turns that into a one-line answer.
for _f in "$EX_DIR"/*; do
  [ -f "$_f" ] || continue
  _b="$(basename "$_f")"
  if [ -f "$HERE/$_b" ] && ! cmp -s "$_f" "$HERE/$_b"; then
    echo "  ERROR: the overlaid $_b differs from the checked-out one -- the build would not compile your tree."
    exit 1
  fi
done
for _d in "${_overlay_dirs[@]}"; do
  [ -d "$EX_DIR/$_d" ] || continue
  for _f in "$EX_DIR/$_d"/*; do
    [ -f "$_f" ] || continue
    _b="$_d/$(basename "$_f")"
    if [ -f "$HERE/$_b" ] && ! cmp -s "$_f" "$HERE/$_b"; then
      echo "  ERROR: the overlaid $_b differs from the checked-out one -- the build would not compile your tree."
      exit 1
    fi
  done
done

# register it in the foreach list (idempotent: only if absent)
if ! grep -q "$EX_NAME" "$EX_LIST"; then
  # insert just before the closing paren of the foreach(EXAMPLE ... ) block that ends with 16_ppu_mixed_dtype_gemm
  sed -i "s|^\( *16_ppu_mixed_dtype_gemm\)\$|\1\n  $EX_NAME|" "$EX_LIST"
fi
grep -q "$EX_NAME" "$EX_LIST" || { echo "ERROR: failed to register example in $EX_LIST" >&2; exit 1; }

# --- tile/warp/stages tuning: forward from the environment (defaults match the stock example) ---
TILE_M="${TILE_M:-32}"; TILE_N="${TILE_N:-32}"; WARP_M="${WARP_M:-16}"; WARP_N="${WARP_N:-16}"; STAGES="${STAGES:-3}"
QUANT="${QUANT:-int4}"   # int4 (default) or uint2 -> bench_cutlass_w4a16's QuantType (W4A16 vs W2A16 perf)
TSK="${TSK:-}"           # TileShapeK override (empty = per-quant default: int2->128, int4->64). Set to force, e.g. TSK=128
echo "[build.sh] TILE=${TILE_M}x${TILE_N} WARP=${WARP_M}x${WARP_N} STAGES=${STAGES} QUANT=${QUANT} TSK=${TSK}"

# --- configure & build just our target ---
BUILD="$ACTLIZE/build_w4a16_compare"
rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
# FORWARD THE SWEEP AXIS KNOBS. They were added to CMakeLists.txt and then not wired through here, so narrowing a sweep was
# impossible from build.sh -- the knob existed and could not be reached, which is worse than no knob because it reads as
# available. Any MOE_* variable in the environment is passed through as a cache var.
_MOE_VARS=()
for _v in MOE_TM_LIST MOE_TN_LIST MOE_WM_LIST MOE_FORMATS MOE_CORES; do
  if [ -n "${!_v:-}" ]; then _MOE_VARS+=("-D$_v=${!_v}"); echo "[build.sh] $_v=${!_v}"; fi
done
cmake .. -DPPU_SDK_ROOT="$PPU_SDK_ROOT" -DCUTLASS_PPU_ARCHS="$ARCH" \
  -DTILE_M="$TILE_M" -DTILE_N="$TILE_N" -DWARP_M="$WARP_M" -DWARP_N="$WARP_N" -DSTAGES="$STAGES" -DBENCH_QUANT="$QUANT" -DTSK="$TSK" \
  -DPPU_EXTRA_DEFS="${PPU_DEFS:-}" "${_MOE_VARS[@]+"${_MOE_VARS[@]}"}" \
  >cmake.log 2>&1 || { tail -40 cmake.log; exit 1; }
# cmake's stdout is redirected above, so its message(STATUS ...) never reaches the terminal. Surface the extra
# defines here instead -- telling someone to "check that the line appeared" when it cannot appear is worse than
# not printing it at all.
if [ -n "${PPU_DEFS:-}" ]; then
  echo "PPU_DEFS applied: $PPU_DEFS"
  grep -F "PPU_EXTRA_DEFS ->" cmake.log || echo "  WARNING: cmake did not report PPU_EXTRA_DEFS -- the defines did NOT reach the build"
fi
TARGET="${TARGET:-bench_cutlass_w4a16}"

# #13: NO SHIPPING TARGET MAY REACH THE LEGACY PACKERS. fold_derivation/legacy_pipeline.hpp still holds
# nfold_regroup_gmem and nfold_place_bits_int1_tk64, on purpose -- they are the INDEPENDENT reference the l58/l61/l64
# gates diff xplane::place_derived against, and deleting them would turn those gates into "the derived walk equals the
# derived walk". But nfold_regroup_gmem moves whole uint32 words, so one word carries one logical column, so it is
# correct only at warp-N extent 32 -- and w64x32 is worth +7 to +9 points, i.e. exactly where the tuning is going. A
# harness that picked it up would miscompute silently.
#
# The gates are host-only files under fold_derivation/ that no CMake target builds, so today nothing shipping can reach
# them. This makes that structural instead of merely true: the header can only be reached by naming its path, so grep
# for the path in the sources CMake actually compiles.
#
# TWO THINGS THIS GREP GOT WRONG ON THE FIRST WRITING, both from stating what grep emits instead of reading it:
#   * `grep -rln <dir>` prints `fold_derivation/l11...` with NO leading slash, so `grep -v "/fold_derivation/"` matched
#     nothing and every gate came back as a violation. --exclude-dir is the option that actually does this.
#   * matching the bare filename also matches the ~30 lines of COMMENTARY that explain why the packers are quarantined,
#     so the two files documenting the rule were reported as breaking it. Anchor on the #include.
_leg=$(grep -rln '^[[:space:]]*#[[:space:]]*include.*legacy_pipeline\.hpp' \
         --include=*.cu --include=*.cuh --include=*.hpp \
         --exclude-dir=fold_derivation "$(dirname "$0")" 2>/dev/null || true)
if [ -n "$_leg" ]; then
  echo "  ERROR: a CMake-built source includes fold_derivation/legacy_pipeline.hpp:"
  printf '           %s\n' $_leg
  echo "         nfold_regroup_gmem is correct only at WN=32. Use xplane::place_derived (see #13); the legacy"
  echo "         packers exist ONLY as the gates' independent reference."
  exit 1
fi

# JOBS overrides -j. Needed since the MoE bench became 9 sources: each hgcc instantiates ~42 kernels and wants GBs, and
# running all of them at once OOM-killed the front end locally -- an OOM kill looks like a compile failure with an empty
# log, which is the least informative way for a build to fail. `JOBS=4 ./build.sh` when memory is tight.
make -j"${JOBS:-$(nproc)}" "$TARGET" 2>&1 | tee make.log

# CMAKE RECEIVING THE DEFINES IS NOT THE SAME AS THIS TARGET GETTING THEM, and the difference is invisible in a perf
# number. The defines used to be attached to three targets by hand, so
#   PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_q3_bconcat_bench ./build.sh
# configured cleanly, printed "PPU_DEFS applied", printed no warning, and produced a binary WITHOUT the macro. The run
# that followed compared a binary against itself and read as "the change does nothing".
#
# LOOK IN build.make, NOT make.log. The first version of this check grepped make.log and cried wolf on every build: the
# device compiles are add_custom_command with a COMMENT, so make.log holds "[100%] [hgcc] foo.cu" and never a compile
# line, meaning the grep could not succeed even when the flag WAS present. cmake's generated build.make does carry the
# full command, so that is what gets checked -- and it is checked for THIS target's directory, which is the claim that
# matters ("cmake received it" is a different and weaker claim, already covered above).
if [ -n "${PPU_DEFS:-}" ]; then
  _bm=$(find . -path "*${TARGET}.dir/build.make" | head -1)
  if [ -z "$_bm" ]; then
    echo "  WARNING: no build.make found for $TARGET -- cannot verify the defines reached it."
  else
    for _d in $PPU_DEFS; do
      if grep -qF -- "-D$_d" "$_bm"; then
        echo "PPU_DEFS verified on $TARGET's compile command: -D$_d"
      else
        echo "  WARNING: -D$_d is NOT on $TARGET's compile command -- THIS BUILD DOES NOT HAVE IT."
        echo "           Any A/B against it is a binary compared with itself. (checked $_bm)"
      fi
    done
  fi
fi

BIN="$(find "$BUILD" -name "$TARGET" -type f -perm -u+x | head -1)"
echo
echo "built: $BIN"
# Per-target run hint. The old line advertised --m=/--mode= for EVERY target, but only the compare example
# parses that shape: on a positional-arg target each flag becomes atoi("--m=2048") == 0, so copying the hint
# silently selects L=0 or Mb=0 and the test passes vacuously. One such copy already cost a box round.
case "$TARGET" in
  test_moe_grouped_verify)
    echo "run:   $BIN [L] [Mb] [ragged?] [gs]      # POSITIONAL, no --flags"
    echo "       $BIN 8 1                          # Mmax==1, required by PPU_A_CUBE_H=1"
    echo "       MOEG_SMEM=1 MOEG_DUMP=/tmp/d.bin $BIN 8 1     # then MOEG_CHECK=/tmp/d.bin on the other build" ;;
  test_moe_splitk_bench|test_lowbit_moe_bench)
    echo "run:   $BIN <L> <rows> <N> <K> <gs> <mode>   # POSITIONAL; mode 3 => rows is top-k"
    echo "       $BIN 64 8 2048 2048 32 3"
    echo "       SPLITK_CFG='16x128:256 w16x16 s2' SPLITK_S=1 $BIN 64 8 2048 2048 32 3   # select one acu row" ;;
  test_gemv_perf)
    echo "run:   $BIN [shape]                      # GEMV_FMT=/GEMV_CFG= select rows" ;;
  *)
    echo "run:   $BIN --m=2048 --n=4096 --k=4096 --g=128 --mode=1 --iterations=100" ;;
esac
