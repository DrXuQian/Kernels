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
# This box's hgcc takes `--gpu-architecture=ppu001|ppu0015|all` (verified via `hgcc --help`); ppu001 is our
# chip. The shipped v1.0.0 CMake instead emits the OLD naming `-arch=ppu_10`, which this hgcc does not accept
# and silently mis-targets -- the kernel came out as a non-ppu001 ELF and the runtime aborted with
# "e_machine ... probably a NV binary / Failed to query occupancy". actlize_ppu001.patch makes CMake pass the
# arch name straight through (ppu001 -> -arch=ppu001, library arch 80a). Applied to the submodule before the
# build and reverted after, so the pinned submodule content is unchanged.
ARCH="${PPU_ARCHS:-ppu001}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}"
PATCH="$HERE/actlize_ppu001.patch"

if [ ! -x "$PPU_SDK_ROOT/bin/hgcc" ]; then
  echo "ERROR: hgcc not found at $PPU_SDK_ROOT/bin/hgcc. Set PPU_SDK=<path> and re-run." >&2
  exit 1
fi
export PATH="$PPU_SDK_ROOT/bin:$PATH"

cleanup() {
  # Restore everything we touched so the submodule's pinned content stays clean: the arch patch, the example
  # list, and the overlay dir.
  git -C "$ACTLIZE" checkout -- CMakeLists.txt cmake/PPUToolchain.cmake examples/CMakeLists.txt 2>/dev/null || true
  rm -rf "$EX_DIR"
}
trap cleanup EXIT

# --- retarget the toolchain to this box's ppu001 arch naming (idempotent) ---
if git -C "$ACTLIZE" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "[build.sh] actlize_ppu001.patch already applied"
elif git -C "$ACTLIZE" apply --check "$PATCH" 2>/dev/null; then
  git -C "$ACTLIZE" apply "$PATCH"
  echo "[build.sh] applied actlize_ppu001.patch (ppu_10 -> ppu001)"
else
  echo "ERROR: actlize_ppu001.patch does not apply to the submodule at $ACTLIZE" >&2
  echo "       the pinned submodule may have moved off v1.0.0." >&2
  exit 1
fi
echo "[build.sh] CUTLASS_PPU_ARCHS=$ARCH"

# --- overlay our example into the actlize example tree ---
mkdir -p "$EX_DIR"
cp "$HERE/bench_cutlass_w4a16.cu" "$HERE/unfused_weight_dequantize.hpp" "$HERE/helper.h" "$HERE/CMakeLists.txt" "$EX_DIR/"

# register it in the foreach list (idempotent: only if absent)
if ! grep -q "$EX_NAME" "$EX_LIST"; then
  # insert just before the closing paren of the foreach(EXAMPLE ... ) block that ends with 16_ppu_mixed_dtype_gemm
  sed -i "s|^\( *16_ppu_mixed_dtype_gemm\)\$|\1\n  $EX_NAME|" "$EX_LIST"
fi
grep -q "$EX_NAME" "$EX_LIST" || { echo "ERROR: failed to register example in $EX_LIST" >&2; exit 1; }

# --- configure & build just our target ---
BUILD="$ACTLIZE/build_w4a16_compare"
rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
cmake .. -DPPU_SDK_ROOT="$PPU_SDK_ROOT" -DCUTLASS_PPU_ARCHS="$ARCH" >cmake.log 2>&1 || { tail -40 cmake.log; exit 1; }
make -j"$(nproc)" bench_cutlass_w4a16 2>&1 | tee make.log

BIN="$(find "$BUILD" -name bench_cutlass_w4a16 -type f -perm -u+x | head -1)"
echo
echo "built: $BIN"
echo "run:   $BIN --m=2048 --n=4096 --k=4096 --g=128 --mode=1 --iterations=100"
