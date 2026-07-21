# CMakeLists for the actlize-side W4A16 comparison bench. This is NOT built by the marlin Makefile -- actlize
# uses a different toolchain (hgcc device compiler + the PPU SDK's hggc runtime, driven by
# third_party/actlize/cmake/PPUToolchain.cmake), not the bare nvcc our marlin kernels build with.
#
# build.sh overlays this directory into actlize's examples/ tree as a new example and builds just this target,
# so we ride actlize's proven example build exactly (see build.sh for why the overlay).

# Tuning knobs (defaults match the stock example: 32x32 tile, 16x16 warp, 3 stages). build.sh forwards these
# from the environment as -D cache vars. They must reach BOTH compiles: the host g++ (TileShape is used in
# host code for strides/args) via target_compile_definitions, and the device hgcc via DEV_COMPILE_FLAGS
# (hgcc is driven by a custom command that does not see target_compile_definitions).
set(TILE_M  "32" CACHE STRING "threadblock tile M")
set(TILE_N  "32" CACHE STRING "threadblock tile N")
set(WARP_M  "16" CACHE STRING "warp tile M")
set(WARP_N  "16" CACHE STRING "warp tile N")
set(STAGES  "3"  CACHE STRING "pipeline stages")

set(_W4A16_DEFS
  -DTILE_M=${TILE_M} -DTILE_N=${TILE_N} -DWARP_M=${WARP_M} -DWARP_N=${WARP_N} -DSTAGES=${STAGES})

cutlass_example_add_executable(
  bench_cutlass_w4a16
  bench_cutlass_w4a16.cu
  DEV_COMPILE_FLAGS ${_W4A16_DEFS}
)

target_compile_definitions(bench_cutlass_w4a16 PRIVATE
  TILE_M=${TILE_M} TILE_N=${TILE_N} WARP_M=${WARP_M} WARP_N=${WARP_N} STAGES=${STAGES})
target_compile_options(bench_cutlass_w4a16 PRIVATE -w)
