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
set(TILE_M  "64" CACHE STRING "threadblock tile M")
set(TILE_N  "64" CACHE STRING "threadblock tile N")
set(WARP_M  "32" CACHE STRING "warp tile M")
set(WARP_N  "32" CACHE STRING "warp tile N")
set(STAGES  "4"  CACHE STRING "pipeline stages")

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

# Official-structured finegrained launcher (fpA_intB_ppu.cuh) compile+run gate. Build with:
#   TARGET=test_fpA_intB_ppu ./build.sh
cutlass_example_add_executable(
  test_fpA_intB_ppu
  test_fpA_intB_ppu.cu
)
target_compile_options(test_fpA_intB_ppu PRIVATE -w)
# The dequant+cublas candidate needs cublas from the CUDA_SDK (host lib). CUDA_SDK is next to the box nvcc
# (which nvcc -> .../PPU_SDK/CUDA_SDK/bin/nvcc). Override with -DCUDA_SDK_ROOT= if elsewhere.
set(CUDA_SDK_ROOT "/usr/local/PPU_SDK/CUDA_SDK" CACHE STRING "CUDA SDK root providing cublas")
target_include_directories(test_fpA_intB_ppu PRIVATE "${CUDA_SDK_ROOT}/include")
target_link_directories(test_fpA_intB_ppu PRIVATE "${CUDA_SDK_ROOT}/lib64" "${CUDA_SDK_ROOT}/lib")
target_link_libraries(test_fpA_intB_ppu cublas)
