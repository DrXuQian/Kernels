# CMakeLists for the actlize-side W4A16 comparison bench. This is NOT built by the marlin Makefile -- actlize
# uses a different toolchain (hgcc device compiler + the PPU SDK's hggc runtime, driven by
# third_party/actlize/cmake/PPUToolchain.cmake), not the bare nvcc our marlin kernels build with.
#
# build.sh overlays this directory into actlize's examples/ tree as a new example and builds just this target,
# so we ride actlize's proven example build exactly (see build.sh for why the overlay).
#
# The stock actlize example helper macro is cutlass_example_add_executable, available once this dir is an
# add_subdirectory target of actlize/examples/CMakeLists.txt.

cutlass_example_add_executable(
  bench_cutlass_w4a16
  bench_cutlass_w4a16.cu
)

target_compile_options(bench_cutlass_w4a16 PRIVATE -w)
