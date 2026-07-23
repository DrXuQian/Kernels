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
# Root-cause cross-check is now a RUNTIME flag on this same binary (--xcheck): it runs the grouped kernel (L=1)
# on the verified bench data (block_A / shuffled block_B_buff / block_scale) and compares vs block_ref_D and the
# mixed kernel's block_D. No separate target / no -D (the two-stage PPU build did not propagate -DMOEG_XCHECK to
# host main reliably). Build stock: ./build.sh ; run: <bin> --xcheck --m=256 --n=4096 --k=4096 --g=128 --mode=1.

# Official-structured finegrained launcher (fpA_intB_ppu.cuh) mixed-input sweep. Build with:
#   TARGET=test_fpA_intB_ppu ./build.sh
# dequant+cublas is deliberately NOT here: cublas drags cuda_fp16 whose __half2 clashes with the hggc headers
# hgcc auto-includes, and our fast dequant is cuda-only device code that cannot link with hgcc device .o. It
# lives on the nvcc side instead: ../marlin_ppu dequant_cublas.
cutlass_example_add_executable(
  test_fpA_intB_ppu
  test_fpA_intB_ppu.cu
)
target_compile_options(test_fpA_intB_ppu PRIVATE -w)

# STEP 1 of MoE W4A16: batched (uniform m_per_expert) mixed-input GEMM. Build: TARGET=test_moe_gemm_ppu ./build.sh
cutlass_example_add_executable(
  test_moe_gemm_ppu
  test_moe_gemm_ppu.cu
)
target_compile_options(test_moe_gemm_ppu PRIVATE -w)

# STEP 2a of MoE W4A16: grouped mixed-input via GroupScheduler. Build: TARGET=test_moe_grouped_ppu ./build.sh
cutlass_example_add_executable(
  test_moe_grouped_ppu
  test_moe_grouped_ppu.cu
)
target_compile_options(test_moe_grouped_ppu PRIVATE -w)

# STEP 2b correctness gate: multi-expert/ragged grouping vs grouped-L=1 oracle. TARGET=test_moe_grouped_verify ./build.sh
cutlass_example_add_executable(
  test_moe_grouped_verify
  test_moe_grouped_verify.cu
)
target_compile_options(test_moe_grouped_verify PRIVATE -w)

# ROUTING PROBE: MOEG_PROBE=1 makes the grouped kernel write (expert+1) instead of the GEMM, to decode which
# D-plane each expert's tiles land in. TARGET=test_moe_grouped_probe ./build.sh ; run: MOEG_PROBE=1 <bin>
cutlass_example_add_executable(
  test_moe_grouped_probe
  test_moe_grouped_probe.cu
)
target_compile_options(test_moe_grouped_probe PRIVATE -w)

# DATA-SLICE probe: controlled per-expert data (B/scale/A each varied in turn) to pin WHICH input the collective
# fails to slice by l_coord. TARGET=test_moe_grouped_dataslice ./build.sh ; run: <bin> (MOEG_PROBE unset)
cutlass_example_add_executable(
  test_moe_grouped_dataslice
  test_moe_grouped_dataslice.cu
)
target_compile_options(test_moe_grouped_dataslice PRIVATE -w)

# REAL-WEIGHT gate: loads a .bin (real Qwen3.5 GPTQ / Q4_K expert weights from real_weight/dump_real_weights.py),
# runs the grouped kernel, compares vs the Python native golden. TARGET=test_moe_grouped_real ./build.sh
#   run: <bin> real_weight/real_gptq_gate_L0.bin   (or the Q4_K .bin)
cutlass_example_add_executable(
  test_moe_grouped_real
  test_moe_grouped_real.cu
)
target_compile_options(test_moe_grouped_real PRIVATE -w)

# W2A16 (uint2b_t) single-expert correctness driver: synthetic Q2 -> pack 4 uint2/byte -> preprocess PACKED_INT2
# -> filter_and_run<...,uint2b_t> vs host dequant golden. Build: TARGET=test_w2a16_grouped ./build.sh
cutlass_example_add_executable(
  test_w2a16_grouped
  test_w2a16_grouped.cu
)
target_compile_options(test_w2a16_grouped PRIVATE -w)

# W2A16 LAYOUT DIAGNOSTIC: identity A + decodable W=(k%4)+10*(n%4) to read the int2 layout permutation (mod 4)
# on a MISMATCH. Build: TARGET=test_w2a16_diag ./build.sh
cutlass_example_add_executable(
  test_w2a16_diag
  test_w2a16_diag.cu
)
target_compile_options(test_w2a16_diag PRIVATE -w)

# Ground-truth ppu ldmatrix.swzl probe (standalone, no cutlass): dumps smem-word -> (lane,reg) map to derive
# the int2 interleave. Build: TARGET=swzl_ldmatrix_probe ./build.sh
cutlass_example_add_executable(
  swzl_ldmatrix_probe
  swzl_ldmatrix_probe.cu
)
target_compile_options(swzl_ldmatrix_probe PRIVATE -w)

# INT4 REFERENCE diag: same params as test_w2a16_diag but int4b_t, so the W2DBG cute-layout print for the
# KNOWN-GOOD int4 path can be compared line-by-line vs int2. Build: TARGET=test_w4a16_diag ./build.sh
cutlass_example_add_executable(
  test_w4a16_diag
  test_w4a16_diag.cu
)
target_compile_options(test_w4a16_diag PRIVATE -w)
