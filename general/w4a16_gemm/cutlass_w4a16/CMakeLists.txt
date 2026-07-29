# A/B hook: PPU_DEFS=<space-separated defines> ./build.sh applies them to every target defined below. Added because
# comparing a collective change against the path it replaces otherwise means editing a TRACKED header between runs,
# and the box builds from COMMITTED state -- so an A/B would need two commits and two pulls. e.g.
#   PPU_DEFS=PPU_SCALE_BCAST=0 TARGET=test_int1_sweep ./build.sh
# NOTE the two forms. hgcc (the DEVICE compiler) is driven by a custom command that does NOT see
# target_compile_definitions -- see the comment further down -- so a define meant for device code MUST go through
# DEV_COMPILE_FLAGS as a verbatim -D flag. Using add_compile_definitions here would have silently reached only the
# host compile, and an A/B of a collective change would then have compared two IDENTICAL device binaries.
set(_PPU_EXTRA_DEV)     # verbatim -D flags for hgcc
set(_PPU_EXTRA_HOST)    # bare NAME=VALUE for target_compile_definitions
if(PPU_EXTRA_DEFS)
  string(REPLACE " " ";" _PPU_EXTRA_HOST "${PPU_EXTRA_DEFS}")
  foreach(_d IN LISTS _PPU_EXTRA_HOST)
    list(APPEND _PPU_EXTRA_DEV "-D${_d}")
  endforeach()
  message(STATUS "PPU_EXTRA_DEFS -> device ${_PPU_EXTRA_DEV} / host ${_PPU_EXTRA_HOST}")
endif()

# EVERY target gets them, via a wrapper -- NOT by listing targets. The previous code attached _PPU_EXTRA_DEV to three
# targets by hand while the comment above claimed "every target defined below", and the two are not the same thing:
#   PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_q3_bconcat_bench ./build.sh
# configured cleanly, printed "PPU_DEFS applied: PPU_B_CHUNK=1", printed no warning (cmake DID receive it), and built a
# binary without the macro -- because test_q3_bconcat_bench was not one of the three. The perf run that followed
# compared a binary against ITSELF and reported "no change", which is indistinguishable from "the change does nothing".
# Enumerating call sites is what made that possible, so the fix must not enumerate.
function(ppu_w4a16_executable NAME)
  cmake_parse_arguments(A "" "" "DEV_COMPILE_FLAGS" ${ARGN})
  cutlass_example_add_executable(${NAME} ${A_UNPARSED_ARGUMENTS}
      DEV_COMPILE_FLAGS ${A_DEV_COMPILE_FLAGS} ${_PPU_EXTRA_DEV})
  target_compile_definitions(${NAME} PRIVATE ${_PPU_EXTRA_HOST})
endfunction()

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

set(BENCH_QUANT "int4" CACHE STRING "bench QuantType: int4 (W4A16) or uint2 (W2A16)")
set(TSK "" CACHE STRING "TileShapeK override (empty=default int2->128/int4->64; set e.g. 128 to force, incl. int4 for apples-to-apples)")
set(_QUANT_DEFS)
if(BENCH_QUANT STREQUAL "uint2")
  set(_QUANT_DEFS -DBENCH_UINT2)
elseif(BENCH_QUANT STREQUAL "uint1")
  set(_QUANT_DEFS -DBENCH_UINT1)
endif()
if(NOT TSK STREQUAL "")
  list(APPEND _QUANT_DEFS -DBENCH_TSK=${TSK})
endif()

set(_W4A16_DEFS
  -DTILE_M=${TILE_M} -DTILE_N=${TILE_N} -DWARP_M=${WARP_M} -DWARP_N=${WARP_N} -DSTAGES=${STAGES} ${_QUANT_DEFS})

ppu_w4a16_executable(
  bench_cutlass_w4a16
  bench_cutlass_w4a16.cu
  DEV_COMPILE_FLAGS ${_W4A16_DEFS}
)

target_compile_definitions(bench_cutlass_w4a16 PRIVATE
  TILE_M=${TILE_M} TILE_N=${TILE_N} WARP_M=${WARP_M} WARP_N=${WARP_N} STAGES=${STAGES} ${_QUANT_DEFS})
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
ppu_w4a16_executable(
  test_fpA_intB_ppu
  test_fpA_intB_ppu.cu
)
target_compile_options(test_fpA_intB_ppu PRIVATE -w)

# STEP 1 of MoE W4A16: batched (uniform m_per_expert) mixed-input GEMM. Build: TARGET=test_moe_gemm_ppu ./build.sh
ppu_w4a16_executable(
  test_moe_gemm_ppu
  test_moe_gemm_ppu.cu
)
target_compile_options(test_moe_gemm_ppu PRIVATE -w)

# STEP 2a of MoE W4A16: grouped mixed-input via GroupScheduler. Build: TARGET=test_moe_grouped_ppu ./build.sh
ppu_w4a16_executable(
  test_moe_grouped_ppu
  test_moe_grouped_ppu.cu
)
target_compile_options(test_moe_grouped_ppu PRIVATE -w)

# STEP 2b correctness gate: multi-expert/ragged grouping vs grouped-L=1 oracle. TARGET=test_moe_grouped_verify ./build.sh
ppu_w4a16_executable(
  test_moe_grouped_verify
  test_moe_grouped_verify.cu
)
target_compile_options(test_moe_grouped_verify PRIVATE -w)

# ROUTING PROBE: MOEG_PROBE=1 makes the grouped kernel write (expert+1) instead of the GEMM, to decode which
# D-plane each expert's tiles land in. TARGET=test_moe_grouped_probe ./build.sh ; run: MOEG_PROBE=1 <bin>
ppu_w4a16_executable(
  test_moe_grouped_probe
  test_moe_grouped_probe.cu
)
target_compile_options(test_moe_grouped_probe PRIVATE -w)

# DATA-SLICE probe: controlled per-expert data (B/scale/A each varied in turn) to pin WHICH input the collective
# fails to slice by l_coord. TARGET=test_moe_grouped_dataslice ./build.sh ; run: <bin> (MOEG_PROBE unset)
ppu_w4a16_executable(
  test_moe_grouped_dataslice
  test_moe_grouped_dataslice.cu
)
target_compile_options(test_moe_grouped_dataslice PRIVATE -w)

# REAL-WEIGHT gate: loads a .bin (real Qwen3.5 GPTQ / Q4_K expert weights from real_weight/dump_real_weights.py),
# runs the grouped kernel, compares vs the Python native golden. TARGET=test_moe_grouped_real ./build.sh
#   run: <bin> real_weight/real_gptq_gate_L0.bin   (or the Q4_K .bin)
ppu_w4a16_executable(
  test_moe_grouped_real
  test_moe_grouped_real.cu
)
target_compile_options(test_moe_grouped_real PRIVATE -w)

# W2A16 (uint2b_t) single-expert correctness driver: synthetic Q2 -> pack 4 uint2/byte -> preprocess PACKED_INT2
# -> filter_and_run<...,uint2b_t> vs host dequant golden. Build: TARGET=test_w2a16_grouped ./build.sh
ppu_w4a16_executable(
  test_w2a16_grouped
  test_w2a16_grouped.cu
)
target_compile_options(test_w2a16_grouped PRIVATE -w)

# W2A16 LAYOUT DIAGNOSTIC: identity A + decodable W=(k%4)+10*(n%4) to read the int2 layout permutation (mod 4)
# on a MISMATCH. Build: TARGET=test_w2a16_diag ./build.sh
ppu_w4a16_executable(
  test_w2a16_diag
  test_w2a16_diag.cu
)
target_compile_options(test_w2a16_diag PRIVATE -w)

# W1A16 LAYOUT DIAGNOSTIC: identity A + random q1/scale/zero, uint1b_t (8/byte, PACKED_INT1). Build: TARGET=test_w1a16_diag ./build.sh
ppu_w4a16_executable(
  test_w1a16_diag
  test_w1a16_diag.cu
)
target_compile_options(test_w1a16_diag PRIVATE -w)

# W1A16 MULTI-EXPERT + RAGGED grouped gate: grouped-L vs grouped-L=1 oracle (contiguous ptr-array D), uint1b_t,
# TK=256, int1's winner tile 32x128. Build: TARGET=test_w1a16_grouped ./build.sh ; run: ./<bin> [L] [Mb] [ragged] [gs]
ppu_w4a16_executable(
  test_w1a16_grouped
  test_w1a16_grouped.cu
)
target_compile_options(test_w1a16_grouped PRIVATE -w)

# Q3_K BIT-PLANE CONCAT on REAL GGUF weights: int2(low2, zero=-4dl) + int1(high1, scale=4dl) summed vs the native
# Q3_K golden. Validates int1 AND int2 on real bits AND the A-concat at once. Bin from real_weight/dump_q3_concat.py.
#   Build: TARGET=test_q3_concat_real ./build.sh ; run: ./<bin> real_weight/real_q3k_concat.bin
ppu_w4a16_executable(
  test_q3_concat_real
  test_q3_concat_real.cu
)
target_compile_options(test_q3_concat_real PRIVATE -w)

# Q3_K B-CONCAT: ONE GEMM with TWO B bit planes (dedicated 2-plane mainloop), vs the native Q3_K golden.
#   Build: TARGET=test_q3_bconcat_real ./build.sh ; run: ./<bin> real_weight/real_q3k_concat.bin
ppu_w4a16_executable(
  test_q3_bconcat_real
  test_q3_bconcat_real.cu
)
target_compile_options(test_q3_bconcat_real PRIVATE -w)

# Q6 (int4+int2) and Q5 (int4+int1) B-CONCAT on the SAME 2-plane mainloop, synthetic full code range vs a CPU golden.
# The converter and the offline are one closed form over (LowBits, HiBits), gated locally in fold_derivation/l65-l67;
# what this exercises is the plumbing -- builder fold factors, gmem strides, the scale path -- which no local gate reaches.
#   Build: TARGET=test_q65_bconcat_real ./build.sh   (PPU_DEFS=PPU_B_CHUNK=1 for the chunked path)
ppu_w4a16_executable(
  test_q65_bconcat_real
  test_q65_bconcat_real.cu
)
target_compile_options(test_q65_bconcat_real PRIVATE -w)

# MULTI-EXPERT (L>1) + RAGGED gate for every LOW-BIT format on the grouped path: the two-plane ones (Q3 = int2+int1,
# Q6 = int4+int2, Q5 = int4+int1) and the single-plane ones (int2, int1, int4). Named for what it covers -- it was
# test_q65_grouped, which read as "Q6 and Q5 only" even though Q3 was the first entry, and the name is the only part a
# reader sees. The oracle is the L=1 kernel run per expert, so a mismatch is per-expert ADDRESSING and nothing else --
# notably plane 2's own L-stride, a hand-written byte count that had never been tested. Every expert gets DIFFERENT
# weights in every plane and different scales, because with identical experts "every expert read plane 0" passes
# silently. int1 is included as a check on the HARNESS: it already has an independent multi-expert gate
# (test_w1a16_grouped), so int1 failing here while passing there would mean this harness is wrong, not the kernel.
#   Build: TARGET=test_lowbit_grouped ./build.sh ; run: ./<bin> [L] [ragged] [Mb] [N] [K] [gs]
ppu_w4a16_executable(
  test_lowbit_grouped
  test_lowbit_grouped.cu
)
target_compile_options(test_lowbit_grouped PRIVATE -w)

# B-CONCAT CONTROLLED-INPUT PROBE: A=identity + scale=1/zero=0 + index-labelled bit planes, so D reads back WHICH
# (n,k) each consumed bit came from. Rungs 0/1/2 separate mantissa PLACEMENT from bit PERMUTATION; rung 3 decodes
# the permutation. Same technique that cracked the int2 sigma_n bug (sim layouts are NOT trustworthy here).
#   Build: TARGET=test_q3_bconcat_probe ./build.sh ; run: ./<bin> [N] [K]
ppu_w4a16_executable(
  test_q3_bconcat_probe
  test_q3_bconcat_probe.cu
)
target_compile_options(test_q3_bconcat_probe PRIVATE -w)

# Q3_K B-CONCAT ABLATION on the REAL .bin: runs the 2-plane kernel on the real numbers with a HOST golden recomputed
# per ablation (real low+high / high=0 / low=0 / synthetic A) to localize a bug the synthetic probe cannot reproduce.
#   Build: TARGET=test_q3_bconcat_ablate ./build.sh ; run: ./<bin> real_weight/real_q3k_concat.bin
ppu_w4a16_executable(
  test_q3_bconcat_ablate
  test_q3_bconcat_ablate.cu
)
target_compile_options(test_q3_bconcat_ablate PRIVATE -w)

# Q3_K B-CONCAT vs A-CONCAT speed: synthetic large shape, interleaved same-session A/B timing (us/MFU/HBM).
#   Build: TARGET=test_q3_bconcat_bench ./build.sh ; run: ./<bin> [M] [N] [K]
ppu_w4a16_executable(
  test_q3_bconcat_bench
  test_q3_bconcat_bench.cu
)
target_compile_options(test_q3_bconcat_bench PRIVATE -w)

# GROUPED int1 (vs int2 ref) single-plane PERF verification: matches the validated test_w1a16_grouped call
# (FinegrainedScaleOnly, PACKED_INT1, 32x128:256:s3) but timed + acu-able. Isolates whether grouped int1 is genuinely
# ~10x the dense record or a harness artifact.  Build: TARGET=test_grouped_int1_perf ./build.sh
ppu_w4a16_executable(
  test_grouped_int1_perf
  test_grouped_int1_perf.cu
)
target_compile_options(test_grouped_int1_perf PRIVATE -w)

# N-FOLD first box run: single-plane int2 folded to TileShape.K=64 (FoldF=2 -> operand run 128 elems = 32B, reusing
# the validated int2@TK128 swzl/converter). Target geometry is int4@TK64 (measured 55.8% gs=32 / 53.1% gs=16) vs
# int2@TK128's 37.1%/30.9%. A=identity/scale=1/zero=0 so D == the weight code (controlled-input check).
#   Build: TARGET=test_fold_int2 ./build.sh ; run: ./<bin> [N] [K] [gs]
ppu_w4a16_executable(
  test_fold_int2
  test_fold_int2.cu
)
target_compile_options(test_fold_int2 PRIVATE -w)

# int1 tile sweep, kept OUT of test_fold_int2 so the verified correctness harness does not pay for the extra
# instantiations. Explores the two knobs left after the box overturned the scale model -- occupancy (TM*TK per stage)
# and reload hiding (atoms per iteration = TK/16) -- plus FOLD_ZDIAG=1, which separates whether the ScaleZero cost is
# the per-reload smem copy or the per-atom transform. Those two want completely different fixes.
#   Build: TARGET=test_int1_sweep ./build.sh ; run: ./<bin> [N] [K] [gs] [M]
ppu_w4a16_executable(
  test_int1_sweep
  test_int1_sweep.cu
)
target_compile_options(test_int1_sweep PRIVATE -w)

# Ground-truth ppu ldmatrix.swzl probe (standalone, no cutlass): dumps smem-word -> (lane,reg) map to derive
# the int2 interleave. Build: TARGET=swzl_ldmatrix_probe ./build.sh
ppu_w4a16_executable(
  swzl_ldmatrix_probe
  swzl_ldmatrix_probe.cu
)
target_compile_options(swzl_ldmatrix_probe PRIVATE -w)

# INT4 REFERENCE diag: same params as test_w2a16_diag but int4b_t, so the W2DBG cute-layout print for the
# KNOWN-GOOD int4 path can be compared line-by-line vs int2. Build: TARGET=test_w4a16_diag ./build.sh
ppu_w4a16_executable(
  test_w4a16_diag
  test_w4a16_diag.cu
)
target_compile_options(test_w4a16_diag PRIVATE -w)

# W2A16 REAL-WEIGHT gate: reads a real Q2_K slice .bin (real_weight/real_q2k_ffn_gate_L0.bin, from
# dump_real_weights.py on the 3B ffn_gate) + native golden; runs W2A16 at gs=16. TARGET=test_w2a16_real ./build.sh
ppu_w4a16_executable(
  test_w2a16_real
  test_w2a16_real.cu
)
target_compile_options(test_w2a16_real PRIVATE -w)

# WIDTH ISOLATION for acu: int1/int2/int4 at ONE shared tile config, so bit width is the only variable. int1's best
# (50.2%) and int4's production config (55.9%) run different tiles, and every model says the gap should NOT be width
# (same cvt/mma, same regs, HBM favours int1, and the fold makes the B smem bank pattern identical by construction).
#   Build: TARGET=test_width_acu ./build.sh ; run: ./<bin> [bits] [M] [N] [K] [gs] ; ACU_ONE=1 for a single launch
ppu_w4a16_executable(
  test_width_acu
  test_width_acu.cu
)
target_compile_options(test_width_acu PRIVATE -w)
