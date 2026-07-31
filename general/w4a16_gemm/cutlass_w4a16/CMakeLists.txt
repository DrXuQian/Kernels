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

# Low-bit weight-only GEMV (gemv_lowbit/). The decode-band kernel: no shared memory in the main loop and K
# split Threads ways, so the grid is ~2048 CTAs x 128 threads instead of the grouped GEMM's 512 x 64.
# Build: TARGET=test_gemv_lowbit ./build.sh    Run: $BIN/test_gemv_lowbit  (add `quick` to skip the long tail)
ppu_w4a16_executable(
  test_gemv_lowbit
  test_gemv_lowbit.cu
)
target_compile_options(test_gemv_lowbit PRIVATE -w)

# ---------------------------------------------------------------------------------------------------
# Decode-band perf: the CUDA-core-only GEMV winner. ONE GENERATED .cu PER CONFIG, for the same reason the MoE
# sweep is generated: one .cu is one compiler invocation, so a single-TU sweep compiles strictly serially no
# matter what -j says. Build: TARGET=test_gemv_perf ./build.sh
#
# A row is  name|WFormat|WLayout|tier  and the tier names the (StepK, Threads, CtaN, Chunk) list below. The
# tier is set by the SPARSEST plane: StepK*min(bits) >= 32 is a 4-byte floor per plane per thread, so (8,256)
# is legal for int4/int8, drops out at 2 bits, and only (32,64) survives with an int1 plane.
set(_GEMV_GROUPS
  "i4-native|WFormat::Int4|WLayout::Native|wide"
  "i4-tileK|WFormat::Int4|WLayout::TileK|wide"
  "i2-native|WFormat::Int2|WLayout::Native|mid"
  "q6-native|WFormat::Q6_42|WLayout::Native|mid"
  "q3-native|WFormat::Q3_21|WLayout::Native|narrow"
  "i1-native|WFormat::Int1|WLayout::Native|narrow"
)
# COMMA-separated inside a row: ';' is CMake's own list separator and would split the row itself. The MoE
# generator learned this the expensive way -- a stray ';' produced 8 malformed iterations that still summed to
# the right unit COUNT, so the log looked correct.
set(_GEMV_TIER_wide   "8,256,2,2" "8,256,4,2" "8,256,8,2"
                      "16,128,2,2" "16,128,4,2" "16,128,8,2" "16,128,8,4"
                      "32,64,2,2" "32,64,4,2" "32,64,8,2")
set(_GEMV_TIER_mid    "16,128,2,2" "16,128,4,2" "16,128,8,2" "16,128,8,4"
                      "32,64,2,2" "32,64,4,2" "32,64,8,2")
set(_GEMV_TIER_narrow "32,64,2,2" "32,64,4,2" "32,64,8,2" "32,64,8,4")

if(GEMV_GROUPS)   # narrow by group name, e.g. -DGEMV_GROUPS="i4-native;i2-native"
  set(_keep "")
  foreach(_row IN LISTS _GEMV_GROUPS)
    string(REPLACE "|" ";" _g "${_row}")
    list(GET _g 0 _nm)
    list(FIND GEMV_GROUPS "${_nm}" _idx)
    if(NOT _idx EQUAL -1)
      list(APPEND _keep "${_row}")
    endif()
  endforeach()
  if(NOT _keep)
    message(FATAL_ERROR "GEMV_GROUPS='${GEMV_GROUPS}' matched no row of _GEMV_GROUPS")
  endif()
  set(_GEMV_GROUPS ${_keep})
endif()

set(_GEMV_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gemv_units")
file(MAKE_DIRECTORY "${_GEMV_GEN_DIR}")
set(_GEMV_UNIT_SRCS "")
set(_GEMV_UNIT_DECLS "")
set(_GEMV_UNIT_CALLS "")
set(_GEMV_GROUP_NAMES "")
set(_GEMV_GROUP_IDX -1)

foreach(_row IN LISTS _GEMV_GROUPS)
  string(REPLACE "|" ";" _g "${_row}")
  list(LENGTH _g _nf)
  if(NOT _nf EQUAL 4)
    message(FATAL_ERROR "_GEMV_GROUPS row '${_row}' has ${_nf} fields, expected 4 (name|fmt|layout|tier)")
  endif()
  list(GET _g 0 _nm)
  list(GET _g 1 _fmt)
  list(GET _g 2 _lay)
  list(GET _g 3 _tier)
  math(EXPR _GEMV_GROUP_IDX "${_GEMV_GROUP_IDX} + 1")
  list(APPEND _GEMV_GROUP_NAMES "${_nm}")
  # TileSizeK is 256 for every instantiation here (KernelDetails' default); the group needs it at RUNTIME to
  # pack, so it is passed to make_bufs rather than inferred there.
  set(_GEMV_UNIT_CALLS "${_GEMV_UNIT_CALLS}  {\n    std::printf(\"  --- ${_nm} ---\\n\");\n    Bufs const _bf = make_bufs(${_fmt}, ${_lay}, 256, sh);\n")
  foreach(_cfg IN LISTS _GEMV_TIER_${_tier})
    string(REPLACE "," ";" _c "${_cfg}")
    list(LENGTH _c _cn_f)
    if(NOT _cn_f EQUAL 4)
      message(FATAL_ERROR "tier ${_tier} entry '${_cfg}' has ${_cn_f} fields, expected 4 (StepK,Threads,CtaN,Chunk)")
    endif()
    list(GET _c 0 _sk)
    list(GET _c 1 _th)
    list(GET _c 2 _cn)
    list(GET _c 3 _ch)
    string(REPLACE "-" "_" _nm_id "${_nm}")
    set(_fn "gemv_unit_${_nm_id}_s${_sk}_t${_th}_n${_cn}_c${_ch}")
    set(_src "${_GEMV_GEN_DIR}/${_fn}.cu")
    file(WRITE "${_src}.in"
      "// GENERATED by CMakeLists.txt -- do not edit. One unit = one GEMV config.\n"
      "#define UNIT_FN  ${_fn}\n"
      "#define UNIT_FMT ${_fmt}\n"
      "#define UNIT_LAY ${_lay}\n"
      "#define UNIT_SK  ${_sk}\n"
      "#define UNIT_TH  ${_th}\n"
      "#define UNIT_CN  ${_cn}\n"
      "#define UNIT_CH  ${_ch}\n"
      "#include \"gemv_perf_unit.inc\"\n")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}.in" "${_src}")
    list(APPEND _GEMV_UNIT_SRCS "${_src}")
    set(_GEMV_UNIT_DECLS "${_GEMV_UNIT_DECLS}void ${_fn}(const Shape&, const Bufs&, Best&);\n")
    set(_GEMV_UNIT_CALLS "${_GEMV_UNIT_CALLS}    ${_fn}(sh, _bf, b[${_GEMV_GROUP_IDX}]);\n")
  endforeach()
  set(_GEMV_UNIT_CALLS "${_GEMV_UNIT_CALLS}  }\n")
endforeach()

list(LENGTH _GEMV_UNIT_SRCS _GEMV_UNIT_N)
list(LENGTH _GEMV_GROUP_NAMES _GEMV_GROUP_N)
message(STATUS "GEMV sweep: ${_GEMV_UNIT_N} generated units in ${_GEMV_GEN_DIR} (${_GEMV_GROUP_N} groups)")
if(NOT MOE_CORES)
  set(MOE_CORES 192)
endif()
if(_GEMV_UNIT_N GREATER MOE_CORES)
  math(EXPR _w "(${_GEMV_UNIT_N} + ${MOE_CORES} - 1) / ${MOE_CORES}")
  message(STATUS "GEMV sweep: ${_GEMV_UNIT_N} units > MOE_CORES=${MOE_CORES} -> ${_w} compile waves. "
                 "Narrow with -DGEMV_GROUPS=... to get back to one.")
endif()
string(REPLACE ";" "\", \"" _GEMV_GN_INIT "${_GEMV_GROUP_NAMES}")
file(WRITE "${_GEMV_GEN_DIR}/gemv_perf_units.inc.in"
     "// GENERATED by CMakeLists.txt -- do not edit.\n"
     "#define GEMV_UNIT_COUNT ${_GEMV_UNIT_N}\n"
     "#define GEMV_GROUP_COUNT ${_GEMV_GROUP_N}\n"
     "static const char* const gemv_group_names[] = {\"${_GEMV_GN_INIT}\"};\n"
     "${_GEMV_UNIT_DECLS}"
     "static inline void gemv_run_all(const Shape& sh, Best* b) {\n"
     "  (void)sh; (void)b;\n"
     "${_GEMV_UNIT_CALLS}"
     "}\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_GEMV_GEN_DIR}/gemv_perf_units.inc.in" "${_GEMV_GEN_DIR}/gemv_perf_units.inc")

ppu_w4a16_executable(
  test_gemv_perf
  test_gemv_perf.cu
  ${_GEMV_UNIT_SRCS}
  DEV_COMPILE_FLAGS -I${_GEMV_GEN_DIR}
)
target_compile_options(test_gemv_perf PRIVATE -w)
target_include_directories(test_gemv_perf PRIVATE ${_GEMV_GEN_DIR})

# ---------------------------------------------------------------------------------------------------
# Decode-band perf: grouped GEMM at splitk=1 vs splitk>1, the other two winners. ONE GENERATED .cu PER CONFIG:
# each row instantiates the AIU mixed-input collective, and on the box the expensive stages are LLVM opt/llc,
# single-threaded and minutes per kernel -- five rows in one TU is five of those in series.
# Build: TARGET=test_moe_splitk_bench ./build.sh
#
# A row is TM,TN,TK,WM,WN,Stages, comma-separated (';' is CMake's list separator). The S ladder {1,2,4,8} is a
# RUNTIME loop inside each unit: split-K changes no template argument, so sweeping S costs no compile.
# THE BLOCK-SIZE LADDER AT FIXED TileK. acu says the CU holds 9 blocks (shared memory) of 2 warps = 18 warps of
# 64, i.e. 28% theoretical occupancy, while Block Limit Registers is 20 -- so registers have 2.2x of slack and
# cutting them buys nothing until shared memory stops binding.
#
# A's smem term is TM*TK*2 and does NOT grow with TileN, while warps/block = (TM/WM)*(TN/WN) does. So raising
# TileN at fixed WarpN amortises A's padding (62% of the block's 26,624 B at decode, where every expert has ONE
# row against TileM=16) over more warps:
#
#   TN=32  w16x16  2 warps/blk  26,624 B  9 blk/CU  18 warps/CU  28%   A = 62% of smem
#   TN=64  w16x16  4 warps/blk  36,864 B  7 blk/CU  28 warps/CU  44%   A = 44%
#   TN=128 w16x16  8 warps/blk  57,344 B  4 blk/CU  32 warps/CU  50%   A = 29%
#
# This is a DIFFERENT lever from lowering TileK, and the difference is why both are here. WN*TK*bits, the AIU
# delivery bound, is 16*256*4 = 16384 for all three -- untouched -- whereas TK=64 sits at exactly 4096, the
# bound itself, which is the suspected reason 16x32:64 s4 measured 19% SLOWER at 19 blocks/CU than 16x32:256 s2
# at 9. So: if the TileN ladder improves, occupancy is the lever and TK=64 lost on delivery granularity; if it
# does not, occupancy is not the lever for this kernel and the mainloop surgery to get A out of shared memory is
# not worth starting.
set(_SPLITK_CFGS
  "16,32,256,16,16,2"    # the recorded decode winner: 2 warps/blk, 18 warps/CU
  "16,64,256,16,16,2"    # 4 warps/blk, 28 warps/CU
  "16,128,256,16,16,2"   # 8 warps/blk, 32 warps/CU
  "16,128,256,16,32,2"   # same smem, 4 warps/blk -- isolates warps/block from smem
  "16,64,256,16,32,2"
  # STAGES LADDER at the S=1 winner's tile. The bench's own banner says occupancy here is bounded by WORK, and the
  # measured row proves it: cta/L 128 over 64 CUs is 2 blocks/CU, 8 warps each, achieved 14.2 of a 16-warp ceiling.
  # So freeing shared memory cannot buy warps -- but it means each block may spend up to 128 KB, and s2 at this
  # tile uses only 57 KB. Depth is the one thing a latency-bound kernel can still buy with that budget, and unlike
  # PPU_A_CUBE_H it needs no change to the delivery contract. Footprints: s3 = 86,016 B, s4 = 114,688 B, both
  # still admitting the 2 blocks/CU the grid actually supplies.
  "16,128,256,16,16,3"
  "16,128,256,16,16,4"
  "16,64,256,16,16,4"    # 4 warps/blk at depth 4, in case 8 warps/blk is what limits the pipeline
  "16,32,64,16,16,4"     # the smem-for-occupancy trade that LOST: 19 blk/CU and 19% slower
  "32,64,64,32,32,4"
  "64,128,64,32,32,4"    # prefill-shaped, for contrast
  # TILEK = 128, THE UNTESTED MIDDLE (TODO #22). The TK ladder had exactly two rungs and both are confounded: TK=256
  # wins, TK=64 lost 19% but sits with WN*TK*bits = 16*64*4 = 4096 EXACTLY ON the swzl delivery bound, so that row
  # measured "small TK" and "starved delivery" at once. TK=128 is 8192, twice the bound, so it separates them.
  #
  # TILEK IS NOT AN OCCUPANCY LEVER, and saying otherwise is the mistake this comment exists to prevent. The grid is
  # ppu_tile_scheduler_group.hpp's total_ctas = SUM_e ceil(M_e/TileM) * ceil(N/TileN), times slices on gridDim.z --
  # TileK does not appear in it. Shrinking TileK lowers smem per CTA and so raises the CAPACITY for blocks/CU, but the
  # grid supplies what it supplies, and capacity is not occupancy. An earlier version of this comment quoted "9
  # blocks/CU at TK=128" as though the grid would deliver them; it delivers 2. What TK=128 actually tests is (a) the
  # delivery bound in isolation, since TK=64's 4096 sat exactly on it, and (b) DEPTH: 28,672 B at s2 leaves room for
  # stages that TK=256's 57,344 B does not, and it is what lets splitk's block count fit at all -- S=4 wants 8
  # blocks/CU and TK=256 admits 4.
  #
  # AND THE OCCUPANCY STORY IS ALREADY REFUTED BY THE TN LADDER, which IS the lever that moves the grid:
  #     16,32,256  -> 512 CTAs, 8 blocks/CU, 2 warps/blk   (once the recorded winner; now loses)
  #     16,64,256  -> 256 CTAs, 4 blocks/CU, 4 warps/blk   (does not win)
  #     16,128,256 -> 128 CTAs, 2 blocks/CU, 8 warps/blk   (WINS, 20.11 us)
  # The row with the FEWEST blocks wins, and it is the row with the MOST warps per block. So the Little's Law
  # arithmetic -- 21 MB of necessary traffic in 20.11 us is 1.04 TB/s of a ~2.77 TB/s peak (38%); in-flight bytes per
  # CU are blocks/CU * Stages * bytes-per-stage = 2 * 57,344 = 114,688 of the 256 KB smem allows (45%); only 73% of
  # those are useful because 15/16 of the A tile is padding at Mmax == 1; and 45% * 73% = 33% lands on the measured
  # 38% -- is right about the HEADROOM and wrong about the lever. Three points of the TN ladder pulled that lever and
  # it moved the other way. What TN=128 winning with 8 warps/block points at instead is the per-CTA DEPENDENT CHAIN:
  # more warps per loaded B tile, not more tiles in flight. So the levers left in this band are depth (Stages), the
  # scale reload chain (TODO #11) and the last-wave tail (TODO #10) -- not block count.
  # NOTE these rows keep the fp16 scale path: at gs=32, TK=128 gives Scale_TileK = 4, so kPackedScaleOn is false.
  #
  # THEY ARE ALSO THE FALSIFIER FOR THE PERSISTENT-KERNEL IDEA, which is why they are worth more than a TK datapoint.
  # The pipeline-fill share of a CTA's timeline is Stages/(K/TK + Stages): 2/(8+2) = 20% at TK=256 and 2/(16+2) = 11%
  # at TK=128. Same total work, half the prologue share. So if the fill is the term that a persistent kernel would
  # amortise, these rows should already show roughly 9%; if they do not move, the prologue is not the term and
  # persistence cannot reach it either -- a one-build answer instead of a new dispatch tag.
  #
  # For the record on what persistence would and would not buy here. It would NOT buy the usual thing: 8 active
  # experts x 16 n-tiles = 128 tiles against 64 CUs x 2 blocks/CU = 128 CTAs is EXACTLY one wave, so there is no tail
  # to reclaim in this band (TODO #10's tail is a different shape). It would buy fill amortisation, and only if the
  # grid is deliberately SMALLER than the tile count (grid = #CU * blk/CU) so a CTA chains tiles -- which is what the
  # hand-written AIU MoE did to reach DeepGemm parity. The machinery exists in actlize already:
  # ppu_aiu_gemm_array_persistent_overlap_prologue.hpp, selected by Schedule tag
  # KernelAiuMultistageBatchArrayOverlapPrologue and requiring PersistentSchedulerPPU0015 (the only
  # IsDynamicPersistent = true in the tree; ppu_tile_scheduler_group.hpp is false despite its class name). It is not a
  # drop-in: our Schedule is KernelAiuMultistageMixedInputFinegrainedGs32, so it needs a new tag plus a builder
  # specialisation, and the real work is in the mainloop -- load_init bakes n_coord/l_coord into the gmem tensors once
  # (gSp = local_tile(..., make_coord(n_coord, 0, _))), and a persistent CTA must re-derive them per tile.
  "16,128,128,16,16,2"
  "16,128,128,16,16,4"
  "16,64,128,16,16,2"
  "16,64,128,16,16,4"
)
if(SPLITK_CFGS)
  set(_SPLITK_CFGS ${SPLITK_CFGS})
  message(STATUS "split-K bench: config list overridden")
endif()

set(_SK_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/splitk_units")
file(MAKE_DIRECTORY "${_SK_GEN_DIR}")
set(_SK_SRCS "")
set(_SK_DECLS "")
set(_SK_CALLS "")
foreach(_cfg IN LISTS _SPLITK_CFGS)
  string(REPLACE "," ";" _c "${_cfg}")
  list(LENGTH _c _nf)
  if(NOT _nf EQUAL 6)
    message(FATAL_ERROR "_SPLITK_CFGS row '${_cfg}' has ${_nf} fields, expected 6 (TM,TN,TK,WM,WN,Stages)")
  endif()
  list(GET _c 0 _tm)
  list(GET _c 1 _tn)
  list(GET _c 2 _tk)
  list(GET _c 3 _wm)
  list(GET _c 4 _wn)
  list(GET _c 5 _st)
  set(_fn "splitk_unit_${_tm}x${_tn}_${_tk}_w${_wm}x${_wn}_s${_st}")
  set(_src "${_SK_GEN_DIR}/${_fn}.cu")
  file(WRITE "${_src}.in"
    "// GENERATED by CMakeLists.txt -- do not edit. One unit = one grouped config, all S values.\n"
    "#define UNIT_FN ${_fn}\n"
    "#define UNIT_TM ${_tm}\n"
    "#define UNIT_TN ${_tn}\n"
    "#define UNIT_TK ${_tk}\n"
    "#define UNIT_WM ${_wm}\n"
    "#define UNIT_WN ${_wn}\n"
    "#define UNIT_ST ${_st}\n"
    "#include \"moe_splitk_unit.inc\"\n")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}.in" "${_src}")
  list(APPEND _SK_SRCS "${_src}")
  set(_SK_DECLS "${_SK_DECLS}void ${_fn}(const Band&, const SkCtx&, SkBest&, SkBest&);\n")
  set(_SK_CALLS "${_SK_CALLS}  ${_fn}(bd, cx, b1, bS);\n")
endforeach()
list(LENGTH _SK_SRCS _SK_N)
message(STATUS "split-K bench: ${_SK_N} generated units in ${_SK_GEN_DIR}")
file(WRITE "${_SK_GEN_DIR}/moe_splitk_units.inc.in"
     "// GENERATED by CMakeLists.txt -- do not edit.\n"
     "#define SPLITK_UNIT_COUNT ${_SK_N}\n"
     "${_SK_DECLS}"
     "static inline void splitk_run_all(const Band& bd, const SkCtx& cx, SkBest& b1, SkBest& bS) {\n"
     "  (void)bd; (void)cx; (void)b1; (void)bS;\n"
     "${_SK_CALLS}"
     "}\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_SK_GEN_DIR}/moe_splitk_units.inc.in" "${_SK_GEN_DIR}/moe_splitk_units.inc")

ppu_w4a16_executable(
  test_moe_splitk_bench
  test_moe_splitk_bench.cu
  ${_SK_SRCS}
  DEV_COMPILE_FLAGS -I${_SK_GEN_DIR}
)
target_compile_options(test_moe_splitk_bench PRIVATE -w)
target_include_directories(test_moe_splitk_bench PRIVATE ${_SK_GEN_DIR})

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

# END TO END with the gguf's OWN scale/zero form: the plane is built host-side with the kernel's own put_code, so the bit
# map exists once, and the GEMM is compared against the fixture's golden. Build it BOTH ways -- with the macro the device
# reads 16 B units, without it the same codes as fp16 planes -- and both must MATCH.
#   PPU_DEFS=PPU_PACKED_SCALE=1 TARGET=test_q4k_packed_gemm ./build.sh ; run: ./<bin> real_weight/q4k_packed.bin
ppu_w4a16_executable(
  test_q4k_packed_gemm
  test_q4k_packed_gemm.cu
)
target_compile_options(test_q4k_packed_gemm PRIVATE -w)

# NATIVE Q4_K SCALE/ZERO, LOADED AND DECODED ON THE DEVICE. Everything else on this path consumes fp16 planes that the
# offline pre-multiplied; this loads the gguf's own form (6-bit sc/mn as int8, one fp16 d and dmin per 256 k) and runs
# gguf_scale_decode.hpp on the hardware. The decode is the only new device arithmetic the packed scale channel needs, so
# gating it in isolation means the collective change inherits a verified decode instead of debugging two things at once.
#   fixture: python3 real_weight/dump_packed_scale.py <file.gguf> --tensor blk.11.ffn_down.weight --out real_weight/q4k_packed.bin
#   Build: TARGET=test_q4k_native_scale ./build.sh ; run: ./<bin> real_weight/q4k_packed.bin
ppu_w4a16_executable(
  test_q4k_native_scale
  test_q4k_native_scale.cu
)
target_compile_options(test_q4k_native_scale PRIVATE -w)

# DO THE TWO f16x2 ASM OPS EQUAL THE SCALAR OPS THEY REPLACE? The one question the host gate structurally cannot ask,
# because it compiles under nvcc where CUTLASS_GGUF_PACKED_F16X2_ASM selects the fallback. No GEMM, no shared memory,
# no mma -- so a failure here cannot be confused with tile plumbing.
#   Build: TARGET=test_ppu_f16x2_probe ./build.sh ; run: ./<bin> [real_weight/q4k_packed.bin]
ppu_w4a16_executable(
  test_ppu_f16x2_probe
  test_ppu_f16x2_probe.cu
)
target_compile_options(test_ppu_f16x2_probe PRIVATE -w)

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

# MoE SHAPE BAND perf sweep, all low-bit formats. Separate from the correctness harness on purpose. Prints m_tiles and the
# masked-row fraction NEXT TO the time, because the dense winner cannot be inherited: TN=128 won on dense by halving A's
# re-read count (A traffic), while MoE is weight-bound and the recorded rule is to minimise the TOTAL m-tile count -- and
# large TileM then wastes rows at ~128 rows per expert. The bench decides which effect dominates instead of assuming one.
#   Build: TARGET=test_lowbit_moe_bench ./build.sh ; run: ./<bin> [L] [rows] [N] [K] [gs] [ragged]
# ONE TRANSLATION UNIT PER SHAPE, GENERATED. cutlass_build_dev_kernels emits one add_custom_command per .cu, so a sweep
# in a single file compiles strictly serially however large -j is. The granularity is set by MEASUREMENT, not taste:
# locally the header alone parses in ~5 s and each two-plane kernel adds ~2.5 s, so a unit costs about 5 + 2.5*kernels.
# One shape (3 stage counts, <= 3 kernels) is ~12 s; eight format-sized units of 42 kernels each were ~109 s; the old
# single file with 30 kernels was ~80 s. On a 192-core box 128 units of one shape therefore finish in less wall clock
# than the ORIGINAL 30-row sweep did, while instantiating 11x as many configurations.
#
# Stages deliberately stay INSIDE a unit. That axis is a runtime cost rather than a compile one: the offline pack is per
# shape and stages do not change the shape, so all three share one pack (31 ms per two-plane shape, 24 ms single-plane,
# ~3 s over the sweep -- already the same order as the GPU time). Splitting stages would triple it for nothing.
#
# THIS TABLE IS THE ONLY PLACE THE ENUMERATION LIVES. Each row is name|2plane|lo_elem|hi_elem|lo_bits|hi_bits|WarpN list;
# the illegal combinations are not listed anywhere because moe_ok<> in lowbit_moe_bench.hpp rejects them and the unit
# compiles to an empty function -- notably WarpM > TileM, and the per-plane delivery bound WN*TK*bits >= 4096 whose
# SPARSEST plane binds (which is what pins q3/q5 to WarpN=64 and lets q6 reach 32).
# THE WarpN LIST IS COMMA-SEPARATED, NOT SEMICOLON-SEPARATED, and that is not cosmetic: `;` IS CMake's list separator, so
# "q6|1|int4_t|uint2_t|4|2|32;64" is not one element containing a semicolon -- `foreach(IN LISTS)` splits it into
# "q6|1|int4_t|uint2_t|4|2|32" and a stray "64", and the stray row has one field so every list(GET) past 0 goes out of
# range. Worse, the generator still printed "128 generated units" while doing it, because 8 broken iterations x 16 happens
# to equal the 5 correct formats x 16 -- so the count that exists to catch a shrunken enumeration agreed with the right
# answer for the wrong reason. Hence the field-count check below: a malformed row must stop the configure, not produce a
# plausible number.
set(_MOE_FORMATS
  "q3|1|uint2_t|uint1_t|2|1|64"
  "q5|1|int4_t|uint1_t|4|1|64"
  "q6|1|int4_t|uint2_t|4|2|32,64"
  "i2|0|uint2_t|half_t|2|0|16,32,64"
  "i4|0|int4_t|half_t|4|0|16,32,64"
)
# 16 IS THE HARDWARE FLOOR, and it was missing. Every MMA atom in cute/atom/mma_traits_ppu0015.hpp has M = 16
# (Shape_MNK = Shape<_16,...>, all eight of them), so TileM must be a multiple of 16 and TileM=8 is not reachable with these
# instructions -- the FA path's ppu.mma.m8n16 is raw asm in a different kernel family, not a cute atom here. TileM=16 IS
# reachable and this project has run it (the Q3 B-concat sweep used 16x32 warps at TM=16, which fold_traits.hpp still notes),
# but the list started at 32 so the whole row was absent from every sweep.
#
# At decode it is the largest smem term: A = TM*TK*2 is 2048 of 3328 B/stage at (32,64,32), so TM=16 takes ifl from 78 to 113
# and TM=16 with TileN=32 to 157 -- roughly 2x the in-flight loads -- while the CTA count does NOT change, because at one row
# per expert mt = ceil(1/TM) = 1 whatever TileM is.
set(_MOE_TM_LIST 16 32 64 128 256)
# SCOPE A SWEEP TO ITS QUESTION. Measured cost of one unit (nvcc front end, local): 4.2 s of header parsing plus ~3.0 s per
# kernel of template instantiation; CODEGEN IS 5.6% of the total, so this is entirely a front-end cost and nothing about the
# backend or the link will move it. Wall clock is ceil(U/cores) * (4.2 + kernels_per_unit*3.0), which at 642 kernels gives
# 19.2 s at U=128, 14.2 s at U=192 and 19.1 s at U=360 -- so the optimum is U ~= the core count, and "one shape per unit" is
# only optimal while the shape count stays under it. 360 shapes crossed that line and bought a second wave.
#
# Rather than re-tune the split, narrow the PRODUCT: a decode question does not need TileM 64/128/256 (at one row per expert
# TileM only sets smem, so larger is strictly worse), and a prefill question does not need the deep stages. Override any axis
# from build.sh, semicolon-separated:
#   PPU_DEFS=... MOE_TM_LIST="16;32" MOE_STAGES="4;6;8;12" TARGET=test_lowbit_moe_bench ./build.sh
# MOE_FORMATS keeps only the named formats (semicolon-separated, e.g. "i4" or "i4;i2"). This is the axis that cuts the most
# when the question is about ONE format -- the decode work is all i4 right now, and carrying q3/q5/q6/i2 through it is 4/5 of
# the compile for none of the answer.
if(MOE_FORMATS)
  set(_keep "")
  foreach(_row IN LISTS _MOE_FORMATS)
    string(REPLACE "|" ";" _f "${_row}")
    list(GET _f 0 _nm)
    # list(FIND) rather than IN_LIST: IN_LIST needs policy CMP0057 NEW, which a `cmake -P` slice of this file does not
    # inherit -- so the generator gate would fail on syntax the real configure accepts. Depending on ambient policy makes the
    # gate and the build behave differently, which is the one thing a gate must not do.
    list(FIND MOE_FORMATS "${_nm}" _idx)
    if(NOT _idx EQUAL -1)
      list(APPEND _keep "${_row}")
    endif()
  endforeach()
  if(NOT _keep)
    message(FATAL_ERROR "MOE_FORMATS='${MOE_FORMATS}' matched no row of _MOE_FORMATS")
  endif()
  string(REPLACE ";" " " _show "${MOE_FORMATS}")
  message(STATUS "MoE sweep: formats restricted to ${_show}")
  set(_MOE_FORMATS ${_keep})
endif()
foreach(_ax TM TN WM)
  if(MOE_${_ax}_LIST)
    string(REPLACE ";" " " _show "${MOE_${_ax}_LIST}")
    message(STATUS "MoE sweep: MOE_${_ax}_LIST restricted to ${_show}")
    set(_MOE_${_ax}_LIST ${MOE_${_ax}_LIST})
  endif()
endforeach()
# 32 is here for DECODE. At batch 1 the grid is 8 m-tiles x (N/TN), so TileN is the only knob that adds CTAs, and the decode
# measurement says the band is latency/occupancy-bound rather than bandwidth-bound. It self-filters where WarpN > TileN.
# 16 IS HERE BECAUSE acu SAID THE GRID CANNOT FILL THE MACHINE. At the decode winner (i4 16x64:256 w16x32 s2) acu reports
# theoretical occupancy 21.88% limited by SHARED MEMORY (Block Limit Shared Mem = 7, registers would allow 12) and achieved
# occupancy 10.90% -- exactly half -- because the grid is mt*(N/TN) = 8*32 = 256 CTAs against a capacity of 72*7 = 504.
# TileN fixes BOTH layers at once: it shrinks smem/block (so blocks/CU rises) and multiplies the CTA count.
#   TN=64  36 KB/blk  7 blk/CU   256 CTAs   51% of capacity   <- the current winner
#   TN=32  26 KB/blk  9 blk/CU   512 CTAs   79%
#   TN=16  21 KB/blk  12 blk/CU  1024 CTAs  full (and register-limited from here)
set(_MOE_TN_LIST 16 32 64 128)
# WarpM=16 comes with it (TileM=16 admits no other), and is offered at every TileM rather than only at 16: cvt/mma = 128/WM
# is 8 there, which the skill records as a ~13-point cliff on PREFILL -- but decode runs at 0.4% MFU where the converter is
# not the bottleneck, so that cost is a prediction to test, not a reason to exclude the axis.
set(_MOE_WM_LIST 16 32 64)

set(_MOE_UNIT_SRCS "")
set(_MOE_LAST_NM "")
set(_MOE_FMT_NAMES "")
set(_MOE_FMT_IDX -1)
set(_MOE_UNIT_DECLS "")
set(_MOE_UNIT_CALLS "")
set(_MOE_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/moe_units")
file(MAKE_DIRECTORY "${_MOE_GEN_DIR}")
foreach(_row IN LISTS _MOE_FORMATS)
  string(REPLACE "|" ";" _f "${_row}")
  list(LENGTH _f _nf)
  if(NOT _nf EQUAL 7)
    message(FATAL_ERROR "_MOE_FORMATS row '${_row}' has ${_nf} fields, expected 7 "
                        "(name|2plane|lo_elem|hi_elem|lo_bits|hi_bits|WarpN,list). A ';' anywhere in a row splits it.")
  endif()
  list(GET _f 0 _nm)
  list(GET _f 1 _2p)
  list(GET _f 2 _lo)
  list(GET _f 3 _hi)
  list(GET _f 4 _lob)
  list(GET _f 5 _hib)
  list(GET _f 6 _wns)
  string(REPLACE "," ";" _wns "${_wns}")   # comma -> real CMake list, only now that the row is safely split
  foreach(_wn IN LISTS _wns)
    foreach(_tn IN LISTS _MOE_TN_LIST)
      foreach(_wm IN LISTS _MOE_WM_LIST)
        foreach(_tm IN LISTS _MOE_TM_LIST)
          set(_fn "moe_unit_${_nm}_tn${_tn}_wn${_wn}_tm${_tm}_wm${_wm}")
          set(_src "${_MOE_GEN_DIR}/${_fn}.cu")
          # CONFIGURE_DEPENDS is not needed: the content is a pure function of this loop, so a changed table means a
          # re-configure, and file(WRITE) leaves the timestamp alone when the bytes are identical... it does not, so
          # write through a compare to keep make from rebuilding all 128 units on every configure.
          file(WRITE "${_src}.in"
            "// GENERATED by CMakeLists.txt -- do not edit. One sweep unit = one shape, all stage counts.\n"
            "#define UNIT_NAME    ${_nm}\n"
            "#define UNIT_FN      ${_fn}\n"
            "#define UNIT_2PLANE  ${_2p}\n"
            "#define UNIT_LOELEM  ${_lo}\n"
            "#define UNIT_HIELEM  ${_hi}\n"
            "#define UNIT_LOB     ${_lob}\n"
            "#define UNIT_HIB     ${_hib}\n"
            "#define UNIT_TM      ${_tm}\n"
            "#define UNIT_TN      ${_tn}\n"
            "#define UNIT_WM      ${_wm}\n"
            "#define UNIT_WN      ${_wn}\n"
            "#include \"moe_bench_unit.inc\"\n")
          execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}.in" "${_src}")
          list(APPEND _MOE_UNIT_SRCS "${_src}")
          set(_MOE_UNIT_DECLS "${_MOE_UNIT_DECLS}void ${_fn}(const Band&, Best&);\n")
          # A section header when the format changes, so 336 rows are still readable. Emitted by the generator rather
          # than by the units: only the generator knows the order they will be called in.
          if(NOT _nm STREQUAL _MOE_LAST_NM)
            set(_MOE_UNIT_CALLS "${_MOE_UNIT_CALLS}  std::printf(\"  --- ${_nm} ---\\n\");\n")
            set(_MOE_LAST_NM "${_nm}")
            # ONE Best PER NUMERIC PRECISION. The verdict used to be "two-plane best" and "single-plane best", which is a
            # grouping by IMPLEMENTATION, not by anything a caller chooses: nobody picks "a two-plane format", they pick
            # Q5 or Q6. And it hid the interesting result -- with i2 and i4 sharing one slot, i4's own best never printed.
            math(EXPR _MOE_FMT_IDX "${_MOE_FMT_IDX} + 1")
            list(APPEND _MOE_FMT_NAMES "${_nm}")
          endif()
          set(_MOE_UNIT_CALLS "${_MOE_UNIT_CALLS}  ${_fn}(bd, b[${_MOE_FMT_IDX}]);\n")
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
list(LENGTH _MOE_UNIT_SRCS _MOE_UNIT_N)
message(STATUS "MoE sweep: ${_MOE_UNIT_N} generated units in ${_MOE_GEN_DIR}")
# Crossing the core count costs a whole extra wave of ~(4.2 + kernels_per_unit*3.0) seconds, and that is invisible in a build
# log otherwise -- the first time it happened the only symptom was "compiling got slower".
if(NOT MOE_CORES)
  set(MOE_CORES 192)
endif()
if(_MOE_UNIT_N GREATER MOE_CORES)
  math(EXPR _waves "(${_MOE_UNIT_N} + ${MOE_CORES} - 1) / ${MOE_CORES}")
  message(STATUS "MoE sweep: ${_MOE_UNIT_N} units > MOE_CORES=${MOE_CORES} -> ${_waves} compile waves. "
                 "Narrow an axis (MOE_TM_LIST / MOE_TN_LIST / MOE_WM_LIST / MOE_STAGES) to get back to one.")
endif()
# main() includes this instead of naming 128 functions by hand. PLAIN CODE, not a multi-line macro: that needs a trailing
# backslash on EVERY line and the first version emitted one, which would have been a build break generated by the build.
# MOE_UNIT_COUNT is compiled in so the binary prints how many units it was linked against -- a silently shrunk
# enumeration then shows up in the log instead of as a quietly smaller sweep.
list(LENGTH _MOE_FMT_NAMES _MOE_FMT_N)
string(REPLACE ";" "\", \"" _MOE_FMT_INIT "${_MOE_FMT_NAMES}")
file(WRITE "${_MOE_GEN_DIR}/moe_bench_units.inc.in"
     "// GENERATED by CMakeLists.txt -- do not edit.\n"
     "#define MOE_UNIT_COUNT ${_MOE_UNIT_N}\n"
     "#define MOE_FMT_COUNT ${_MOE_FMT_N}\n"
     "static const char* const moe_fmt_names[] = {\"${_MOE_FMT_INIT}\"};\n"
     "${_MOE_UNIT_DECLS}"
     "static inline void moe_run_all(const Band& bd, Best* b) {\n"
     "  (void)bd; (void)b;\n"
     "${_MOE_UNIT_CALLS}"
     "}\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_MOE_GEN_DIR}/moe_bench_units.inc.in" "${_MOE_GEN_DIR}/moe_bench_units.inc")

# -I must go through DEV_COMPILE_FLAGS, NOT target_include_directories: the device compiles are add_custom_commands that
# never see target properties (the same reason PPU_DEFS has to be forwarded twice). The generated units find the .inc via
# the custom command's own -I<source dir of the file>, but test_lowbit_moe_bench.cu lives in the source tree and would
# not find moe_bench_units.inc in the binary tree without this.
ppu_w4a16_executable(
  test_lowbit_moe_bench
  test_lowbit_moe_bench.cu
  ${_MOE_UNIT_SRCS}
  DEV_COMPILE_FLAGS -I${_MOE_GEN_DIR}
)
target_compile_options(test_lowbit_moe_bench PRIVATE -w)

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
