# Plan: other 4-bit formats + offline reorder (actlize mixed-input on ppu001)

Reference architecture is the PPU-official acext fpA_intB wrapper (dense W4A16/W8A16): a void* `RunnerInterface`
+ templated `Runner<T,WeightType,QuantOp>` + `dispatchGemmToCutlass` (`ENABLE_CUTLASS3` + `if constexpr
(uint4b_t)` → `dispatchGemmToCutlass3`) + heuristic/LUT config selection. Our `bench_cutlass_w4a16.cu` is the
same GEMM (actlize mixed-input) minus the runner/LUT wrapper.

## The format axis = QuantOp

"Supporting another 4-bit format" is almost entirely the acext QuantOp axis plus a host-side unpack. No new
GEMM kernel except for non-linear codebooks (IQ4).

| format | mixed-input mode / QuantOp | work |
|---|---|---|
| GPTQ symmetric (u4b8) | mode 1 / FINEGRAINED_SCALE_ONLY | done (our comparison path) |
| AWQ asymmetric (u4 + zero) | mode 2 / FINEGRAINED_SCALE_AND_ZEROS | wire the existing `GemmScaleWithZeroPoint` into the tactic path |
| per-column (gs=-1) | PER_COLUMN_SCALE_ONLY | add the QuantOp branch |
| **GGUF Q4_K** | → AWQ form (mode 2) | host unpack the 6-bit (d·s6, dmin·m6) fields → fp16 group **scale + zero @ gs=32**, then the mixed-input GEMM. No new kernel. |
| GGUF Q4_0 / Q4_1 | scale-only / scale+zero @ gs=32 | host unpack (Q4_0 symmetric, Q4_1 affine) |
| IQ4_NL / IQ4_XS | — | needs a custom `NumericConverter` (int4→fp16 through the NL codebook); the stock mixed-input converter is linear only |

**Unifying intermediate format:** `(int4 weights, fp16 group scale [+ fp16 group zero], group_size)`. Every
GGUF/GPTQ/AWQ 4-bit format decodes to this on the host; the GEMM is then format-agnostic.

**gs=32 caveat (corrected against the official acext launcher).** There are TWO actlize mixed-input paths:
- The **generic** schedule `KernelTmaWarpSpecializedCooperativeMixedInput` with a **runtime** group size
  (`options.g`) — this is what example 16 and our bench use. It has NO `block_k >= group_size` constraint
  (our 64x64 winner, TileShapeK=64, passed at gs=128), and plausibly runs gs=32 directly. Verify.
- The **official high-perf** path (`cutlass3/fpA_intB_gemm_template_cutlass3.cu`) uses **compile-time**
  `KernelAiuMultistageMixedInputFinegrainedGs128/Gs64` with an explicit `ScaleTileShape =
  Shape<CTA_N, ceil(CTA_K/gs)>`, a `cute::tuple<TileShape, ScaleTileShape>` mainloop arg, and
  `switch(group_size){case 128; case 64; default: throw}`. It also validates `block_k >= group_size`. So it
  supports ONLY gs 128/64. **Q4_K's gs=32 needs a new `FinegrainedGs32` specialization on this path** (or
  stays on the generic runtime-g schedule, which is what our bench proves out).

## Offline reorder — required

`preprocess_weights_for_mixed_gemm` (the interleave-256 / mixed_gemm_B_layout) is a **weight-only, M- and
activation-independent** transform. The bench runs it in `initialize()` every invocation; a real deployment
must move it offline:

- **Offline (model load / conversion):** GGUF Q4_K on-disk layout → (1) decode 6-bit fields to int4 weights +
  fp16 group scale/zero, (2) `preprocess_weights_for_mixed_gemm` interleave, (3) store. The Q4_K unpack and
  the reorder fuse into this one step; the runtime path is then pure GEMM.
- **"不能两份在显存":** the interleaved B replaces the original weight in HBM — one copy, not two.
- Consistent with the marlin note (a GPTQ/vLLM checkpoint's B is directly edible there), except actlize
  mixed-input wants its OWN interleave-256, not Marlin's layout — so the offline conversion is mandatory here.
- **The interleave is conditional on shape.** The official launcher picks `ColumnMajorInterleaved<256>` only
  when `n % 256 == 0 && k % 256 == 0` (and a16w4), else plain `ColumnMajor`. Our bench hard-codes the
  interleaved layout (fine for qwen's 4096-multiples); a general loader must branch on divisibility.

## Reference: the official cutlass3 launcher (for a real runner)

`cutlass3/fpA_intB_gemm_template_cutlass3.cu` is the mature version and the template for a real runner. Beyond
what our bench has:
- **split-k**: `cutlass::gemm::SplitKSerialScheduler` + `gemm_config.split_k_factor` (a real config axis;
  source of fpA_intB's `sm80:MxNxK:stages:split_k` naming).
- **LUT config selection**: `get_gemm_lut<T,WeightType>(device_info)` keyed by `{m,n,k}` → a `GemmLoopHelper`
  compile-time loop over `TypeCfgArray` configs, with `dispatchGemmConfig` heuristic as fallback. Validity gate
  `!(isFinegrained(QuantOp) && block_k < group_size)`. This is the grown-up form of our shape-keyed tactic
  cache + `W4A16_DISPATCH`.
- **arg layout is NOT swapped** (passes A,B at {m,n,k} directly, `stride_C = make_shape(m,0,1)` broadcast
  bias); example 16 (and our bench) use the swap-and-transpose formulation instead. Both work; ours is the
  one that compiled green.
- `EpilogueSimtVectorizedWithoutEvt`, `ClusterShape = WarpShape` (cluster unused on ppu1.0).

## Path to a runner (when this graduates from a bench)

Mirror acext: a `RunnerInterface` (void* ABI, stable across dtypes/QuantOps) + `Runner<T,WeightType,QuantOp>`
templated impl + explicit-instantiation .cu per {fp16,bf16}×{int4}×{per_col,fg_scaleonly,fg_scalebias}. Our
current in-binary tactic registry (`supported_configs()` + `W4A16_DISPATCH`) becomes the config-selection
layer; the shape-keyed tactic cache is the poor-man's version of acext's per-device LUT `.ini` files.
