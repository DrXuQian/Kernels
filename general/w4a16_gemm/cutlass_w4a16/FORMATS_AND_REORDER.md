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
GGUF/GPTQ/AWQ 4-bit format decodes to this on the host; the GEMM is then format-agnostic. gs=32 rides
actlize's **runtime** group path (`options.g`, StaticGroupSize=0), not a compile-time FinegrainedGs64/128 —
needs one validation run at g=32.

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

## Path to a runner (when this graduates from a bench)

Mirror acext: a `RunnerInterface` (void* ABI, stable across dtypes/QuantOps) + `Runner<T,WeightType,QuantOp>`
templated impl + explicit-instantiation .cu per {fp16,bf16}×{int4}×{per_col,fg_scaleonly,fg_scalebias}. Our
current in-binary tactic registry (`supported_configs()` + `W4A16_DISPATCH`) becomes the config-selection
layer; the shape-keyed tactic cache is the poor-man's version of acext's per-device LUT `.ini` files.
