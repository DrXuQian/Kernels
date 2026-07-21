# Dense W4A16 prefill: actlize (PPU cutlass3) vs our marlin_gguf kernel, gs=128

This compares dense W4A16 prefill on ppu001 between:

- **actlize** — T-Head's PPU cutlass3 fork (`third_party/actlize`, pinned v1.0.0). Its mixed-input GEMM
  (`KernelAiuMultistageMixedInput`, fp16 × int4, group scale) run at **g=128, mode=1 (scale-only)**.
- **marlin_gguf** — our hand-written W4A16 kernel (`../marlin_ppu/marlin_gguf_ppu.cuh`), same fp16 × int4,
  gs=128 symmetric.

The two kernels each pack the weights their own way and each verify against their own reference; the
comparison is **time / weight-bandwidth at the same M/N/K**, not a byte-identical cross-check.

## What actlize does and does NOT already provide (v1.0.0 survey)

| piece | present? | where |
|---|---|---|
| W4A16 single GEMM on the AIU (ppu001) | ✅ | `KernelAiuMultistageMixedInput`, example 16 |
| grouped scale, runtime group size (g=128) | ✅ | `MainloopPPUAiuMixedInput`, `options.g` |
| static fine-grained group specializations | ✅ but only **gs 128 / 64** | `KernelAiuMultistageMixedInputFinegrainedGs128/64` (no Gs32) |
| grouped / array GEMM (MoE) | ✅ but **plain dtype, not mixed-input** | `ppu_aiu_gemm_array_group.hpp` |
| **W4A16 grouped GEMM (= MoE W4A16)** | ❌ **not combined** | the two mainloops don't share a schedule base |

So: the **dense** W4A16 path is ready-made here; the **MoE** W4A16 path is not — it's a port of the
mixed-input mainloop onto the array/group kernel.

## Build (needs the PPU SDK, not our nvcc)

actlize builds with the PPU toolchain — `hgcc` device compiler + the `hggc` runtime — driven by
`third_party/actlize/cmake/PPUToolchain.cmake`. This is a **different toolchain** from the bare `nvcc`
(no `-arch`) the marlin kernels use, which is why this bench is NOT a target in the marlin Makefile.

```bash
git submodule update --init third_party/actlize
# PPU_SDK defaults to /sim/eec/shared/junfu.qx/PPU_SDK (this box); set it only if elsewhere.
./build.sh
```

`build.sh` (1) applies `actlize_ppu001.patch`, which fixes only the hgcc arch-flag spelling: this box's hgcc
wants `-arch=ppu0010` verbatim, while the shipped v1.0.0 CMake emits `-arch=ppu_10`, which it rejects — an
unpatched build silently mis-targets and the runtime aborts ("probably a NV binary / Failed to query
occupancy"); (2) overlays `bench_cutlass_w4a16.cu` into actlize's `examples/` as a new example; (3) builds
just that target (`CUTLASS_PPU_ARCHS=ppu0010`, override with `PPU_ARCHS=`); (4) restores the submodule (patch,
example list, overlay) on exit, so the pinned submodule content stays clean.

## Run the comparison

```bash
# actlize side (this dir, after build.sh):
<build>/bench_cutlass_w4a16 --m=2048 --n=4096 --k=4096 --g=128 --mode=1 --iterations=100

# marlin side (built by the marlin Makefile, bare nvcc). It sweeps gs {128,32} x aff {false,true} itself:
cd ../marlin_ppu && make bench_marlin_gguf && ./bench_marlin_gguf 2048 4096 4096
#   -> read the gs=128, aff=false line (symmetric dense W4A16), same shape as the actlize --mode=1 run
```

Both print a `us` / weight-`GB/s` line; the actlize line is labelled `[CUTLASS gs=128]`. Divide bandwidth by
the **achievable** read bandwidth (~2200 GB/s from `bw_probe`), not the 2766 nameplate.

## Entry file

`bench_cutlass_w4a16.cu` is example 16 (`16_ppu_mixed_dtype_gemm`) **verbatim**, with exactly two changes,
both marked in-file: `MmaType` bf16 → **half_t** (our W4A16 is fp16), and the `Options` defaults
(mode=1, g=128, iterations=100, qwen35moe-ish shape). Staying this close to the shipped example is
deliberate — the number is only trustworthy if the actlize side is known-good code.
