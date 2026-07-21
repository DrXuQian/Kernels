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

`build.sh` (1) overlays `bench_cutlass_w4a16.cu` into actlize's `examples/` as a new example; (2) builds just
that target through actlize's proven example machinery (`CUTLASS_PPU_ARCHS=ppu0010`, override with
`PPU_ARCHS=`); (3) restores the submodule (example list, overlay) on exit, so the pinned submodule content
stays clean.

## Run the comparison

```bash
# actlize side (this dir, after build.sh):
<build>/bench_cutlass_w4a16 --m=2048 --n=4096 --k=4096 --g=128 --mode=1 --iterations=100

# marlin side (built by the marlin Makefile, bare nvcc). It sweeps gs {128,32} x aff {false,true} itself:
cd ../marlin_ppu && make bench_marlin_gguf && ./bench_marlin_gguf 2048 4096 4096
#   -> read the gs=128, aff=false line (symmetric dense W4A16), same shape as the actlize --mode=1 run
```

At a prefill M the GEMM is compute-bound, so the metric is **MFU vs the 500 TFLOP/s fp16 peak**, not %HBM
(that only binds at decode M~1). Both lines report it; the actlize line is `[CUTLASS gs=128]`.

## Result (2048x4096x4096, gs=128, scale-only)

Tuned, both actlize paths BEAT the hand-written marlin kernel:

| path | best tile | MFU |
|---|---|---|
| generic runtime-g (bench) | 64x64x**64** / s4 | **61%** |
| official finegrained (fpA_intB_ppu.cuh) | 64x64x**128** / s3 | **56.6%** |
| marlin gs=128 sym | — | 43% |
| actlize generic, stock 32x32 tile | — | 25% |

The generic path edges the official finegrained one by ~4.4 pts here: the official finegrained gs=128 path is
forced to block_k>=gs=128, a deeper K tile with worse occupancy, while the generic runtime-g path uses K=64.
Stages interact with that — s4 helps K=64 (61 vs 59) but hurts K=128 (46 vs 57), i.e. the extra buffer blows
shared at the larger tile. The 128x128x128 config is 12.9% (shared blown). Official runtime does NOT sweep;
it reads a per-device LUT keyed by {m,n,k} (offline-built) and falls back to an occupancy heuristic.

The stock 32x32 tile (2x2=4 warps) starves the MMA pipe; the win is entirely tile choice. Non-obvious: 64x64
is the sweet spot, not the biggest tile — 128/256 tiles undersubscribe the 72 CUs at these shapes, and 4
stages help 64x64 but collapse 128x128 (24%, shared/occupancy). 64x64/s4 is now the compiled default.

## Tuning the actlize tile

`bench_cutlass_w4a16.cu` exposes `TILE_M / TILE_N / WARP_M / WARP_N / STAGES` (defaults = the stock
32/32/16/16/3). build.sh forwards them from the environment to both the host and device compiles:

```bash
TILE_M=128 TILE_N=128 WARP_M=64 WARP_N=64 STAGES=4 ./build.sh
TILE_M=128 TILE_N=256 WARP_M=64 WARP_N=64 STAGES=3 ./build.sh
```

Sweep tile/warp/stages and read the `[CUTLASS gs=128]` MFU line. Not every combination will compile or run
on the mixed-input AIU mainloop — keep WARP a divisor of TILE and a multiple of the 16x16 MMA atom.

### Autotune — in-binary tactics (machete / fpA_intB style)

The binary now compiles a **fixed set of tile configs** (see `supported_configs()` / the `W4A16_DISPATCH`
if-chain) and selects one at **runtime** — no recompile to switch. Same model as `../machete_standalone` and
`../fpA_intB_standalone`. Build once, then:

```bash
./build.sh    # one build with all configs baked in
BIN=$(find "$PWD/../../../third_party/actlize/build_w4a16_compare" -name bench_cutlass_w4a16 -type f)

$BIN --list_configs                                                    # enumerate compiled tactics
$BIN --m=2048 --n=4096 --k=4096 --g=128 --config=64x64:32x32:s4        # force one
$BIN --m=2048 --n=4096 --k=4096 --g=128 --search_configs \             # in-process sweep, pick best, run it
     --save_tactic=tactics_ppu001.cache
$BIN --m=2048 --n=4096 --k=4096 --g=128 --tactic=tactics_ppu001.cache  # load best for this shape from cache
```

`--search_configs` times every compiled config in-process (skips any that don't verify), prints a table,
optionally writes the shape-keyed cache, and runs the winner. The cache is exact-match text
`m,n,k,g|config=<name>,tflops=<x>` — add a shape by searching it, load it by `--tactic`. To add/remove
candidate tiles, edit `supported_configs()` **and** the `W4A16_DISPATCH` arms (each arm is a compiled
instantiation), then rebuild once.

`sweep.sh` (build-time recompile-per-config) is kept as a fallback for exploring tiles not compiled into the
binary; the in-binary `--search_configs` is preferred for anything in `supported_configs()`.

## Entry file

`bench_cutlass_w4a16.cu` is example 16 (`16_ppu_mixed_dtype_gemm`) **verbatim**, with exactly two changes,
both marked in-file: `MmaType` bf16 → **half_t** (our W4A16 is fp16), and the `Options` defaults
(mode=1, g=128, iterations=100, qwen35moe-ish shape). Staying this close to the shipped example is
deliberate — the number is only trustworthy if the actlize side is known-good code.
