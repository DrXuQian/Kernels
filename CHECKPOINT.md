# CHECKPOINT — 2026-08-01, before the restructure

A snapshot taken immediately before reorganising this tree for public release as
`DrXuQian/quactlize`. It records what is validated, what is in flight, what every switch does, and what must
survive the move. Numbers here are measured unless the line says otherwise.

---

## 1. Where the code lives, and the thing that blocks publication

| | |
|---|---|
| this repo | `Kernels`, branch `ppu_dev`, 980 commits, 185 MB of git |
| the collective | `third_party/actlize`, branch `ppu-w4a16-dev` — **a PRIVATE submodule (`DrXuQian/actlize`)** |
| the target hardware | T-Head PPU / ZW810, chip `ppu001`, **72 CUs**, 256 KB shared per CU, HBM peak ~2766 GB/s |
| the box | `aiswu96`, `/sim/eec/shared/junfu.qx/Kernels` — no PPU SDK locally, so nothing here builds for the device off-box |

**Most of the W4A16 work is inside actlize, not here.** `ppu_mma_aiu_multistage_mixed_input.hpp` is the mainloop;
this repo holds the harnesses, the offline, the probes and the docs. A public repo that submodules a private one is
unusable to anyone else, so the split has to answer that before anything is pushed. Three options, none yet chosen:
vendor the handful of headers actually needed, publish actlize as well, or release only the parts that do not need it.

Also unresolved: this tree is full of PPU-specific detail (`ppu.mma`, `ppu.ldmatrix.swzl`, the AIU bulk-load path,
`acu` counter names) and of box paths. Whether that may be published is not mine to decide.

## 2. Scale of the tree

| directory | files | source lines | destination |
|---|---:|---:|---|
| `general/` (mostly `w4a16_gemm/cutlass_w4a16`) | 1976 | 83,550 | split |
| `moe_ffn/` | 195 | 53,920 | split |
| `linear_attn/` | 92 | 18,483 | split |
| `studies/` | 45 | 12,182 | dev |
| `sampling/`, `helpers/`, `flash_attn/` | 24 | 12,577 | split |
| **`general/w4a16_gemm/cutlass_w4a16/fold_derivation/`** | **1676** | **227,875** | **dev only** |

`fold_derivation` is the derivation and probe area — l1..l98, the plan, the TODO, the syntax-check baselines. It is
larger than everything else combined and is a working record, not a product. It does not belong on `main`.

## 3. What is validated and would ship

* **W4A16 grouped GEMM (MoE) and dense**, int4 weights, fp16 activations, per-group scale and zero, on the PPU
  cutlass3 (actlize) mixed-input collective. Validated against real GPTQ and GGUF weights.
* **Sub-byte formats through bit-plane decomposition**: int2, int1, and the two-plane composites Q3 (int2+int1),
  Q5 (int4+int1), Q6 (int4+int2). All numerically MATCH against native goldens.
* **Q3_K and Q6_K drop the zero channel entirely** — the converter's own bias carries the format's centre
  (`kSymBias2Plane = 1 << (W-1)`), so a symmetric k-quant needs no zero plane at all.
* **GGUF Q4_K W4A16** at gs=32 with the affine min folded in for free.
* **The offline**: `dump_real_weights.py` / `dump_packed_scale.py` produce the fixtures; the reorder is derived from
  the fragment layout (`pi = frag.layout()^-1`) rather than hand-written.
* **The GGUF-native scale channel runs end to end on hardware** — `test_q4k_packed_gemm` rowA/rowB/rowC all MATCH
  with `PPU_PACKED_SCALE=1`, on real `blk.11.ffn_down.weight`.

## 4. What is in flight, and its honest state

The native scale channel (plan #20) works numerically and is **not worth shipping on performance**. The measurements,
one run, one binary set, six binaries verified byte-distinct, pinned row `16x128:256 w16x16 s2 S=1`, decode band
(L=64, top-k 8, N=K=2048, gs=32):

| variant | us | vs base |
|---|---:|---:|
| base | 23.67 | — |
| `PPU_SCALE_SWIZZLE=1` | 25.33 | +7.0% |
| `PPU_B_DEQUANT_NOP=1` | **21.05** | **−11.1%** |
| `PPU_PACKED_SCALE=1` | 24.23 | +2.4% |
| `+ PPU_PACKED_SCALE_NOP=1` | 24.78 | *invalid, see below* |
| `+ PPU_PACKED_SPLIT_GROUPS=1` | 26.92 | +13.7% |

**The single largest real term anyone has measured on this kernel is the int4→fp16 dequant pipeline at 11.1%**
(`base − bdqnop`). It is 43% of dynamic instructions and 11% of time, i.e. those instructions issue largely in
memory's shadow. That is the ceiling for TODO #18 and for every dequant idea.

Everything else measured is zero or negative. The swizzle is also **incorrect** — `test_moe_grouped_verify` with
`PPU_SCALE_SWIZZLE=1` alone dies with a device-side assert inside `copy_B_and_extra_info`. The `packnop` row is
**invalid**: it wrote the unit's own `d` into the scale plane, and `d` is subnormal for 80.5% of superblocks, so it
timed the hardware's denormal path. Fixed in the tree, not yet re-measured.

**Cross-run comparison is void.** `base` is 23.67 here against 20.11 in an earlier run of the same pinned
configuration — 17% apart, wider than the ~13% spread on record. So `pack − base` is +2.4% here and +12.9% there, and
the native-format tax is not determined.

## 5. Every switch, and what it is for

None of these belong on `main`. Recorded so the dev branch keeps their meaning.

| macro | purpose |
|---|---|
| `PPU_PACKED_SCALE` | consume the gguf's own 16 B scale unit instead of two pre-multiplied fp16 planes |
| `PPU_PACKED_SCALE_NOP` | timing-only: keep the transport and stores, drop the decode arithmetic |
| `PPU_PACKED_PAIR=0` | bisect: force the scalar per-group decode instead of the f16x2 one |
| `PPU_F16X2_EARLYCLOBBER=0` | bisect: restore `"=r"` on the two f16x2 asm operands |
| `PPU_PACKED_SPLIT_GROUPS` | eight warps decode four groups each instead of four decoding eight |
| `PPU_B_DEQUANT_NOP` | timing-only: drop the baseline int4→fp16 conversion and affine chain |
| `PPU_SCALE_SWIZZLE` | XOR the scale tile's address to take the read from 4-way to 1-way conflicted |
| `PPU_SCALE_PAD` | the additive alternative to the swizzle; measured and lost |
| `PPU_SCALE_PREFETCH` | prefetch the next group's scale; measured at 0.7% of a 7.3% channel |
| `PPU_A_PACK`, `PPU_A_CPASYNC` | collapse A's 15/16 padding at Mmax == 1 |
| `SK_QUANT` | 2 ScaleZero (ships), 1 ScaleOnly, 0 PerColScaleOnly — prices the scale channel by removing it |
| `SPLITK_CFG`, `SPLITK_S`, `SPLITK_ACU` | pin one bench row; **`SPLITK_ONLY` matches the whole tag and selects ~228** |
| `MOEG_SMEM`, `MOEG_DUMP`, `MOEG_CHECK` | grouped-GEMM diagnostics |
| `PPU_FORCE_INSTANTIATE` | make the local front-end check see units the main TU never builds |

## 6. Open questions, in priority order

1. **The swizzle asserts on hardware.** Reproduce on a named shape (`MOEG_*` narrows the verifier), then find the
   view. Every local check was on TN=128/Stages=2; the verifier sweeps others.
2. **Re-measure `packnop`** now that it no longer feeds the dequant subnormals. If the decode arithmetic really is
   free, the packed path's remaining cost is the 1024 explicit `tsm.st` per CTA per k-tile that cp.async gets for
   nothing — a 32-lane × 2 B store using 16 of 32 banks, which has a known fix.
3. **Repeated interleaved runs.** Nothing about the native-format decision can rest on numbers 17% apart.
4. **rowC's intermittency has no known cause.** It was bad=128, then 724, then MATCH across four commits with no
   semantic change, and restoring `"=r"` did not bring it back. Four hypotheses died: FTZ (refuted by simulating it,
   88.5% predicted damage against 17.7% observed), the `"=r"` aliasing, an out-of-range partition (the caller already
   wraps), and the paired-store race (real, but removed before the 724). Keep `PPU_PACKED_PAIR` for when it returns.
5. **TODO #18** — fold the dequant constants into one `hfma2`. Now bounded at 11.1%, the largest target left.
6. `dump_real_weights.py` has no `unpack_q6k`, so Q6_K's `−32` centre has never been checked against real weights.

## 7. What must survive the move

* `fold_derivation/PLAN_task20_scale.md` — the full history of the native-scale work, every refutation included.
* `fold_derivation/HANDOFF_packed_scale.md`, `fold_derivation/TODO.md`, `fold_derivation/README.md`.
* `fold_derivation/l91..l98` — the local gates. They are what makes a claim checkable without the box.
* `fold_derivation/syntax_check.sh` and its baselines — the local front-end check.
* `run_batch.sh` — the measurement recipe, with the traps built in.
* `real_weight/*.py` and the committed fixtures.
* This file.

## 8. The reference

`https://github.com/IST-DASLab/qutlass` — a pip-installable python package wrapping CUDA kernels:
`setup.py` (182 lines) does the build, `CMakeLists.txt` is 26 lines, the package dir holds `csrc/` with one `.cu`
per kernel plus `bindings.cpp`, and `tests/` and `benchmarks/` are python, one file per format. Its only submodule is
public NVIDIA cutlass. Nothing in this tree is packaged that way today: the harnesses are C++ mains driven by a
bespoke `build.sh` that overlays into actlize's example tree.
