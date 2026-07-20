# MoE decode GEMV (Q4_K W4A16) on PPU — handoff

Branch `ppu_dev`, repo `DrXuQian/Kernels`, dir `general/w4a16_gemm/marlin_ppu/`.
Everything below was measured on **ppu001 (PPU-ZW810)**. Head at handoff: `ed82efb`.

> **Repo layout trap.** `/root/marlin_ppu/*.cu` is an UNTRACKED duplicate of the real tree at
> `.../Kernels/general/w4a16_gemm/marlin_ppu/`. Editing the copy compiles and looks fine and never
> reaches git. Work in the `Kernels/...` path.

---

## 1. What the kernel is

`gemv_w4a16_ppu.cu`, kernel `moe_gemv_rows` (LDG) and `moe_gemv_rows_aiu` (AIU staging).

MoE decode = grouped GEMV. Expanded rows = `tokens × topk`; `row_expert[r]` picks the B/scale plane,
`row_token[r]` picks the A row. Weights are Marlin-format INT4, GGUF Q4_K semantics (gs=32, affine).
Target model qwen35moe: FC1 N=1024 K=2048, FC2 N=2048 K=512, 256 experts, topk 8.

Grid = `(N/64) × SPLIT_K × ceil(n_rows/MOE_ROWS_PER_BLOCK)`. `N/64` is pinned by the B layout: one int4
covers exactly 64 columns × 4 k-values, so a narrower block re-reads bytes it discards.

Run it:
```bash
make gemv_w4a16_t128_aiu
GEMV_TARGET_BLOCKS=1024 ./gemv_w4a16_t128_aiu moe                    # default N=1024 K=2048 tokens=1 topk=8
./gemv_w4a16_t128_aiu moe 1024 2048 32 8 256                          # tokens=32
GEMV_GROUPSIZE=128 ./gemv_w4a16_t128_aiu moe                          # groupsize control
A=/sim/eec/shared/junfu.qx/asight/bin/acu
GEMV_NCU=1 $A --set full -f -o out ./gemv_w4a16_t128_aiu moe          # GEMV_NCU=1 => single launch, no check
```

**Current best: `gemv_w4a16_t128_aiu`, 1024 blocks, sk=8 → 10.14–10.64 us.** (Run-to-run ~5%.)
Correctness MATCHes at rel ~3e-04 for rows=8/16.

Note: an older standalone kernel (`marlin_moe_gemv_ppu.cuh` + `test_moe_gemv.cu`, T=32/sk=16/U=2) once
measured **9.11 us**, still the record, never reproduced on the merged kernel. Those two files are
otherwise superseded and were slated for deletion.

---

## 2. THE OPEN PROBLEM — read this first

`bw_probe` measures the machine's real achievable streaming read bandwidth:

| buffer | GB/s |
|---|---|
| 8–64 MB | 2600–4800 (**LLC hits, do not use**) |
| 128 MB–1 GB | **~2200 (the plateau — this is the denominator)** |

The nameplate 2766/2700 GB/s is NOT the right denominator. Divide by ~2200.

The kernel achieves:

| shape | data | time | GB/s | % of 2200 |
|---|---|---|---|---|
| tokens=1 (rows=8) | 8 MB | 10.6 us | 789 | 36% |
| tokens=32 (rows=256) | ~170 MB | 193 us | 901 | 41% |

**A ~900 GB/s ceiling that does not move with problem size, grid size, occupancy, or groupsize.**
Every optimization so far has hit it. That is the thing to explain and break.

### Leading hypothesis at handoff (NOT yet tested)

The AIU main loop is **single-buffered**:

```
issue 4 KB → aiu_wait0 → __syncthreads → compute → __syncthreads → issue next → wait → ...
```

Memory is idle while computing and the ALU is idle while waiting; they never overlap. With H=8 and 16
ktiles per slice there are only **two** batches, so the kernel is literally "fetch half, compute half,
fetch half, compute half". This matches acu's persistently dominant `Memory Dependency` stall, and it
explains why H=16 is worse than H=8 and H smaller is better — shrinking H partially fakes a pipeline
without ever building one.

**Proposed next step: double-buffer the AIU stage** (prefetch batch i+1 while computing batch i),
`MOE_AIU_STAGES=2`, shared cost 2 × 4 KB at H=8. Combine with smaller H so there are enough batches to
pipeline. `fa_ppu.cu` in the holmes/cutlass3 tree already does multi-stage AIU and is the reference.

The LDG path has the same disease in weaker form: `GEMV_UNROLL=2` means only 2 outstanding loads per
thread.

---

## 3. Measurements that CLOSED lines of attack

Do not re-run these. Each is a control, not a guess.

### Occupancy / warps — CLOSED
- LDG T=64: Achieved 26.21 warps/CU. `Block Limit Registers` 20 blk/CU vs grid supplying 14.2 → **grid-limited, registers slack.**
- LDG T=128: limit drops to 10 blk/CU → **register-limited.** Warps 31.05.
- AIU T=128: registers 90 → **64** (compiler's own choice), `Block Limit Registers` 10 → **16** = `Block Limit Warps` 16 = the T=128 ceiling. Warps **48.84**, occupancy 76.3%.
- **Warps +86% bought −4% time (10.57 → 10.14).** Occupancy is not the bottleneck.

### Registers / `__launch_bounds__` — CLOSED, and it was NEGATIVE
| build | time |
|---|---|
| `aiu` (no launch_bounds) | **10.14** |
| `aiu` + `GEMV_MIN_BLOCKS=12` | 11.14 |
| `aiu` + `GEMV_MIN_BLOCKS=16` | 10.44 |

AIU alone removed all the pressure; the compiler picked 64 regs unaided. `MIN_BLOCKS=12` *allows* 85 regs
— looser than what it already chose — so it only perturbed scheduling. **I predicted these would stack;
they did not.** The tell was visible before running: asking for 12 blocks/CU when the build already
achieves 16 is asking to go backwards. Targets have been deleted.

### Grid size — CLOSED
- `Waves Per CU = 0.89` at 1024 blocks (capacity 16 blk/CU × 72 = 1152 slots) → occupancy ceiling is 88.9%, not 100%.
- sk=16 → 2048 blocks → **12.01 us** (worse). No reversal even after capacity doubled from 10 to 16 blk/CU.
- tokens=32 → 4096 blocks, 3.6 waves, grid fully filled → still only **901 GB/s**.
- Grid can only grow via SPLIT_K (taxed by partial+reduce); `N/64` is pinned by the B layout and `rows` is already 1 block/row.

### groupsize gs=32 — CLOSED (my hypothesis, refuted)
| | time | HBM |
|---|---|---|
| gs=32 | 10.64 / 10.75 | 28.5% / 28.2% |
| gs=128 | **9.82** | 30.9% |

Only 8%. I had claimed ~40 points based on **dense prefill Marlin** (gs=-1 87.9% vs gs=32 47.2%) — that
number does not transfer to this GEMV; different kernel, different structure. Retracted.

### AIU itself — the T=64 vs T=128 control
| | LDG | AIU |
|---|---|---|
| T=64 (registers slack) | 10.57 | 10.89 (**worse**) |
| T=128 (register-limited) | ~10.6 | **10.23** (better) |

AIU wins *only* where registers bind. That is the control proving the gain is a register trade, not a
bandwidth one. (The dense GEMV AIU probe lost outright: 9.38 → 10.23; see the long comment above
`gemv_w4a16_aiu` in the source.)

### Earlier, also closed
`MOE_ROWS_PER_BLOCK`>1 monotonically worse; fused last-CTA reduce and fp32 atomics both slower than
partial+reduce; persistent grid no gain; `GEMV_UNROLL`=2 is the optimum; more blocks monotonically worse
at the old capacity.

---

## 4. Method (the user's standing instruction)

> **每次都要看 acu,要不然可能改动是要累加才有效果的**

Judge each change by **whether the metric it targeted moved**, not by wall time — one change can only
remove one constraint, and time only reflects the currently-binding one. Keep a change if its metric
moved and time didn't regress (the constraint is "banked"); roll it back if the metric didn't move (the
mechanism was misunderstood).

Counterexamples both directions, from this work: the int4 scale read zeroed bank conflicts for 1.2% time;
the cross-group hoist fixed warp starvation with no time change at T=64; and `launch_bounds` was assumed
to stack with AIU and was actually redundant *and* negative.

### Traps hit repeatedly (all real, all cost a cycle)
- `nvcc ... | grep error` followed by unconditional `echo BUILD OK` → pushed uncompilable commits. Use `set -e` + `test -f $BINARY` after `rm`-ing it.
- Header-only `-x cu` compile never instantiates templates and catches nothing.
- **Off-box front-end check**: `nvcc -arch=compute_80 -ptx` catches C++/template errors without running ptxas (which rejects `ppu.*` asm). Without `-arch` it defaults below sm_53 and every `__hfma2` "fails" spuriously.
- Duplicate/truncated Makefile targets from substring `replace` anchors (`moe_aiu:` is a substring of `bench_moe_aiu:`). `make` warns; read the warnings.
- Silent config substitution: unknown T / U / sk falling through to a default, so the change under test never ran.
- Verification polluting an ncu capture, or verifying a different template instantiation than the one profiled (a different SPLIT_K is a different kernel).
- Cache-warm timings. `%HBM` on a kernel whose cold and warm times are identical is meaningless.
- **Any percentage >100 or identically 0 means a broken denominator** — as here, where 2766 was the wrong one all along.
- Clock skew on the box ("modification time in the future") can make `make` skip a rebuild, which masquerades as "the change had no effect". `make -B` to rule out.

---

## 5. Build constraints

- **Build for ppu001 with NO `-arch`.** Any forced `sm_XX` routes to ppu0015 and rejects ppu001-only asm.
- 72 CUs, 256 KB shared/CU (hard max), 131072 regs/CU, 64 warps/CU, 4 schedulers, 1.7 GHz.
- fp16 tensor peak 500 TFLOP/s, int8 1000 TOP/s. **Achievable HBM read ~2200 GB/s** (not 2766).
- **5090 is not a performance proxy for PPU.** Correctness only.
- Weights must not be duplicated in HBM ("不能两份在显存").
- Do not modify `marlin_classic_ppu.cuh` — it is the one validated reference.

---

## 6. Not started

- Q6_K has no kernel (Q4_K_M uses it for some tensors).
- No validation against a real GGUF file — every check so far is self-consistent (our quantizer, our packing, our reference).
- `marlin_moe_gemv_ppu.cuh` + `test_moe_gemv.cu` are superseded and should be deleted once the merged path is confirmed.
