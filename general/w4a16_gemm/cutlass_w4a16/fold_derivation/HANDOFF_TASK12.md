# Task #12 — chunked B conversion on the 2-plane path. Recipe, with the hard part already settled.

Method, traps and the perf model: the `ppu-cutlass-mixed-gemm` skill. This file is only the recipe.

## Why this is the shipping target

int1 never ships standalone — it is the sparse plane of the GGUF bit-plane decomposition (Q3 = int2+int1,
Q5 = int4+int1; Q6 = int4+int2 has no int1). `TK` is set by the **sparsest** plane, so a combination containing int1
obeys int1's delivery bound `WN*TK >= 4096`: **Q3/Q5 at `WN=32` must run `TK >= 128`**. And `TK=128` is exactly where
chunking gained the most standalone:

| config | unchunked | chunked | Δ |
|---|---|---|---|
| `(32,128,128) w32x32 s2` | 46.5% | 56.6% | +10.1 |
| `(64,128,128) w32x32 s3` | 41.0% | 54.8% | +13.8 |

The standalone int1 63.7% is a *methodological* result — it proves the mechanism. The shipping impact is here.

## The hard part is done: ONE gate covers BOTH planes

`l37_2plane_as_layouts.cu` — the correspondence is a tuple of two layouts over one `(t, v)` domain (0/32 differ):

```
lo     = MixGemmEmit<2>::index(t, v)                       already a Layout
hi_src = 64*(v>>1) + 8*(v&1) + t   ==  Layout (8,(2,2)):(1,(8,64))
```

`l38_2plane_chunk.cu` — the consequence, verified for `NChunk` 2 and 4: each `_E2` line reads the low crumb **and**
the high bit and writes **one** `h2` slot, so **gating the line gates both planes**. Exact partition, every `h2` slot
written once per chunk, nothing straddles. **No high-plane predicate is needed** — the 2-plane gate is
`MixGemmChunkEmit<2, Chunk, NChunk>::keep/at` verbatim. That is what makes this a small change rather than a fourth
hand-derived index map.

## Two corrections found before writing any code

**(a) Why `TK=256` came back bit-identical, chunked and unchunked. Not a skipped branch inside the fold collective --
the fold collective is not selected at all.** `moe_grouped_ppu.cuh` picks the schedule from
`MOEG_FOLD = 32/(TK*Bits/8)`, and wraps in `KernelAiuFold` only when `MOEG_FOLD > 1`. int1 at `TK >= 256` already has a
32 B contiguous K-run, so `MOEG_FOLD == 1` and the plain `KernelAiuMultistageMixedInputFinegrained*` schedule is used.
`PPU_B_CHUNK` exists ONLY in `ppu_mma_aiu_fold.hpp`, so it cannot apply: the A/B compared the same non-fold kernel
twice. Nothing to fix in the chunk code. To chunk int1 at `TK=256` on the single-plane path the chunking has to go into
the non-fold collective too -- which is the same work as this task, since the 2-plane collective is also non-fold.

**(b) `FragL = decltype(tCrB_mma.layout())` -- what step 1 below used to say -- is NOT valid here, and the fold path
cannot see the difference.** `l39_2plane_frag.cu`: every fold-path config has `MMA_N == MMA_K == 4`, so the fragment's
`MMA_N` stride 32 is simultaneously `8*MMA_K` and `8*MMA_N` and **no fold measurement distinguishes k-inner from
k-outer**. At the 2-plane's locked `TK=256` the fragment is `((2,2,2),2,16):((1,2,4),128,8)`: `MMA_N` stride 128,
`MMA_K` stride 8, so one copy step's atoms span two n-groups 128 apart and `size(FragLayout)` is 256 against int2's
`kOut = 64`. `MixGemmChunkEmit`'s `static_assert(size(FragLayout) == kOut)` therefore rejects it, correctly: the
emission index space is ONE DELIVERY, not the whole fragment.

What the emission space actually is cannot be settled offline -- this collective's `tiled_mma` carries the builder's
`PermutationM/N`, and `tCrB_load` is partitioned through the **int8 `m16n16k32`** atom, not the fp16 `k16` one (hence
`CPY_K = 2` for the low plane at `TK=256`, and `P2_DIV = 2`). A `PPU_MMA_PROBE=1` block in the 2-plane collective now
prints `tCrB_mma`, `tCrB_load`, `tCrB_copy_view`, `tCrB2_load` and `cvt_in`, plus the `MMA_N` stride beside `8*MMA_K`
and `8*MMA_N`. **Run that first** -- the same "let the kernel report its own indices" ladder that settled the
single-plane gate after three wrong guesses. Note the existing 2-plane convert is numerically CORRECT (Q3 all MATCH),
so if `cvt_in`'s mode-1 stride disagrees with `tCrB_mma`'s `MMA_N` stride, the wrong object is my model, not the code.

```
PPU_DEFS=PPU_MMA_PROBE=1 TARGET=test_q3_bconcat_bench ./build.sh && ./test_q3_bconcat_bench 2048 4096 4096 16 2>&1 | head -30
```

## Recipe

1. **`fast_numeric_conversion_for_mix_gemm.h`, `MixGemm2Plane_uint2_uint1`.** Add
   `<int Chunk = -1, int NChunk = 1, bool Rebase = true, class FragLayout = ...>` and template the per-vreg emission on
   `V` (the `for (int v = 0; v < 4; ++v)` loop must go — `if constexpr` cannot depend on a runtime variable; this is
   the blocker that cost a round on the single-plane one). Then gate each of the 8 `_E2` lines with
   `if constexpr (MixGemmChunkEmit<2, Chunk, NChunk, Rebase, FragLayout>::keep(T, V))` and index
   `h2[MixGemmChunkEmit<...>::at(T, V)]`. `kOut` is **64** here, not 128.
   Keep the unchunked path delegating to `<-1, 1>` so the two cannot drift — same pattern as the single-plane
   converters.

2. **`ppu_mma_aiu_mixed_input_2plane.hpp`.** Mirror the fold collective:
   * `kBChunkMode` / `kBChunk` gate, `constexpr bool` + `if constexpr`, **never `#if`** (an `#if` left the other branch
     un-type-checked and shipped an int1-only emitter instantiated for `uint2b_t` — 576 errors)
   * `transform_B_atom<RealB, Chunk, NChunk, Rebase, FragL>` converting one k-atom into
     `tCrB_one = make_fragment_like(tCrB_mma(_,_,Int<0>{}))`, using `raw_pointer_cast(t.data())` before any
     `reinterpret_cast` (subbyte iterators), and `FragL = decltype(tCrB_mma.layout())` passed in, not restated
   * reuse `apply_scale_atom<FINE, APG>` — do **not** write a second copy of the FINE/APG_/reload rule
   * capture `b_consume_stage = smem_pipe_read` **before** the `++smem_pipe_read` block; at `K_BLOCK_MAX == 1` that
     block fires every iteration and sits before the mma loop
   * this collective has TWO packed sources live, so the packed cost is 8 registers not 4 — still negligible against
     the `4*MMA_N*(MMA_K-1)` fp16 saving

3. **Correctness first, then acu.** The 2-plane numeric harnesses are `test_q3_bconcat_*` / `test_q3_concat_real`.
   Use a **varying** scale (period coprime to 8/16/32 — a period-8 probe is blind to the displacements a broken
   fragment map produces; see `FOLD_SVARY` in `test_fold_int2.cu`). Only then measure.

## Expected, and how to falsify it

`B` drops from `4*MMA_N*MMA_K` to `4*MMA_N`. At Q3/Q5's forced `TK=128, WN=32`: `MMA_N=2, MMA_K=8`, so 64 → 8
registers, a saving of 56 — the same saving that moved the standalone `(32,128,128) w32x32` rows by +10 to +14.

It will **not** help if the 2-plane config's `cvt/mma` is 8 (i.e. `WM=16`): that axis is a throughput ceiling and
freeing registers underneath it measured **−0.5 … +1.0** across six standalone rows. Check `WM >= 32` first — if the
shipping 2-plane config runs `WM=16`, this whole task is worth nothing and should be dropped rather than measured.

---

# Per-plane N-fold (landed, UNTESTED on the box)

`Block_K` for the 2-plane path is no longer pinned to 256. Three pieces:

* **builder** `ppu_mma_builder.inl` — `DefaultOperandB2` gets `(Block_N/P2Fold, P2Fold*Block_K)`; `P2Fold` is the extra
  fold plane 2 needs on top of plane 1's.
* **collective** `ppu_mma_aiu_mixed_input_2plane.hpp` — `SmemLayoutB2` physical `(TN/P2Fold, P2Fold*TK, Stages)`,
  `P2Fold` read off the atom; `load_init_B2` folds shape AND stride in both branches; new `dB2`/`dB2_valid`.
* **caller** `moe_grouped_ppu.cuh` — builds `dB2` from `(n/P2_FOLD, k*P2_FOLD, L)` when `P2_FOLD > 1`.
* **bench** `test_q3_bconcat_bench.cu` — `pack_plane<..., FoldTN, FoldTK>` folds a plane when its run is under 32 B;
  six `BC128` rows beside the TK=256 sweep, which is unchanged and acts as the control.

`MmaPermK` needed NO change at Block_K=128: the non-fold rule gives `32*8/2 = 128 == TileShape.K`, so both rules
coincide there. **At Block_K=64 they do not** — that is Stage 2 and it also needs plane 1 to fold (F1=2), the shared
logical mma view, and a re-derived chunk gate (l41's `at_plain/4` is valid only at the non-fold `MmaPermK`).

## What to measure

`build.sh` does `rm -rf` on its build dir and emits to
`$ACTLIZE/build_w4a16_compare/examples/99_kernels_w4a16_compare/<target>` — it is NOT in the source dir, and the
script's closing `built: ...` line prints the full path. **Two builds write the same path, so the second overwrites
the first**: any A/B must copy the binary aside in between. (An acu capture of "the best config" that came back at
~50% instead of 63.7% is what this footgun looks like.)

```
BIN=/sim/eec/shared/junfu.qx/Kernels/third_party/actlize/build_w4a16_compare/examples/99_kernels_w4a16_compare
TARGET=test_q3_bconcat_real  ./build.sh && $BIN/test_q3_bconcat_real                 # numerics FIRST
TARGET=test_q3_bconcat_bench ./build.sh && $BIN/test_q3_bconcat_bench 2048 4096 4096 16

# A/B against the chunked build -- copy aside, or the second build eats the first
TARGET=test_q3_bconcat_bench ./build.sh && cp $BIN/test_q3_bconcat_bench /tmp/bench_plain
PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_q3_bconcat_bench ./build.sh && cp $BIN/test_q3_bconcat_bench /tmp/bench_chunk
```

The TK=256 `BC` rows must reproduce their previous numbers exactly — they are the control that the change did not
disturb the unfolded path. Only the `BC128` rows are new.

## Deferred, in order

1. **A-concat with fold** — the bench's single-plane `i1`/`i2` rows run UNFOLDED, so they are forced to their unfolded
   minimum `TK` (int2 128, int1 256) and measure 30.9% / 26.7% against records of 53.2% / 63.7%. int4 is the built-in
   control: its home `TK=64` is legal unfolded and it measures 53.2% against a record of 55.9%, the 2.7 points being
   gs=16. **So the bench's "B-concat wins 1.16x" verdict is invalid** — extrapolating the folded records gives
   A-concat ~474 us against B-concat's 823. Fix by giving `i1`/`i2` the same `pack_plane` fold treatment.
2. **Fuse (-1024, zero-point, 2^-b) into one (s', b') pair.** `w = h_raw*s' + b'` with `s' = s*2^-b`,
   `b' = z - 1024*s'`. Kills the per-atom `hmul2` and `hadd2` and the whole zero fragment: 4 ops/half2 -> 2 for
   ScaleZero, 3 -> 2 for ScaleOnly. Cost: `s'` varies with the slot's compile-time `b`, so the live count is
   (distinct b per chunk) x MMA_N -- **2** for int2, to be derived for int1. Register-neutral for ScaleZero
   (b' replaces zero), +1 register for ScaleOnly.
3. **Sign-magnitude encoding** (int1 plane = sign, int2 = magnitude). Merge becomes one XOR on bit15/31 AFTER the bias
   is removed -- order is load-bearing. The real win is that the low plane then uses the EXISTING validated
   single-plane int2 converter and `MixGemm2Plane_uint2_uint1` disappears. Generalises to Q5, **not** to Q6 (2-bit high
   plane is not a sign). **Hard constraint: symmetric +-0..3 is 7 values; Q3_K is 3-bit with a -4 centre, and -4 has no
   sign-magnitude representation.** So this is our own W3A16 format, not bit-exact GGUF Q3_K.
4. **Stage 2, Block_K=64** (F1=2, F2=4, WN must be 64). Needs plane 1's fold, the shared logical mma view, the fold
   `MmaPermK`, and the chunk gate re-derived onto the fold family (where `MixGemmChunkEmit`'s `right_inverse`
   composition is the correct one -- the two gates converge there).
5. **Q6 converter** (`int4 + int2`). Needs NO fold at all: at Block_K=128 both planes are F=1. It is the only format
   where B-concat and A-concat can both run at their best shape, so it gives the clean verdict on which wins.

## OPEN: plane 2's gmem tile rank (box build, 2plane.hpp:808)

```
error: no matching function for call to object of type Tensor<..., Layout<((32,256),1,1,1,int), ...>>
    copy_aiu(gmem_tiled_copy_B2, tB2gB2(_,_,_,k_iter_crd), tB2sB2(_,_,_,smem_pipe_write), warp_idx);
```

`tB2gB2` has FIVE modes where the slice passes four. Diagnosis: in the interleaved branch `mB2_nk` is
`(N2, (kCon=256, K2/kCon))` — K is a NESTED mode — and `local_tile` divides it by the tiler's K.

* plane 1 at `TK=128`: `128 < kCon`, so `(256, K/256)` splits into `(128, 2, K/256)` and the rest carries **two** modes
* plane 2 at `F2*TK = 256`: **exactly `kCon`**, so the rest carries **one**

The two planes therefore disagree in rank, and the `(_,_,_,k)` slice is hardcoded for plane 1's. `P2Fold == 1` keeps
them equal, which is why nothing showed before.

Candidate fixes, in order of preference:
1. Slice plane 2 rank-agnostically — derive the number of leading `_` from `rank(tB2gB2)` instead of hardcoding, e.g.
   take the last mode by `tB2gB2(repeat<rank(tB2gB2)-1>(_), k_iter_crd)` (check cute's spelling).
2. Give plane 2 a tiler K that does NOT coincide with `kCon`, so both planes split the nested mode the same way. This
   trades a compile error for a silently different gmem walk — only do it if 1 is impossible.
3. Flatten/coalesce plane 2's K mode before `local_tile` so the rest is always rank 1.

Check the single-plane fold collective first: `int1 @ TK=64, F=4` also lands on `FoldF*TKe == 256 == kCon`, so it has
already solved this exact case — mirror whatever it does rather than inventing a fourth answer.

**Local gates cannot see this.** The stub makes `ppu_mma_builder.inl` fail, so nothing downstream of `CollectiveMma`
instantiates and the whole mainloop is invisible to `syntax_check.sh` (it now says so out loud). l42's shape check
catches tile-EXTENT mismatches but not tile-RANK ones; extending it to compare `rank(local_tile(...))` per plane is
the cheap way to close that too.

---

# Cross-plane offline for the folded plane 2 — state, and the exact next step

## Established

* **l44 — the root cause of `bad=15010/32768`.** With different fold factors the planes disagree on thread→column by
  construction: the fold halves plane 2's physical row count and the mma maps threads to PHYSICAL rows. At
  `(64,64,128) w32x32` thread 0 holds low codes for logical N `{0,8,32,40}` and high bits for `{0,1,16,17}` — the data
  it needs is in another thread's registers. **No `hi_p` offset repairs this**; the guess
  `hi_p = base + g(ii, F2)` was the wrong shape of answer. It also explains why the `TK=256` control is right: both
  planes are `F=1` there and their thread→row maps coincide.
* **l45/l46 — feasibility.** Counts balance and the target map `T2 : plane-2 physical bit → logical (n,k)` is
  consistent (`conflicts=0`, `unclaimed=0`, multiplicity `== WOM`) at `(64,64,256) F2=1`, `(64,64,128) F2=2` and
  `(32,128,128) F2=2`, under the converter's TRUE pairing (not slot order):
  ```
  line (t,v):  LOW  code 16*v + (t%4) + 4*(t/4) [+8]   of the 64-code chunk
               HIGH bit  64*(v>>1) + 8*(v&1) + t [+16] of the 128-bit chunk
  ```
  So nothing is ruled out: a placement exists, and it lives OFFLINE — the kernel needs only the indexing noted below.
* **Multiplicity is not a conflict.** B is split across the N warps only, so all `TM/WM` M-warps read the same B and
  every element is legitimately demanded `WOM` times. Requiring 1 wrongly calls `(64,64,128)` infeasible.

## NOT established — the buffer emitter is wrong

`l46::emit_and_diff` produces a buffer that differs from the shipped one in **16384 of 16384 bytes** at the `F2=1`
control, where it must be identical. The cause is structural, not a typo: **l20's `tile_map` does not take the
destination address from `partition_B`.** It uses the explicit swzl TV formula

```cpp
row = inst*RPI + 16*warp_n + (v/2)*8 + lane/4;      wd = (v%2)*4 + lane%4;
```

and uses `part(pi(flat))` only to recover the LOGICAL `(n,k)`. l46 conflated the two. Adding `pi` to the
`partition_B` lookup (tried) does not fix it, because the destination side is wrong to begin with.

## Next step, concretely

Mirror l20's structure exactly, with plane 2's own `Ng = TN/F2`, `CPW = 32`, `RPI = WON*16`, `VEC = 128`:

1. **Destination**: `(row, wd, j)` from the swzl TV formula above — plane 2's, not plane 1's.
2. **Which high bit that is**: within the delivered chunk, vreg `v`, code `j` → bit index `h = 32*v + j`.
3. **Which line consumes it**: solve `h = hi_vreg0 + 2*(v'>>1)` for the vreg and `j = 8*(v'&1) + t + 16*half` for
   `(v', t, half)`, with `hi_vreg0 = (kb % P2_DIV) + P2_DIV*(ii / MMA_N2)`.
4. **The paired low code**: plane-1 chunk index `16*v' + (t%4) + 4*(t/4) + 8*half`, whose logical `(n,k)` comes from
   plane 1's own `part(pi(flat))` chain.
5. **Write** `m[(row*8 + wd)*CPW + j] = n_local*TK + k_local`, then run it through l20's buffer walk.

**Gate it on the control.** `F2=1` must reproduce the shipped plane-2 buffer byte for byte before the `F2=2` map is
trusted — that diff is what caught this, twice.

## Kernel-side change this assumes

```cpp
hi chunk   = cvt_hi(_, ii % MMA_N2)
uint32 off = (k_block % P2_DIV) + P2_DIV * (ii / MMA_N2)
```
`P2_DIV == 1 && MMA_N2 == MMA_N1` reproduces the shipped expression exactly, so the unfolded path is untouched.

---

# Block_K=64 (Stage 2) — root cause found, and it makes #12 a CORRECTNESS prerequisite

Box: `Block_K=256` bad=0 (control), `128` bad=0, **`64` bad=13689/32768**.

`l55_write_side.cu` — the side no earlier derivation modelled: where the CONVERTED fp16 is written.
`transform_B_kblock` aliases `tCrB_mma`'s registers through the LOAD fragment's layout:

```cpp
cvt_out = make_tensor(tCrB_mma(_,_,kb*K_ATOM_PER_COPY).data(), cvt_in.layout());
```

Valid only while `cvt_in`'s mode-1 stride equals `tCrB_mma`'s `MMA_N` stride. Measured, with the two
working configs as controls:

| config | tCrB_mma | cvt_in | |
|---|---|---|---|
| TK=256 unfolded | MMA_N=2 stride 128 | mode1=2 stride 128 | MATCH (bad=0) |
| TK=128 unfolded | MMA_N=2 stride 64 | mode1=2 stride 64 | MATCH (bad=0) |
| **TK=64 F1=2** | **MMA_N=4 stride 32** | **mode1=2 stride 64** | **MISMATCH (bad=13689)** |

Folding plane 1 puts `tCrB_mma` on the fold-in-N LOGICAL view: `MMA_N` doubles and its stride halves,
while `cvt_in` stays physical. The flat write then scatters into the wrong registers. The collective's
own comment warns about exactly this ("Write through the TENSOR ... NOT through a linear pointer") — it
does use a tensor, but with the wrong layout.

**The fix is #12.** `tCrB_one = make_fragment_like(tCrB_mma(_,_,Int<0>{}))` is a COMPACT single-atom
buffer, so a flat write into it is valid by construction, and the mma consumes `tCrB_one` directly. So
chunked B conversion is not a perf lever at Block_K=64 — it is required for the path to be correct at
all. Do #12 before measuring Stage 2.

Note the chunk gate must be re-derived for this shape: at `Block_K=64` `tCrB_mma` is
`((2,2,2),4,4):((1,2,4),32,8)`, the FOLD family, where `MixGemmChunkEmit`'s `right_inverse` composition
is the correct gate — not l41's `at_plain/4`, which was derived at the non-fold `MmaPermK`. l42 already
showed the two converge there.

---

# Rung 5 (Block_K=64): the launch-geometry class is swept clean

Rungs 1-4 are `bad=0` after capping the scale copy's thread extent. Rung 5 is the second, independent defect.

**The whole launch-geometry class audited, rung 4 (passes) vs rung 5 (fails)** — the class that produced the rung-4
bug, so worth sweeping rather than sampling:

| quantity | rung 4 | rung 5 | |
|---|---|---|---|
| warps / threads per CTA | 2 / 64 | 2 / 64 | unchanged |
| `aiu_warp_group_thread_idx = warp_idx*32` | {0,32} | {0,32} | unchanged |
| scale copy `ThrH*ThrW` vs threads | 64 <= 64 | 64 <= 64 | both fixed |
| `P2_DIV = CPY_K1/CPY_K2` | 1 | 1 | unchanged |
| kernel `N2_` vs model `NI2` | 2 = 2 | 1 = 1 | agree |
| kernel `NumIter` vs model `NI1` | 4 = 4 | 2 = 2 | agree |
| `stride1` (cvt_in mode-1) | 64 | 64 | unchanged |
| swzl `CUBE_H` plane1/plane2 | 128 / 64 | 64 / 32 | both have working precedents (single-plane int2 F=2 is 32, int1 F=4 is 32) |
| `cvt_out` flat write covering `tCrB_mma` | [0,256) bijective | [0,128) bijective | both correct |

Nothing in this class changes its relationship to the launch geometry between the two rungs.

**What is left cannot be settled locally, and precisely why.** Everything remaining is a place where the model and the
code share an assumption, so the reference and the subject come from the same derivation. l56's closed loop over the B
path is clean at rung 5, and l58 pins `plane_map` at **F=4** against `nfold_place_bits_int1_tk64` — the offline the
63.7% single-plane int1 config actually runs — byte for byte.

**The one link with no hardware-level reference at all:** plane 1's offline at `F1=2` with `TN=128 / WN=64`. int2's F=2
has only ever run at `(64,64,64) w32x32` (the 53.2% config); int1's F=4 has only run at `(64,128,64) w64x64`. The
combination "int2 folded AND TN=128/WN=64" has never executed on hardware.

So `PROBE=lo` / `PROBE=hi` is a two-way discriminator with a stated prior, not a fallback:

* `PROBE=lo` dirty -> plane 1's F1=2 offline at TN=128/WN=64 (the unreferenced link above)
* `PROBE=hi` dirty -> plane 2's cross-plane composition (`tile_map_int1`'s F2=4 branch)
* both dirty with the SAME deltas -> the physical->logical relation both planes share

`D[0][n]/MULT` prints the permutation itself, with no model in between — which is the only way to break an assumption
the model and the code hold in common.

---

# Measured state after chunking + w64x32, at BOTH group sizes

`test_q3_bconcat_bench 2048 4096 4096 <gs>`, PPU_B_CHUNK=1, PEAK 500 TFLOP/s, L=1 dense.

| | gs=16 (Q3_K's real granularity) | gs=32 | cost of gs=16 |
|---|---|---|---|
| B-concat (int3) | **262.19 us / 52.4%** | 255.47 / 53.8% | **+2.6%** |
| int2            | 247.90 | 233.76 / 58.8% | +6.0% |
| int1            | 224.62 | 215.23 / 63.9% | +4.4% |
| int4            | 234.16 | **211.33 / 65.0%** | +10.8% |

**gs=16 costs Q3 only 2.6%.** APG = gs/16 = 1 there, i.e. a scale reload from smem at every mma atom, and it is nearly
free -- so #11 (prefetch the next group's scale) and #18 (fold the dequant constants into one hfma2) are worth at most
2.6% on the shipping path and drop in priority. int4 pays the most (10.8%), which fits: it is the most mma-dense and
does the least conversion, so the reload is relatively most visible.

**"int4 is no longer the ceiling" was WRONG and is retracted.** It was merely untuned. With its own w64x32 row it goes
243.54 -> **211.33 us / 65.0% MFU**, +8.6 points, and is again the fastest of the four. The error was comparing against
a reference I had not tuned on the same grid -- the same shape as this session's other mistakes: trusting a quantity I
had not read off the thing itself.

So, tuned, at gs=32: int4 211.33 (65.0%) < int1 215.23 (63.9%) < int2 233.76 (58.8%) < B-concat 255.47 (53.8%).
B-concat / int4 = 1.21x at gs=32 and **1.12x at gs=16** -- a 3-bit two-plane GEMM for 12% over a single 4-bit GEMM.

w64x32 elsewhere: int1 at TK=128 improves to 219.70 / 62.6% but does not beat TK=64's 215.23; the three B-concat TK=128
rows do not beat TK=64 either, so the TK=64 winner stands.

---

# Q6 (int4+int2) and Q5 (int4+int1) -- implemented, locally gated, awaiting the box

Q3/Q6/Q5 are now ONE object, `MixGemm2Plane<LowBits, HiBits, Chunk, NChunk, Rebase, FragLayout>`. Q3's names are
aliases, so nothing downstream changed, and the builder and `moe_grouped_ppu.cuh` needed no change at all -- both already
derive their fold factors from the plane element widths.

```
kPairs     = 16/LowBits                                     half2 pairs per low vreg
kVregRatio = LowBits/HiBits                                 high/low vreg ratio inside ONE delivery
hshift(T,V)= HiBits * (T + kPairs * (V % kVregRatio))        where the high plane's bits sit
hi vreg    = kVregRatio * (V / kVregRatio)
at_plain   = MixGemmEmit<LowBits>::index(T,V) / 2            the destination is the LOW plane's own emission
```

At (2,1) this reproduces Q3's shipped `8*(V&1)+T` and `2*(V>>1)` exactly, and Q3's hand-written `AtLayout` turns out to
have been `MixGemmEmit<2>::index/2` all along.

**int4 was never a different scheme -- only a bias.** `MixGemmChunkEmit`'s mask and mul already matched; only `add`
needed `-(2^(10-bpos) + Bias)`, i.e. `0x8000 | ((25-bpos)<<10) | (Bias ? 1<<(bpos+3) : 0)`, which emits exactly the
0xE408 / 0xD480 the shipped int4 converter hardcodes.

**The int4 bias is a uniform offset.** With low = q & 15 and high = q >> 4 the converter emits `q - 8` for EVERY q, no
wrapping, so the caller folds `+8*dl` into the zero point. Q3 needed none of this because int2 carries no bias. The
offline must write `q & 15` through `place_derived` DIRECTLY -- the shim's +8 exists to reproduce the legacy pipeline.

**Two quantities were both called P2_DIV, and separating them cost a round.** `PDcopy = DL1/DL2` is the COPY STEP ratio
(drives `kb` and `base`); `VR = LowBits/HiBits` is the VREG ratio inside one delivery (drives `v2` and `j2`). They
coincide only when neither plane folds -- both are 2 for Q3 at Block_K=256 -- so the first version of the generalisation
left exactly that row passing while every other Q3 row changed. That reads like a subtle regression and is actually two
things sharing a name. The converter's member is now `kVregRatio`.

**Q6 and Q5 are structurally simpler than Q3.** int4's contiguous run is already 32 B at Block_K >= 64, so the LOW plane
never folds (F1 = 1, interleave-256 walk) and only the high plane does. Delivery bounds: low int4 needs WN >= 1024/TK,
Q6's int2 high WN >= 2048/TK (32 at TK=64, so **w64x32 is legal** -- the shape worth +7 to +9 points on the single-plane
sweeps), Q5's int1 high WN >= 4096/TK (64 at TK=64).

Local gates, all green before anything reached the box:

* `l65` five checks: int4's two constants, `at_plain == MixGemmEmit/2`, `hshift` and `hvreg` reproduce Q3's, and the
  pairing is a bijection for all three formats.
* `l66` the arithmetic EMULATED on the host -- `lop3` with immLut 0xEA is `(a&b)|c`, `ppu.fma.rtte.f16x2` is a half2 FMA
  -- requiring `lo + 2^LowBits*hi` for every (t,v,lane) of all three formats, plus Q3 identical to the old members for
  Chunk -1..7. Q6 and Q5 have no working reference anywhere, so this is the only local source of correctness for them.
* `l67` `tile_map_hi`: Q3 byte-identical to the original int1-only body at four configurations, Q6 and Q5 complete
  bijections at seven.
* The compile gate was VERIFIED rather than assumed: planting `static_assert(kLowBits == 99)` in the mainloop makes
  `test_q65_bconcat_real` report 72 errors, so the collective really is instantiated for every configuration.

## Box commands

```
PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_q65_bconcat_real  ./build.sh && $BIN/test_q65_bconcat_real
PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_q3_bconcat_real   ./build.sh && $BIN/test_q3_bconcat_real
PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_q3_bconcat_bench  ./build.sh && $BIN/test_q3_bconcat_bench 2048 4096 4096 16
```

`test_q65_bconcat_real` is synthetic with the FULL code range and a double-precision CPU golden: 5 Q6 configurations and
4 Q5. Q3's own test must still MATCH -- everything about Q3 is required to be byte-identical, and l66/l67 check that
locally, so a Q3 regression there would mean a plumbing change the local gates cannot see.

## Box results, gs=16, PPU_B_CHUNK=1

Q3 is CLEAN -- control plus all five rungs `bad=0/32768` against the native Q3_K golden, so the whole cute-ification
(d, a, c, b, e, f) and the converter generalisation are behaviour-preserving on hardware, not just in the local gates.

Perf, `2048 4096 4096 16`:

| | best | us | MFU | vs int4 alone |
|---|---|---|---|---|
| int4 (1 plane, 4-bit) | (64,64,64) w64x32 s3 | 230.73 | 59.6% | 1.00x |
| Q3 = int2+int1 | (64,128,64) w64x64 s2 | 262.15 | 52.4% | **1.14x** |
| Q5 = int4+int1 | (64,128,64) w64x64 s2 | 268.14 | 51.3% | **1.16x** |
| Q6 = int4+int2 | (64,128,64) w64x64 s2 | 281.93 | 48.8% | **1.22x** |
| int2 (1 plane) | (64,64,64) w64x32 s2 | 248.15 | 55.4% | |
| int1 (1 plane) | (64,128,64) w64x64 s3 | 224.73 | 61.2% | |

All three bit-plane formats land within **1.14-1.22x of a single 4-bit GEMM** while carrying 3, 5 and 6 bits. A-concat's
honest sum is 472.88 us, so B-concat/A-concat is 0.55x. All three peak at the SAME geometry, `(64,128,64) w64x64 s2`.

Q6/Q5 at Block_K=128 are far worse (417-441 us) than at 64, mirroring Q3.

## The Q6/Q5 numeric failure was the TEST, and the residual proved the kernel right

First Q6/Q5 run: every configuration `bad=32768/32768`, with values IDENTICAL across configurations and Q6's residual
equal to Q5's. `got - exp` came out -15.25, -106.73, -91.53, -76.25, -61.0, -45.75 for n=0..5 -- exact multiples
1,7,6,5,4,3 of -15.25, which is the shape of the `dl` generator `(&7)+1`. So

    got - exp = -16 * sum_k A*dl        with NO q dependence whatsoever

`apply_scale_atom` is `multiplies` then `plus`, i.e. `w = dl*emitted + zero`, and the test set `zero = -8*dl` while the
converter emits `q - 8`; the correct zero is `+8*dl`. Substituting gives exactly `-16 * sum_k A*dl`.

The useful part is what the residual's q-INDEPENDENCE forces: `sum_k A*dl*(emitted - q) = -8*sum_k A*dl` for every
(m,n), hence `emitted == q - 8` exactly, i.e. **the two-plane combination lo + 16*hi was already correct on hardware for
both Q6 and Q5**. A whole-output mismatch whose residual does not depend on the quantised data is never the kernel.

Added 12 bench rows around the two winners, including the w*x32 family for Q6 -- its int2 high plane needs only
WN >= 2048/TK = 32 and its int4 low plane WN >= 1024/TK = 16, so all of w*x32 is legal for Q6 and none of it was sampled.
Q5's int1 high plane pins it to w*x64 at Block_K=64.

## After (g)(h)(i)(j): both position chains are closed, and it cost nothing

Box, gs=16, PPU_B_CHUNK=1. Q6/Q5 numerics: 9 configurations, all bad=0/32768. Perf, before -> after the whole
cute-ification (g: pairing layouts shared, h: destination slice, i: descriptor extents off the tensor, j: scale/zero
coordinates):

| | before | after |
|---|---|---|
| Q3 = int2+int1 | 262.60 | 261.48 |
| Q5 = int4+int1 | 268.13 | 267.97 |
| Q6 = int4+int2 | 282.07 | 281.62 |
| int2 (1 plane) | 248.08 | 248.02 |
| int1 (1 plane) | 224.56 | 224.96 |
| int4 (1 plane) | 228.13 | 227.35 |

All inside noise, which is what an equivalence refactor should measure. gs=16 is the sharpest operating point for (j) --
APG = gs/16 = 1 there, i.e. a scale reload at EVERY mma atom, so ScaleSplit and ScaleThrDupL are exercised as hard as
they ever will be, and nothing moved.

WHAT IS NOW CUTE, END TO END. B operand: MixGemmEmit -> ChunkPlace -> right_inverse(frag.layout()) -> partition_B ->
CubeTV (partition_S + LogicalTV) -> HiPlaneSrc -> LoCodeL/HiCodeL/HVregL -> place_from_map's two destination layouts ->
MixGemmMmaPermK. Scale/zero: the copy view's (group, stage) as coordinates, ScaleSplit for the divide/mod, ScaleThrDupL
for the thread wrap. No rule is stated twice, no stride is hand-multiplied, and no multi-coordinate index is flattened
and then recomputed.

WHAT IS DELIBERATELY NOT CUTE, and should stay that way: the VALUE constants (mask/mul/add/bpos and the 0x64006400 base)
describe what a code becomes, not where it goes; the capacity bounds (F = 32/contig, RPS, WN*TK*Bits >= 4096) are
inequalities, not maps; AiuDesc is a hardware struct, and its inputs now come off the tensor.

TASK #7 SHOULD BE CLOSED, NOT DONE. A write-side LogicalTV would be the identity on the cube: both AIU instructions carry
.swzl and the two cancel, so the read atom's LogicalTV is already the map of write-then-read. Splitting that cancellation
would mean inventing a factorisation neither instruction exposes. Recorded on the traits themselves.

NOT SEEN: test_q3_bconcat_real's own output for this build. Q3 is the one format running real GGUF weights under a
byte-identity requirement, so it is worth one explicit look.

## #17 multi-expert: CLOSED for all six formats, max_rel exactly zero

`test_lowbit_grouped`, L=4 ragged and L=8 uniform, 11 rows each: **22/22 MATCH with max_rel = 0.000e+00**. Not "within
tolerance" -- bit-exact, which is the correct signature for a grouped-vs-L=1 oracle, since the two runs perform identical
arithmetic and differ only in per-expert addressing.

  Q3 / Q6 / Q5   two Block_K each  -- plane 2's own per-expert L-stride, a hand-written byte count
                                      (int64_t(N)*int64_t(K)*sizeof_bits<PlaneB2>/8), validated for the FIRST time
  int2           three configs      -- the actual gap; it had single-expert coverage only
  int1           one config         -- the HARNESS check, and it earned its place on first use (below)
  int4           one config         -- cross-checkable against test_moe_grouped_verify

WHAT THE SELF-CHECK ROW BOUGHT. The first run failed 11 of 11 INCLUDING int1, which has an independent passing
multi-expert gate, so "the harness is wrong, not the kernel" was available before any debugging. A second free signal
pointed at the reference: for e=1 the grouped run returned the same 38.1250 at L=4 and L=8 while the ORACLE returned
30.5000 and 34.3125, and both runs read the same A row (offs[1] == 64 in both). Root cause was mine --
cutlass::uint2b_t / uint1b_t / int4b_t all have sizeof == 1, so `ptr + e*K*N` advances BYTES, over-advancing 4x / 8x / 2x
and putting expert 1 past the end of the allocation. Eight earlier instances of that same failure class each cost a full
box round; this one cost one read, and the only difference was having a row whose answer was already known.

## build.sh's PPU_DEFS check was crying wolf, and the chunked grouped path is still unverified

The post-make check grepped make.log for `-D<define>`. The device compiles are add_custom_command with a COMMENT, so
make.log holds `[100%] [hgcc] foo.cu` and NEVER a compile line -- the grep could not succeed even when the flag was
present. So the warning on the passing run above is spurious, and, more importantly, whether that build had
PPU_B_CHUNK cannot be determined from it. **The grouped runs so far should be treated as UNCHUNKED until re-run.**

Fixed to read cmake's generated `CMakeFiles/<target>.dir/build.make`, which does carry the full command, and to check it
for THIS target's directory -- "cmake received the defines" is a weaker claim and was already covered separately.

## #17b measured: the tile is not the lever, cvt/mma is worth more than on dense, and #9 is closed

`test_lowbit_moe_bench`, L=64 experts, ~128 rows each, N=K=2048, gs=32, PPU_B_CHUNK=1, **skewed** rows (arbitrary counts,
8 zero-row experts, total=9941, Mmax=417).

**Verdict: two-plane q5 (64,128,64) w64x64 s2 = 355.50 us; single-plane i4 (64,64,64) w64x32 s3 = 382.76 us.** Q5 BEATS
int4 in the MoE band -- the two-plane overhead does not merely shrink here, it inverts.

**THE TILE IS NOT THE LEVER.** TileM 32/64/128/256 at fixed warpM=64: the winner TM=64 has neither the smallest m-tile
count (TM=32 has 256) nor the least masking (TM=32 has 8.1%). The two candidate explanations point at opposite ends and
the winner sits between them. The recorded weight-bound rule "minimise the TOTAL m-tile count" does **not** hold -- more
m-tiles measured faster. And since `A ~= mt*TM*K*2` with `mt*TM ~= total_rows` (so A is roughly TileM-independent) while
`B ∝ mt ∝ 1/TM`, the configurations that move the FEWEST bytes are the slowest. What is left is occupancy:
A-smem = `TM*TK*2` = 4/8/16/32 KB per stage across the sweep.

**cvt/mma IS WORTH MORE IN MoE THAN ON DENSE.** Matched (TileM, TileN), only warpM moving, so it is separated from the
m-tile count (the first version of this sweep confounded them):

| | w32x64 (cvt/mma 4) | w64x64 (cvt/mma 2) | |
|---|---|---|---|
| Q3 | 539.57 | 429.19 | +20.5% |
| Q5 | 476.30 | 355.50 | +25.3% |
| Q6 | 473.69 | 361.53 | +23.7% |

against eleven points on dense. Consistent: MoE collapses the useful A work, so conversion is a larger share of what is
left.

**#9 CLOSED BY MEASUREMENT, no code needed.** Q6's high plane is int2, so Q6 can legally run w*x32 today: 408.28 us
against w64x64's 361.53 -- **WN=32 loses by 13%**. Halving `accum = WM*WN/32` was the hoped-for occupancy win and it does
not pay, because the n-tile count doubles and `cvt/mma = 128/WM` is untouched by WN. So int1 being pinned to WN=64 costs
nothing and relaxing the delivery bound would unlock only worse configurations.

**Q3 IS THE OUTLIER IN BOTH REGIMES.** Same grid: Q3 429.19 vs Q5 355.50, i.e. Q3 20.7% slower in MoE against 27% on
dense. Q3 remains the only format whose LOW plane also folds (F1=2, where Q5 and Q6 are F1=1). That correlation now holds
across two regimes and two shapes, which makes it a hypothesis to test -- next instrument is acu on the two configs, not
more sweeping.

**The %HBM column was garbage in this run** (116-181%) and is fixed to a compulsory floor plus a noreuse ratio; see the
commit. On the worst row the floor is 17.6%, so "bandwidth-bound" is not established and this model cannot settle it.

## The MoE sweep was not a sweep, and the compile was serial. Both fixed by measurement.

**IT WAS 30 HAND-WRITTEN ROWS.** TileK appeared only as 64; TileN only as one value per family; WarpN only as each
format's minimum; and **stages was not an axis** -- it was baked into each row. That last one had already corrupted the
verdict: int4's single-plane winner ran s3 while int2's IDENTICAL shape ran s2, so `i4 382.76 vs i2 420.83` was measuring
stages, not formats. The `(TileM, WarpM)` grid was five points on an L-shape, so `(128,32)` and `(256,32)` were absent --
and "WarpM=32 is predictably worse because `cvt/mma = 128/WM`" is a PREDICTION, the same kind that was wrong about int4
not being the ceiling and about `w64x32` being unnecessary.

Now: the full `(TileM, WarpM) x TileN x WarpN x stages{2,3,4}` product, filtered by ONE constexpr predicate rather than a
hand-maintained list. **336 rows at TileK=64, 304 at 128** -- verified two ways that agree on a DISCRIMINATING case (a
python mirror of the predicate, and a static_assert probe compiled through nvcc: 42 rows per format-unit at TK=64, 38 at
TK=128). `FoldF` comes from `fold::FoldTraits` now; it used to be hand-passed as 2/4, 1/2, 1/4, which are the TileK=64
values, so TileK could not have become an axis without them silently being wrong.

**COMPILE PARALLELISM: THE CEILING WAS THE SOURCE COUNT, NOT `-j`.** `cutlass_build_dev_kernels` emits one
`add_custom_command` per `.cu`, so a sweep in one file is strictly serial however large `-j` is. Granularity was chosen by
measurement, not taste (nvcc front end, local):

| unit size | wall |
|---|---|
| 0 kernels (header only) | **5 s** -- the fixed cost |
| 3 kernels (one shape, 2-plane) | **25 s** |
| 21 kernels | 57 s |
| 42 kernels | 109 s |
| 30 kernels (the ORIGINAL single file) | ~80 s modelled |

Marginal cost is not linear: a 3-kernel unit pays 20 s for its first kernel because the whole collective instantiates
once, then ~2.5 s each. So the useful floor is one shape per unit. **128 generated units, one per shape (all three stage
counts inside), 112 non-empty, critical path ~25 s on a 192-core box** -- less wall clock than the ORIGINAL 30-row sweep
while instantiating 11x as many configurations.

**Stages deliberately stay INSIDE a unit**, because that axis is a runtime cost, not a compile one: the offline pack is
per SHAPE and stages do not change the shape, so all three share one pack. Measured: 31 ms per two-plane shape (19.2 int2
+ 11.8 int1), 24 ms single-plane, ~3 s over the sweep -- already the same order as the GPU time, so splitting stages would
triple it for nothing. TileM/TileN/WarpM/WarpN are all part of the shape, so splitting on those is free.

**THE TRAP THAT COST A ROUND: `if constexpr (false)` DOES NOT SUPPRESS INSTANTIATION IN A PLAIN FUNCTION.** It only does
so inside a templated entity. So `moe_ok` returning false did NOT make an illegal shape compile away -- three of four
probe units built clean and the `(TileM=32, WarpM=64)` one produced **99 errors** headed by
`gemm_operands.hpp: division by zero`, because `warpOnM = TM/WM = 0` and the collective builder degrades to `int` (the
failure already recorded in the skill for TM=16/WM=32). Fix: the unit body is a `template <int Dummy>` function.
**The evidence was already in hand** -- a static_assert probe I had written minutes earlier fired from inside a discarded
branch, which is the same rule. Only the consequence was missed.

Two guards kept for the split: each unit **votes on its own `PPU_B_CHUNK`** at static-init time and main reports the
tally, because a unit missing the `-D` would otherwise run the unchunked collective under a banner (compiled in a
different TU) saying chunked; and `MOE_UNIT_COUNT` is compiled in from the generator so a silently shrunk enumeration
shows up in the log.

Also: the pack ran once per EXPERT on byte-identical input -- 268 M positions per row to produce one row's worth of
information. Pack expert 0, memcpy the other 63.

## 336 rows: the verdict changes, and one recorded finding is retracted

Full product sweep, L=64 skewed, N=K=2048, gs=32, PPU_B_CHUNK=1, MOE_TK=64.

| format | best | us | MFU |
|---|---|---|---|
| **i2** | `64x128:64 w64x32 s3` | **300.26** | **55.5%** |
| q5 | `64x128:64 w64x64 s2` | 340.75 | 48.9% |
| q3 | `64x128:64 w64x64 s2` | 349.38 | 47.7% |
| q6 | `64x128:64 w64x64 s2` | 354.79 | 47.0% |
| i4 | `64x64:64 w64x32 s3` | 362.14 | 46.1% |

**The single-plane winner changed from i4 to i2 and the whole band got 21.5% faster** (382.76 -> 300.26), entirely from
configurations the 30-row table did not contain. **int2 now beats int4 by 17%**, the reverse of dense AND the reverse of
the old verdict -- which had int4 ahead only because int4's row was s3 while int2's IDENTICAL shape was s2. The old
comparison was measuring the stage count.

**The optimal stage count is format- and shape-dependent.** q3/q5 want s2, i2/q6 want s3, i4 wants **s4**, and s4 was in
no row of the old table: `i4 64x128:64 w64x32` is s2 418.64 / s3 402.79 / **s4 378.23**, while `q3 64x128:64 w64x64` is
**s2 349.38** / s3 407.38 / s4 536.83 -- a 1.54x spread on one shape. No single hard-coded value works.

**RETRACTED: "Q3 is 20.7% slower than Q5 in MoE and 27% on dense, and it is the only format whose LOW plane also folds
(F1=2) -- a correlation holding across two regimes, worth an acu investigation."** It was the GRID, not the format. q3's
best is 349.38 against q5's 340.75, a **2.5%** gap. The 30-row table measured q3 at a configuration that suited it worse
than the one it gave q5, and I read the difference as a property of the format. Everything built on the F1=2 correlation
goes with it.

**Nothing in the band is bandwidth-bound.** The compulsory floor is 5-29% of HBM on all 336 rows, `noreuse` 4.5-13.5x.
And `mt`/`msk` remain non-predictive at 336 rows exactly as at 30: the winner has neither the smallest m-tile count
(TM=256) nor the least masking (TM=32). The lever is occupancy/latency, so acu is the next instrument -- now reachable
without an edit-and-rebuild via `MOE_ONLY=<tag> MOE_ACU=1`, which issues exactly one launch.

**A 23% CROSS-RUN DRIFT IS UNEXPLAINED.** The identical config `q3 64x128:64 w64x64 s2` measured 429.19 us in the 30-row
run and 349.38 here, same data, same shape, same chunk state. Testable hypothesis: the old sweep packed 64 experts per row
(~1.2-1.5 s of host time) where this one packs once and memcpys (~30 ms), so the old run idled the GPU for a second before
every timing loop and these are "hot clock" numbers. Until that is checked, compare only WITHIN a run -- which is what
every conclusion above does.
