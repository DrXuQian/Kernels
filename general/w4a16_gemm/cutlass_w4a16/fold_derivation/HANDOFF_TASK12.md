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
