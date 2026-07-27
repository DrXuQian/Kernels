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
