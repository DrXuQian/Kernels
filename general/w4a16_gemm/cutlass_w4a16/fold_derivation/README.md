# Where the N-fold's limits come from

Four standalone programs. **None of them needs the box** — that is the point. They replace probe-fitting on
ppu001 with a derivation you can re-run in seconds, and they agree with every configuration ever measured there.

```
g++ -O2 -std=c++17 leg1_runword.cpp   -o leg1  && ./leg1
nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include leg2_frag.cu -o leg2 && ./leg2
g++ -O2 -std=c++17 leg3_predicate.cpp -o leg3  && ./leg3
g++ -O2 -std=c++17 ft_check.cpp       -o ftchk && ./ftchk
g++ -O2 -std=c++17 sweep_shapes.cpp   -o sweep && ./sweep
```

`sweep_shapes.cpp` is the "will this break the build" check. The kernel-side guard depends on nothing but
`(Bits, TileShape.K)`, so enumerating the distinct pairs the tree uses is exhaustive — section 1 does that, and
section 2 runs all 45 concrete `(Bits, TM, TN, TK, Stages, WM, WN)` tuples through the full `FoldTraits`,
including the invariants the guard leaves out. All 45 pass. Four sit exactly at the I1 boundary
(`delivery == slots`): correct today, no headroom, so lowering their TN would silently drop weights.

`stub_inc/` holds the `hggc*.h` headers cute pulls in through `cute/util/debug.hpp`. Nothing here runs on a
device — leg2 only asks cute to partition a layout on the host.

## The two legs

**leg1** is a copy of `ppu_tsm_ld_swzl_sim`'s `SWAP=true` branch (`cute/arch/copy_ppu0010_aiu.hpp:365-413`) —
`SWAP=true` is what the B operand instantiates, and it is the variant that hardware calibration corrected to
(see the w2a16 cube-width bug: the delivery is `[low N | high N]`, i.e. `src0,1` low and `src2,3` high). The AIU
write's `.swzl` and this read's `.swzl` are a matched hardware pair, so the two swizzle terms
(`^ vreg_line_idx%2`, `+ slice_start_vec`) cancel. What is left is the *logical* word — the gmem word, which is
what the offline relayout gets to choose:

```
row      = (v/2)*8 + lane/4
run-word = (v%2)*4 + lane%4
```

A row's 32 B run is a **2x4 grid**: `v%2` selects a 4-word half, `lane%4` selects the word inside it. Nothing
else reaches a run-word (checked over all 128 `(lane, v)` pairs).

**leg2** asks cute directly, via `partition_B` on the real `TiledMMA`, which logical `(n, k)` each thread
demands. The answer, for every tile shape tried: **N depends on `lane/4` and the value bits only.** `lane%4`
moves K at stride 2 and never touches N.

## The theorem

The four lanes of a `lane/4` group demand the *same* N, and they receive the four *different* words of a half.
So every word of a half must carry the same logical column — a folded column can never be narrower than a
half-run:

```
TK * Bits >= 128        (and F <= 2 follows, since F = 256 / (TK*Bits))
```

One-sided on purpose. A column *wider* than a run is fine; it just spans several 32 B slices, which is the
ordinary unfolded case (int4 at TK=128 is 64 B per column).

Smallest legal TK: **int1 → 128, int2 → 64, int4 → 32.**

**leg3** checks the predicate against all nine ppu001 reference points — seven that ran correctly and two that
returned garbage. Zero mismatches. **ft_check** does the same through `FoldTraits` itself, so the header's
`static_assert`s are what is being tested, not a restatement of them.

## What this rules out

`F=4` is **not** a converter limitation. The converter's bases `{0,32,2,34}` look like a two-way N split, and
the obvious fix is a four-way variant with `{0,16,32,48}`. It cannot work: at `F=4` a column is 8 B, so
`f = 2*(v%2) + (lane%4)/2`, and the third and fourth columns of a run are delivered to **different lanes**. A
converter only relabels registers inside one thread.

The practical consequence is that int1 is pinned at TK=128, so at gs=16 it carries `SK=8` where int2 and int4
carry `SK=4`. That lines up with int1's gs=16 gap (ScaleOnly 54.3% at gs=32 against 45.3% at gs=16, while int2 is
flat across the two), which points at the scale path rather than the fold — see the last section for the run
that would actually confirm it.

## Knock-on: which TK each two-plane (B-concat) format may use

Both planes of a B-concat share one `TileShape.K`, so the predicate has to hold for the **narrower** plane too.
That fixes the choice with no search:

| format | planes | TK=32 | TK=64 | TK=128 | verdict |
|---|---|---|---|---|---|
| Q6_K | int4 + int2 | int2 8 B ✗ | int4 F=1, int2 **F=2** ✓ | int2 F=1 (no fold) | **TK=64** — and the int2 plane does fold |
| Q3_K | int2 + int1 | ✗ | int1 8 B ✗ | int2 F=1, int1 **F=2** ✓ | **TK=128** |
| Q5_K | int4 + int1 | ✗ | int1 8 B ✗ | int4 F=1 (64 B col), int1 **F=2** ✓ | **TK=128** |

So Q6 gets what it needed — its int2 plane folds at TK=64, which is the 53%-class geometry. Q3 and Q5 are pinned
at TK=128 by their int1 plane, and at gs=16 that means `SK=8`.

## Open question this leaves, and the cheap way to settle it

int1 loses 9.0 points going gs=32 → gs=16 at a fixed tile (ScaleOnly 54.3 → 45.3, `SK` 4 → 8) while int2 loses
0.1 at its fixed tile (`SK` 2 → 4). The natural reading is that `SK=8` is where the FINE scale reload starts to
hurt, but that is an inference from two formats, not a measurement.

One box run settles it without touching the fold: **int4 at TK=128, gs=32 vs gs=16.** Same tile, same occupancy,
same converter — only `SK` moves 4 → 8. If int4 also drops ~9 points there, the cost is `SK`-driven and
format-independent, and the scale path is the next thing to work on. If it stays flat, something int1-specific
is going on and the 9 points are elsewhere.
