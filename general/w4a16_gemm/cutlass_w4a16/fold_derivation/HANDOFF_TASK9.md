# Task #9 — chunked B conversion. Everything needed to implement it, and one blocker I hit.

Written at the end of a long session. Nothing here is speculation: every number is measured or derived in a harness
in this directory. The code change is **not** in — a partial edit was reverted deliberately (see *Where I stopped*).

## What the change is for, in one paragraph

One swzl delivery is a fixed **16 bytes**, so it carries `D = 128/Bits` codes = **`A = 16/Bits` mma atom-slots** of B
(an atom's B operand is 8 fp16 per thread). Two consequences from the same constant:

* **int1's advantage** — one read feeds 16 atom-slots against int4's 4. The width-isolation run confirms the
  ordering: at the shared config `(32,128,64) w32x64 s2`, int1 **49.9%** > int2 48.1% > int4 45.9%.
* **int1's handicap** — the fp16 fragment must hold a whole delivery, so `MMA_N*MMA_K >= 16/Bits`, and since
  `B_regs = 4*MMA_N*MMA_K`, int1 is **forced to spend ≥ 64 registers on B**. int2 ≥ 32, int4 ≥ 16.

Those 64 registers are what push int1's best config over the power-of-two billing boundary:

| c | accum | A | B | S | total | billed | blk | warps/CU | cell |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 32 | 64 | 4 | 164 | 256 | 4 | 16 | bad — **today, measured 48.4%** |
| 2 | 64 | 32 | 32 | 4 | 136 | 256 | 4 | 16 | bad (does not cross) |
| **4** | 64 | 32 | **16** | 4 | **120** | **128** | 8 | **32** | **good** |

`accum = WM*WN/32 = 64` is the output and lives across the whole K loop, so B is the only movable term.
Chunking **decouples** "how many atom-slots the delivery covers" from "how many are fp16 at once": all 16 still get
converted and used, in `c` batches. **No wasted delivery, no change to the mma count or total converter work — only
the emission order.**

Target: `(64,128,64) w32x64 s2`, c=4. Expected ~54% against today's 50.2%, anchored on int4 52.7% / int2 47.8% at
ladder rung 3 plus int1's consistent 4–6 point margin over int2 on shared rungs.

## The two facts the implementation rests on, both verified

**Chunk axis is K, not N.** `cute::gemm(tiled_mma, tCrA(_,_,atom), tCrB_mma(_,_,atom), accum)` consumes one k-atom's
`(8, MMA_N)` = 32 fp16 per call, while `tCrB_mma` holds all `MMA_K=4` atoms = 128 fp16 = 64 regs. `MMA_N = MMA_K = 4`
so either axis gives 64→16, but K is far simpler: the mma loop is **already** per-k-atom (`k_loop` at
`ppu_mma_aiu_fold.hpp:721`), so the transform just moves inside it.

**The chunk predicate is compile-time static** (`l32_chunk_predicate.cu`, 0 mismatch on all three widths × MMA_N
4 and 2, pairs split exactly evenly). `tCrB_mma` is compact `(8, MMA_N, MMA_K)` so k-atom `a` owns the contiguous
range `[32a, 32a+32)` at MMA_N=4, and from `MixGemmEmit<1>`

```
e = bit4 + 2*b0 + 8*b1 + 16*b2 + 32*bit3 + 64*(v&1) + 4*(v>>1)
```

every term except `32*bit3` and `64*(v&1)` is below 32, hence

```
e / 32  ==  bit3(code) + 2*(vreg & 1)
```

For the int1 converter's 16 pairs (pair `t` carries codes `t` and `t+16`, which share `bit3`), the chunk is
`c = bit3(t) + 2*(v&1)`, i.e. 16 of the 64 (t,v) pairs per chunk:

| chunk | vreg | pair t |
|---|---|---|
| 0 | 0, 2 | 0–7 |
| 1 | 0, 2 | 8–15 |
| 2 | 1, 3 | 0–7 |
| 3 | 1, 3 | 8–15 |

This is why **#5 was a real prerequisite, not hygiene**: against the old hand-written offset table there is nothing
to gate on.

## STATUS: CORRECT ON HARDWARE. bad=0/131072 with the varying-scale probe.

```
[fold] ... TileShape=(64,128,64) warp=32x64 FoldF=4 | slots=128 delivery=128  [BITPACK]  [SVARY]
  fold int1 TK=64 (64,128,64) w32x64 vs host codes x scale(g,n): bad=0/131072 MATCH
```

Behind `PPU_B_CHUNK` (default off; the default path is byte-identical, and chunking is gated to 1-bit so int2/int4
take the unchanged path even with the flag on).

### Three bugs, all found by DECODING the printed values, none by reading the code

**1. 576 compile errors.** `transform_B_atom` called the int1 emitter unconditionally, so with the flag on it was
instantiated for `uint2b_t`. Gated now on `sizeof_bits<RealInternalElementB>::value == 1`, and every branch became
`if constexpr` instead of `#if` — with `#if` the other branch is never type-checked, which is how it reached the box.

**2. `bad=85545`.** Decoding each mismatching output against the probe's own
`scale(g,n) = 1 + (1/16)*((5n+3g) mod 13)` gave **g = 2 for every line** where g = 0 was correct. One smem stage is
`Scale_TileK = 2` groups, so it was a stage off-by-one, not a permutation. Cause: with `K_BLOCK_MAX == 1` the
`++smem_pipe_read` block fires **every** iteration and sits **before** the mma loop, so a per-atom transform placed in
that loop reads an already-advanced stage. Fixed by capturing `b_consume_stage` before the advance.

**3. `bad=57976`.** The pattern was `MMA_N` atom 0 correct and atoms 1–3 wrong — the signature of a wrong `MMA_N`
stride. Printing the layout (`l34_fragment_layout.cu`):

```
tCrB_mma : ((2,2,2), MMA_N, MMA_K) : ((1,2,4), 32, 8)
```

**`MMA_N` stride 32, `MMA_K` stride 8** — not the compact `(8, 32)` I had assumed. So `e = val + 32n + 8k` and
`e/32 == n_atom`, meaning the code was chunking by **N** while telling `cute::gemm` the buffer was one k-atom of
`(val, MMA_N)`. Corrected to `keep = ((e/8) % MMA_K) == Chunk`, `at = (val + 8*n_atom)/2`.

**The lesson is not the arithmetic.** `l32` had *verified* its split — correctly, of the wrong model. A harness that
confirms a wrong assumption is worse than no harness, because it reads as evidence. What was missing was ever
*printing* the layout being reasoned about. `l34` now does.

### Measured: registers 186 -> 142, real but not enough. And "the lost overlap" never existed.

acu on the chunked build at `(32,128,64) w32x64 s2`: **`Regs = 142`**, down from 186 — a real 44-register saving,
unlike the scale broadcast which measured as a no-op. The estimate tracks: measured = estimate + 22 at both points
(164 -> 186, 120 -> 142).

**But 142 still bills at 256, so occupancy is unchanged.** Power-of-two billing is confirmed rather than assumed: at
rung 4 acu reported `Regs=186, 128 thr/blk, theoretical 16 warp/CU`, and `131072/16 = 8192` regs/warp = 256
regs/thread. An 8- or 16-aligned model would predict 20 warps/CU there and is refuted by that point. So
`131072/(256*64) = 8` blocks x 2 warps = **16 warps/CU**, same as before.

**A CORRECTION.** Earlier notes in this file and in several commit messages said the first experiment "loses the B
copy/mma overlap, so only the register count is meaningful". That was wrong. At `K_BLOCK_MAX == 1`:

```cpp
if (K_BLOCK_MAX > 1) { ...register prefetch... }      // skipped entirely
auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;   // == 0 == k_block
```

so there is **no register-level prefetch to lose**. The original order is `copy_B(this tile) -> convert ALL -> mma
ALL`; the chunked one is `copy_B(this tile) -> (convert atom -> mma atom) x 4`. The copy is still hoisted; what moved
is the converter's ALU work, from all-before to interleaved-with-mma — which is *better* overlap, not worse.

**So this build's MFU is meaningful**, and acu's 263.26 us against the harness's ~274-276 us for the same config is
plausibly a real ~4% gain, with a mechanism: in a latency-bound kernel, interleaving converter ALU between mma issues
fills slots. It still needs the harness's own timing to confirm, since acu and the harness measure by different paths.

### The remaining lever: chunk A as well

To cross into the 128 bucket the estimate must reach <= 106 (measured = estimate + 22). It is 120 now, so 14 short.
The only movable term left is A:

```
tCrA = (8, MMA_M=2, MMA_K=4) = 64 fp16 = 32 regs      live across the whole K loop
per k-atom                   =  8 fp16 =  4 regs      saves 28
=> estimate 92 -> measured ~114 -> billed 128 -> warps/CU 32
```

A is *easier* than B: it is already fp16, so there is no converter, no chunk predicate and no scale — only
`copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next))` to move into the mma loop. And by
the same argument as above, at `K_BLOCK_MAX == 1` there is no A prefetch to lose either.

`accum = WM*WN/32 = 64` stays: it is the output and must live across the whole K loop.
