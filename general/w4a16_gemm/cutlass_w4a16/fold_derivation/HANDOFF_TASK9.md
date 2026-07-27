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

## STATUS: the chunked path is written, and the "assumptions" turned out to be derivable

Behind `PPU_B_CHUNK` (default off; the default path is byte-identical and both harnesses parse clean without it).

**A correction worth reading, because it was a process error.** The first version hardcoded `tCrB_load(_,_,Int<0>{})`
in `transform_B_atom` and then added `static_assert(K_BLOCK_MAX == 1)` to protect that shortcut — which made a
limitation of *my implementation* look like a property of the *design*. It is not. `tCrB_load` has **one slot per copy
step** (the copy writes `tCrB_copy_view(_,_,k_block_next)` while the conversion reads `(_,_,k_block)`), so deferring
the conversion into the mma loop cannot see overwritten codes, and "convert inside the loop instead of all at once"
works for **any** `K_BLOCK_MAX`. Fixed: `k_block` and `NChunk` are passed in.

And `CPY_K` never needed probing either — it is `slots / delivery` in fp16 units:

| | slots | delivery | `K_BLOCK_MAX` | k-atoms per copy step |
|---|---|---|---|---|
| int1 | 128 | 128 | **1** | 4 |
| int2 | 128 | 64 | 2 | 2 |
| int4 | 128 | 32 | 4 | 1 |

int1 at the target shape has `delivery == slots`, so one copy fills the whole fragment — `K_BLOCK_MAX = 1` is
arithmetic, not an assumption. Note int4 lands at 1 k-atom per copy step, i.e. `NChunk = 1`: nothing to chunk, which
is correct since its B is already only 16 registers.

The only constraint left is the emitter's own: 128 outputs must split evenly into `NChunk` chunks.

`PPU_MMA_PROBE=1` is kept as a cheap confirmation of the derivation on hardware, but it is no longer load-bearing:

```bash
PPU_DEFS=PPU_MMA_PROBE=1 TARGET=test_fold_int2 ./build.sh
FOLD_SVARY=1 FOLD_BITS=1 FOLD_TK=64 FOLD_BITPACK=1 $D/test_fold_int2 256 512 32
# expect: [mma probe] K_BLOCK_MAX(CPY_K)=1  K_ATOM_PER_COPY=4  MMA_K(tCrB_mma)=4  MMA_N=4  Scale_TileK=2
```

## Where I stopped originally, and the first blocker

I edited `MixGemmNumericArrayConverter<half_t, uint1b_t, 128>`'s emission loop to `if constexpr (kEmit(t)) _E(...)`
and **reverted it**, because:

> **`v` is a runtime loop variable** (`for (int v = 0; v < 4; ++v)`), so `if constexpr` cannot depend on it.

Two ways out, in preference order:

1. **Template the per-vreg emission.** Move the `_E` macro out of `convert` and add
   `template <int V> CUTLASS_DEVICE static void emit_vreg(uint32_t reg, uint32_t* h2)`, called four times with
   `Int<0..3>`. Then `if constexpr (Emit::in_chunk(t, V))` is well-formed. Costs a small restructure of one
   converter.
2. **Plain `if` instead of `if constexpr`**, relying on `CUTLASS_PRAGMA_UNROLL` over `v` plus dead-code elimination.
   One word of change, but the register benefit then *depends on* the compiler folding the branches — which is
   exactly the kind of assumption the scale-broadcast episode punished. If this route is taken, the acu check below
   is not optional.

There is also a **plumbing** question: the converter is selected by array size `N` through `convert_tensor`, so a
chunked variant cannot simply add template parameters to the existing specialisation. Cleanest is a separate
`convert_chunk<Chunk, NChunk>(in, out)` helper that `transform_B_kblock` calls, leaving `convert_tensor` and the
unchunked converter untouched.

## Mainloop wiring, and the one ordering hazard

`ppu_mma_aiu_fold.hpp` around lines 690–727 currently does, per `k_block`:

```
copy_B_and_extra_info(..., k_block_next, ...)      // load NEXT k_block's packed codes into tCrB_load
transform_B_kblock(..., k_block_next, ...)         // convert them immediately (whole delivery)
...
for k_loop in 0..K_ATOM_PER_COPY-1:                // consume the CURRENT k_block
    gemm(tiled_mma, tCrA(_,_,atom), tCrB_mma(_,_,atom), accum)
```

Moving the transform into `k_loop` means it must read the **current** `k_block`'s codes — but `tCrB_load` has already
been overwritten with `k_block_next`'s. `tCrB_load` is a single buffer. Options:

* move the copy to **after** the `k_loop` — correct with one buffer, but **loses the B copy/mma overlap**;
* double-buffer `tCrB_load` (it is only 4 registers per k_block, so 8 total) and keep the prefetch.

For the **first experiment** take the simple one and accept the lost overlap, because:

> **The MFU from that build is not the answer. Only the register count is.**

## The acu check that decides whether to continue

Run the first build and look at **exactly two numbers**:

* `Registers Per Thread` must fall into the **128** bucket (from 186 measured at c=1, estimate 164)
* `warps/CU` must read **32** (from 16)

If they do not move, the compiler was already staggering `tCrB_mma`'s live ranges across `k_block`s and the whole
idea is dead — **revert and stop**, do not spend a second round on it. Unlike the scale fragment, B's 128 values are
all *distinct* so the compiler cannot coalesce them, but it can still reorder, and that is precisely the untested
assumption.

Only if the registers drop is it worth restoring the copy/mma overlap and measuring MFU.

```bash
D=/sim/eec/shared/junfu.qx/Kernels/third_party/actlize/build_w4a16_compare/examples/99_kernels_w4a16_compare
A=/sim/eec/shared/junfu.qx/asight/bin/acu
cd /sim/eec/shared/junfu.qx/Kernels && git pull --ff-only origin ppu_dev
git submodule update --init third_party/actlize        # NOT optional -- see PPU_SCALE_FRAGMENT_API
cd general/w4a16_gemm/cutlass_w4a16
TARGET=test_fold_int2 ./build.sh
FOLD_SVARY=1 FOLD_BITS=1 FOLD_TK=64 FOLD_BITPACK=1 $D/test_fold_int2 256 512 32   # correctness FIRST
TARGET=test_width_acu ./build.sh
ACU_ONE=1 $A --set full -f -o chunk $D/test_width_acu 1 2048 4096 4096 32
```

## Method notes worth carrying, earned expensively this session

* **Print candidates beside ground truth; do not swap expressions and diff totals.** Four rounds went to the latter
  on `l31` and produced nothing; the former produced each answer in one step. Same technique that localised the
  ladder's 10 points and the int2 pairing bug.
* **A cute layout describes what the *program* asks for; whether the hardware does it is a codegen question.** For
  register-resident, fully-unrolled, provably-equal values the compiler usually wins first — that is why the scale
  broadcast measured as a no-op. Check that a cute-level redundancy survives to the ISA *before* trading idiom for
  it, which on this toolchain means acu (actlize's cute will not compile for device under nvcc, so the PTX cannot be
  read).
* **One passing config is not evidence.** `(32,128,128) w32x32` passed `l31` while six others failed, purely because
  with two warps the warp-order swap is a no-op.
* **Never assume `git submodule update` ran.** A stale submodule made an A/B compare two identical binaries with no
  trace in any log; `PPU_SCALE_FRAGMENT_API` now turns that into a compile error.
