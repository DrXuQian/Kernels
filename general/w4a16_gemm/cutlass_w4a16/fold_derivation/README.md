# Where the N-fold's limits come from

Four standalone programs. **None of them needs the box** — that is the point. They replace probe-fitting on
ppu001 with a derivation you can re-run in seconds, and they agree with every configuration ever measured there.

```
g++ -O2 -std=c++17 leg1_runword.cpp   -o leg1  && ./leg1
nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include leg2_frag.cu -o leg2 && ./leg2
g++ -O2 -std=c++17 leg3_predicate.cpp -o leg3  && ./leg3
g++ -O2 -std=c++17 ft_check.cpp       -o ftchk && ./ftchk
g++ -O2 -std=c++17 sweep_shapes.cpp   -o sweep && ./sweep
./gen_guard_check.sh          # run this before pushing anything the box will compile
nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l2l3_layouts.cu -o l2l3 && ./l2l3
nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include leg5_perthread.cu -o leg5 && ./leg5
nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l5_slots.cu -o l5s && ./l5s
```

`l5_slots.cu` is the load-bearing one — it is the only probe that builds the builder's **real** `TiledMma` shape
rather than a stub, and correcting that is what settled the question twice over. `leg2`/`leg4`/`leg5` all use the
stub layout, which corresponds to `WN=64`; their layout algebra is fine but their per-thread numbers describe a
warp shape the fold tests do not use.

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

## The one hard limit

Both legs above are sound measurements, but the conclusion I first drew from them was not, and the fix came from
a third measurement: **`l5_slots.cu` builds the builder's real `TiledMma`** — `WarpOnN = TN/WN`,
`PermN = WarpOnN*16` — instead of the stub warp layout the earlier probes used. With that, the per-thread B
fragment is

```
slots = WN * TK / 32        measured over 12 configs; note it does NOT contain TN
```

`slots` being independent of `TN` is the crux, and it is why widening the tile never bought anything: B is split
across the warps in N, so a wider tile is more work per block, not a bigger fragment per thread.

One predicate then separates all nine ppu001 reference points, **9/9**:

```
delivery <= slots        i.e.   WN * TK * Bits >= 4096
```

Over-delivery is unrecoverable — a thread cannot use more codes than its fragment has slots, and the surplus is
never fetched. At the `WM=WN=32` every fold test passes, this reduces to `TK*Bits >= 128`.

## Two wrong turns, and what each got right

| version | claim | verdict |
|---|---|---|
| v1 | `TK*Bits >= 128`, therefore `F <= 2`, therefore int1 can never use TK=64 | right inequality, **wrong reason**, and the "never" was wrong |
| v2 | that bound is not real; the offline packer is what fails | right that int1@TK=64 is reachable, **wrong** that the bound is not real |

v1's error: LEG 2 shows the four lanes of a `lane/4` group demand the same *set* of N, and I read it as the same
*single* N, then concluded a folded column must fill a half-run. A thread demands many columns, so nothing forces
that. v2's error: it also used the stub warp layout, so its `cols_per_word` came out 2 for a config where the
real layout gives 1 — at `WN=32` the packer was never the binding constraint.

**Where the escape actually is:** `slots` scales with `WN`. Same int1 at TK=64 —

| WN | slots | delivery | verdict | cols/word |
|---|---|---|---|---|
| 32 | 64 | 128 | over-delivery — impossible | 1 |
| 64 | 128 | 128 | tight — **feasible** | 2 |
| 128 | 256 | 128 | under — feasible, with headroom | 4 |

The price is the thing v2 found, which is real but only bites once `WN > 32`:

```
cols_per_word = WN / 32     how many logical columns must share one 32-bit word
```

`nfold_regroup_gmem` moves whole `uint32`s (`dst[dst_w] = src[src_w]`), so it can only ever do 1. **int1 at TK=64
needs both a higher WN and a bit-granular packer** — the two constraints pincer it, and neither alone suffices.

Smallest legal TK:

| | WN=32 | WN=64 | WN=128 |
|---|---|---|---|
| int4 | 32 | 16 | 16 |
| int2 | 64 | 32 | 16 |
| int1 | **128** | **64** | 32 |

## What this rules out

`F=4` is not a converter limitation. The converter's bases `{0,32,2,34}` look like a two-way N split, and the
obvious fix is a four-way variant with `{0,16,32,48}`. It would not have helped: at `WN=32` the config is
over-delivering, so the data does not arrive at all, and a converter only relabels registers inside one thread.

## Knock-on: which TK each two-plane (B-concat) format may use

Both planes share one `TileShape.K` and one warp shape, so the bound has to hold for the **narrower** plane:

| format | planes | at WN=32 | at WN=64 (needs the bit-granular packer) |
|---|---|---|---|
| Q6_K | int4 + int2 | **TK=64** — int2 is the binding plane | TK=32 |
| Q3_K | int2 + int1 | **TK=128** — int1 is binding | **TK=64** |
| Q5_K | int4 + int1 | **TK=128** — int1 is binding | **TK=64** |

Q6 needs nothing new: its int2 plane already folds at TK=64. Q3 and Q5 sit at TK=128, and at gs=16 that means
`SK=8`, so the WN=64 route matters to them as much as to standalone int1.

## Open question, and the cheap way to settle it

int1 loses 9.0 points from gs=32 to gs=16 at a fixed tile (ScaleOnly 54.3 → 45.3, `SK` 4 → 8) while int2 loses
0.1 at its own fixed tile (`SK` 2 → 4). The natural reading is that `SK=8` is where the FINE scale reload starts
to hurt — but those int1 numbers are themselves unverified (the harness measured a different tile than it
checked), so this needs re-measuring before it means anything.

One box run isolates the mechanism regardless: **int4 at TK=128, gs=32 vs gs=16.** Same tile, same occupancy,
same converter — only `SK` moves 4 → 8.


## The index chains, as cute Layouts (`l2l3_layouts.cu`)

Three maps stand between a weight bit in gmem and the mma. Each is currently open-coded in a different file,
which is where every fold bug has hidden. Two of them are plain linear layouts, and the third is cute's own:

| | map | expressed as |
|---|---|---|
| **L2** | `(lane, vreg)` → logical 32-bit word in the cube | `((4,8),(2,2)):((1,8),(4,64))` |
| **L3** | `(bit, vreg)` → fp16 element of the converter's 128 | `((2,2,2,2,2),(2,2)):((2,8,16,32,1),(64,4))` |
| **L4** | fp16 element → logical `(n, k)` | `partition_B`, indexed linearly |

Each is checked against an independent model rather than a restatement of itself — L2 against
`ppu_tsm_ld_swzl_sim`'s own arithmetic, L3 against the sixteen `_E()` mask constants read straight out of the
instruction semantics. Both come out with **0 mismatches over all 128 (lane, vreg) / (bit, vreg) pairs, and both
are bijections.**

L3 is the interesting one. The converter looked non-linear because of its `off8 = {0,1,4,5,8,9,12,13}` lookup
table, but `off8[b] == b0 + 4*b1 + 8*b2` — so the whole dequant emission order is a layout. That is exactly the
thing [[ppu-swzl-cute-modelable]] argued should be modelled instead of the copy atoms: **the offline permutation
becomes derivable rather than probe-fitted.**

L4 needs no work at all: the converter writes through
`make_tensor(tCrB_mma(_,_,k_block*K_ATOM_PER_COPY).data(), cvt_in.layout())`, i.e. consecutive elements from a
pointer into a compact register fragment, so the converter's element index *is* `partition_B`'s linear index.

### What L5 still needs

Composing L2∘L3∘L4 and inverting gives the offline placement directly. The one missing input is from the
collective, not from cute: when a fragment takes several copy-atom instances (int1 at TK=128 takes two, since
`tCrB_load` is 32 B per thread while one swzl delivers 16 B), which `coord_h` / slice feeds each group of four
vregs. That is a read of `retile_D` and the copy loop, not a derivation.


## Before pushing: `gen_guard_check.sh`

`fold_traits.hpp` is plain C++ and the dispatch ladders are right there in the sources, so **a guard failure on the
box is always something that could have been found locally.** One was not: `CORR_DISPATCH(64,64,64)` at `fbits==1`
is int1 at TK=64 and WN=32, which over-delivers, and the ladder instantiates it regardless of the runtime `ftk`.
`sweep_shapes.cpp` section 1 missed it because that list is hand-written from reading the code.

`gen_guard_check.sh` extracts the instantiations mechanically instead:

* dispatch macros -> every sub-byte width crossed in, warp shape taken from *which* macro (CORR/FOLD at 32x32,
  BITPACK at 32x64). A rejected combination is fine **if** the macro gates on `fold::deliverable`.
* explicit `filter_and_run<...>` -> width is named and nothing gates it, so a rejected one is a hard failure.

The gate and the guard are the same expression — `fold::deliverable<Bits,TN,TK,WM,WN>` — so they cannot drift. A
hand-written equivalent condition (`TKV >= 128`) was the first fix and was replaced for exactly that reason.


## Before pushing anything the box compiles: two local gates

```
./gen_guard_check.sh     # every B-operand instantiation vs fold::deliverable
./syntax_check.sh        # nvcc front end over the harness sources
```

Both exist because a failure they would have caught reached ppu001 instead:

* `gen_guard_check.sh` — `CORR_DISPATCH(64,64,64)` at `fbits==1` is int1 at TK=64 and WN=32, which over-delivers,
  and the ladder instantiates it whatever the runtime `ftk` is. `sweep_shapes.cpp` section 1 missed it because that
  list is hand-written from reading the code.
* `syntax_check.sh` — `return bad == 0 ? 0 : 1;` inside a block above where `bad` is declared. An
  undeclared-identifier error in host code, i.e. the most local kind there is.

`syntax_check.sh` runs `nvcc -cuda`, which is the front end only: inline PPU asm is an opaque string at that stage,
so the file parses without an assembler for the target. `-D__HGGCCC__` is required or `CUTLASS_DEVICE` degrades to
host `inline` and `__syncthreads` lands in host code. The actlize headers then emit a fixed set of host/device
qualifier complaints that the real hgcc does not; those are ignored, and so is one stub artefact
(`CUTLASS_PPU_CHECK`'s `std::cerr <<` becoming ambiguous against the stub runtime). **Only errors attributed to the
source files count.** Verified in both directions: reintroducing the `bad` bug makes it exit 1, removing it exit 0.

It is a syntax gate, not a build. It says the file parses; it says nothing about the kernel being correct.
