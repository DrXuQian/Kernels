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


## Step 1: the derived offline (`l20_derived_offline.cu`)

L18 pinned down why an offline relayout exists at all:

> the ONLY reason is that `MixGemmEmit != identity`.

fp16 needs none, because there the swzl read already delivers the mma operand order (`L2 o fragment = I`). For
sub-byte the converter sits in between and breaks that correspondence, and the offline is exactly the compensation.

So the placement is derivable: for each physical `(row, word, bit)` the chain says which logical `(n, k)` belongs
there — `LogicalTV` for the swzl, `MixGemmEmit` for the converter, `pi = frag.layout()^-1`, `partition_B` for the
mma — and the offline is that walk. The one non-positional piece is int4's `+8`, applied explicitly, since a layout
says where a code goes and not what it becomes.

**It is proven bit-identical to the shipped offline before anything is replaced**, on every configuration the tree
uses, fold and unfolded, several N-tiles and k-tiles:

| config | bytes differing |
|---|---|
| int1 fold (32,128,128) | 0 / 16384 |
| int2 fold (64,64,64) | 0 / 32768 |
| int2 fold wideN (64,128,64) | 0 / 32768 |
| int4 fold TK=32 | 0 / 65536 |
| **int4 (64,64,64) — production** | **0 / 32768** |
| int2 unfolded (64,64,128) | 0 / 16384 |
| int1 unfolded (64,64,256) | 0 / 8192 |

Bit-identical output means swapping the five steps for the derived walk is safe *by construction* rather than by
argument. The five steps stay in place until the box regression is green — the point of this file is that the
decision no longer rests on reasoning.

## The performance model, and two prescriptions it got wrong (`l26_convert_amort.cu`)

The tile sweep took int1 from 42.0% to 50.2% MFU, but it took two refuted predictions to get the objective right.

**Refuted #1 — "more atoms per k-iteration hides the scale reload better", so TK=256.** Measured 23.0-32.1%, and
**4.5%** with zero. The cause is the register file, not hiding: TK=256 costs 320 regs/thread (352 with zero) against
a 256 budget, so it spills. `TK` controls the register file, not overlap.

**Refuted #2 — minimise `regs_per_thread`, so `w16x64`.** `accum = WM*WN/32` depends only on the warp shape and the
delivery bound constrains only `WN`, so `w16x64` keeps int1's `TK=64` legal on 128 regs instead of 176. Measured
**39.7% against 50.0%** — fewer registers, comparable occupancy, 10.3 points worse.

**What actually separates all 20 points**, with no overlap and no crossing:

| group | measured | note |
|---|---|---|
| `WM=32`, regs ≤ 256 | 40.9 – **50.2%** | 10 points |
| `WM=16`, regs ≤ 256 | 31.9 – 39.8% | 8 points; the best of these has `blk=32`, the *highest* occupancy in the sweep |
| regs > 256 | 4.5 – 39.0% | spilled |

The best `WM=16` config has the highest occupancy in the sweep and still loses to the *worst* `WM=32` config
(`blk=4`). Occupancy orders configs within a group; it cannot cross between them.

**The mechanism, counted from cute rather than argued.** Every B fragment element must be converted (lop3 + fma) and
scaled before it can enter an mma. Per thread per k-tile:

```
mma instructions  = (WM/16)*(WN/16)*(TK/16)
B elems to cvt    = 8*(WN/16)*(TK/16)
cvt elems per mma = 128/WM            <- WN and TK cancel EXACTLY; only WM survives
```

Each B fragment feeds `WM/16` mma instructions down M, so `WM/16` is the converter amortisation factor. `WM=16`
means every converted element feeds exactly one mma. This is the quantised-B analogue of arithmetic intensity, at
the register file rather than at HBM — and it is why buying registers by cutting `WM` was exactly backwards.

### `cvt/mma` is a threshold, not a rate — and the register knee is not at 256

The prediction from the model above was that `w64x64` (`cvt/mma=2`, the only int1 shape that reaches it) would
collapse on its 272 estimated registers. It ran, and the result corrects the model twice over:

| config | cvt/mma | regs | blk | measured |
|---|---|---|---|---|
| `(64,128,64) w32x64 s2` | 4 | 176 | 13 | 48.4% |
| `(64,128,64) w64x64 s2` | **2** | **272** | 13 | **48.7%** |

Half the converter work *plus* 96 extra registers, at identical occupancy, nets **+0.3 points**.

**(a) `cvt/mma=8` is throughput-bound on the B convert path; `cvt/mma<=4` is not.** The signature is occupancy
sensitivity, over the same ~5× `blk` span at `regs<=176`, `TK<=128`:

| | `blk` span | MFU | spread |
|---|---|---|---|
| `cvt/mma=8` | 8 → 32 (4×) | 38.4 – 39.8% | **1.4 pts — flat** |
| `cvt/mma=4` | 4 → 23 (5.8×) | 40.8 – 50.2% | 9.4 pts |

A flat response to 4× the occupancy is what a throughput ceiling looks like. Once past it, there is nothing left for
a smaller `cvt/mma` to recover — which is why `WM=64` returns +0.3 and not another 10 points.

**(b) the register penalty is graded, and its knee sits above 272.** Against a same-`blk` 176-reg peer:

| regs | cost |
|---|---|
| 272 | +0.3 (nothing) |
| 288 | −2.6 |
| 320 | −19.9 |

256 was a guess that happened to land below the real knee. Treat >288 as the danger zone.

**Prescription:** reach `cvt/mma=4` (`WM >= 32`), stay under ~288 regs, then maximise `blocks`. int1's measured
optimum `w32x64 s2` at `TK=64` is exactly that.

**Retracted:** *"int1 is capped at `amort=2`, and int4 `(64,64,32) w64x32` is an untried lever."* The cap is real —
`WM=64` needs 272 regs at every shape legal under `WN*TK >= 4096`, asserted in `ft_check.cpp` — but it is
**harmless**, because `cvt/mma=2` is worth +0.3 points. int4's production config is already at `cvt/mma=4`, so there
is nothing to gain there either. Do not spend a box run on it.

## Step 2 was going to interleave `sS`/`sZ`. It should not be built (`l27_scale_contiguity.cu`)

The zero diagnostic pointed at the copy rather than the transform (the ScaleZero delta grows with `gs`: +49.6 us at
`gs=32`, +83.0 us at `gs=16`, while the transform count is `gs`-independent), so the plan was to interleave `sS` and
`sZ` into one tensor with a trailing extent-2 mode — one copy of twice the width instead of two. A 2×.

**The prerequisite checks out**: interleaving is a *loss* if the reads are already vectorized, because it splits a
contiguous run apart. Measured max contiguous run is **1 half in every config**, so nothing is broken by it. (A
permuted layout that tries to group a thread's own elements only reaches max-run 2 — my "transpose `n` from `(b,a)`
to `(a,b)` makes it 8" derivation was wrong about the per-thread `n` set.)

**But the number that makes it irrelevant**: one scale reload asks for 64–256 slots per thread to fetch 4–8 distinct
smem elements.

| config | slots asked | distinct | redundancy | fragment reg-halves |
|---|---|---|---|---|
| int4 `(64,64,64) w32x32` — production | 64 | 4 | **16×** | 16 for 4 values |
| int1 `(32,128,64) w32x64` — best | 128 | 8 | **16×** | 32 for 8 values |
| int1 `(32,128,256) w32x32` | 256 | 4 | **64×** | 64 for 4 values |

The cause is structural and sits in the partition layout:

```
((_1,(_2,_2,_2,_4)),_4,_1,_2):((_0,(_1@1,_8@1,_8@0,_16@1)),_32@0,_0,_1@2)
                                      ^^^^   ^^^^   ^^^^^
```

Three val modes walk mode-1 of the `(TN, 1, SK)` scale tensor — and that mode has **extent 1**. `sS` is
`k`-invariant, but `make_tiled_copy_B` builds the copy for the *full* B tile (`TN × TK`), so every `k`-walking mode
collapses to stride 0 and re-requests the same element. The fragment mirrors B's shape for the same reason, so the
replication is materialised in registers too — 4× in every config.

The replication is not an accident: it is what makes the transform a shuffle-free elementwise
`transform(tCrB_mma(_,_,atom), tCrS(_,_,0), ..., multiplies{})` against a B fragment of the same shape. The price is
up to 256 smem requests and 4× the scale registers.

**The fix that subsumes the interleave.** Keep the replicated *shape* the transform needs, stop materialising it:

```cpp
tCrS_ld = compact fragment of `ne` elements                                   // the copy targets this
tCrS_bc = make_tensor(tCrS_ld.data(), <B-fragment shape, stride 0 on k modes>) // handed to transform
```

**CORRECTION to the 16–64× figure.** cute's `copy(Copy_Atom<...>, src, dst)` already auto-filters: it computes
`nullspace(layout<1>(dst_v))` and divides both operands by it, so redundant iterations along the **destination's**
stride-0 modes are skipped — and this is in the general CopyAtom overload, not gated on `AutoFilter`, so the
collective's `copy(smem_tiled_copy_S, ...)` gets it today. The destination fragment distinguishes 32 registers, so
what is actually issued is ~32 requests for 8 distinct values, i.e. **4× redundant, not 16–64×**. The 128 figure is
what the partition *asks for*, not what the copy *issues*.

| | before | after |
|---|---|---|
| smem requests per reload | ~32 (of 128 asked) | 8 (**4×**, an upper bound) |
| scale registers | 16–64 halves | 4–8 (**4×**, definite, and tier-1 — TK=256's 320 regs included 32 here) |
| ScaleZero | two full copies | both halves shrink, beating the interleave's 2× on its own |

This also explains *why* the fix works mechanically and needs no hand-written filtering: making the destination's
nullspace bigger is exactly the lever AutoFilter keys on, and `zipped_divide` handles the correspondence.

It touches neither the transform, the B path, nor the offline — which makes it a *safer* change than the interleave,
not merely a bigger one. The earlier fused-FMA attempt regressed 52.3% → 33.5% precisely because it rewrote the
transform into a scalar loop.

One thing this could not settle locally: whether the PPU compiler already CSEs the redundant `ld.shared`. actlize's
cute gates `CUTE_HOST_DEVICE` and its global functors on `__HGGCCC__`, so it will not compile for device under nvcc
and the PTX cannot be counted here. The fix makes the question moot by removing the redundancy structurally.
