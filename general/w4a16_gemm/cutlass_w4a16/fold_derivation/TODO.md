# Low-bit / bit-plane mixed-input GEMM: open items

Kept here because the list has so far lived only in conversation, which does not survive a context compaction. Numbers
are the session task ids, so the handoffs and commit messages that cite `#9`, `#17b` etc. still resolve.

## Where things stand

Dense, gs=16, `PPU_B_CHUNK=1`, M=2048 N=K=4096, PEAK 500 TFLOP/s, L=1:

| format | us | MFU |
|---|---|---|
| int1 | 224.96 | 62% |
| int4 | 227-228 | 60% |
| int2 | 248 | 55% |
| Q3 (int2+int1) | 261.48 | 52% |
| Q5 (int4+int1) | 267.97 | 51% |
| Q6 (int4+int2) | 281.62 | 49% |

MoE band, L=64 experts, ~128 rows each, **skewed** (arbitrary counts, 8 zero-row experts), N=K=2048, gs=32, from the
**336-row** product sweep (the earlier 30-row hand-written table is superseded and its verdict was wrong):

| format | best | us | MFU |
|---|---|---|---|
| **i2** | `64x128:64 w64x32 s3` | **300.26** | **55.5%** |
| q5 | `64x128:64 w64x64 s2` | 340.75 | 48.9% |
| q3 | `64x128:64 w64x64 s2` | 349.38 | 47.7% |
| q6 | `64x128:64 w64x64 s2` | 354.79 | 47.0% |
| i4 | `64x64:64 w64x32 s3` | 362.14 | 46.1% |

**int2 beats int4 by 17% in the MoE band**, the reverse of dense, and the reverse of the old sweep's verdict -- which had
int4 winning only because int4's row was s3 and int2's identical shape was s2.

**The optimal stage count is format- AND shape-dependent**, which no single hard-coded value could have found:
q3/q5 want s2, i2/q6 want s3, **i4 wants s4** -- and s4 appeared in no row of the old table.
`i4 64x128:64 w64x32`: s2 418.64 / s3 402.79 / **s4 378.23**. `q3 64x128:64 w64x64`: **s2 349.38** / s3 407.38 / s4 536.83.

**Nothing in the band is bandwidth-bound**: the compulsory floor is 5-29% of HBM on every one of 336 rows, with
`noreuse` 4.5-13.5x. The lever is occupancy/latency. And `mt`/`msk` are both non-predictive on 336 rows as they were on
30: the winner has neither the smallest m-tile count (TM=256 does) nor the least masking (TM=32 does).

Correctness: six formats x multi-expert, 22/22, max_rel exactly 0 (`test_lowbit_grouped`). Q6/Q5 dense 9 configs,
max_rel 4.88e-04 (`test_q65_bconcat_real`).

## Open

**#20 -- shrink the scale channel.** The biggest remaining budget, and it is TRAFFIC, not the 2.6% latency that #14
measured. `S/B = 32/(gs*bits)` is the scale bytes per weight byte: Q2_K 1.00, Q3_K 0.67, Q6_K 0.33, Q4_K 0.25,
Q5_K 0.20. Two steps: (a) fold the redundant zero for Q3_K/Q6_K by generalising the converter's `kBias` to a template
parameter -- `B=4` is exact at every int2 bpos (0x6404 / 0x5C10 / 0x5440 / 0x4D00); (b) compress the GGUF scale to
int8+d instead of widening it to fp16.
**The fp16 scale path must NOT be deleted.** fp16 IS the native scale form for GPTQ and AWQ, so it is the backup
whenever the weights do not come from GGUF. `ElementScale` stays a template parameter defaulting to `half_t`, and the
GPTQ regression in `real_weight/` is what proves nothing was traded away.

**#10 -- last-wave tail, ~11%.** Tile tuning cannot reach it; needs stream-K.

**#11 / #18 -- scale prefetch, and folding the dequant constants into one `hfma2`.** Both capped at 2.6% by #14's
measurement (`APG = gs/16`, so gs=16 forces APG=1 and it is nearly free anyway). Deprioritised: real, small, and the
traffic in #20 is the same channel with a 10x larger budget.

**Why int2 beats int4 by 17% in MoE while losing to it on dense.** i2 300.26 vs i4 362.14 at their own best configs.
i2 moves half the weight bytes for the same mma count, but the band is not bandwidth-bound (floor 5-29%), so that is not
the explanation. Next instrument is **acu on the two configs**, not more reasoning:
`MOE_ONLY="i2 64x128:64 w64x32 s3" MOE_ACU=1` and `MOE_ONLY="i4 64x64:64 w64x32 s3" MOE_ACU=1` each emit exactly one
launch. Note the two winners have DIFFERENT tiles, so also profile i4 at i2's shape to separate format from tile.

**#17b -- MoE band.** Instrument is shipped and the correctness half is closed. The sweep is now a 336-row product
across 128 generated translation units (one per shape) instead of 30 hand-written rows; `build.sh` takes `JOBS`. Read
`floor %HBM` as conclusive in one direction only, check that all units agree on `PPU_B_CHUNK`, and run `MOE_TK=128`
separately for the TileK half. The `%HBM` column that printed 116-181% was an upper bound whose A term assumed every
n-tile column re-reads all of A from DRAM; it is a compulsory FLOOR plus a `noreuse Nx` ratio now, and needs one
confirming run.

**TileK=32 is reachable for i4/i2/q6, and the code's comment saying otherwise was wrong.** `moe_grouped_ppu.cuh` carried
"TK=32 still won't compile (AIU needs TK>=64)" for a long time; the three folded configurations the delivery bound allows
build with zero errors through the front end that DOES fire the collective's static_asserts. The claim looked plausible
because **B's smem K-extent is `FoldF*TK`, not `TK`**: at TK=32 int2 folds by 4, so the run is 128 elements and the >=64
requirement lands on the folded extent.

This is the axis worth trying next, and the reason is the sweep's own verdict: nothing is bandwidth-bound (floor 5-29%),
so the lever is occupancy, and occupancy is driven by **A-smem = TM*TK*2** -- 4 KB/stage at (TM=64, TK=32) against 8 KB at
TK=64. It is also exactly the mechanism by which foldN paid off on dense for int1 and int2: the fold is what makes a small
TileK legal for B at all.

Row counts, from the predicate (both the python mirror and a static_assert probe agree):

| | rows | q3 | q5 | q6 | i2 | i4 |
|---|---|---|---|---|---|---|
| MOE_TK=32 | 168 | 0 | 0 | 42 | 42 | 84 |
| MOE_TK=64 | 336 | 42 | 42 | 84 | 84 | 84 |
| MOE_TK=128 | 304 | 38 | 38 | 76 | 76 | 76 |

**TileK=32 MEASURED, and it is the best band so far.** i2 `64x128:32 w64x64 s4` = **295.08 us (56.5% MFU)**,
i4 `64x128:32 w64x64 s4` = 317.26 (52.6%), q6 `64x128:32 w64x64 s4` = 357.71 (46.6%); q3/q5 filtered as designed.
Against TileK=64: i2 300.26 -> 295.08 (+1.7%), **i4 362.14 -> 317.26 (+12.4%)**, q6 354.79 -> 357.71 (-0.8%).
TileK=128 was worse across the board.

**The mechanism is stages, i.e. occupancy.** Every TileK=32 winner is **s4**, where TileK=64's winners were s2/s3 -- A-smem
= TM*TK*2 halves to 4 KB/stage, so a fourth stage fits. That is the same lever the 336-row sweep pointed at when it found
nothing bandwidth-bound.

**F is still NOT isolated.** TileK 64->32 changes F for all three formats (i4 1->2, i2 2->4, q6 (1,2)->(2,4)) at the same
time as A-smem, so i4's 12.4% cannot be attributed. The clean datum remains **i4 at TileK=64 vs 128**, where int4's F stays
1 at both (contig 32 B and 64 B, both >= 32) so only A-smem moves. That number is still needed to decide whether
`F > F_min` is worth opening as an axis.

q3/q5 self-filter to zero at TK=32 rather than breaking the build: their int1 plane needs `WN >= 4096/32 = 128`, and
w64x128's accumulator alone wants `WM*WN/32 = 256` registers per thread against a 256 ceiling. `moe_ok` now carries that
register bound (`WM*WN/32 <= 192`) so the dead config is a filter, not a compile error.

**foldN's coefficient is threaded but is NOT an axis.** The offline uses `fold::FoldTraits::F` and the kernel derives
`MOEG_FOLD` / `P2_FOLD` from the same closed form (verified equivalent: the kernel's
`P2_CONTIG = MOEG_RUN_B*P2_BITS/MOEG_BITS == TK*hi_bits/8`), so offline and kernel cannot disagree -- but `filter_and_run`
takes no `FoldF` parameter, so F is always its MINIMUM legal value and `F > F_min` has never been tried. It is a free
knob in the two quantities that usually bind: `b_smem = Ng*(F*TK*bits/8)` with `Ng = TN/F` is `TN*TK*bits/8`, INDEPENDENT
of F; and the delivery bound `WN*TK*bits >= 4096` does not contain F either. Opening it needs a template parameter on
`filter_and_run` plus relaxing `fold_traits.hpp`'s `contig_bytes*F == 32` to `>= 32`. Decide after the TK=32/128 halves,
which already move F as a side effect.

## DECODE batch=1: TileK is the only axis that moved, and the bus is still 3x from the roof

8 active experts x 1 row of L=64, N=K=2048, gs=32, PPU_B_CHUNK=1, i4. Traffic at decode is LOCKED (mt == active), so %HBM is
exact, not a bound.

| config | us | %HBM | run | kit |
|---|---|---|---|---|
| `32x64:32 w32x32 s4` (the first measurement) | 32.15 | 24.8% | 32 B | 64 |
| `16x32:32 w16x32 s4` (TileM=16, the smem-minimal corner) | 29.63 | 26.3% | 32 B | 64 |
| **`16x64:256 w16x32 s2`** | **23.54** | **33.1%** | **128 B** | **8** |

**Every tile / warp / stage knob together bought under 8%; TileK alone bought 21%.** Deep pipelines (s6/s8/s12) lost, TileN=32
lost, TileM=16 won 7.8% and TileM=8 is not buildable (every MMA atom has M=16). The roofline time is
21.55 MB / 2766 GB/s = 7.79 us, so 23.54 us is still **3.0x off the memory roof** on a shape whose AI is 3 FLOP/B against a
ridge of 181 -- it is latency or transaction efficiency, not bandwidth.

**TileK CONFOUNDS THE TWO CANDIDATES** and cannot separate them: 32 -> 256 takes the AIU contiguous run from 32 B to 128 B AND
the k-iteration count from 64 to 8. **The one experiment that separates them is FoldF at fixed TileK.** i4 at TileK=32 has
F_min = 2 (run 32 B); forcing F = 4 gives run 64 B with kit unchanged at 64. If that recovers about half the TileK=256 gain the
mechanism is transaction size; if it recovers nothing, it is the iteration count.

That is the `F > F_min` axis recorded earlier as untried, and it now has a purpose rather than being merely available. It needs
a `FoldF` template parameter on `filter_and_run` (default 0 = keep the current derivation) and `fold_traits.hpp`'s
`contig_bytes*F == 32` relaxed to `>= 32`. Both quantities that usually bind are indifferent to F:
`b_smem = Ng*(F*TK*bits/8)` with `Ng = TN/F` is `TN*TK*bits/8`, and the delivery bound `WN*TK*bits >= 4096` does not contain F.

Cheap intermediate while that is built: **TileK=128** (run 64 B, kit 16) as the midpoint of a 3-point curve. With the sweep
narrowed to `MOE_FORMATS=i4 MOE_TM_LIST=16 MOE_WM_LIST=16 MOE_TN_LIST=64` plus `MOE_STAGES_2` that is one kernel.

**COMPILE COST IS THE BINDING CONSTRAINT AT LARGE TileK, and it changes how these sweeps must be run.** On the box the
expensive stages are LLVM `opt` and `llc`, single-threaded, minutes of CPU per kernel with ~700 MB RSS -- and only 2 ran
concurrently during a 40-minute build, not the 192 the core count would allow. Compile cost scales with the unrolled mainloop,
i.e. with `MMA_K = TK/16` (2 atoms at TileK=32, 16 at 256), so the product sweep that is affordable at TileK=32/64 is not at
256. At minutes per kernel the budget is the KERNEL COUNT, not the unit count, and the right instrument is a few hand-picked
configs -- the ladder discipline this work used before -- not a product. My earlier "front end is 94%, codegen is 5.6%" was
measured with nvcc/ptxas and does not transfer to hgcc.

## acu on the decode winner, twice: the limiter is the GRID, and split-K must be paired with a smaller TileK

Two captures of the same shape family, `MOE_ONLY=<tag> MOE_ACU=1` (one cold launch, no warmup):

| | `16x64:256 w16x32 s2` | `16x32:256 w16x16 s2` |
|---|---|---|
| harness / acu duration | 23.39 / 22.77 us | **20.74 / 19.55 us** |
| DRAM Throughput | 33.39% | **38.94%** |
| Compute (issue) Throughput | 32.79% | 39.98% |
| Theoretical occupancy | 21.88% (14 warps/CU) | **28.13% (18)** |
| Achieved occupancy | 10.90% (6.97) | **21.33% (13.65)** |
| Block Limit SMem / Registers | 7 / 12 | **9 / 20** |
| Regs per thread | 148 | **102** |
| **Memory Dependency** | 0.451 | **1.015** |
| **Instruction Fetch** | **0.471** (top) | 0.433 |
| grid | (8,32,1) x (64,1,1) | (8,64,1) x (64,1,1) |

**Two model predictions confirmed, twice each.** `grid warps = mt*N*TM/(WM*WN)` predicts 14.2 warps/CU and acu measured
13.65; the smem expression predicts `Block Limit Shared Mem = 9` and acu measured 9. Both are now safe to reason with.

**THE STALL PICTURE INVERTED, and that is the useful part.** Memory Dependency went 0.451 -> 1.015 and is now 2.3x Instruction
Fetch, which was previously the top stall. The kernel moved from fetch/issue-limited to MEMORY-LATENCY-limited -- exactly the
direction wanted, since doubling occupancy put more requests in flight -- and DRAM followed, 33.4% -> 38.9%. Registers are not
a constraint anywhere near here: 102 used, and the register-occupancy curve is flat until ~168.

**THE GRID IS THE LIMITER AND TileK CANNOT MOVE IT.** achieved = min(theoretical 18, grid 14.2), measured 13.65. TileK does
not appear in the grid identity, so TileK=128 would take smem 26 KB -> 13 KB and theoretical 18 -> 40 while achieved stays at
14.2. Conversely split-K alone raises the grid to 56.9 and achieved stops at the smem-limited 18. **They must be paired:**

| | theoretical | grid | achieved |
|---|---|---|---|
| now | 18 | 14.2 | **13.65 (21%)** |
| + split-K S=4 only | 18 | 56.9 | 18 (28%) |
| + TileK=128 only | 40 | 14.2 | 14.2 (22%) |
| **TileK=128 AND split-K S=4** | 40 | 56.9 | **40 (62.5%)** |

One raises the ceiling, the other raises the floor, and neither alone gets past ~28%.

**#20 Phase 1 re-enters the picture here.** Dropping the zero tile takes smem/stage 13 KB -> 12.5 KB and `blk` 9 -> 10,
i.e. +11% theoretical. That was irrelevant at the previous config (theoretical was far above the grid) and is not now
(theoretical is only 27% above it).

Full decode progression, i4, 8 active experts x 1 row, N=K=2048, gs=32:

| config | us | %HBM | what changed |
|---|---|---|---|
| `32x64:32 w32x32 s4` | 32.15 | 24.8% | starting point |
| `16x32:32 w16x32 s4` | 29.63 | 26.3% | TileM=16 |
| `16x64:256 w16x32 s2` | 23.39 | 33.3% | TileK=256 |
| **`16x32:256 w16x16 s2`** | **20.74** | **37.5%** | WarpN=16, TileN=32 |

**-35.5% cumulative, and 2.66x from the memory roof (7.79 us).** TileM=32/WarpM=16 was in the sweep and LOST -- it doubles the
grid warps but also doubles A-smem and takes masking from 15/16 to 31/32 -- so of the two routes to occupancy only WarpN was
free, and WarpN=16 is the MMA atom floor. That is why split-K is now the only remaining lever rather than one option among
several.

## split-K REFUTED on the dense ladder: raising grid warps through K costs 9x

`test_fpA_intB_ppu 8 2048 2048 32` (m=8, the decode shape), int4, gs=32, scale-only. The ladder at `16x32x64/16x16/s2`:

| spk | GB/s | grid warps/CU |
|---|---|---|
| 1 | 185 | 1.8 |
| **2** | **214** | 3.6 |
| 4 | 197 | 7.1 |
| 8 | 129 | 14.2 |
| 16 | 64 | 28.4 |
| 32 | 24 | 56.9 |

`16x64x64` is the same shape of curve (211 -> 22). And **on the configuration that actually wins, TileK=256, split-K is
negative from S=2 onward**: `16x32x256` gives spk1 266 / spk2 251 / spk4 208 / spk8 129. Overall winner
`16x64x256/16x16/s2/**spk1**` at 273 GB/s -- so **split-K's contribution to the real winner is zero**; the small gain at
TileK=64/spk2 sits on a config already 30% behind.

**Mechanism.** Serial split-K serialises the S slices of one (m,n) tile at the EPILOGUE through the semaphore, so the
parallelism won in the mainloop is given back there; and every slice reads and writes the whole D tile, making
`D traffic = S * 2 * M*N*2` -- 2 MB at S=32 against 2.1 MB of weights, i.e. the total traffic roughly doubles on top of a
32-deep serial chain. A PARALLEL split-K (fp32 partials + reduction) pays the same in workspace traffic.

**This kills the "TileK=128 + split-K S=4 -> 62% occupancy" plan**, and with it the grouped split-K specialization -- several
hundred lines and multiple box rounds, cancelled by one dense measurement that needed no new kernel. That is why the cheap
dense ladder was the right first step rather than writing the grouped kernel.

**DO NOT OVER-GENERALISE THIS.** The ladder refutes *obtaining* warps through K-slicing, not occupancy as a lever. The
grouped kernel's 14.2 warps/CU come from 8 INDEPENDENT experts with no epilogue serialisation; the dense ladder's 14.2 come
from 8 slices of ONE tile, fully serialised. Those are different objects with the same warp count.

**Where that leaves decode.** 20.74 us, 37.5% of the memory roof, 2.66x off it. Every tile/warp/stage knob together bought
under 8%; TileK alone bought 21%; split-K is refuted. Within the grouped-GEMM structure decode is finished. Going further
needs a different STRUCTURE -- B from gmem straight to registers with no smem staging, blocks partitioning N, no masked mma
-- which is llama.cpp's `mul_mat_vec_q` shape and the one the PPU's own dense bf16 GEMV already runs at 82% of HBM. The
recorded gap is that this GEMV covers dense FFN and attention but NOT MoE experts (`mul_mat_id`, 3D), and llama.cpp's answer
to that same gap is one line: `channel_x = ids[channel_dst]`.

## Retracted

**"Q3 is 20.7% slower than Q5 in MoE and 27% on dense, and it is the only format whose LOW plane also folds (F1=2) --
a correlation across two regimes worth an acu investigation."** WRONG, and it was the GRID, not the format. On the
336-row sweep q3's best is 349.38 against q5's 340.75: a **2.5%** gap. The 20.7% came from the 30-row table measuring q3
at a configuration that suited it worse than the one it gave q5. Anything built on the F1=2 correlation should be dropped.

**A 23% CROSS-RUN DRIFT IS UNEXPLAINED, so only WITHIN-run comparisons are safe.** The identical config string
`q3 64x128:64 w64x64 s2` measured 429.19 us in the 30-row run and 349.38 us here, same data and same shape. Testable
hypothesis: the old sweep packed 64 experts per row (~1.2-1.5 s of host time) where this one packs once and memcpys
(~30 ms), so the old run gave the GPU a second of idle before every timing loop and these numbers are "hot clock" ones.
Until that is checked, do not compare any number here against a number from a different run -- including the dense MFU
table above.

## Closed, with the reason

**#9 -- chunk the conversion in N to relax the delivery bound.** Closed BY MEASUREMENT, no code. Q6's high plane is
int2, so Q6 can legally run `w*x32` today: 408.28 us against `w64x64`'s 361.53 -- **WN=32 loses 13%**. Halving
`accum = WM*WN/32` was the hoped-for occupancy win; it does not pay, because the n-tile count doubles while
`cvt/mma = 128/WM` is untouched by WN. int1 being pinned to WN=64 therefore costs nothing.

**#7 -- AIU write copy traits in cute.** Closed as SHOULD NOT BE DONE: both asm forms carry `.swzl`, so write-then-read
is a byte-level identity and the read atom's `LogicalTV` already IS write∘read. What the sub-byte offline compensates
for is the converter's fixed emission order, not the copy.

**#13 -- retire the legacy packers.** `test_fold_int2` was the last non-gate consumer and is migrated (one
`FOLD_CONFIGS` table generating the offline, the banner, the correctness launch and the perf launch, which were four
separate ladders that disagreed). The packers stay in `fold_derivation/legacy_pipeline.hpp` as the gates' INDEPENDENT
reference -- deleting them would make l58/l61/l64 compare the derived walk with itself -- and `build.sh` now fails the
build if any CMake-built source includes that header.

**#14 -- re-measure at gs=16.** Done; `APG = MMA_KA_/Scale_TileK = gs/16` is tile-independent, so gs=16 forces APG=1
and costs 2.6% on Q3 (int4 pays 10.8%: densest mma, least conversion, so the reload is relatively most visible).
