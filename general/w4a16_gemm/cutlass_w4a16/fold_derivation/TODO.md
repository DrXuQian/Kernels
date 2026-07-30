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

## WITHDRAWN: the dense split-K ladder was run on an EMPTY machine, so it refutes nothing

acu on `16x32x64/16x16/s2/spk1` at m=8 reports **`Size (1,64,1)x(64,1,1)` -- 64 CTAs on 72 CUs -- with DRAM Throughput
4.43% and Compute 7.00%**. Less than one CTA per CU: the machine is idle, and the 19.88 us is latency on an empty device.
Every conclusion below was drawn against that baseline and none of them stands.

**Why the shape was wrong.** m=8 with TileM=16 gives `mt = ceil(8/16) = 1`, while grouped decode has `mt = 8` (eight
experts). That factor of 8 is the entire difference between grouped's 512 CTAs and this test's 64, so the ladder compared
split-K against a pathological baseline rather than against the loaded regime the grouped kernel runs in. The traffic-vs-
serialisation decomposition (1.37x traffic, 4% serialisation at S=8) rests on the same bad baseline and is withdrawn too.

**The shape that CAN answer it is m=128.** `mt = ceil(128/16) = 8` and `ntile = 2048/32 = 64` gives **512 CTAs**, and
`gw/CU = mt*n*TM/(WM*WN)/72 = 8*2048*16/(16*16)/72 = 14.2` -- exactly the 13.65 acu measured on the grouped decode winner.
So m=128 reproduces the grouped grid and occupancy at spk1, and only from there does the ladder measure what split-K adds:

    $BIN/test_fpA_intB_ppu 128 2048 2048 32     # then read gw/CU from 14.2 upward

GB/s still climbing => occupancy remains a lever that grouped cannot reach (its 14.2 already exhausts WarpN>=16 and
TileM/WarpM<=2), so grouped split-K is worth writing, parallel with fp16 partials. GB/s already peaked at spk1 => saturated
near 14.2 warps/CU and split-K has nowhere to go -- but concluded on the right baseline this time.

## (withdrawn, kept for the record) split-K "REFUTED" on the dense ladder

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

**Mechanism, DECOMPOSED -- and the first version of this attributed it to the wrong term.** `wbytes` in the harness is a
constant, so GB/s is exactly inverse time. With output elements `E = mt*TM*N = 32768` and a baseline of ~2.43 MB
(weights + scale + A), serial split-K adds `E*2*(2S-1)` of D traffic:

| S | traffic ratio | measured time ratio | residual = serialisation |
|---|---|---|---|
| 2 | 1.06x | **0.86x (FASTER)** | -- |
| 8 | 1.37x | 1.43x | **1.04x** |
| 16 | 1.79x | 2.89x | 1.61x |
| 32 | 2.63x | 7.71x | 2.93x |

**At S=8 the serialisation costs 4%; the whole 43% is PARTIAL TRAFFIC.** So a PARALLEL split-K with a separate lightweight
reduction -- which removes only the serialisation -- would land at ~1.37x slower, not better. And its traffic is not lower:
per output element, serial is a fp16 read+write per slice (~4S*E), parallel with fp16 partials is a write per slice plus one
read by the reduction (~4S*E, IDENTICAL), and parallel with fp32 partials is ~8S*E, i.e. TWICE serial. The lightweight reduce
removes a term that was already negligible at the useful S.

**S=2 DOES win, and the first version of this missed it**: 185 -> 214 GB/s at TileK=64, +16%, because the occupancy gain
1.8 -> 3.6 warps/CU beats a 6% traffic cost. So split-K is not useless -- it is useful only at S=2, and only where the kernel
is latency-starved. On the configuration that actually wins it is negative from S=2 onward: `16x32x256` 266 -> 251 (-6%),
`16x64x256` 273 -> 259 (-5%). Consistent reading: **TileK=256 already removed the latency starvation (kit 8 rather than 32),
so split-K has no occupancy left to buy there and only traffic to pay.**

**This kills the "TileK=128 + split-K S=4 -> 62% occupancy" plan**, and with it the grouped split-K specialization -- several
hundred lines and multiple box rounds, cancelled by one dense measurement that needed no new kernel. That is why the cheap
dense ladder was the right first step rather than writing the grouped kernel. The decisive number is not the S=32 collapse
(which is mostly serialisation and would be fixed by a parallel reduce) but **S=8, where serialisation is 4% and an 8x
occupancy gain still lost 43% to partial traffic**.

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

---

## Stream-K: Marlin already IS stream-K, and what that costs us (from the user's Marlin notes, 2026-07-30)

**Marlin's scheduler is stream-K.** The notes quote it directly, and even use the word "stripe":

```c
int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);  // stripe length per CTA (in K-tiles)
int slice_row     = (iters * blockIdx.x) % k_tiles;            // where in K this CTA starts
int slice_col_par = (iters * blockIdx.x) / k_tiles;            // which (N, M-region) it starts in
```

The work unit is one K-tile of one output tile; the total `k_tiles * n_tiles * parallel` is divided into
equal CONSECUTIVE stripes, one per CTA. A stripe may start mid-slice, end mid-slice, cross N-tile boundaries
and cross M regions (the notes note CTA=9 crossing two `rest M`). Tiles split across CTAs are combined with
`locks[] + barrier_acquire/release + global_reduce`, ordered by `slice_idx` among `slice_count` contributors.
That is Stream-K (Osama et al.) under a different name.

### Why the recorded persistent-scheduler failure does NOT refute stream-K here

`ppu_aiu_gemm_mixed_input_group.hpp:160` records: the persistent GroupScheduler launched `grid=(72,1,1)` =
one block per CU and measured 2 active warps/CU, 3.1% achieved occupancy, 16% CU throughput. That is a real
measurement and it kills ONE block per CU -- not stream-K.

The difference is CTA WIDTH. Marlin runs 256 threads = 8 warps per CTA, so one CTA fills an SM and
grid = #SMs is a full wave. Our mixed-input collective runs 64-128 threads = 2-4 warps, so grid = #CU is
2 warps/CU by construction -- exactly the 3.1% that was rejected. Stream-K on this collective needs

    gridDim = CU_count * blk,    blk from fold::warps_per_cu_chunked

i.e. 72 * 9 = 648 CTAs at the decode winner, which is 648 * 2 = 1296 warps = **18 warps/CU** -- precisely the
theoretical occupancy acu reported for that config (28.13%). Untried, and distinct from what failed.

### The ceiling, so nobody expects more than is there

Decode today: 512 CTAs * 2 warps = 1024 warps = 14.2 warps/CU, and acu measured 13.65 achieved -- every warp
of the launch resident at once, no second wave. Stream-K at grid=648 raises the resident count to 18/CU.
That is **1.27x, and it is the whole ceiling**, because 18 warps/CU IS the theoretical occupancy. 20.74 us
would become ~16.3 us, still ~2.1x off the memory roof.

The GEMV committed this session is in a different regime, not a better constant: 2048 CTAs * 4 warps = 8192
warps of work = ~7 waves, no shared memory in the main loop. For decode, GEMV replaces this question.

**Where stream-K actually pays for us is #10** -- the prefill/MoE band's ~11% last-wave tail, which is load
imbalance across a grid that ALREADY exceeds one wave. That is what stream-K exists to fix and what tile
tuning provably cannot reach.

### Fitting Marlin's scheduler into actlize: what exists, what it costs

Already present:
  * `make_splitk_coord_iterator(shape, start_k, k_step)` -- an arbitrary K start and stride in the mainloop
    (dense split-K serial path). This is Marlin's `slice_row` and it is the key primitive.
  * `cutlass::Semaphore` -- Marlin's `locks[]`.
  * The flat `blockIdx.x -> (expert, m_tile)` prefix-scan decode -- the same shape of computation as Marlin's
    `slice_col_par / n_tiles` M-region switch.
  * As of f103c8d: the fp32 fixup reduce and the `k_full` stride/shape separation, both gated (l71).

Missing, and the honest cost:
  1. **Flat work-unit decode** over `SUM_e mt_e * n_tiles * k_tiles`. ~15 lines, Marlin's `init_slice` plus the
     expert dimension. Cheap.
  2. **Mid-stripe re-entry into a NEW n-tile.** This is the expensive part and it is specific to mixed input:
     crossing an N-tile boundary means re-priming the B swzl/AIU base, the per-group SCALE iterator, and the
     fold/interleave offsets. In an fp16 GEMM this is a pointer bump; here the scale tile and the fold factor
     make it a real re-initialisation. Marlin pays it too, but its B layout has neither.
  3. **A choice.** Allow N-crossing stripes (maximum balance, pay 2) or restrict a stripe to one output tile
     (= split-K with a DERIVED S, which is what f103c8d already is). The restricted form gets most of the
     balance for K-divisible shapes at none of the re-prime cost, so it is the thing to measure first.

### Take Marlin's SCHEDULER, not Marlin's TILE SHAPE

Separable decisions, and the tile shape is already known to be wrong for us: Marlin caps `thread_m_blocks` at
4 (64 rows) because larger blows up registers, and splits warps over n and k only, never m. For MoE the
quantity to minimise is the TOTAL m-tile count (weights are the bottleneck), so a 32-64 row m-tile is the
wrong shape -- recorded in ppu-moe-gemm-design. The scheduler carries none of that.

One thing in our favour that Marlin does not have: PPU's 256 KB of shared memory against Marlin's 96 KB. A
bigger tile or a deeper pipeline means FEWER work units, which makes the tail relatively larger -- an argument
FOR stream-K, not against.

---

## GEMV on the box: ALU-bound, not bandwidth-bound -- and it does NOT beat the tensor-core GEMM at decode (2026-07-30)

First real numbers for the CUDA-core GEMV, 42 generated units, ppu001.

### The decode band, shape [0] = L=8 active experts x 1 row, N=K=2048, gs=32, ScaleZero

| implementation | time | %HBM |
|---|---|---|
| grouped mixed-input GEMM, `i4 16x32:256 w16x16 s2` (recorded earlier) | **20.74 us** | 37.5% |
| GEMV `int4 native s16/t128 N2 C2` | 22.27 us | 34.1% |
| GEMV `int4 tileK  s32/t64  N2 C2` | 22.26 us | 34.2% |
| GEMV `int2 native s32/t64  N2 C2` | 21.83 us | 20.9% |
| GEMV `int1 native s32/t64  N4 C2` | 21.72 us | 14.1% |
| GEMV `q3(2+1) native s32/t64 N2 C2` | 27.04 us | 22.5% |
| GEMV `q6(4+2) native s32/t64 N2 C2` | 27.49 us | 38.7% |

**The GEMV is 7% SLOWER than the tensor-core GEMM here.** The prediction that it would win because its grid is
~7 waves against the GEMM's single resident wave is REFUTED. Occupancy was not the binding constraint for a
kernel that already has enough of it.

### The observation that settles why, without acu

int1 and int4 take the SAME TIME (21.72 vs 22.27 us, 2.5% apart) while int4 moves **4x the weight bytes**. They
have the same element count, the same loop trip count and the same number of mma hfma2s; only the extraction
differs slightly. So the time is set by per-ELEMENT work, not by bytes: **this kernel is ALU/latency-bound.**

The %HBM column being in exact inverse bit-width order across the formats (q6 38.7 > i4 34.1 > i2 20.9 > i1
14.1) is the same fact seen from the other side -- with the time pinned, %HBM only reports the bit width.

**This also predicts split-K will not help**, for the same reason: split-K buys CTAs, and CTAs are not what is
short. That prediction is exactly what test_moe_splitk_bench measures, so it is still worth running.

### Where the GEMV does look good

Shape [5], dense m=1 N=12288 K=4096, gs=128 ScaleOnly: `int4 native s32/t64 N4 C2` at 18.48 us and **50.8%
HBM** -- the best efficiency in the whole table. At m=1 a larger N gives more independent columns per unit of A
traffic, so efficiency rises with N/K.

### An open question, not to be hand-waved

Dense prefers CtaN=8 (shape [3]: `s16/t128 N8 C4`, 256 CTAs); MoE prefers CtaN=2 (shape [0]: `N2 C2`, 8192
CTAs). The prediction in the bench header -- that DENSE would need SMALL CtaN to buy parallelism -- is backwards.
"Larger CtaN amortises the activation broadcast, so fewer ops/element" explains dense under an ALU-bound
reading but then fails to explain why MoE goes the other way. For acu, not for a story.

### If acu confirms ALU-bound, the lever changes from occupancy to OPS PER ELEMENT

At CtaM=1 each element of each column currently costs roughly 1 shift + 1 and + 1 or (extraction) + 1 hfma2
(affine) + a half-share of 1 hfma2 (mma) -- four to five ops per useful fma. The tensor-core GEMM pays the same
dequant but amortises it over a 16x16x16 mma.

The largest single reduction available: **move the affine from per-element to per-group.** With (s, z) constant
inside a group,

    sum_k a_k*(q_k*s + z) = s * (sum_k a_k*q_k) + z * (sum_k a_k)

so accumulate the RAW integer-code dot product plus one column-independent `sum a` shared by all CtaN columns,
and apply (s, z) once per group per column. The affine term drops from StepK*CtaN/2 hfma2 per iteration to
CtaN/2 -- 16x at StepK=16. Needs its own numeric gate: the accumulation now runs on unscaled codes, so the
partial magnitudes rise (int4: q<=15 x depth 16 -> ~240; Q6: ~1008, both inside fp16's exact-integer range, but
that is an argument for measuring rather than a proof).

Ordering: confirm with acu first. The 4x-bytes-same-time evidence is strong but indirect; acu says which pipe.

## The TileM padding freedom, and its three forms (2026-07-30)

At decode every expert has one row against TileM >= 16 (forced: every MMA atom in mma_traits_ppu0015.hpp is
Shape<_16,...>). Two consequences, both measured:

  * 16x of the ARITHMETIC is on padded rows. acu: v.mma.f32.f16.m16n16k16 = 131,072, which equals
    mt*N*TM*K/16^3 exactly and is the MINIMUM for a 16-row atom; useful MACs are 8*2048*2048 = 33.6 M against
    536 M delivered. S cancels out of that formula, so split-K neither adds nor removes it.
  * 62% of the block's SHARED MEMORY is A's padding: (16*256*2 + 32*256/2 + 32*8*4)*2 = 26,624 B with A's term
    16,384 B. 262144/26624 = 9 reproduces acu's measured Block Limit Shared Mem exactly.

The padding rows' results are discarded by the epilogue's residue mask, so their INPUTS are don't-care. Three
forms of spending that, in increasing generality -- and they are not interchangeable:

  1. **stride-0 on A's m dimension.** All TileM rows read the expert's row 0. Correct ONLY at M_e == 1: at
     M_e = 3 it would map rows 1 and 2, which are real, onto row 0. Implemented, gated, refused above Mmax == 1.
     Costs nothing, changes no smem, and is a pure traffic/locality win (A's L2->L1 volume 33.5 MB -> 2.1 MB) --
     IF the collective's A copy is not already predicated on the m residue, in which case it is a no-op. One
     measurement decides that (SPLITK_ABCAST=1).
  2. **Clamp the row index to min(r, M_e-1).** Correct for every M_e <= TileM, but it is not a stride, so it
     needs the collective's copy changed.
  3. **Unpredicated over-read, plus TileM rows of padding on the A allocation.** Because a grouped A is
     GATHERED, expert e owns rows [off_e, off_e + M_e) and reading off_e .. off_e+TileM-1 spills into expert
     e+1, whose results are masked anyway. This is the CHEAPEST general form -- a uniform, fully vectorised copy
     for every expert shape, no predicate arithmetic -- and its only requirement is allocation padding. Also a
     copy change, not a stride.

None of the three touches shared-memory FOOTPRINT, so none of them changes occupancy. Occupancy needs either
fewer bytes per block or more warps per block; the TileN ladder now in _SPLITK_CFGS is the second, and it is
free of A's padding because A's smem term does not grow with TileN.

### CORRECTION: the smem saving IS available, and it is the biggest lever found so far

The entry above says none of the three forms changes shared-memory FOOTPRINT. That is true only of the one
implemented -- stride 0 on A's GMEM m-stride, which still copies TM rows into a TM x TK smem tile. It is false
of the idea itself, and the code says so:

    ppu_mma_aiu_multistage_mixed_input.hpp:271
      cute::ArrayEngine<RealInternalElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    :205
      using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(TM, TK, Stages)));

A's allocation is sized by cosize_v<SmemLayoutA>, NOT by TM*TK. So a stride-0 M mode in that layout shrinks the
allocation 16x automatically, with no change to SharedStorage. The obstacle is only that tile_to_shape produces a
compact (bijective) layout, so the decode case needs its SmemLayoutA written directly instead.

    config                    A      B     scale+zero  x Stages  blk/CU  warps/CU  theoretical occ
    16x32:256 s2 (now)      8192   4096      1024       26,624      9       18          28%
    + A smem stride-0        512   4096      1024       11,264     23       46          72%
    + TN=64  w16x16          512   8192      2048       21,504     12       48          75%
    + TN=128 w16x16          512  16384      4096       41,984      6       48          75%

2.6x, against 1.78x for the TileN ladder alone. And a useful bound falls out: once A's padding is gone B is the
dominant term, so the occupancy ceiling from smem work is ~48 warps/CU (75%) and raising TileN further does not
move it -- all three combinations land on 48.

Work, and where the risk is:
  1. an alternative SmemLayoutA with a stride-0 M mode for Mmax == 1, bypassing tile_to_shape. cosize does the
     rest.
  2. the A COPY must become 1 x TK as well. Left as TM x TK it writes 16 rows into the same 512 B -- values
     identical (gmem is stride-0 too) so the race is benign, but it wastes 16x the stores AND its interaction
     with the swizzle is the part most likely to be wrong: SmemLayoutAtomA is a bank-conflict swizzle atom and a
     stride-0 mode composed with a swizzle is not obviously safe.
  3. the tsm.ld.swzl side needs nothing: 16 rows reading the same 512 B is the intent.

DO THE TileN LADDER FIRST. Both changes test the same hypothesis -- that occupancy is the lever -- and TileN
already buys 28% -> 50% with code that is already written and gated. If 1.78x of occupancy does not convert into
time, 2.6x will not either, and there is already one counter-example on record: 16x32:64 s4 reaches 38 warps/CU
and measures 19% SLOWER than 16x32:256 s2 at 18. One number decides whether the swizzle work is worth starting.

### RESULT: the TileN ladder works but is small (1.066x), and the A smem stride-0 FAULTS

Both from the same box run, L=64 top-k=8, N=K=2048, gs=32, mode 3.

**TileN ladder, S=1, all w16x16 s2 unless noted:**

    16x32:256   22.68 us   cta 512   wkwrp/CU 14.2
    16x64:256   22.16 us   cta 256   wkwrp/CU 14.2
    16x128:256  21.28 us   cta 128   wkwrp/CU 14.2   <-- winner, 1.066x over 16x32
    16x64:256  w16x32  24.47 us   wkwrp/CU 7.1
    16x128:256 w16x32  24.73 us   wkwrp/CU 7.1
    16x32:64   s4      26.59 us
    32x64:64   s4      30.67 us
    64x128:64  s4      39.78 us

Two things confirmed and one bounded:
  * wkwrp/CU is 14.2 for EVERY w16x16 row, exactly as warps = mt*N*TM/(WM*WN) predicts -- TileN cancels out of
    the total work, it only redistributes it into fewer, wider blocks.
  * more warps per block is the right direction: at identical smem, w16x16 (8 warps/blk) beats w16x32 (4) by
    1.16x at TN=128 and 1.10x at TN=64.
  * but the WIN IS 1.066x, not the 1.78x the occupancy arithmetic allows. Occupancy is a WEAK lever for this
    kernel.

CROSS-RUN ABSOLUTE TIMES ARE NOT COMPARABLE and that nearly produced a wrong claim. Every S=1 row in this run is
1.03-1.07x slower than the same row in the previous run (16x32:256 was 21.16, now 22.68). It is a uniform shift,
not per-row noise, so WITHIN-run orderings hold and the 1.066x stands -- but any comparison that spans runs does
not. State which run a number came from.

**A smem stride-0 (PPU_A_BCAST): reverted, it faults.** cosize_v<SmemLayoutA> would have shrunk the allocation
16x, but InternalSmemCopyAtomA is a tsm.ld.swzl atom that derives its byte addresses from the swizzled compact
layout and walks past a stride-0 one -- illegal memory access at
`tsm.ld.swzl.b32x4.s0.t1.trans0 vreg[64:67], [sreg63] @sreg27`. nvcc's front end accepted it with
PPU_FORCE_INSTANTIATE, every static_assert passing, so the front end was no evidence here.

What would be needed: the layout plays two roles -- allocation-plus-copy, and the mma's read -- and only the
second wants stride 0. Splitting them is a change to the copy atom's contract. Given occupancy measured as a
1.066x lever, that surgery is probably not worth starting.

The GMEM half survives and is unaffected: a_row_broadcast still cuts A's L2->L1 volume (33.5 MB -> 2.1 MB) with
no smem change, and MOE_ABCAST / SPLITK_ABCAST still switch it.

### REOPENED (see the next section): A's smem CAN be shrunk -- the override was in the wrong struct

The motivation was sound and stays on record: at decode every expert has ONE row against TileM >= 16 (every MMA
atom is Shape<_16,...>), so 15/16 of A's smem tile is padding whose results the epilogue's residue mask discards,
and A is 62% of the block's 26,624 B. Removing it would take 9 blocks/CU to 23, 18 warps/CU to 46.

**Attempt 1, stride 0 on SmemLayoutA.** Illegal access. l74_swzl_coord_not_stride.cu measures why: the mma-side
read is partition_S(make_mix_tensor_like(sA)), a mix tensor carries a COORDINATE, and the coordinate at (m,0,0) is
(0,m,_0,0) for the compact AND the stride-0 layout -- identical. Strides never reach the addressing.

**Attempt 2, shrink CUBE_H so one cube is one row.** Illegal access again, with the disassembly's M step still
512 B where CUBE_H=1/CUBE_W=64 would give 128 B. TWO GAPS in my local verification produced a false green light:

  * l76 exercised `DefaultGemm_AIU_Operand` DIRECTLY rather than through the builder. The mixed-input path's A
    operand is `MixGemm_AIU_Operand`, which hardcodes `CUBE_H = Block_MN{}` and has no override point -- so the
    override very likely never reached A's atom.
  * l77 probed `Mainloop::SmemCopyAtomA`, but the collective uses
    `InternalSmemCopyAtomA = conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>`, and SwapAB is TRUE here
    because the operand that goes through the converter occupies the "A" slot. The atom printed back as
    `integer_subbyte<4>` -- the QUANTIZED one. So l77's CPY_M, tCsA layout and fragment-size readings were all
    for the wrong atom and are withdrawn.

The only reading that survives is that cosize_v<SmemLayoutA> follows the layout, which was never in doubt.

**Not attempting a third time, on the measured payoff.** The TileN ladder raised theoretical occupancy 28% -> 50%
at constant total work and bought 1.066x within one run (22.68 -> 21.28 us). Occupancy is a weak lever for this
kernel, so the ~2.6x of theoretical occupancy this would unlock is worth well under 1.1x -- against a code path
that has now faulted twice and that the local toolchain provably cannot verify (symbolic ScaledBasis strides,
address resolved by the asm, SwapAB renaming the operands).

**Do this instead, and first.** A dummy-padding occupancy sweep: add `char pad[N]` to the kernel's own
SharedStorage (ppu_aiu_gemm_mixed_input_group.hpp:77, a LOCAL file -- no collective change) and sweep N so
blocks/CU walks 9 -> 8 -> 7 -> 6 -> 5 -> 4. That measures dTime/dOccupancy in the direction that IS reachable,
single-variable, with no correctness risk. If time is flat from 9 down to 4 blocks, then 9 -> 23 gains nothing
and this whole direction is closed by measurement rather than by two faults. Pad values for 16x32:256 s2
(26,624 B): 2560 -> 8 blk, 6656 -> 7, 11264 -> 6, 17408 -> 5, 26112 -> 4.

What survives from the attempt: the CubeH override on DefaultGemm_AIU_Operand (inert for this path, harmless),
and the gmem-side a_row_broadcast, which cuts A's L2->L1 volume 33.5 MB -> 2.1 MB with no smem change and is
still switchable via MOE_ABCAST / SPLITK_ABCAST.

### The A-smem override was in the wrong struct, and the two withdrawals above are themselves withdrawn

The section above closed this line and blamed SwapAB. Both are wrong, and one printout settled it. Printing the
atom the collective ACTUALLY uses on sA -- InternalSmemCopyAtomA, not SmemCopyAtomA:

    default          PPU0010_TSM_LD_SWZL<half_t, 16, 64, true, false, 4>
    PPU_A_CUBE_H=1   PPU0010_TSM_LD_SWZL<half_t,  1, 64, true, false, 4>

`half_t` there settles that **SwapAB is FALSE** and the A slot really is the activations. My "SwapAB is true, so
l77 probed the quantized operand" explanation came from reading `integer_subbyte<4>` out of an UNRESOLVED
conditional_t branch and taking it for the selected type. So l77's readings (cosize 8192 -> 512, CPY_M = 1) were
for the right object all along and their withdrawal is retracted. CPY_M staying 1 is expected, as the user said.

And `16, 64, 4` matches the builder's **MixGemm_AIU_Operand** generic form -- (Block_MN, AiuContElemSize, InstNum)
-- not DefaultGemm_AIU_Operand, which is where I had put the CubeH override. That is the whole reason attempt 2
faulted with the disassembly's M step unchanged at 512 B: the override was inert, so the allocation shrank 16x
while the instruction still read 16 rows. The override now lives in MixGemm_AIU_Operand, where A's atom is built.

THE DIFFERENCE FROM THE TWO FAILED ATTEMPTS, stated as a discriminator rather than a hope: in both of those, A's
atom parameters were IDENTICAL with and without the switch, so an out-of-bounds read was guaranteed. This is the
first time that instruction's geometry actually changes. That is necessary, not sufficient -- the box decides.

Order stays: PPU_DEFS=PPU_A_CUBE_H=1 TARGET=test_moe_grouped_verify ./build.sh, and only then timing. Mmax > 1
cases are REFUSED by launch(), so expect them excluded rather than passing.

### PPU_A_CUBE_H=1 runs, and A's smem is 64x smaller. Plus: I read a passing signature as a failing one.

First hardware run that neither faulted nor was refused (Mb=1, so Mmax==1 as this path requires):

    PPU_DEFS verified on test_moe_grouped_verify's compile command: -DPPU_A_CUBE_H=1
    [moe_grouped] smem/block = 13456 B  (A = 768 B = 384 elems, 6%)  PPU_A_CUBE_H = 1
    [moe_grouped]   blocks/CU at 256 KB = 19
    verify: L=8 uniform Mb=1 ... Mmax=1   max_rel=0.000e+00 bad=0 -> MATCH

A = 384 elems = TileK 128 x Stages 3, i.e. exactly one row, against 64*128*3 = 24576 elems by default: 49152 B
-> 768 B, and the block from ~62 KB to 13456 B. Read off SharedStorageSize and cosize_v<SmemLayoutA>.

MY LIVENESS CHECK WAS WRONG, and it failed that very run. rel = |got-gold|/(|gold|+1e-3), so a BIT-EXACT pass
gives rel == 0 for every element and `if (rel > max_rel)` never fires -- worst_e keeps its initial -1. That is
the signature of a PASS, and the L=1 oracle is bit-exact by construction. I had asserted the opposite one turn
earlier ("worst e=-1 is the tell") and built the check on it. Vacuity now keys on gold_absmax == 0, i.e. on the
golden VALUES, which is what I claimed to be testing all along.

What still stood from that turn: the earlier default-Mb run really did verify nothing, but the evidence for that
is the refusal COUNT (5 launches refused), not worst_e.

Also fixed: the vacuity return preceded the MOEG_DUMP/MOEG_CHECK block, so the one oracle that does not share
this binary's collective was unreachable exactly when it mattered. Cross-build compare now runs first.

### NaN defeats every comparison-based check, and it produced TWO simultaneous MATCHes on garbage

The box printed three readings that have no solution over the reals:

    grouped-L=8 vs grouped-L=1 oracle: max_rel=0.000e+00 (worst e=-1) bad=0 -> MATCH
    cross-build vs /tmp/d_off.bin:     max_rel=0.000e+00 (worst idx=0) bad=0 -> MATCH   <- reference judged LIVE
    *** VACUOUS: the oracle never produced a nonzero value ***                          <- golden judged ALL ZERO

All-zero golden plus bit-exact equality plus a nonzero reference cannot hold together. NaN reconciles all three:
every `if (x > y)` is FALSE when x is NaN, so rel = |got-NaN|/(NaN+1e-3) = NaN never updates max_rel and never
trips `rel > 5e-2`; abs(NaN) > gold_absmax fails so the golden reads as all-zero; and `g != 0` is TRUE for NaN,
which is how an all-NaN reference passed the liveness test. A comparison-based checker reports a PERFECT MATCH on
a buffer full of NaN -- on both sides at once.

Fix: non-finite values are counted, not compared, and are bad by definition on either side; both absmaxes, the
non-finite counts and the first four values of each buffer are printed unconditionally, because a verdict derived
from comparisons cannot distinguish 'equal' from 'both zero' from 'both NaN'.

This is the same class as the refused-launch pass, one level deeper: the check was structurally incapable of
seeing the failure it was written to catch. Both times the tell was an internal inconsistency between two
printed numbers, not a value that looked wrong on its own.

Open question the next run answers: WHICH side is non-finite, and whether it predates PPU_A_CUBE_H -- the OFF
build's dump was judged live, but under NaN that judgement is worthless, so the baseline at Mb=1 is now also
unverified. Mb=1 had never been run before this line of work.

### CLOSED FOR A READ-OFF REASON: A's smem floor is TileM x TileK x Stages, and CUBE_H is not a footprint knob

l78 (fold_derivation/l78_cubeh_delivery.cu), all values as template arguments out of the compiler:

                     cosize<SmemLayoutA>   Src/Dst bits   size(tCsA)   size(tCrA)   mma atom
    CUBE_H = 16              8192           4096 / 4096      256          128       (16,.,16)
    PPU_A_CUBE_H = 1          512           4096 / 4096      256          128       (16,.,16)

The allocation shrinks 16x and NOT ONE of the three delivery quantities moves. CUBE_H is the M extent of the
instruction's cube, so changing it changes the swzl permutation: the same 4096 bits land in different registers,
the 128-element fragment is filled from the wrong positions, and the output is wrong -- NaN once uninitialised
registers join in -- WITHOUT faulting, because the addresses fold into the single row. At CUBE_H=16 the same
512-element allocation faults instead, since the instruction still sources 16 rows. That is all three box
failures from one cause, and the user's reading confirms it: ON is wrong, OFF is correct.

So A's floor is TileM x TileK x Stages with TileM >= 16 forced by the MMA atom shape. The decode winner
16x128:256 s2 ALREADY sits at that floor (16*256*2*2 = 16,384 B), so A's 62% share is irreducible at fixed
(TileK, Stages) and the whole line was chasing something that does not exist.

The lever that does exist is TileK, which cuts A AND B AND the scale channel together:

    16x128:256 s2   A 16384 + B 32768 + sz 8192 = 57,344 -> 4 blocks/CU
    16x128:128 s2   A  8192 + B 16384 + sz 4096 = 28,672 -> 9 blocks/CU

That is TODO #22, promoted from a sweep point to the principled next step. Constraints to respect: TK >= gs (SK
= ceil(TK/gs) <= 2) and the AIU 32B contiguous-K run, TK*bits/8 >= 32, which at int4 means TK >= 64.

PPU_A_CUBE_H stays in the tree, off, and now prints a KNOWN-WRONG banner whenever it is compiled in.

### PPU_A_CUBE_H removed from the tree; the route that survives is PPU_A_IN_REG (A never enters shared memory)

The user's instruction: do not keep code that produces NaN when switched on. All four sites are gone -- the
SmemLayoutA #if in the collective, the CubeH constant in the builder's MixGemm_AIU_Operand, the four
DefaultOperandA call sites, and the ninth template parameter of DefaultGemm_AIU_Operand. l76 and l77, which only
existed to drive that macro, are deleted; the numbers they produced are recorded above. What stays is prose at
each site saying why the knob cannot exist, so the next person does not re-derive it from scratch.

PPU_A_IN_REG replaces it, and it is a different KIND of change: it uses no A copy atom at all, so nothing about
any instruction's delivery contract moves.
  - load_init builds gA as a PLAIN tensor, not make_mix_tensor_like. That wrapper carries (ptr, coordinate) for the
    AIU descriptor and has no addressable strides (l74), so partitioning it for a register load would have been the
    same error as the stride-0 attempt: allocation right, addressing wrong.
  - Both copy_aiu calls drop to the B-only overload; A's gmem->smem stage and its AIU partitions are gone.
  - SharedStorage allocates ONE element for A. sA survives only as a layout, for the shape asserts and
    partition_fragment_A, and is never dereferenced.
  - tCgA_all = thr_mma.partition_A(gA) sources the fragment. That partitioning equals what partition_fragment_A
    allocated -- three CUTE_STATIC_ASSERT_V check it and they fire locally, since syntax_check.sh compiles with
    PPU_FORCE_INSTANTIATE=1 and instantiates the whole mainloop. The equivalence holds because the AIU write
    composed with the swzl read is a byte identity for fp16, which is also why fp16 A needs no offline relayout.
    Sub-byte B could NOT be sourced this way.
  - a_tile_iter lags the prefetch iterator and names the tile being CONSUMED. The main loop runs exactly K_TILES
    iterations -- (K_TILES-(Stages-1)) live plus (Stages-1) drain -- so it advances K_TILES-1 times and is never
    dereferenced past the end.
  - launch() forces a_row_broadcast (A m-stride 0) and keeps the Mmax==1 refusal. Here that is not about footprint
    -- there is no tile to alias -- but because the fragment spans TileM rows while the expert owns one: stride 0
    points every slot at the real row, removing both the read past the last expert and any dependence on padding.

Not pipelined: A's load sits in the innermost loop with no second buffer. The bet is that it needs none, since
every slot reads the same TileK-long row and it is L1-resident after the first touch. If acu shows a stall on that
load, the fix is to hoist it to one load per k-tile, not to restore the smem stage.

Unmeasured on hardware. Correctness first: test_moe_grouped_verify 8 1, with MOEG_CHECK against a dump from a
build without the macro.

### The first PPU_A_IN_REG run faulted, and the cause was my one-element placeholder for smem_a

    Exception AIU_ld TSM size out of range
    Got bad device status: an illegal memory access was encountered

Not A's load -- A no longer touches shared memory. It was B's AIU load, broken by A's leftover member.
cute::array_aligned's default alignment is 16 B and smem_b sits immediately after smem_a in SharedStorage. At the
real size (cosize_v<SmemLayoutA> * 2 B, always a multiple of 32) smem_b happens to land 32-B aligned, which is what
PPU0010's AIU load requires (align_bytes = 32 in gemm_operands.hpp). My 2-byte placeholder put smem_b at offset 2
and the descriptor became invalid.

The user's question was the fix: why is smem_a still there at all? It is gone now -- the member is compiled out
entirely, so smem_b is at offset 0 and inherits the smem allocation's alignment, which is stronger than before.
sA survives as a NULL-pointer tensor carrying only SmemLayoutA, used by the shape asserts and
partition_fragment_A, neither of which touches the pointer.

Worth keeping in mind beyond this bug: shrinking a shared-memory member is not a smaller version of removing it.
Everything after it moves, and on this hardware the AIU's 32-B alignment is load-bearing and silent -- it held by
arithmetic coincidence, not by any declared alignment.

### PPU_A_IN_REG PASSES: A into the mma fragment from gmem, shared memory untouched, bit-exact

    non-finite: gold=0 got=0   |gold|max=21.72  |got|max=21.72
    gold[0..3]=1.5752 2.87695 2.33789 -4.1875   got[0..3]= identical
    MOEG_CHECK: |ref|max=21.72 non-finite=0
    cross-build vs /tmp/d_off.bin: max_rel=0.000e+00 bad=0 -> MATCH

The reference came from a build WITHOUT the macro, so this is not the same collective judging itself. Three
attempts at A's shared memory: stride-0 layout faulted, CUBE_H=1 returned NaN, and not staging A at all is right.

Why this one works where those did not: it uses no A copy atom. partition_A on the global tile equals what
partition_fragment_A allocated, and the AIU write composed with the swzl read is a byte identity for fp16, so the
fragment's logical map IS the mma's. The other two tried to keep the swzl instruction and change its geometry.

Default path proven untouched by preprocessing both ways: tCgA_all and a_tile_iter appear 0 times without the
macro, and each original A construct loses exactly one instance with it (storage.smem_a 30->29,
copy(smem_tiled_copy_A 49->47, gmem_tiled_copy_A/tAgA 27->25) -- the remainder belong to the 2plane and
overlap_prologue collectives, which are not touched.

Unmeasured: timing. Expect ~0 from occupancy at decode (work-bound at 16 warps/CU, measured 14.2) and read the
mainloop instruction count instead -- A's AIU write and swzl read are gone, a per-atom gmem load is added.

Also settled while checking: split-K is NOT a separate loader. Same kernel, runtime `int splitk = 1`, and every
site degenerates at 1 -- the iterator's step is 1 from idx 0 (identical to a plain coord iterator), the grid's z is
L*1 or 1, expert = z/1, slice = 0, epilogue plane = expert + 0. make_splitk_coord_iterator is vendor code
(ppu_stride.hpp, last touched by 'ACTLIZE v1.0.0 for PPU'), already used by ppu_aiu_gemm_parallel.hpp. The one real
cost on the default path is that S is a runtime int, so blockIdx.z/S and %S are runtime divisions -- once per
block, in the prologue, not in the k loop.
