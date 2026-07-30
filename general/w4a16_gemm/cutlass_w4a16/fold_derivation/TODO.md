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

### PPU_A_IN_REG is CORRECT and 1.14-1.85x SLOWER. Quantified: A's delivery goes from 4 instructions to 64

Same sweep, S=1, both builds (sk_off.log / sk_reg.log):

    16x128:256 s2   22.07 -> 38.46 us   1.74x     16x32:256 s2   22.86 -> 34.53   1.51x
    16x64:256 s2    22.34 -> 36.20      1.62x     64x128:64 s4   40.42 -> 74.91   1.85x
    16x128:256 s3   22.91 -> 38.46      1.68x     16x128:256 s4  23.57 -> 39.20   1.66x

Two predictions held. wkwrp/CU did not move at all (14.2 / 7.1 in both), confirming the shape is work-bound at
1024 warp-tiles / 64 CUs = 16 warps/CU, so smem 57,344 -> 40,960 bought exactly nothing. And the allocation really
did shrink by 8192*Stages bytes at every row.

The cost is instruction count, measured locally (l79_a_gmem_vector.cu): max_common_vector for the
(gmem partition, fragment) pair is 2. The fragment is 128 elements per thread per k-tile, so the replacement copy
is 64 loads where the swzl atom delivered the same 4096 bits in InstNum = 4. 16x more instructions, in the
innermost loop, immediately before the mma. The vector is 2 and not 8 because m-stride 0 makes fragment slots that
differ in m share an address, so the common contiguous run breaks at 2.

THE STRUCTURAL REASON, which closes the line rather than inviting a tuning attempt: A's 16x redundancy is
BETWEEN THREADS, not within one. A warp's 4096 fragment slots cover 256 distinct addresses (row 0, k = 0..255), but
each thread's m is fixed by the mma's A layout, so a thread cannot fill its own slots from fewer loads. The only
hardware that shares one value across threads is shared memory -- which is exactly what the AIU write plus swzl
read exist to exploit. Taking A out of smem discards a 16x reuse to save a footprint that nothing was waiting on.

Kept, off by default and proven inert by preprocessing, as the measurement's record. Not a candidate.

So all three A-smem routes are closed, and the reasons are now different and specific: stride-0 layout faults
(coordinate addressing), CUBE_H=1 corrupts (permutation, not footprint), gmem-to-register is correct but pays 16x
the instructions (inter-thread reuse lost). What remains for decode is not A: it is the 15/16 of the mma work spent
on padding rows (TileM >= 16 against one row per expert) and the per-element op count.

### acu on the A-in-register build: issue-bound on the vector memory pipe, and NOT a coalescing problem

    Stall Vector Memory Pipe Busy   0.596   <- dominant
    Stall Pipe Busy                 0.628
    Memory Dependency               0.233
    Instruction Fetch               0.207
    Stall AIU Pipe Busy             0
    Stall Shared Memory Pipe Busy   0.003
    Memory Throttle                 0

Memory Throttle at 0 rules out the coalescing story: the memory system is not being flooded with sectors. It could
not be -- m-stride 0 collapses the whole warp onto one 512-B row, so these are about the most sector-efficient
loads in the kernel. What saturates is the vector memory PIPE, i.e. instruction issue, exactly as
max_common_vector = 2 predicted (64 loads per thread per k-tile against the swzl atom's 4).

The pair worth keeping is AIU Pipe Busy 0 and Shared Memory Pipe Busy 0.003. The swzl/AIU route does not merely use
fewer instructions -- it uses a DEDICATED engine that nothing else contends for, and this change moved A onto the
general vector memory pipe and filled it. That second effect was missing from my instruction-count argument.

Ceiling check, so this is not revisited as a tuning task: even at a perfect 8-element vector it would be 16 loads
against 4, still on the contended pipe. Closed.

### The whole A line was attacking 0.6% of the instructions. Instruction mix, per mma (131,072 mma):

    int4 unpack     v.shll 12.11 + v.lop3 8.17 + v.or 4.01 + v.bfi 2.06 + v.shrl 1.11   = 27.5   42%
    s.wait                                                                              = 10.2   15%
    affine          v.fma.f16 6.19 + v.add.f16 2.06                                     =  8.3   13%
    addressing      v.add.co/ci 4.04 + v.mov.i 2.28 + v.madw/madl 1.32                  =  7.6   12%
    scalar control  s.add/mov/shll/cbr/cmp/csel                                         =  4.4    7%
    smem reads      tsm.ld 2.08 + tsm.ld.swzl 0.26 + smem.ld 0.25                       =  2.6    4%
    v.mma.f32.f16.m16n16k16                                                             =  1.0   1.5%
                                                                                       ~66 total

A's ENTIRE chain is 0.15 (vmem.acp.commit.grp, the AIU bulk gmem->smem) + ~0.26 (tsm.ld.swzl) = 0.4 per mma, i.e.
0.6%. Three routes, two faults and one silent-NaN round went after that. A's shared memory is 29-62% of the block
by SPACE, and on a work-bound shape space does not convert into time -- the bench's own banner said so and the
measurement confirmed it (wkwrp/CU 14.2 with 57,344 B and with 40,960 B).

Two corrections to what I recommended off the back of this mix:

The mma padding waste is NOT the order-of-magnitude term I called it. mma is 1.5% of instructions, so eliminating
15/16 of it saves 1.4%. What TileM = 16 actually does is act as the DENOMINATOR for anything moved to the
accumulator side, which is why the affine idea below is weak.

Moving affine to per-group does not pay at gs = 32. Currently ~6.2 hfma2 per mma atom over 8 B codes; on the
accumulator it is 8 floats x 2 fma applied every gs/16 = 2 atoms = 8 per atom, i.e. WORSE. The accumulator is
16x16 per atom, the same size as the B fragment, and the group is fine. At gs = 128 it becomes 2 per atom, ~3x on
that term (~8% overall). And the -1024 must stay in the converter regardless: folding it into the zero was already
refuted for fp16 (measured 1.9e-2..4.6e-2 at s = 0.01), and accumulating the biased codes in fp32 costs ~7 of 24
mantissa bits at K = 2048.

Option 'A stays in smem, read one gmem row' is also withdrawn, on the user's recall that it produced NaN. Likely
mechanism: a_row_broadcast puts the m-stride at 0 and that stride is what initialises the AIU descriptor
(desc_.init<>(nullptr, M, K, dA)), and a bulk-DMA row pitch of 0 is not a defined descriptor. Different mechanism
from the PPU_A_CUBE_H NaN. Worth 0.2% even if it worked, so not worth the risk.

Next: the unpack's 27.5. The uniform formula needs 3 instructions per half2 -- 8 codes per atom per thread is 4
half2, so 12 -- against 27.5 measured. 2.3x slack, and v.bfi / v.shrl / the extra v.or point at the chunked-B or
interleave-256 handling rather than at the formula.

### 'A stays in smem, load one row' IS ALREADY THE DEFAULT, and a_row_broadcast was a second mechanism fighting it

Read off cute/arch/copy_aiu_base.hpp, AiuDesc::init:

    dim_h  = MN                    <- the tensor's row EXTENT
    dim_w  = get<0>(stride)        <- the row PITCH
    cube_h = Block_MN              <- rows per transfer, a template argument

and the instruction is ppu.cp.async.aiu.bulk.tensor.shared.global.padz... -- padz, so rows beyond dim_h land in
shared memory as ZERO. The grouped kernel passes the PER-EXPERT M (ppu_aiu_gemm_mixed_input_group.hpp:243,
M = get<0>(pe)), so with one row per expert dim_h is already 1. The AIU already reads exactly one row per k-tile
and already zero-fills the other TileM-1 rows of the cube, deterministically, not as garbage.

That is the feature, and it explains the 0.15 instructions per mma for A's gmem->smem: it was never moving 16 rows.

It also explains the NaN exactly. a_row_broadcast asked for 'one row' by zeroing the m-stride, and that stride is
what dim_w is initialised from -- the row PITCH. The result is a descriptor claiming 16 rows spaced zero bytes
apart, which is malformed, and NaN is what came out. Nothing to do with precision.

So launch() now REFUSES a_row_broadcast on the AIU path, with that explanation, and SPLITK_ABCAST is removed from
the bench. It stays available only under PPU_A_IN_REG, which issues no AIU copy for A and uses the zero stride
solely so cute can see the fragment's m aliasing.

The general lesson, and it is the same shape as the CUBE_H one: these AIU parameters are not independent knobs.
dim_h expresses 'how many rows exist', dim_w 'how far apart they are', cube_h 'how many to move'. Asking for a row
count through the pitch produces a descriptor no hardware contract covers.

### Both NaN mechanisms are now unreachable, by deletion rather than by a fix

Two separate bugs, both understood, neither reachable from the tree:

  1. PPU_A_CUBE_H=1 -- shrinking the swzl atom's cube changed the PERMUTATION, not the footprint, so the right
     bits landed in the wrong registers. Deleted at all four sites (SmemLayoutA's #if, MixGemm_AIU_Operand's CubeH,
     four DefaultOperandA call sites, DefaultGemm_AIU_Operand's ninth template parameter).
  2. a_row_broadcast -- zeroing A's m-stride put 0 into AiuDesc's dim_w, the row PITCH, leaving a descriptor that
     claimed TileM rows spaced zero bytes apart. The PARAMETER is deleted, from launch(), filter_and_run,
     moe_splitk_ppu and the bench (SPLITK_ABCAST with it), so the malformed descriptor cannot be constructed.

Neither is "fixed" in the sense of the intended effect now working, and neither needs to be: dim_h already comes
from the per-expert M and the instruction is ...padz..., so the AIU already reads exactly one row per k-tile at
decode and zero-fills the rest of the cube. There is no functionality gap left behind.

PPU_A_IN_REG is deleted too, on the user's instruction -- correct but 1.14-1.85x slower, and the reason is
structural (A's 16x reuse is between threads). 14 conditionals stripped from the collective, 101 lines net removed,
plus l79/l80. The measurements and the reasons stay recorded here.

What survives from three routes and five box round trips, as facts rather than code:
  - A's whole chain is 0.41 instructions per mma, 0.6% -- there was never a lever here
  - the decode shape is work-bound at 1024 warp-tiles / 64 CUs = 16 warps/CU, so freed smem buys no warps
  - dim_h / dim_w / cube_h are not interchangeable; a row COUNT cannot be expressed through the PITCH
  - shrinking a shared-memory member is not a weaker form of removing it (smem_b's 32-B alignment held by
    arithmetic coincidence)
  - a comparison-based checker reports MATCH on all-NaN buffers, on both sides at once

### RETRACTION: the PPU_A_CUBE_H NaN was a ONE-SIDED edit, not a hardware limit. The route is open again.

l81_aiu_pair.cu prints the write and read parameters side by side. MixGemm_AIU_Operand derives THREE things from
Block_MN, and both asm forms carry .swzl, so write-then-read is a byte identity only while all three agree:

    write payload   bits_per_aiu = Block_MN * AiuContElemSize * bits   ->  PPU0010_AIU_LOAD<C<16384>, ...>
    write cube      GmemTiledCopy's value layout (Block_MN, AiuCont)   ->  Tiler_MN (16,64), hence desc_.cube_h
    read cube       PPU0010_TSM_LD_SWZL<Element, Block_MN, AiuCont>    ->  CUBE_H = 16

PPU_A_CUBE_H changed the READ leg alone. A 16-row swizzled cube was written and reinterpreted as a 1-row cube. That
is the NaN, and my recorded conclusion -- 'CUBE_H is not a footprint knob, the route is structurally dead' -- is
WITHDRAWN. It was dead because I only moved one leg of a matched triple.

PPU_A_CUBE_MN now moves all three, verified locally:

                write bits   write cube M   cosize A   CPY_M   CPY_K   frag
    default        16384          16          8192       1       4      128
    CUBE_MN=1       1024           1          8192       1       4      128

CPY_M staying 1 matters: cute does NOT tile a 1-row read atom 16x in M, so the read issue count is unchanged and
the change cannot lose on read instructions. That was the obvious way for this to be a net loss and it is not.

Deliberately split into two steps, because step 2 is where attempt #1 faulted:
  step 1 (this commit)  the triple at Cube_MN = 1. Only the smem WRITE shrinks, 16384 -> 1024 bits per instruction.
                        The allocation is untouched, so smem rows 1..15 go stale; they feed accumulator rows the
                        epilogue masks at Mmax == 1, which is what makes it harmless.
  step 2 (next)         SmemLayoutA down to one row, for the 16x allocation saving. The m >= 1 read coordinates are
                        what went out of bounds before.

What I cannot check locally: whether write-then-read is still a byte identity at Cube_MN = 1. Three legs agreeing
is NECESSARY -- the old attempt lacked it -- and not sufficient. test_moe_grouped_verify 8 1 with MOEG_CHECK against
a default-build dump decides it.

### A's shared tile is 16 rows because ONE swzl instruction spans 16 rows. Read off LogicalTV, not asserted.

Moving all three legs of the triple removed the NaN and produced wrong values instead:

    non-finite: gold=0 got=0        |gold|max=10.4  |got|max=10.4     (self-consistent, matches its own L=1 oracle)
    cross-build vs default build:   max_rel=8.685e+02  bad=8055/8192  |ref|max=21.72

Finite, deterministic, a permutation rather than missing data. The reason is in
Copy_Traits<PPU0010_TSM_LD_SWZL>:

    SrcLayout = (32 lanes, 128 bits) : (128, 1)          <- no CUBE_H in it at all
    LogicalTV row index = lane/4 + 8*(v/2),  lane/4 in [0,8), v/2 in [0,2)  ->  row in [0,16)

so one instruction's (thread, vreg) structure spans SIXTEEN ROWS by construction -- 32 lanes x 4 vregs arranged as
8 rows x 2. CUBE_H reframes the hardware cube; it does not shrink that register footprint. And the .swzl
cancellation that makes write-then-read a byte identity needs the WRITE to frame the same 16-row cube, which is why
cube_h = 1 corrupted row 0 too, not only the padding rows.

The traits file already stated the boundary and I had read past it: "stock cute covers ANY cube WIDTH". Width, not
height. Height is fixed at 16 by the TV structure.

This also explains the floor we kept bumping into from the other side: A's read is 4 instructions per k-tile because
ONE instruction already covers the whole 16-row tile. There is no smaller unit, which is why A's entire chain is
0.41 instructions per mma, 0.6% of the stream.

Three attempts, three distinct and now precisely-stated failures:
    stride-0 SmemLayoutA  -> fault    (the swzl read is addressed by coordinate; strides never reach it)
    read leg only         -> NaN      (16-row cube written, 1-row cube assumed; .swzl no longer cancels)
    all three legs        -> wrong    (the instruction still delivers 16 rows; the write reframed, the read cannot)
plus A-in-register        -> correct, 1.14-1.85x slower (16x the loads, on the contended pipe, because A's reuse is
                                                         between threads)
PPU_A_CUBE_MN is removed. The record is the deliverable.

### Settled locally: the swzl ldmatrix CANNOT read one row, and cute-ifying its addressing would not change that

l82_swzl_rows.cpp replays ppu_tsm_ld_swzl_sim's arithmetic verbatim on the host (SWAP=true, the fp16 A branch). That
simulator is the object LogicalTV was derived from and is validated 0-mismatch against hardware in l2l3/l17/l7/l10/
l12/l13/l16, so this is not a fresh inference:

    CUBE_H=16   16 rows touched, 512 distinct 32-bit words, 0 aliased reads      <- bijective
    CUBE_H=1    16 rows touched, 152 distinct words, 360 of 512 reads ALIASED
    (lane,vreg) -> row identical in 128 of 128 cases

Two conclusions. CUBE_H does not change which row any (lane, vreg) reads -- the 16 rows come from
vreg_row_idx = (v/2)*8 + lane/4 + coord_h, the instruction's lane/vreg structure, and 'm16n16' is in the mnemonic.
And CUBE_H only scales the slice stride (slice_base += CUBE_H*8*slice_idx), so shrinking it collapses the slices onto
each other: 360 of 512 reads alias, which is exactly the max_rel = 868 measured on the box.

So the answer to 'would cute-ifying the addressing let it read one row' is NO. Which row a lane/vreg gets is not
computed from an address at all; it is how the instruction distributes what it fetched. The minimum valid smem region
for one swzl read is 512 distinct words = 2048 B as 16 rows x 128 B.

BUT the same question has a yes for a different instruction. A PLAIN smem load is per-lane addressed from a real
layout, which cute expresses natively and which has no cube and no swizzle, so a one-row A tile is legal there. The
trade, at 16x128:256 s2:

    A's smem      16384 B -> 1024 B                (block 57,344 -> 41,984)
    instructions  0.26 tsm.ld.swzl -> ~2 smem.ld per mma   (+1.7 of 66, +2.6%)
    pipe          dedicated TSM -> shared memory pipe, measured 0.003 busy
    reuse         PRESERVED -- 32 threads read the same 512 B row from smem, which is what A-in-register lost

Worth nothing at decode (work-bound, measured) and worth the 16x at prefill. Implementing: SmemLayoutA to one row,
A's gmem->smem via plain cp.async (the scale/zero path in the same collective is the existing template), A's
smem->reg via a plain load into the m == 0 fragment slots, then a register broadcast. No cube, no swizzle, no
descriptor -- every one of the four earlier failure mechanisms is inapplicable by construction.

### PPU_A_CPASYNC: A in shared memory, ONE row, plain cp.async in and a plain load out

The swzl route is closed for a reason l82 nails down: the read's 16 rows are the instruction's lane/vreg structure
(m16n16 in the mnemonic) and the asm has NO stride operand, so a stride-0 layout has nowhere to be expressed. A
plain copy is addressed from the layout, so there the zero works.

Five parts, all in the mixed-input collective, all behind the macro:
  1. SmemLayoutA gets a stride-0 M mode -> cosize_v = TileK * Stages, so SharedStorage allocates ONE row.
     16x128:256 s2: 16,384 B -> 1,024 B; the verify's 64x64x128 s3: 49,152 B -> 768 B. Both multiples of 32, so
     smem_b keeps the 32-B alignment its AIU descriptor needs (that alignment holds by arithmetic, not declaration).
  2. SmemLayoutAFrag, a compact twin that is never allocated, shapes the mma fragment. partition_fragment_A on the
     stride-0 layout would INHERIT the zero and allocate fewer registers than the mma reads -- measured on the gmem
     variant, which came back ((2,2,2),1):((1,2,0),0), cosize 4 against 8.
  3. gA is built PLAIN, not make_mix_tensor_like: that wrapper carries (ptr, coordinate) for the AIU and has no
     addressable strides (l74), and cp.async needs real ones.
  4. gmem->smem for A is GmemTiledCopyACp, a 1-D cp.async over TileK elements at 8 per thread, atom and
     thread_idx % n slicing copied from GmemTiledCopyScale, which already moves a small operand this way in this
     same collective and rides the same cp_async_fence. Surplus threads are guarded out rather than re-issuing.
     B keeps the AIU, now through the single-operand copy_aiu overload.
  5. smem->reg for A is make_tiled_copy_A(Copy_Atom<DefaultCopy, ...>, tiled_mma) over the PLAIN sA. DefaultCopy
     rather than AssumedAlignment<128> because a per-thread offset that is not 16-B aligned would fault.

launch() refuses Mmax > 1: the stride-0 M mode aliases every row onto the real one, which is only correct there.

Compile-checked both ways on four harnesses with PPU_FORCE_INSTANTIATE=1, so the mainloop instantiates and the
CPY_M / CPY_K static asserts on the new partitioning actually fired.

Expected: A's smem 16x smaller, tsm.ld.swzl 0.26 per mma replaced by roughly 2 smem.ld per mma (+2.6% instructions)
on the shared-memory pipe, which measures 0.003 busy. Nothing at decode, since that shape is work-bound; the 16x is
for prefill. Unmeasured -- correctness first.

### PPU_A_CPASYNC PASSES: A's shared tile is ONE ROW and the result is bit-exact against a separate compilation

    A path: A in smem, ONE row, plain cp.async + DefaultCopy
    smem/block = 13456 B   (A = 768 B = 384 elems, 6%)      blocks/CU at 256 KB = 19
    non-finite: gold=0 got=0     |gold|max = |got|max = 21.72
    cross-build vs /tmp/d_off.bin: max_rel=0.000e+00 bad=0 -> MATCH

384 elems = TileK 128 x Stages 3, exactly one row. A goes 49,152 -> 768 B (64x), the block 61,840 -> 13,456 B
(4.6x), blocks/CU 4 -> 19. The reference came from a build WITHOUT the macro, so this is not the collective judging
itself.

Six attempts; the one that works is the one that stops using the AIU/swzl pair for A. Every earlier failure
mechanism is inapplicable by construction here: no cube, no swizzle to cancel, no descriptor (dim_w/cube_h never
enter), no 16-row footprint, and the data stays in shared memory so the inter-thread reuse that killed the
gmem-to-register variant is preserved.

Next: timing. Expect ~0 at decode -- that shape is work-bound at 16 warps/CU and freeing smem measurably bought no
warps before -- and the 64x is for prefill, where the grid supplies far more blocks than the smem limit admits.
Prefill also needs the Mmax > 1 case, which this path currently refuses because the M mode is stride-0; serving it
means predicating per m rather than aliasing.

### CORRECTED profile of the DEFAULT build, and where the mma's overhead actually is

The earlier "unpack 27.5 per mma, 2.3x slack" table was measured on the A-IN-REGISTER build, not the default. It is
withdrawn. The default build (cpa_off, v.mma = 131,072) is:

    v.fma.f16 5.69   v.lop3.i 4.17   s.add 2.68   v.add.f16 2.56   tsm.ld 2.08   v.bfi.i 2.06   s.wait 1.93
    tsm.ld.swzl 1.29   v.mov.i 1.21   s.mov 1.14   v.shrl.i 1.10   v.mma 1.00   s.shll 0.72   s.cbr 0.56
    s.cmp 0.54   v.mul.f16 0.50   s.csel 0.41 ...        total ~34.3 per mma, of which v.shll.i is 0.09 not 12.11

Every arithmetic block is AT ITS FLOOR for this scheme, and the numbers close exactly:

    unpack            4 lop3 + 1 shrl per atom       floor 5      measured 5.27
    converter fp16    2 sub + 2 fma per atom         floor 4      part of add 2.56 + fma 5.69
    group affine      4 hfma2 per atom               floor 4      the rest of fma 5.69
                      (the separate multiplies{}/plus{} passes ARE fused by the compiler --
                       v.mul.f16 is only 0.50 per mma, which is the proof)

So per half2: 1 lop3 + 2 fma = about 3.3 ops against a floor of 2, and the missing 1 is exactly the fold that fp16
cancellation kills. Moving affine to the accumulator is 2x WORSE at gs=32 (8 floats x 2 fma x 8 groups = 128 per
k-tile against 64) and 2x better only at gs >= 64.

Structural point worth keeping: B's dequant work is proportional to TileN * TileK and INDEPENDENT of M, so TileM=16's
15/16 padding wastes only the 1 mma in 34 and does not inflate dequant. Decode pays the same dequant per useful
output as prefill because M=1 has no reuse -- decode is dequant-bound, and the only lever is ops per weight element.

That leaves the ~36% that is not arithmetic on data: scalar overhead 9.05, v.bfi.i 2.06, v.mov.i 1.21.

### First cut at it: transform_B_kblock took k_block as a runtime int and erased the caller's Int<x>

    void transform_B_kblock(..., int const k_block, ...)

The callers already hold static values -- for_each hands out Int<x>, and k_block_next is (Int<x> + _1) % K_BLOCK_MAX,
also static. Taking it as an int meant atom_idx, g = atom_idx / APG_ and the `atom_idx % APG_ == 0` guard all became
runtime work despite APG_ and K_ATOM_PER_COPY being constants. The profile shows the shape of it: s.cmp 0.54 +
s.csel 0.41 + s.cbr 0.56 + s.shra 0.09 + s.mull 0.16 = 1.8 per mma, about 5%.

Now templated as KBlockT, all four inner loops use for_each so the index is Int<i>, the guard folds to if constexpr
and disappears, and the division and modulo fold away. Constant folding only -- numerics unchanged. Four harnesses
compile clean both with and without PPU_A_CPASYNC.

### PPU_A_PACK: overlap the swzl cubes so A's smem is ~7x smaller with the ldmatrix read UNCHANGED

The user's idea, and it is the only one of the seven that leaves the read instruction alone. Row 0 of a cube owns
just 32 of its 512 words; the rest only has to be READABLE, and its garbage lands in accumulator rows the epilogue
masks. So pack the cube BASES together instead of changing anything about the cube.

Measured locally first, in this order:
  l84  row 0 owns 32 of 512 words, at word offsets 0-7 | 144-151 | 264-271 | 408-415  (CUBE_H=16, 4 slices)
  l85  packing the bases 8 words apart is COLLISION-FREE for 8 sub-tiles; span 568 words against 4096 unpacked
  l86  the logical->physical map for row 0 is FOUR CONTIGUOUS RUNS and k advances with the word inside each run,
       so the writer needs no shuffle at all -- four plain 32-byte copies per cube

Numbers for 16x128:256 s2 (8 sub-tiles = 4 cubes x 2 stages): A goes 16,384 B -> 2,272 B, about 7.2x.

What made this workable where six earlier attempts failed: only the DISTANCE between cube bases changes. The cube
geometry, the swizzle, the write/read pairing and the read's 4-instruction cost are all untouched. PPU_A_CUBE_MN
changed the geometry and corrupted the data; PPU_A_CPASYNC replaced the read and paid 64 loads plus 17 address ops
per mma.

The one thing that forces a new writer: the AIU write is .padz, so it writes the whole 2048-byte cube including 15
rows of zeros, and packed cubes would erase each other's row 0. A's write is therefore a cp.async of row 0's four
runs -- one 128-bit transfer per thread over kACubes * kASlices * 2 threads, so one instruction on one warp, cheaper
than the AIU it replaces. B keeps the AIU untouched.

Implementation, five places:
  - PPU0010_TSM_LD_SWZL gains a CubePitch template parameter (0 = natural CUBE_H*CUBE_W). A parameter and NOT a
    macro because B reads through the same atom and must keep its natural pitch
  - Copy_Traits follows the new parameter
  - MixGemm_AIU_Operand passes pitch 16 for A's atom under the macro
  - SharedStorage sizes smem_a to the packed span
  - copy_A_packed_row0 writes the four runs; the run offsets are COMPUTED from the sim's arithmetic
    (2*(CUBE_H*8*s + ssv(s)*4), ssv = 0,4,2,6), which reproduces l86's table exactly and generalises over CUBE_H --
    the first draft hardcoded CUBE_H=16 and the compiler caught it on a 64x64x128 config

Guards: l85's collision check is now a constexpr static_assert over (cube, stage) pairs, plus asserts that
AiuContElemSize is 64 and CUBE_H is TileM. launch() refuses Mmax > 1.

Expected: no gain at decode (work-bound, four measurements) -- this banks the capability and the geometry knowledge.
Unmeasured on hardware.

### The local gate accepts a static_assert in an incomplete class body; hgcc does not

PPU_A_PACK=1 compiled clean under the local nvcc front end on four harnesses and then failed on the box with

    error: no type named 'SharedStorage' in cutlass::gemm::collective::CollectiveMma<...>

Cause: static_assert(aPackDisjoint(), ...) sat in the CLASS BODY and calls a member function of that same class,
which is still incomplete there. EDG accepts it, clang/hgcc rejects it, and the rejection surfaces as the whole
class failing -- so the reported error names SharedStorage and says nothing about the assert.

Moved to the top of mma(), where the class is complete. Same numbers checked, same guard.

This is a NEW blind spot in syntax_check.sh, distinct from the ones already recorded: not a missing include, not an
uninstantiated template, but a construct EDG is more permissive about than clang. Worth remembering as a class --
the local gate proves a translation unit PARSES under nvcc, and permissiveness differences are exactly what it
cannot see.

Also settled this round, and it invalidated a measurement: the box was at ef838e4, one commit BEFORE PPU_A_PACK
landed, so grep -c PPU_A_PACK on moe_grouped_ppu.cuh returned 0. build.sh verified -DPPU_A_PACK=1 was on every
compile line -- host and device, main file and all eleven generated units -- and the run was still the default
build, because nothing in that checkout responds to the macro. sk_pack.log is void.

So build.sh checks that the flag is PASSED, never that anything USES it. A misspelled macro name or a stale
checkout both pass silently, which is the same shape as the accident its own comment at line 218 records. The fix
belongs in the kernel, not the build script: when a macro is meant to change SharedStorageSize, compare it against
the unpacked arithmetic at launch and treat equality as a FAILURE. Then 'the macro did nothing' is a refusal
instead of a plausible-looking set of timings.

Next session, in order: (1) that self-report check plus host/device consistency in build.sh, (2) re-measure
PPU_A_PACK for real, (3) redo #11 passing the second scale fragment as NAMED parameters -- appending to the
cute::tuple<Ts...> would not deduce.

### The packed path faulted because the read's pitch and the write's were two separate literals

Symbol name from the box: PPU0010_TSM_LD_SWZL<half_t, 64, 64, true, false, 2, 16> -- CubePitch 16, while the
collective's writer had moved to 64. Written at one spacing, read at another, "invalid VA". I had flagged this exact
coupling when adding the parameter ("the builder's kCubePitchA must match the collective's kAPackPitch -- that's a
coupling risk, let me add a static_assert") and then did not add it.

Both now come from ONE macro, PPU_A_PACK_PITCH, defined identically in the two files. Not a static_assert -- the
first attempt at one named a member that does not exist (Copy_Struct) and failed to compile. A single definition is
stronger than a comparison anyway: there is nothing left to compare.

Confirmed from the same run that the packing IS active: smem/block 61,840 -> 21,520 B and blocks/CU 4 -> 12. The
banner's "A = 49152 B" is the LAYOUT's cosize, which PPU_A_PACK deliberately leaves alone -- only the allocation
moves -- so that number staying put is correct and the 228% it prints is the two being different things.

### PPU_A_PACK measured: the winner does not move, but it removes smem from the constraint list

Verified active first (banner "PACKED cubes", smem/block down on every row), per the checklist.

    16x64:256 s2  (winner)  22.36 -> 22.12   -1.1%      smem 36,864 -> 23,424
    16x128:256 s2           22.40 -> 22.33   -0.3%      smem 57,344 -> 43,904
    16x128:256 s4           23.84 -> 23.34   -2.1%      smem 114,688 -> 85,888   (2 -> 3 blocks/CU)
    16x64:256 s4            32.48 -> 23.56  -27.5%      smem 73,728 -> 44,928    (3 -> 5 blocks/CU)
    32x64:64 s4             32.05 -> 32.83   +2.4%
    64x128:64 s4            40.58 -> 41.54   +2.4%

Everything is drift except 16x64:256 s4, and that row is the point: it was the ONE decode config genuinely
smem-limited, and packing bought 1.38x there. The rest were work-bound already (wkwrp/CU 14.2 and 7.1 unchanged
across the whole table, the fifth measurement saying the same thing), so freeing space could not help them.

The two prefill-shaped rows LOSE 2.4%: at TileM 32 and 64 A is a smaller share, so the packing saves less while
the cp.async writer's threads still cost. This path should gate on TileM == 16 as well as Mmax == 1.

So PPU_A_PACK is not a speedup, it is an enabler -- 'A's shared memory' is no longer a reason a config is
unaffordable. Its immediate consequence is that the Stages ladder is worth revisiting: s3 and s4 all lost to s2
before, and s4 is now 1.2 us behind instead of 10.

### split-K is settled: it DOES deliver the warps, and they turn straight into more waiting

acu on the same tile, S=1 against S=2 (grid identifies them: z is S, and a 1-D grid is the reduce kernel):

                          S=1      S=2
    Achieved warps/CU    14.02    26.84     <- it delivers, 1.9x
    Memory Dependency     0.98     1.772    <- and doubles with them
    DRAM                 38.65%   41.48%
    Duration (bench)     20.18    20.96 us

So the "split-K did not deliver warps" hypothesis is refuted -- and the answer is worse than a bug would have been.
The extra parallelism converted one-for-one into extra waiting: more warps issuing against the same memory path, each
waiting longer. Bank Conflicts 311,296, about 2.4 per mma, sit on that path.

This also retires "compress registers to raise occupancy": occupancy already went 14 -> 26.8 in this experiment and
bought nothing, so pushing the 106-register ceiling to the next step (88 regs -> 40 warps on acu's chart) would only
raise Memory Dependency further. Warp count is not the constraint any more.

SK_QUANT added to the bench to price the scale channel by removing it:
    2 (default) FinegrainedScaleZero    what ships
    1           FinegrainedScaleOnly    no zero: one fewer f16x2 pass per atom, no Z reload
    0           PerColScaleOnly         no per-GROUP reload at all -- the FINE path's 8 dependent smem loads per
                                        k-tile disappear, which is the ceiling on what TODO #11 could ever win
The mode is in the row tag so a log cannot be mistaken for the default.

### Two measurement facts that invalidate earlier comparisons

**acu's Duration and the bench's number measure different things, and the gap is 17%.** The bench runs
`time_it(go, 20)` -- twenty iterations of wall time, launch and sync included -- while `SPLITK_ACU=1` runs
`time_it(go, 0)`, one cold launch, and the row even prints "ONE COLD launch (not a timing)". acu's 18.33 us against
the bench's 21.57 us for the same config is the difference, i.e. roughly 3 us of per-launch overhead on a 20 us
kernel. Both numbers are right for their own question; comparing them is not. Worth knowing for the real workload:
at decode about a sixth of the cost is launch overhead, which no kernel change can touch.

**Same-config run-to-run spread is 13%, not the 3-7% I had recorded.** The same row measured 20.18 (inside a full
sweep), 22.91 and 21.57 (single filtered rows). A filtered single-row run has no warm-up from the rows before it.
So an A/B is only valid inside ONE run, or across runs with identical filtering back to back -- which is what the
SK_QUANT triple did, so its 11.5% / 7.3% split stands, while "20.18 is the best number" does not.

### PPU_SCALE_PREFETCH: the applicability was written as a requirement, and three units failed to compile

`static_assert(GRP <= 2)` where GRP = K_ATOM_PER_COPY / APG_. The TileK=64 configs have GRP > 2, so instead of
keeping the per-group reload they failed the build. Two register sets are an APPLICABILITY CONDITION, not a
precondition of the code.

Now `if constexpr (kPfOk)` with kPfOk = (GRP == 2) && tuple_size<PfPack> == 4, and the original loop in the else.
The GRP expression also guards divisibility -- `K_ATOM_PER_COPY % APG_` non-zero would have truncated to a GRP that
skips groups, and the old assert let GRP == 0 through, which would have loaded no scale at all while passing.

Same shape as the four "two places must agree" errors: a boundary check that guarded one end only.

### #11 CLOSED at 0.7% of a 7.3% ceiling -- the reload's cost is the WORK, not the WAIT

Numerically correct (cross-build MATCH), and back-to-back single-row runs with identical filtering:

    baseline               22.81 us
    PPU_SCALE_PREFETCH=1   22.64 us    -0.7%

SK_QUANT priced the whole per-group reload at 7.3% by removing it. Prefetching only removes the WAITING and
recovers 0.7% of that, so nine tenths of the reload's cost is the loads, the indexing and the two extra scale
fragments -- not stall. Which also says Memory Dependency 0.98 is mostly NOT the scale channel.

The 7.3% is still reachable, but by a SHAPE condition rather than scheduling: FINE triggers on
Scale_TileK > K_BLOCK_MAX, so a copy step that spans exactly one scale group takes the coarse path and reloads
nothing. At TileK=256, K_BLOCK_MAX = 4, so Scale_TileK <= 4, i.e. gs >= 64, is already free.

### And that reframes every decode number this session produced

The bench hardcodes FinegrainedScaleZero at gs=32 -- Q4_K's worst case. The REAL-weight path already selects by
format (test_moe_grouped_real.cu: mode 0 -> FinegrainedScaleOnly), so:

    bench, all session         gs=32,  zero, FINE reload      22.91 us
    GPTQ, real path            gs=128, no zero, COARSE        about the pc row, 18.60 us
    Q4_K, real path            gs=32,  zero, FINE             equals the bench

GPTQ escapes BOTH the 11.5% zero and the 7.3% reload, and it already does -- no kernel change involved. So every
decode figure here overstates GPTQ by roughly 19%, and the 18.8% belongs to Q4_K specifically. The bench's default
should be selectable per format, or the effort keeps landing on costs only one format pays.

### Q4_K's scale channel, split into its two parts (acu, sz against pc on the same tile)

    Shared Load instructions   441,344  (+133%)   -> the scale/zero reads are the +272k, exactly tsm.ld's count
    Shared Load transactions 1,908,736  (+35.9%)  -> +504k
    Bank Conflicts             278,528  (+946%)   -> +252k, about 1.02 per scale read
    Duration                     19.40  (+19.85%)
    % Peak                       28.36            -> shared memory is NOT saturated

So the channel costs +272k instructions (2.08 per mma) and an almost equal number of conflicts, and the conflicts
double its transactions. Per (group, warp, k-tile) a group's scale takes about FOUR shared-load instructions for a
64-half row -- narrow. And with % Peak at 28 it is transactions x latency, not a bandwidth ceiling, which is why
prefetching the wait away (#11) recovered only 0.7% while removing the loads recovered 18.8%.

Q4_K cannot change gs=32 or drop the zero. Two things it CAN change, both layout-level:
  1. widen the scale read -- four instructions per group per warp for 128 bytes says the copy atom is 32-bit-ish;
     this cuts instructions AND transactions, and is the main term
  2. pad SmemLayoutScale to break the bank period -- removes ~252k conflicts, halving the channel's transactions

### PPU_MAXREG: cap registers via __launch_bounds__

device_kernel.h feeds MinBlocksPerMultiprocessor into __launch_bounds__, and the kernel had it at 1. PPU_MAXREG now
sets a REGISTER target and derives the block count as 131072 / (regs * MaxThreadsPerBlock) -- expressed that way
because a hardcoded block count means 102 registers at 128 threads and 51 at 256, which would silently
over-constrain the wider tiles.

Expectation on the record before measuring: at S=1 with a 16x64 tile the grid supplies 4 blocks/CU while 106
registers already allow 9, so registers are not the binding limit and this should do nothing. The S=2 experiment
already showed occupancy rising 14 -> 26.8 for no gain.

### The scale channel, read off the layout (l89_scale_read.cu)

    SmemLayoutScale   = (64, 8, 2) : (1, 64, 512)      <- group stride 64 halfs = 128 B
    max_common_vector = 1                              <- the read is scalar-grade

128 B is exactly 32 banks x 4 B, so CONSECUTIVE GROUPS START ON THE SAME BANK. N is contiguous inside a group, so one
group covers the banks once and is fine on its own; the conflicts come from stepping in the group or stage direction.
That matches acu: +252k conflicts on +272k scale reads, 1.02 each, doubling the channel's transactions to +504k while
shared memory sits at 28% of peak -- transactions times latency, not a bandwidth wall.

PPU_SCALE_PAD adds N halfs to the group stride (8 shifts each group by 4 banks). Data, the gmem->smem copy and every
read all go through this layout, so nothing else changes and the only cost is the extra bytes.

The second half, widening the read, is NOT done: max_common_vector on the pair I built came back 1, but my slice of
tCsS did not reduce to a single group (the printed strides still carried the 512 stage stride), so the 32-elements-
at-width-1 reading disagrees with acu's ~4 instructions per group per warp by 8x. One of the two is wrong and I did
not resolve which. Do that before touching the copy atom -- the width fix has to start from a slice that is
demonstrably one group.

Also settled: PPU_MAXREG raised occupancy and changed nothing, the third independent confirmation that warp count is
not this shape's constraint (after the TileN ladder at 1.066x and split-K's 14 -> 26.8 warps for -4%).

### PPU_SCALE_PAD REFUTED: padding the group stride makes it slower

Back-to-back, same filter:

    PPU_A_PACK=1                    21.10 us
    PPU_A_PACK=1 PPU_SCALE_PAD=8    23.15 us   +9.7%
    PPU_A_PACK=1 PPU_SCALE_PAD=16   21.32 us   +1.0%   (noise)

So "consecutive groups start on the same bank" is not the cause of the conflicts, or not the main one. And the pad
has its own cost: the group stride stops being a power of two, so g*72 or g*80 needs a MULTIPLY where 64 was a
shift, and that arithmetic sits in the inner reload path. The knob stays off; it is kept only as the record of the
refutation.

What the refutation leaves: the conflicts must come from the INTRA-GROUP access pattern -- how the mma's B layout
maps lanes onto N -- not from the distance between groups.

### The one blocking unknown, and why nothing else should be attempted before it

    probe:  max_common_vector = 1 over 32 elements  ->  32 reads
    acu:    about 4 reads per (group, warp, k-tile)
    an 8x disagreement

l89's slice of tCsS did not reduce to a single group -- the printed strides still carried the 512 stage stride -- so
at least one of those two numbers describes something other than what I labelled it. BOTH remaining Q4_K items
(widen the read, remove the conflicts) depend on the lane->N mapping inside one group, which is exactly what that
disagreement is about.

Resolve it first, from a slice that is demonstrably one group. This session already paid twice for changing code on
top of a relation I had written down rather than read off: PPU_SCALE_PAD here, and the four "two places must agree"
faults before it.

### RETRACTION: the "8x disagreement" was two of my own errors multiplied, not a phenomenon

Both readings were wrong, in opposite directions.

acu side: I divided 272,384 by (8 groups x 8 k-tiles x 4 warps x 256 CTAs) and got 4.16 "per group per warp". The
4-warp/256-CTA figures belong to the 16x64 config, but that instruction mix was captured on 16x128 -- 8 warps and 128
CTAs. Corrected:

    16x128:256 s2, gs=32  ->  Scale_TileK = 8, APG_ = 2
    16 copy calls per k-tile per thread (8 reload points x 2 arrays), 128 per thread over 8 k-tiles
    128 x 8 warps x 128 CTAs = 131,072 copy calls
    272,384 / 131,072 = 2.08  ->  about TWO load instructions per copy call

Probe side: l89 built tCsS as make_tiled_copy_B(...).partition_S(sS), while the kernel gets it from
partition_extra_inputs, where the FOURTH mode is the one sk indexes. The printed strides still carried 64 (group) and
512 (stage), so that slice was never one group, and its "32 elements at width 1" describes something else.

So the real picture is 2 loads per copy call and 278,528/272,384 = 1.02 conflicts per load -- essentially one 2-way
conflict on every scale read. A self-consistent explanation, though still an explanation and not a reading:
WarpN = 16 against 32 lanes means TWO LANES SHARE EACH n and therefore each ADDRESS, and if the hardware does not
coalesce that into a broadcast it is a permanent 2-way conflict. It also explains all three failures -- padding
cannot help identical addresses (and adds a multiply), prefetching removes waiting when the cost is issuing, and
removing the loads is the only thing that recovered the 7.3%.

Also retracted, from the same investigation: I claimed the gmem->smem scale copy "already transposes". It does not.
StrideScale is Stride<Int<1>, int64_t, int64_t> and the collective static_asserts is_mn_major, so gmem is N-major and
so is smem. And making smem k-major is not worth it: reads outnumber writes 9.1:1 (272,384 against 29,952 Shared
Store From Global Load), so dividing reads by 8 while multiplying writes by 8 takes 302k instructions to 274k -- 9%,
not 8x.

Next step, now a bounded task: build the scale probe THROUGH partition_extra_inputs / retile_extra_mma_info, and read
the per-thread fragment size and the lane->n map off it. Everything else about this channel is speculation until that
exists.

### The scale read, read off the kernel's own construction (l90_scale_tv.cu) -- and why the channel is closed

Built through the collective's scale_fragment_layout (the host twin of make_scale_fragment) and the same
make_tiled_copy_B, so these are what the kernel asks for, not a hand-rolled partition:

    SmemLayoutScale = (128, 8, 2) : (1, 128, 1024)
    scale fragment  = ((2,2,2), 1, 4) : ((1,2,4), 0, 8)          32 compact registers, cosize 32
    source TV       = ((4,8,8), (1,(2,2,2,4))) : ((256,1,16), (0,(128,1024,8,2048)))

From the thread mode, offset(t) = 256*t0 + t1 + 16*t2 with t0 = t%4, t1 = (t/4)%8, t2 = t/32. Warp 0 therefore reads
{0..7, 256..263, 512..519, 768..775}, and in banks ((offset/2) % 32, two halfs per 4-byte bank):

    0..7      -> 0,0,1,1,2,2,3,3
    256..263  -> the same, because 256 halfs = 512 B is a multiple of the 128 B bank period
    512..519  -> the same
    768..775  -> the same

So a WARP'S 32 LANES LAND ON FOUR BANKS. The concentrating stride is the THREAD stride 256, not the group stride --
which is why padding the group stride did nothing useful. It would have helped in principle (G=72 shifts t0 by 8
banks, 8-way down to 2-way) and still measured 9.7% SLOWER, because inside a 7.3% envelope the conflict share is
worth a couple of points and the pad's non-power-of-two multiply in the inner reload path costs more.

Q4_K's scale channel is therefore closed, with every layer either measured or derived:

    copy calls          16 per k-tile per thread (8 groups x 2 arrays)   fixed by gs=32 and affine
    loads per call      about 2                                          cute asks 32 slots, the compiler CSEs to 8
                                                                         addresses; a hand-built stride-0 layout was
                                                                         tried before and acu called it a no-op
    bank concentration  32 lanes on 4 banks                              changeable, but measured cost > benefit
    total               18.8%  (zero 11.5 + reload 7.3)                  SK_QUANT

All three layout-level routes are now spent: widening (prior work, no-op), padding (-9.7% here), prefetching
(+0.7% here). What is left changes the algorithm (affine on the accumulator, computed 2x worse at gs=32) or the
format. The 18.8% is intrinsic to (gs=32, affine, this mma's B TV layout).
