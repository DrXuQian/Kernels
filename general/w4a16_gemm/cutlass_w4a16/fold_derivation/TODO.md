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
