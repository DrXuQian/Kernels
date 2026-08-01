# #20 — the scale channel in its NATIVE GGUF form, represented in cute

## What this actually buys, stated honestly first

The 336-row and 168-row sweeps both say **nothing in the MoE band is bandwidth-bound** — the compulsory traffic floor is
5–29% of HBM on every row. So the traffic half of this task should **not** be expected to move MoE MFU much, and planning
it as a bandwidth win would be planning against the measurement. Where it does pay, in descending order of confidence:

1. **Dropping the redundant zero for Q3_K/Q6_K is a LATENCY win, not just traffic.** In the FINE path every mma atom
   reloads the scale, and *with* a zero it reloads twice — that is what #2 measured. This is the one part with a
   measurable MoE payoff, and it is independent of everything else here.
2. **smem → occupancy.** The scale tile shrinks by the same factor as the traffic. Occupancy is the lever the sweep
   actually identified: at TileK=32 every winner moved to **s4** (i2 295.08 us / 56.5% MFU) from s2/s3 at TileK=64,
   purely because A-smem halved. A smaller scale tile is more of the same currency.
3. **Decode / small batch**, where the GEMV *is* bandwidth-bound and the scale channel is a real fraction of the read.
4. gs=16 generally, where the channel is largest.

Scale bytes per weight byte, `S/B`:

| format | gs | now (fp16 scale + fp16 zero) | native | ratio |
|---|---|---|---|---|
| Q2_K | 16 | 1.00 | 0.31 | 3.2x |
| Q3_K | 16 | 0.67 | 0.146 | 4.6x |
| Q6_K | 16 | 0.33 | 0.094 | 3.6x |
| Q4_K | 32 | 0.25 | 0.125 | 2.0x |
| Q5_K | 32 | 0.20 | 0.10 | 2.0x |

`native` counts the packed sub-block field **plus** the amortised `d`/`dmin` (one or two fp16 per 256-weight superblock).

**The fp16 scale path is NOT deleted.** fp16 IS the native form for GPTQ and AWQ, so `ElementScale = half_t` stays the
default specialization and the GPTQ regression in `real_weight/` is what proves nothing was traded away.

## DECIDED: repack, because the pipeline stage already exists and literal-raw is traffic-NEGATIVE

Two facts settled this, both read off the code rather than assumed.

**1. Literal-raw GGUF gives the win away, and for Q4_K it is worse than today.** The 12-byte scale block is superblock-
granular (256 weights) while a K-tile is 32-128, and **the block does not slice by group**: Q3_K's low nibble lives in byte
`t + 4*q0` (0-7) and its high 2 bits in byte `8+t` (8-11), so ANY 8-group tile touches all 12 bytes and the next k-tile reads
the same superblock again. Q4_K is worse — groups >= 4 borrow their high 2 bits from bytes 0-3, so a 2-group tile at g=4
needs bytes {0,1,4,5,8,9}. Bytes actually read per group:

| | fp16 today | literal raw GGUF | compact repack |
|---|---|---|---|
| Q3_K gs=16, TK=128 (8 groups/tile) | 2.0 (ScaleOnly, after Phase 1) | **1.75** | 0.875 (4+2 planes) / **1.125 (int8)** |
| Q4_K gs=32, TK=64 (2 groups/tile) | 4.0 | **4.0-5.0** | 2.0 (4+2) / **2.5 (int8)** |

**2. The offline pass, and a repack, ALREADY EXIST.** `dump_real_weights.py` emits `q` as `int8 [N][K]` — one code per byte,
i.e. the weights are fully unpacked and transposed on the host, with the final device relayout done in C++ by
`place_derived`. And it emits `scale` as fp16 `[L][scale_k][N]`, so **the widening is itself an offline step**, and
`[scale_k][N]` is already a repack of GGUF's `[n][superblock][12 B]`. Changing the scale's emitted form is editing an
existing stage, not adding one. The `.bin` is an intermediate; the device layout is decided in C++, which is where the cute
Layout belongs.

So the awkward GGUF bit maps stay on the **HOST**, where they cost nothing, and the kernel sees a trivially affine layout.

**Recommendation: a uniform `int8` scale plane first, not 4+2 bit planes.** The 4+2 split buys 0.25 B/group more on Q3_K
(0.875 vs 1.125) at the cost of a two-plane assembly in the inner loop, and #14 bounds the ENTIRE scale path at 2.6% — so
there is very little to win there and a real risk of spending more than the traffic saves. int8 is also exactly what Q6_K
already is natively, so one of the five formats needs no transformation at all. Traffic with a uniform int8 plane:

| format | gs | fp16 today | int8 plane | ratio |
|---|---|---|---|---|
| Q2_K | 16 | 4.0 | 2.25 | 1.8x |
| Q3_K | 16 | 2.0 (after Phase 1) | 1.125 | 1.8x |
| Q4_K | 32 | 4.0 | 2.5 | 1.6x |
| Q5_K | 32 | 4.0 | 2.5 | 1.6x |
| Q6_K | 16 | 2.0 (after Phase 1) | 1.125 | 1.8x |

Keep the 4+2 plane form documented as the fallback if decode/GEMV turns out to be traffic-bound there — the host-side maps
below are what it would need, and they are already derived.

`ElementScale = int8_t` then becomes a clean specialization: decode is one byte load, an int8->fp16 convert, and one `hmul`
by the tile-constant `d` — amortised over the `APG = gs/16` mma atoms that share the register.

## The GGUF bit maps, for the HOST side (and the 4+2 fallback)

Read off `real_weight/dump_real_weights.py`, which is the decoder already regressed against
`marlin_gguf_ppu.cuh:gguf_q4k_scales` — **not** written from memory.

Every format's per-group scale is a **4-bit low field plus a 2-bit high field** (or a single 4/8-bit field), and each
field's byte index and bit shift is **affine** in a suitably nested group coordinate. That is exactly the shape of
`MixGemm2Plane`'s `LoCodeL` / `HiCodeL` / `HVregL`, so the machinery is not new.

**Q4_K / Q5_K**, 12 bytes → 8 sc + 8 mn of 6 bits, `gs=32`. Group coord `g = (t, h)`, `t = g%4`, `h = g/4`:

| field | plane | byte | shift |
|---|---|---|---|
| sc | low4 | `t + 8h` — `(4,2):(1,8)` | `0` |
| sc | high2 | `t` — `(4,2):(1,0)` | `4 + 2h` — `(4,2):(0,2)` base 4 |
| mn | low4 | `4 + t + 4h` — `(4,2):(1,4)` base 4 | `4h` — `(4,2):(0,4)` |
| mn | high2 | `4 + t` — `(4,2):(1,0)` base 4 | `4 + 2h` — `(4,2):(0,2)` base 4 |

**Q3_K**, 12 bytes → 16 sc of 6 bits, `gs=16`, no min. Group coord `g = 4q + t` with `q` split as `(q0, q1) = (q&1, q>>1)`:

| plane | byte | shift |
|---|---|---|
| low4 | `t + 4·q0` — `(4,2,2):(1,4,0)` | `4·q1` — `(4,2,2):(0,0,4)` |
| high2 | `8 + t` — `(4,2,2):(1,0,0)` base 8 | `2·q0 + 4·q1` — `(4,2,2):(0,2,4)` |

**Q2_K**, 16 bytes, `gs=16`: byte `= g`, sc at shift 0 (4 bits), min at shift 4 (4 bits). One plane each, trivially affine.

**Q6_K**, `int8 scales[16]`, `gs=16`, no min: byte `= g`, shift 0, **signed**. One plane, trivially affine.

**`d` is CONSTANT over a K-tile, and this is GUARANTEED BY THE FILE FORMAT, not merely by our tile choices.** Read off
`src/llama-quant.cpp:360` `tensor_type_fallback`: llama.cpp never pads a K-quant to fit — if `ncols % blck_size != 0` it
changes the TYPE (Q2_K/Q3_K -> Q4_0, Q4_K -> Q5_0, Q5_K -> Q5_1, Q6_K -> Q8_0, the 256-block IQ types -> IQ4_NL, and F16 if
even 32 does not divide), backed by a hard `GGML_ASSERT(nelements % block_size == 0)` at line 254. So **every K-quant tensor
in a GGUF file has K divisible by 256 and there is never a partial superblock.** Every TileK we run (32/64/128) divides 256
and tiles start at k=0, so no tile can straddle a superblock: `d`/`dmin` is one value per `(n, k-tile)`, a `[TileN]` vector.
The per-group work is ONLY the field extraction; the `d` multiply is a per-tile broadcast.

**N, however, has NO alignment guarantee** — it is `ne[1]`, the row count, and nothing constrains it. Since the sub-byte
scale planes pack along N (below), the 2-bit plane's row stride is only a whole number of bytes when `N % 4 == 0`. Adopt
llama.cpp's own idiom for exactly this, from `ggml/src/ggml-cuda/ggml-cuda.cu:818` and its three siblings: **pad the
ALLOCATION, not the data.** `MATRIX_ROW_PADDING = 512`, and `get_alloc_size` over-allocates every quantised tensor's row by
`ggml_row_size(type, 512 - ne0%512)` with the comment "to avoid out-of-bounds memory accesses"; `mmq.cu:112` separately pads
the *activation* row length so the MMQ K-loop is a whole number of blocks. So:

* the FILE stores the exact `ceil(N*bits/8)` bytes per group-row — not one bit more, which is what the size constraint asks
* the DEVICE buffer is padded to whatever the copy atom wants; the tail bytes are garbage that predication ignores
* an n-tile's byte offset is aligned for free, because `n0 = tile_n * TileN` and TileN in {64,128} is a multiple of 4 — only
  the last partial tile reads fewer columns, which is already predicated
* in practice N is a multiple of 64 in every real model shape (2048 / 4096 / 5120 / 11008 / 14336), so the padding is zero

## Where the decode goes: smem holds NATIVE bytes

Two options, and the recommendation is not close:

* **(A) decode in the g2s prologue, smem holds fp16** — saves HBM only, leaves the smem tile as it is.
* **(B) smem holds the native bytes, decode on the s2r path into the fragment** — saves HBM *and* smem.

Take **(B)**: smem/occupancy is the measured lever (point 2 above), the decode is amortised by `APG = gs/16` (at gs=16
APG=1, i.e. once per mma atom — the worst case, and #14 measured the *entire* scale cost there at 2.6%, which bounds what a
few extra ops can cost), and it mirrors the proven B-side design (native in smem, converter on the s2r path) rather than
inventing a second pattern.

## Phases

### Phase 0 — ground truth and a LOCAL gate, no kernel changes

Deliverable `gguf_scale_layout.hpp`: per format, a descriptor carrying `ScaleBits/MinBits/GroupSize/GroupsPerSuper`, the
byte and shift Layouts above, and whether a min exists. Plus:

* `static_assert` that each (byte, shift) pair is a **bijection** over the format's groups — the same collision/miss check
  that `plane_map` gets, because a silently non-injective map is the failure mode that cost rung 5.
* a host gate that decodes native bytes with these Layouts and compares against `dump_real_weights.py`'s output on the
  **real** `real_q2k_ffn_gate_L0.bin` and `real_q3k_concat.bin`, exact-integer on `sc`/`mn`, not approximate on fp16.
* a **known-answer row**: one hand-computed group per format, so the gate cannot be vacuously green.

This phase is where the plan can still be wrong, so it produces evidence before any kernel work.

**PHASE 0 IS DONE AND GREEN.** Deliverables: `gguf_scale_layout.hpp` (new file beside the fp16 path, nothing touched)
and `fold_derivation/l91_gguf_scale_gate.cu` plus `real_weight/dump_scale_blocks.py`.

    Q4_K/Q3_K/Q2_K/Q6_K  bits tile exactly: yes            static_assert, not a test
    exhaustive            zero + all 96/128 unit vectors + 20000 random, 0 mismatches
    known answers         6/6 hand-computed values
    real GGUF             4096 Q4_K blocks x 8 groups against get_scale_min_k4 -> 0 bad

Three things worth keeping from doing it:

* **The tiling check is stronger than the injectivity the plan asked for.** Every K-quant's scale block is exactly
  full -- Q4_K/Q5_K 8*6+8*6 = 96 = 12 B, Q3_K 16*6 = 96, Q2_K 16*4+16*4 = 128 = 16 B, Q6_K 16*8 = 128 -- so "every
  bit claimed exactly once" is necessary AND sufficient, and it catches a MISS as well as a collision.
* **The comparison is a proof, not a sample.** Each decode is a selection of bits, hence linear over GF(2) in the
  input bits, so agreement on the zero block and on every single-bit block settles all 2^96 inputs. The random
  blocks only exist to falsify that argument if it is wrong.
* **The first version of the real-file check was vacuous and I caught it from the output**, not from the code: it read
  `real_q3k_concat.bin` at offset 96 as a raw Q3_K record, and the decoded scales came out `0 32 0 10 0 40 0 20`, every
  other one zero. Those .bin files are the already-decoded harness inputs for `test_moe_grouped_real.cu` and
  286736 % 110 != 0 gives it away. Replaced by `dump_scale_blocks.py`, which imports `dump_real_weights` (so the
  reference stays the regressed one) and dumps raw bytes plus its own decode from a real q4_k_m file. The distinction
  matters: **the exhaustive sweep proves the BIT MAP, only the real-GGUF block proves the RECORD OFFSETS.**

### Phase 1 — generalise `kBias`, then run Q3_K/Q6_K as ScaleOnly

Independent of Phases 2–3 and the only part with a confident MoE payoff.

Q3_K and Q6_K are symmetric: their "zero" is a constant `-bias·d·sc`, i.e. already expressible by the converter's additive
constant. So generalise `MixGemmChunkEmit`'s `kBias` from `(Bits==4) ? 8 : 0` to a template parameter. `B=4` is exact at
every int2 `bpos` — the four constants are already derived: `0x6404 / 0x5C10 / 0x5440 / 0x4D00`. int4's existing `-8` is the
same mechanism (`add = -(2^(10-bpos) + Bias)`), so this is a generalisation of a working thing, not a new one.

Then switch Q3/Q6 to `FinegrainedScaleOnly`. Gate: the existing `test_lowbit_grouped` Q3/Q6 rows must match their
ScaleZero oracle exactly. Payoff: half the scale channel gone *and* one fewer smem reload per mma atom.

Note the synergy with Phase 2: Q4_K's zero is `-dmin·mn + 8·scale` (the int4 `-8` folded in). With `kBias` generalised the
`+8·scale` term moves into the converter and the zero decode becomes purely `-dmin·mn` — one `hfma` per group.

**kBias IS GENERALISED AND GATED (first half of phase 1 done).** `MixGemmChunkEmit` gained a `Bias` template
parameter defaulting to `(Bits == 4) ? 8 : 0`, and `add()`'s mantissa field became `kBias << bpos` -- the old
`1 << (bpos+3)` was the `kBias == 8` special case written out. Exactness is a static_assert
(`kBias << bpos_max < 1024`), not an assumption.

`fold_derivation/l92_kbias_general.cu` verifies the magic-number identity NUMERICALLY rather than by reproducing
constants: for every (Bits, Bias, bpos, code) it checks `x*mul + add == c - Bias` with `x = 1024 + c*2^bpos`, in
double, which decides it because every intermediate is an exactly-representable fp16 integer or power of two in the
ranges used.

    int4 Bias=8   0 bad,  add[0]=0xE408 add[1]=0xD480   bit-identical to the shipped FP16_TOP_MAGIC_NUM / NEG_72
    int2 Bias=0   0 bad     int1 Bias=0  0 bad           int4 Bias=0  0 bad
    int2 Bias=4   0 bad,  0xE404 / 0xDC10 / 0xD440 / 0xCD00 -- the four constants this plan derived by hand
    int1 Bias=1   0 bad                                  the mechanism generalises past the two cases needed

Five harnesses compile clean, so the shipped converter is untouched.

**Still open in phase 1**, and it is the part that needs the box: thread `Bias` through the two-plane path so Q3_K's
int2 plane can name 4, then flip Q3/Q6 to `FinegrainedScaleOnly` and gate on `test_lowbit_grouped`'s Q3/Q6 rows
matching their ScaleZero oracle exactly. The arithmetic for it is now proven; what remains is plumbing plus a
hardware gate.

### Phase 2 — `ElementScale` becomes a packed type, decoded through the Layouts

* `SmemLayoutScale` carries **bytes** instead of `half_t`; the g2s copy shrinks by the ratio in the table.
* `make_scale_fragment` gains a decode step driven by the Phase-0 Layouts — the same relationship `MixGemmEmit` has to the
  B converter. No hand-written shift arithmetic anywhere; that is the whole point of the cute constraint.
* `d` / `dmin` ride along as a `[TileN]` per-tile vector (pending the Phase-0 verification that they are tile-constant).
* `ElementScale = half_t` remains the default specialization, untouched, for GPTQ/AWQ.

### Phase 3 — the offline emits int8 + d instead of widening to fp16

Edit the EXISTING stage: `real_weight/dump_real_weights.py` emits `sc` as `int8 [L][scale_k][N]` (and `mn` likewise where
the format has one) plus `d`/`dmin` as fp16 `[L][K/256][N]`, instead of the pre-multiplied fp16 plane. The host keeps all the
awkward GGUF unpacking it already does; only the OUTPUT form changes. Bump the `.bin` magic (`RWMOE\0\0\0`) so an old
fixture cannot be read as a new one — a silently misread header is the worst failure mode available here.

Precision check to include: today the host computes `d*sc` in fp32 and rounds once to fp16; the new path multiplies
`half(d) * int8->half(sc)` on device, also one rounding, with `sc <= 63` exactly representable. Expected equal or 1 ulp, and
the existing GPTQ + Q4_K real-weight regressions are what confirm it.

**Settled by reading, not assumed**: the scale needs no per-tile permutation. The current path builds the fragment with a
stride-0 broadcast view (`ScaleSplit` / `ScaleThrDupL`, task #1), not a permutation, so only N-tile blocking applies — unlike
the B operand, which does need `place_derived`.

### Phase 4 — measure, in the places where it can show

Not the MoE band first. In order: **gs=16 dense** (largest channel, and where the smem tile is 12 KB today at
TN=128/TK=128/s3), then **decode/GEMV**, then the MoE band with `MOE_ONLY` on the current winners to confirm no regression.
Report the smem delta explicitly — if occupancy moves, that is the mechanism, and if it does not, the traffic saving is real
but invisible here and should be claimed for decode only.

## What reading the scale path actually changed in this plan

The scale goes **gmem -> smem -> registers**, in the multistage pipeline, so the "smem holds native bytes" design is
available as written:

* **g2s** is `Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementScale>` — a plain `cp.async` at 16 B/thread, NOT the AIU
  bulk path — issued next to A and B at lines 1093 / 1112 of the 2-plane collective.
* **smem** is `SmemLayoutScale` via `tile_to_shape`, in `shared_tensors.smem_scale`, with `smem_zero` as a second block.
* **s2r** is `SmemCopyAtomScale` + `make_tiled_copy_B`. Note there are **TWO** layouts: `SmemLayoutScale` (storage) and
  `SmemCopyLayoutScale` (copy view). The in-file comment says conflating them is what produced the `hi_vreg0` defect, so
  Phase 2 must change both, consistently.

**One hardcoded relation to fix on the way**: the copy assert reads "Scale_TileN must split into ThrH threads of a multiple
of **8** elements". That 8 is `16 / sizeof(half_t)` written as a literal; with a 1-byte element it must be 16. Read it off
`sizeof(ElementScale)` — this is exactly the class of bug the METHOD section is about, and it will silently mis-vectorise
otherwise.

**smem delta**: the scale tile halves (fp16 -> int8) and the zero tile disappears entirely for Q3_K/Q6_K after Phase 1. At
TileN=128 / Scale_TileK=8 / Stages=3 that is 12 KB -> 3 KB. Whether that buys anything is an occupancy question, and the
TileK=32 result (every winner moved to s4 once A-smem halved) says occupancy is the currency.

## Risks, and the two things to verify before writing kernel code

1. ~~**`d` tile-constancy**~~ — **DISCHARGED**, not by geometry but by `llama-quant.cpp`'s type-fallback: a K-quant tensor
   whose K is not a multiple of 256 is stored as a DIFFERENT type, so a partial superblock cannot exist in a GGUF file. What
   replaced this risk is the N-alignment question above, and llama.cpp's allocation-padding idiom answers it.
2. **Decode cost vs the 2.6% ceiling.** #14 bounds the entire scale path at 2.6% (gs=16, APG=1), so the decode must stay
   at a couple of ops per scale register. Decode **once per group into the fragment**, amortised over the `APG` mma atoms
   that share it — never per mma atom.
3. A stale claim to re-check while in this file: `moe_grouped_ppu.cuh` says gs=16 "Needs TK=128 (SK=8=TK/16 cap)", but the
   constraint is `SK <= TK/16` with `SK = ceil(TK/gs) = TK/16` at gs=16, which holds for **any** TK. That comment was
   probably true of an older cap. The last comment in this file asserting a TileK limit ("TK=32 still won't compile") was
   wrong, so this one gets tested rather than believed.


---

## RE-PRICED from the decode band (2026-07-31 session): this is the FIRST-order term there, not a traffic side-effect

The framing above prices this task from the MoE-band sweeps, where nothing is bandwidth-bound, and concludes the traffic
half should not be expected to move MFU. That still holds. But the decode band (L=8 active experts, one row each,
N=K=2048, gs=32, ScaleZero) was measured directly this session with a bench knob that removes the channel, and there
the reload is the largest single cost in the kernel:

    SK_QUANT=2  per-group scale + zero (what ships)   22.91 us
    SK_QUANT=1  scale only, no zero                   20.28    -11.5%
    SK_QUANT=0  per-column only, no group reload       18.60    -18.8%

Same tile, same filter, back to back, so the split is trustworthy even though absolute numbers drift 13% across runs.

Item 1 above is confirmed and quantified: with a zero the FINE path reloads twice, and that second reload is 11.5% of
the kernel. What the plan did not price is the part that matters for Q4_K, whose min is REAL data and cannot be
dropped: native co-location. The 12-byte packed field holds all eight sub-block scales AND all eight mins, so under
Phase 2 option (B) -- smem holds native bytes, decode on the s2r path -- one contiguous read serves eight groups and
both arrays. That collapses **16 reload operations per k-tile per thread to 1**, without dropping any value.

Three further measurements from the same session, each closing a cheaper alternative, which is why the native route is
the remaining one:

* **prefetching the next group's scale**: implemented (PPU_SCALE_PREFETCH), numerically correct, **0.7%** against a
  7.3% ceiling. So nine tenths of the reload's cost is issuing the loads, not waiting for them.
* **padding the group stride to break the bank period** (PPU_SCALE_PAD): **9.7% SLOWER**. l90 explains why -- the
  concentrating stride is the THREAD stride, not the group stride: the source TV layout is
  `((4,8,8),(1,(2,2,2,4))) : ((256,1,16),(0,(128,1024,8,2048)))`, so warp 0 reads
  `{0..7, 256..263, 512..519, 768..775}` and every block lands on banks 0..3 because 256 halfs = 512 B is a multiple of
  the 128 B bank period. **A warp's 32 lanes sit on four banks.** Native packing changes this for free: consecutive n
  are then 12 bytes apart instead of 1 half, so the lanes spread.
* **widening the read at the cute level**: already tried in earlier work and recorded in `make_scale_fragment`'s comment
  as an acu-verified no-op -- cute asks for 32 slots, the compiler CSEs them to the 8 distinct addresses, and the
  hardware issues about 2 loads per copy call (272,384 tsm.ld over 131,072 copy calls).

And one arithmetic correction that raises the ceiling further: **the zero's 11.5% is NOT arithmetic.** `v.mul.f16` is
0.50 per mma against `v.fma.f16` at 5.69, so the `multiplies` and `plus` passes are already fused -- ScaleOnly would
issue `hmul2` where ScaleZero issues `hfma2`, the same count. The 11.5% is the Z channel's loads, its cp.async stream
and its smem. All three are what native co-location removes.

So for the decode band the expected payoff is a large fraction of 18.8%, and the mechanism is the reload COUNT, not
traffic. Phase order is unchanged; only the expectation is.

**PHASE 1 IS COMPLETE (1b: the two-plane path).** `MixGemm2Plane` gained `Bias` as parameter 7, defaulted to the
shipped value, threaded into the low plane's `MixGemmChunkEmit`. The reason ONE constant suffices for a two-plane
format is structural and worth writing down: `emit_one` ORs the high plane into the mantissa at `bpos + LowBits`
*before* the fma, so the fma sees the concatenated code `c = low + 2^LowBits*high` and the low plane's single `add`
biases the whole code. Hence `kSymBias2Plane<LowBits,HiBits> = 1 << (LowBits+HiBits-1)`: Q3_K 4, Q6_K 32.

`kCvtBias` in the 2-plane collective picks that rule in `ConvertAndScale` and keeps the shipped constants in
`ConvertAndScaleWithZero`. Deriving it from the MODE rather than adding a template parameter was checked, not assumed:
every VERIFYING 2-plane caller today is ScaleZero, and the only ScaleOnly 2-plane callers were bench rows, where an
fp16 immediate cannot move the instruction count.

Two bugs this phase, both of the session's dominant shape:
* `Cvt2Plane` (line ~300) is not a properties-only alias -- line 1455's unchunked path calls `Cvt2Plane::convert`. I
  had biased only the chunked site and written a comment claiming the other was properties-only. Caught by grepping
  every `::convert(` site rather than the one just edited; fixed by moving the alias below `kCvtBias` so there is one
  biased type and no way to bias one path only.
* **The plan's own gate was wrong.** It said "gate on `test_lowbit_grouped`'s Q3/Q6 rows". That harness's oracle is
  the SAME KERNEL at L=1 -- it isolates per-expert addressing and is structurally blind to a wrong dequant constant.
  The real gates are the harnesses with an INDEPENDENT golden: `test_q3_bconcat_real` (native Q3_K golden out of the
  gguf; rungs 6-7 added, ScaleOnly, and the zero buffer is still passed still holding -4*dl so a double-applied bias
  fails too) and `test_q65_bconcat_real` (rungs added against a new SYMMETRIC golden `dl*(q - qmax/2)`, qmax/2 being
  kSymBias2Plane at both widths: Q6 32, Q5 16).

Local evidence: `l92` extended to sweep the full concatenated range -- Q3 8 codes x 8 bpos, Q6 64 x 4, Q5 32 x 4, all
0 bad, defaults unchanged. nvcc front end clean on all three edited harnesses (the ScaleOnly 2-plane instantiation is
really expanded there, which is the check that would have caught the `GRP <= 2` class of failure).

Box gate still owed: run `test_q3_bconcat_real` and `test_q65_bconcat_real` and require rungs 6/7 and the two
ScaleOnly Q6 rungs to MATCH.

**PHASE 3 DONE (offline), in a new file: `real_weight/dump_packed_scale.py`.** It imports `dump_real_weights` so the
unpackers, the golden and `get_scale_min_k4` stay single-source; what is new is only the output form. Magic bumped to
`RWMOEP\0\0`, header extended with `ktype`, `sb`, `z_mul`, `cvt_bias`. Planes: `sc`/`mn` int8 `[L][scale_k][N]`,
`d`/`dmin` fp16 `[L][K/256][N]`. Scope Q4_K -- the format with a min, i.e. the hard case, and the one carrying the
measured 18.8%.

Four local gates, on real `blk.11.ffn_down.weight` out of qwen2.5-0.5b-instruct-q4_k_m.gguf:
* (a) the vectorised (d,dmin,sc,mn) extraction vs `get_scale_min_k4`, 512 superblocks, **0 bad**
* (b) `f16(f32(d)*f32(sc)) == f16(d)*f16(sc)`, 136192 groups, **0 bad** -- EQUAL, not 1 ulp, because sc <= 63 needs
  6 bits and d is already fp16, so the product is exact in fp32 and both forms round the same real number once
* (c) the weight against fp64 truth, in units of the QUANTISATION STEP
* (d) `d*sc` plane == the reference scale plane element for element (catches a transposed or off-by-one-superblock
  pairing, which a per-block check cannot see), **0 bad**

**(c) corrected two of my own claims, so record the numbers, not the story.** First version compared the two ZEROs to
each other and reported "max_ulp=196, catastrophic fp16 cancellation". Those were ulps *of the zero*, which is small --
the wrong denominator. Second version normalised by |W| instead, which is also wrong: W passes through zero, so
max_rel was dominated by weights that are essentially zero and the CANCELLING form scored best on it. In steps:

    current pipeline, fp32-precomputed zero      max 0.0085 step   mean 0.00089
    packed, Bias=0, zero = -dmin*mn              max 0.0148 step   mean 0.00223
    packed, Bias=8, zero = 8*scale - dmin*mn     max 0.0128 step   mean 0.00194

So: forming the zero on device costs +0.00135 step of mean error (the offline's fp32 zero rounds once, AFTER the
cancellation, and is the most accurate of the three); the two on-device forms differ by 0.00029 step. **Accuracy does
not choose between them.** `Bias=0` is chosen because it is one product instead of a product plus an fma, and needs no
dependency on `scale` -- a cost argument, stated as one. Everything is 30x inside a quantisation step either way.

Note what does NOT change: the B bytes, the packing, `preprocess_weights_for_mixed_gemm`. The stored 4-bit codes are
already the unsigned nib, so `Bias=0` is purely the converter's immediate.

### PHASE 0's C++ GATE IS NOW GREEN ON REAL DATA
`l91` against `real_weight/scale_blocks_q4k.bin`: **4096 superblocks x 8 groups, 0 bad** against
`get_scale_min_k4`, on top of the exhaustive GF(2) checks (96/128 unit vectors + zero + 20000 random per format) and
the four known-answer decodes. The gate that was written in phase 0 and never run has now run.

### PHASE 2 -- the decode unit is DONE and gated; the collective plumbing is NOT landed

**Delivered (new files, fp16 path untouched):**
* `gguf_scale_decode.hpp` -- native GGUF scale bytes -> the `(scale, zero)` an mma atom needs. `Superblock<KType>` is
  the object that replaces eight strided fp16 reads with one contiguous read plus a register decode.
* `Traits::kScaleBias` / `kSigned` added to `gguf_scale_layout.hpp`, and this is where a guess would have been wrong:
  **Q3_K's scale is `d*(sc6 - 32)`**, not `d*sc6` (`unpack_q3k_expert`). Q4_K and Q2_K have no centre; Q6_K's scales are
  signed int8. One constant per format, read off `dump_real_weights.py`, checked in l93 against those formulas.
* ONE conversion rule for all four formats: `int_to_half_small(v)` puts `v+128` in the mantissa of `0x6400` and
  subtracts 1152. The `+128` is what makes the SAME instruction pair cover Q6_K's signed codes -- no second path.
* `fold_derivation/l93_scale_decode.cu`: (1) the conversion exact over its whole claimed range [-128, 895] and DIFFERENT
  at 896, so the bound is real; (2) the four centres against the reference formulas + Q6_K's signed range; (3) 32768
  real Q4_K groups -- field extraction 0 bad, and `(scale, zero)` **bit-identical** to the host's fp32-then-round.
  The first version of (3) drove `d` with powers of two only, which made `max_rel` come out 0.000 because nothing ever
  rounded -- a strong-looking number testing nothing. With non-dyadic `d` it is 3.87e-4 (fp16 eps) and the bit-identity
  still holds; that identity is the claim, and it now actually enters the rounding path.
* Traffic, off the object: Q4_K native **1.5 B/group/col vs 4.0 fp16 (2.67x)**; Q3_K 0.75 vs 2.0 (also 2.67x).

**NOT landed: the collective.** What it takes, so the next attempt starts from the sites and not from scratch:
`SmemLayoutScale`'s element (l353/359), the `ArrayEngine` members (l430/431), `GmemTiledCopyScale`'s and
`SmemCopyAtomScale`'s Copy_Atom element (l159/163/172/494/499), `Params::ptr_S/ptr_Z` (l456), the three s2r sites
(l1173 coarse, l1286 FINE, l1334 prefetch) and both transform branches (l1263-1300, l1301-1345, each with a coarse and
a FINE arm). Plus new gmem plumbing for `d`/`dmin` all the way out through `moe_grouped_ppu.cuh` and the harnesses.
That is ~10 sites in the collective and 3 driver layers, and none of it compiles locally beyond the nvcc front end --
so it is a box-loop change, and landing it blind is how the pitch/gA faults happened.

**One design option is already RULED OUT, which is worth more than a guess.** "Interleave (scale, zero) in N so one read
gets both" does not work: `tCrS` is `make_fragment_like(partition_fragment_B(...))`, so each slot is bound to a specific
`n` of the mma B operand. Interleaving in N makes a lane's 8 halfs into 4 scales and 4 zeros of DIFFERENT n. Any
co-location has to be at `[group][n][2]` granularity (a 32-bit read yielding both for the same n) and then de-interleave
in registers -- which is a different change from the packed-int8 one, not a cheaper version of it.

**A MUCH CHEAPER ROUTE FOR THE COLLECTIVE, found while scoping it: reinterpret, do not resize.**

The packed form's element does not have to be smaller than `half_t` -- it has to CARRY MORE. Interleave `(sc, mn)` as two
int8s in one 16-bit slot, host-side, at `[group][n]` granularity. Then the smem element stays 2 bytes and
`SmemLayoutScale`, `GmemTiledCopyScale`, `SmemCopyAtomScale`, `elements_per_smem_scale`, `make_scale_fragment`,
`Params::ptr_S` and every partition are **byte-identical** -- nothing about the plumbing changes. What changes:

* the ZERO channel disappears as a channel: no `ptr_Z` stream, no Z smem tile, no per-group Z s2r read. That is where
  the measured 11.5% lives.
* `d`/`dmin` ride in on the pointer the zero used to use, at `[K/256][n]` granularity -- one eighth the rows.
* the transform's `multiplies{}`/`plus{}` pair becomes a decode (`gguf_scale_decode.hpp`, already gated by l93) plus the
  same two passes. FOUR sites: ConvertAndScale coarse/FINE and ConvertAndScaleWithZero coarse/FINE.
* the FINE path's TWO reads per group (scale and zero) become ONE, and it costs no extra bytes because 2 int8s occupy
  exactly one fp16.

**Correction to the line above as first written: this is 16 reloads -> 8, NOT -> 1.** At TileK=256/gs=32 the FINE path
does 8 groups x 2 channels = 16 s2r reads per k-tile; interleaving makes it 8. Getting to 1 needs the TRUE superblock
form -- 12 raw bytes read once and decoded for all eight groups -- which requires a byte-granular or k-major scale tile,
i.e. exactly the expensive route this note was trying to avoid. So the cheap route's payoff is: the Z channel's stream
and tile gone (the measured 11.5%) plus half the FINE reload reads (7.3% -> ~3.7%), about 15% of the kernel, and NOT the
full 18.8%. Writing "-> 1" was the same failure this file keeps recording: a relation restated from memory instead of
recounted off the shape.

**THE ONE DESIGN DECISION LEFT: where d/dmin come from.** They are per (superblock, n), i.e. one eighth the rows of the
scale plane, so they do not fit the existing Z tile's granularity.
  (a) a second smem tile at 1/8 the K rows -- needs its own SmemLayoutScale, its own partitioning and its own read
      cadence (once per k-tile at TileK=256). ~4-6 sites, all in the collective.
  (b) load them per lane straight from gmem into registers once per k-tile, no Z smem at all. The Z channel then
      vanishes completely rather than shrinking, the read is tiny (2 fp16 per n-slot) and L2 serves the reuse across the
      eight groups. Sites: one loader plus the four transform arms.
(b) is the smaller change and the bigger saving; its one prerequisite is getting a lane's own `n` coordinates inside the
mma loop, which the gmem side already has (`partition_S(cS)` at l985 exists for predication) but the register side does
not yet -- READ how that partitioning maps before writing it, do not infer it from the fragment shape.

So the change is four transform sites plus the Z tile's K-extent, not ten sites plus three driver layers. The host-side
interleave belongs in `dump_packed_scale.py` (it already emits sc/mn/d/dmin separately) and in the synthetic harnesses.
Verify before writing code, since this note is itself a written-down relation: that `NonVoidElementScale` reaches nothing
but sizing/copies/fragment-element (grep says l159/163/172/385-430/456/1034), and that no arithmetic outside the four
transform arms touches `tCrS`/`tCrZ`.

### THE NATIVE-FORMAT READ NEEDS NO CONVERTER CHANGE. Retracting a prerequisite I invented.

I wrote earlier that Q4_K's packed form needs the converter's `Bias` set to 0 so the zero collapses to `-dmin*mn`, and
started to add `kCvtBias` to the SINGLE-plane collective for it. That path uses `MixGemmNumericArrayConverter`'s
hand-written specializations (hardcoded FP16_TOP_MAGIC_NUM / NEG_72), not the width-templated `MixGemmChunkEmit`, so it
would have meant editing the shipped int4 converter -- which is exactly the code the user said not to touch.

It is also unnecessary, and my own measurement says so: keeping Bias=8 and forming `zero = 8*(d*sc) - dmin*mn` on device
scores **0.0128 step** against fp64 truth, while Bias=0 with `zero = -dmin*mn` scores **0.0148**. The cancelling form is
if anything slightly better. Bias=0 was only ever "one product instead of a product plus an fma".

So the whole job is on the READ path, and it is three things:
1. the smem tile holds 2 bytes of `(sc, mn)` per (group, n) instead of one pre-multiplied fp16 -- IDENTICAL SIZE, so
   SmemLayoutScale, the gmem->smem copy, the fragment partitioning and Params::ptr_S do not change at all;
2. `d`/`dmin` arrive per k-tile in registers (one pair per 256 k, an eighth of the scale plane); the Z smem tile goes;
3. the four transform arms swap `multiplies{}`/`plus{}` for "split two int8s -> gguf_scale_decode.hpp -> the same two
   passes".

The zero disappears as a CHANNEL (its gmem stream, its smem tile, its per-group read). What it does NOT do is fix the
bank conflicts: the concentration comes from the THREAD stride being a multiple of the 128 B bank period, and an int8
element leaves 256 B, still a multiple. `PPU_SCALE_PAD` is the knob for that and has never been timed.

### HARDWARE-VERIFIED (ppu001): the native Q4_K scale/zero loads and decodes bit-exactly

    [rwmoep] q4k_packed.bin: L=1 M=8 N=16 K=4864 gs=32 mode=1 ktype=4 sb=256 z_mul=0 cvt_bias=0
    device decode vs host fp16 reference: 2432 groups | scale 0 bad (max_abs 0.000e+00) | zero 0 bad (max_abs 0.000e+00)
    scale channel bytes: native 6080 vs fp16 9728 (1.60x smaller)     == PASS: 0 ==

First time the gguf's own form reaches the device. The decode the mainloop will call is now verified on hardware, on real
weights, in isolation -- so the collective change gates one thing, not two.

**1.60x, not the 2.67x quoted from bytes_per_group_per_col().** That function counts the 12 raw scale bytes (1.5 B per
group per column) and ignores d/dmin. The fixture stores sc/mn WIDENED TO int8, so it is 2 + 0.5 = 2.5 B against fp16's
4.0. Truly packed 6-bit would be 1.5 + 0.5 = 2.0 B, i.e. 2.0x. The int8 form is a deliberate trade and the reason the
collective change is small: two int8s occupy exactly one fp16 slot, so the scale tile's size, SmemLayoutScale, the
gmem->smem copy, the fragment partitioning and Params are all unchanged while the Z channel disappears outright. The
remaining 0.5 B needs a byte-granular tile -- the expensive route -- and Superblock<KType>::decode already supports that
form (l93 gates it on raw blocks), so it stays available.

Also fixed getting here: build.sh overlays only _overlay_dirs=(gemv_lowbit), so a compiled header under real_weight/ is
absent at box build time while the local front-end check resolves it against the real tree and passes. Compiled headers
live flat next to the harnesses.

NEXT (the mainloop, in this order): (1) the scale tile carries 2 bytes of (sc,mn) -- no layout/copy/fragment/Params
change; (2) d/dmin per k-tile in registers, Z smem tile removed; (3) the four transform arms call rwmoep/gguf_scale_
decode instead of multiplies{}/plus{}.

### OPTION E WINS, and l94 gates it locally. The earlier B recommendation is withdrawn.

B (widen sc/mn to int8 so two codes fill one fp16 slot) dies on a constraint I should have applied from the start: the
device holds the gguf's bytes and nothing else, so an offline widening -- 12 B -> 16 B per Q4_K block, +2.8% model size,
plus a preprocessing pass -- is not available. Online widening just moves the cost.

C (llama.cpp's loader-side decode) is not portable to us either, and for a structural reason: our scale g2s is
`Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementScale>`, and **cp.async cannot do arithmetic between gmem and
smem**. llama.cpp's MMQ is a synchronous tile loader with __syncthreads, so `dm * make_half2(sc8[l], m8[l])` in the
loader costs it nothing; for us it means dropping async on that channel in a multistage pipeline.

E: cp.async the gguf's OWN bytes, decode in registers. The observation that makes it cheap is that a Q4_K superblock's
12 scale bytes are per COLUMN and cover all 8 of that column's groups -- so a lane owning column n reads 3 uint32 ONCE
per k-tile and holds every group's sc and mn. The FINE per-group read does not halve, it DISAPPEARS.

`fold_derivation/l94_native_scale_path.cu`, host-only on l21's stub mma (the CollectiveBuilder cannot be CALLED locally:
that needs -D__HGGCCC__, and then cute's namespace-scope `_` is not device-visible under nvcc):

    today's SmemLayoutScale : (_128,_8,_2):(_1,_128,_1024)      native tile: (_128,_12):(_12,_1)
    (2) DISTINCT addresses per bank over warp0: today 4-way on 4 banks (16 addrs) | native 1-way on 24 banks (24 addrs)
    (1) 1024 (n,group) pairs | codes 0 bad | scale 0 bad | zero 0 bad | non-zero refs 1008
    (3) per lane per k-tile: 12 B of codes (3 uint32) + 1 half2 of (d,dmin) held across 8 groups
        s2r reads per k-tile: today 16 -> native 3 | smem per (group,column): 4.0 B -> 2.00 B

**The bank conflict disappears for free**: the native column stride is 12 B = 3 words and gcd(3,32) == 1, so consecutive
columns spread over banks instead of aliasing. Today's 4-way matches the acu finding of 1.02 conflicts per scale read.

**A metric bug worth recording.** The first version of check (2) counted LANES per bank and reported native as 4-way. But
several lanes hitting the same address is a BROADCAST, and the scale fragment is deliberately k-broadcast (task #1's
stride-0 view), so lanes do share addresses. Counting distinct ADDRESSES per bank is the only version that measures
conflicts; with it, native is 1-way. Same failure family as the rest of this file: a quantity that looks like the one you
want until you ask what the hardware actually does with it.

Caveat: the lane->n map comes from l21's stub mma, i.e. the reconstruction the fold derivation was built on and the box
later validated -- not from the CollectiveOp itself.

### THE ARRANGEMENT IS DECIDED BY MEASUREMENT: 16 B per (superblock, column), all-in-one

Offline reordering is allowed -- B already requires it (`ColumnMajorInterleaved<256>` +
`preprocess_weights_for_mixed_gemm`, i.e. the HBM weights are ALREADY not the gguf's arrangement), so the constraint
that killed option B was never "byte-identical arrangement", it was "do not grow and do not store a second copy".
Reordering does neither.

And a Q4_K block's FIRST 16 BYTES are exactly `d(2) + dmin(2) + scales(12)`. So the plane is `[K/256][N][16]`: one
contiguous 16 B per (superblock, column) carries everything a lane needs, 16 B aligned, and wastes nothing -- 16/8
groups = 2.0 B per group per column, identical to storing 12 and 4 apart. It deletes the (d,dmin) plane, its tile and
`ptr_Z` outright.

l94 measured DISTINCT addresses per bank over warp 0 (shared addresses broadcast; only distinct ones conflict):

    today                    4-way on  4 banks (16 addrs)
    A: 12 B codes only       1-way on 24 banks (24 addrs)   + a separate (d,dmin) plane
    B: 16 B all-in-one       1-way on 32 banks (32 addrs)   one plane, 16 B aligned      <-- chosen

My worry that a 16 B stride would alias (4 words, so n and n+8 share banks) did not happen, and the reason is only
visible by measuring: warp 0's lane set touches **8 CONSECUTIVE columns**, each shared by 4 lanes as a broadcast, so
8 x 4 words fill exactly all 32 banks. With 8 columns spaced 8 apart it WOULD have been 4-way. This is config-dependent:
re-run l94 when the warp shape changes.

Final shape of the change:
    gmem   [K/256][N][16], folded into the existing offline B pass
    smem   (TN, 16) bytes, one 16 B cp.async per column
    reg    one 16 B read per lane per k-tile; per group a shift+mask and two hfma2
    result s2r 16 -> 4 per k-tile, smem 4.0 -> 2.0 B/(group,col), banks 4-way -> 1-way, gmem halved, no extra bytes

NOT transferable to the other formats without re-measuring: 16 is a coincidence of Q4_K's header. Q3_K is
`scales(12) + d(2)` = 14 (pad to 16), Q6_K is `scales(16) + d(2)` = 18 (needs 32 B or two reads), and both have 16
groups, not 8, so the group->superblock divisor and the register count change.

### THE 16 B MUST BE REORDERED TO BE SEPARABLE, and that is now gated too

The native packing's two halves are NOT separable, which a k-tile covering half a superblock exposes:
`get_scale_min_k4` builds `sc[4..7]` from bytes 8-11's low nibbles PLUS bytes 0-3's top 2 bits, and `mn[4..7]` from
bytes 8-11's high nibbles PLUS bytes 4-7's top 2 bits. So groups 0-3 need bytes 0-7 and groups 4-7 need ALL twelve --
at TileK=128 a lane would have to read the whole block for half the groups. There is no contiguous 6-byte half to read.

Since the offline order is ours, make each half self-contained. Still 16 B, nothing grows:

    byte 0-1 d | byte 2-3 dmin | byte 4-9 half0 (groups 0-3) | byte 10-15 half1 (groups 4-7)

A half is 4 sc + 4 mn as 6-bit fields = 48 bits = 6 bytes exactly. ONE Layout gives every position --
`PackBits = Layout<Shape<_4,_2,_2>, Stride<_6,_48,_24>>` with base 32, i.e. bit = 32 + 6*(g%4) + 48*(g/4) + 24*which.

l94 (4), all local:

    PackBits (i,h,which)->bit : (_4,_2,_2):(_6,_48,_24)  base=32
    round trip over 128 columns x 8 groups -> 0 bad | bits outside their own half: 0
    TileK=256: 8 groups/tile, read 16 B (one LDS.128, 16 B aligned) -> 2.00 B per (group,col)
    TileK=128: 4 groups/tile, read 10 B (LDS.32 at 4 + LDS.16 at 8, all aligned) -> 2.50 B per (group,col)
    TileK= 64: 2 groups/tile, read 10 B -> 5.00 B per (group,col)   <-- WORSE than fp16, gate this TileK off

The round trip is the honest chain: native 12 B -> reference sc/mn (the l91-gated `scale_of`/`min_of`) -> new 16 B ->
decode -> must equal the reference. So l91's gate stays valid for the offline leg and the new leg is gated as well.

Padding a half to 8 B would make it one load instead of two, but that is 20 B per superblock, +25% on the channel --
refused. The 6-byte halves sit at byte 4 and byte 10, so every sub-read (4 B at 4, 2 B at 8; 2 B at 10, 4 B at 12) is
naturally aligned.

**TileK=64 must be statically gated back to the fp16 path.** The decode band's winners are TileK=256 (16x128:256 and
16x64:256, where the second number is TN), so they land in the best row; the MoE/prefill sweeps with TileK=64 fall back.

### l95: THE STUB IS THE COLLECTIVE'S OBJECT -- and checking it caught a wrong tile

`fold_derivation/l95_stub_vs_real.cu` asserts TYPE IDENTITY rather than comparing maps: if `SmemLayoutScale`,
`SmemCopyAtomScale` and the mma's `layoutB_TV` / `ThrLayoutVMNK` are the same types, every derived quantity -- including
the lane->n map l94 computes -- is identical by construction. Nothing is called and nothing is printed: every cute entry
point instantiates a device path that references namespace-scope constants nvcc cannot see under `-D__HGGCCC__`, while
`static_assert(is_same_v<decltype(...)>)` is an unevaluated context. **Compiling clean IS the pass condition.**

It immediately caught a real error in l94: I had written the permutation tile as `Tile<_16, 128, 256>`, taking K from
TileK. The collective's is `Tile<C<16>, C<128>, C<64>>` -- **K is 64, not 256**, and the compiler printed both types side
by side. Fixed in both probes. The bank rows happen to be unchanged (warp 0 still touches 8 consecutive columns, still
1-way on 32 banks), but they were previously resting on the wrong object.

`TiledShape_MNK` is not a member of this cute's TiledMMA -- `layoutB_TV` identity covers it.

### l94 (5): the per-format table, computed from Traits and NOT inherited from Q4_K

    Q4_K  G= 8 6+6 bit  12.0 B codes + 4 hdr = 16.0 B/superblock/col -> 2.000 B/(group,col)  vs fp16 4.0 = 2.00x
    Q3_K  G=16 6+0 bit  12.0 B codes + 2 hdr = 14.0 B                -> 0.875               vs fp16 2.0 = 2.29x
    Q2_K  G=16 4+4 bit  16.0 B codes + 4 hdr = 20.0 B                -> 1.250               vs fp16 4.0 = 3.20x
    Q6_K  G=16 8+0 bit  16.0 B codes + 2 hdr = 18.0 B                -> 1.125               vs fp16 2.0 = 1.78x
    TileK=128 halves: 6.0 / 6.0 / 8.0 / 8.0 B -- WHOLE BYTES for all four, so separability holds everywhere

Q4_K and Q3_K fit one 16 B read (Q3_K with 2 B slack). Q2_K and Q6_K do NOT -- their codes alone are already 16 B, so
`d`/`dmin` must be a separate small plane, keeping the code read 16 B aligned. **Q2_K has the largest saving (3.20x), not
Q4_K**: 4+4 bit codes still carry two fp16 today. Q6_K has the smallest (1.78x).

### l94 (6): THE GMEM SIDE IS NOT THE HARD PART AFTER ALL

I called the gmem reshape the one high-risk site. Measured, it is the easiest. Today's scale g2s is
`make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>, Layout<(TN/8, SK)>, Layout<(_8,_1)>)` --
128 threads x 8 halfs = 16 B each, covering TN*SK*2 = **2048 B**. The native tile is TN columns x 16 B = **also 2048 B**,
because the Z tile disappears. So the copy stays one uint128 per thread and only the shape changes:

    Layout<(TN, _1)> threads, Layout<(_1, _16)> values, element uint8_t   ->  thread t reads bytes [16t, 16t+16)

    (6) gmem g2s, native shape: 128 threads x 16 B = 2048 B (today's S tile is 2048 B)
        contiguity 0 bad | 16 B alignment 0 bad | coalescing (t -> 16t) 0 bad | gaps 0 | overlaps 0

That is 16 B aligned per thread AND consecutive threads on consecutive chunks -- one fully coalesced 2048 B burst, where
today's map is strided (thread t covers N-offset 8*(t%16), K-offset t/16). The atom in the probe is DefaultCopy because
what is checked is make_tiled_copy's algebra, which does not depend on the instruction; what the uint128 atom requires is
16 B per thread, which the value layout gives.

**So all three pre-collective unknowns are now closed, locally:** the stub is the collective's object (l95, and it caught
a wrong tile), the per-format numbers come from Traits (l94 (5)), and the gmem reshape is a shape change with the same
byte count and better coalescing (l94 (6)). What remains is wiring, and the largest single edit is now the fragment
scatter -- value -> n, which is the map l94 already computes from partition_S(identity).

### STEPS 1-3b ARE IN (default-off, front-end clean both ways). One shortcut is RULED OUT.

* **1** `SmemLayoutScalePacked` + `GmemTiledCopyScalePacked` as types.
* **2** SharedStorage's scale member becomes bytes over the packed tile and **the zero tile drops to zero elements** --
  `mn` rides in the same unit. Gate: `MOEG_SMEM=1`'s SharedStorageSize must fall by exactly the zero tile.
* **3a** the decode moved INTO actlize as `cutlass/gguf_packed_scale.h` (PackBits, code_of, put_code,
  int_to_half_small, `group_of<ScaleBias, HasMin>`), because the mainloop needs it and the mainloop is there; the harness
  header re-exports it, so l94 still gates the shipped code and the bit map exists once. Plus `load_packed_units`,
  `decode_packed_group`, `packed_fill`, and `(sSp, tCcS)` APPENDED to the extra-info tuple (never inserted -- every
  consumer reads it positionally).
* **3b** all three per-group `copy` sites become `packed_fill`: the coarse site (where `scale_k_idx` is flattened over
  (stage, group) and has to be split), the FINE ConvertAndScale arm, and the FINE ConvertAndScaleWithZero arm where BOTH
  reads vanish. `PPU_SCALE_PREFETCH` is forced off -- it hides a read that no longer happens.

`decode_packed_group` looks each element's column up in the coordinate tensor instead of using l94 (7)'s measured run
length. The run length is real (period 8, two slots) but it moves with the warp shape, and a read-off-the-object version
was available for free.

**RULED OUT, do not retry: making the 16 B unit ONE `uint128_t` element.** It would have kept the gmem tensor at
(N, nsb, L) and every residue/predication line unchanged, which is why it was attractive. cute rejects it:
`Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<uint128_t>, uint128_t>` with a `(_1,_1)` value layout fails
`copy_traits.hpp`'s "dst failed to vectorize into registers" -- an element as wide as the whole vector is not modelled.
The element must stay `uint8_t` with a `(_1,_16)` value layout, which is what l94 (6) verified anyway.

**Still open (step 3c):** the g2s wiring itself -- `Params::ptr_S`, the `mS_nkl`/`gS` construction and the two
`copy(gmem_tiled_copy_scale, ...)` sites. With a byte element the tile's second mode is BYTES, not k, so the
`scale_residue_k` / `scale_valid` arithmetic around `tScS(0,0,0)` has to be re-derived rather than inherited -- that is
the one place where the fp16 path's logic does not carry over unchanged.

### PHASE 1 IS VALIDATED ON HARDWARE (ppu001). Both gates green.

    [q3-bconcat-real] real_q3k_concat.bin M=128 N=256 K=256 gs=16
      1..5 (ScaleZero ladder)                       bad=0/32768   MATCH
      6 (64,128,128) w64x64  ScaleOnly, bias in converter   bad=0/32768 max_rel=5.440e-02 MATCH
      7 (64,128, 64) w64x64  ScaleOnly + F1=2 F2=4          bad=0/32768 max_rel=5.440e-02 MATCH
      last rung (7) vs native Q3_K golden: bad=0/32768 MATCH
    [q65-bconcat] Q6 two ScaleOnly bias 32 rows and Q5 one ScaleOnly bias 16 row: bad=0/32768 MATCH
      => 0 failing configuration(s)

So `kSymBias2Plane` (Q3 4, Q6 32, Q5 16), the `kCvtBias` mode rule, and the `detail::NoZero` slot that made 2-plane
ScaleOnly reachable at all are all correct against an INDEPENDENT golden -- for Q3_K, the native one straight out of the
gguf. The redundant zero channel is gone for the symmetric formats.

### THE PACKED-SCALE SMEM GATE I WROTE WAS UNMEASURABLE, twice over

`MOEG_SMEM=1 test_moe_grouped_verify` printed 61840 B with and without PPU_PACKED_SCALE. Neither number is wrong:

1. that binary was a STEP 1 build, which is types-only and changes SharedStorage by design not at all; and
2. more fundamentally, `test_moe_grouped_verify` runs `FinegrainedScaleOnly`, where `elements_per_smem_zero()` already
   returns 0 -- **there is no zero tile to remove**, and the packed tile is exactly the same size as the fp16 scale tile
   it replaces (16 B carries 8 groups = 2 B per group, which is one fp16 per group).

So the smem saving is ENTIRELY the zero tile, and only a ScaleZero run can show it. The gate has to be
`SK_QUANT=2` (ScaleZero) at `gs=32 / TileK=256`, where the delta is `TN*SK*2*Stages` = 128*8*2*2 = 4096 B.

And a second thing that config exposed: at `gs=128 / TileK=128`, `Scale_TileK == 1` -- a k-tile needs ONE group while the
packed unit carries eight, so packed costs 3072 B against fp16's 384 B. **Packed only pays when Scale_TileK is large**;
gs=32/TileK=256 (8 groups per tile) is the sweet spot and gs=128/TileK=128 is the worst case. That belongs in the static
gate next to the TileK==64 fallback: require `Scale_TileK * gs == TileK` AND `Scale_TileK >= 4`.

## THE PACKED CHANNEL RUNS END TO END ON ppu001, ON REAL Q4_K WEIGHTS

    PPU_DEFS verified on test_q4k_packed_gemm's compile command: -DPPU_PACKED_SCALE=1
    plane round trip through group_of<0,true,8>: ok
    host affine model vs the fixture's golden: max_rel=4.789e-04 ok
    row2 reads the gguf's own 16 B units
    rowA scale-only 64x64:128    bad=0/4096 max_rel=1.211e-01 MATCH
    rowB affine   64x64:128      bad=0/4096 max_rel=3.476e-01 MATCH
    rowC affine   16x128:256     bad=0/4096 max_rel=1.457e+00 MATCH        <-- the packed row
    == PASS ==

The device now receives the gguf's OWN scale form -- one 16 B unit per (superblock, column) holding d, dmin and the 8+8
six-bit codes -- and decodes it in registers. rowA and rowB are a built-in CONTROL: their Scale_TileK is 4, so
kPackedScaleOn is false and they stay on the fp16 path even in the packed build. They still MATCH, so the macro is not
leaking into units it should not touch.

**A second result, independent of this work:** rowB is the first check of the single-plane int4 AFFINE path against an
external golden on hardware. Everything validated before it was ScaleOnly (`bench_cutlass_w4a16::xcheck_grouped` passes
zeros=nullptr), and the plan file recorded test_moe_grouped_real's Q4_K fixture as box-pending. The affine path is correct.

**Not yet explained:** rowC's max_rel is 1.457 against rowB's 0.348 on the same math. bad=0 because the tolerance has an
absolute floor and the large ratios sit on near-zero goldens, but the 4x wants attributing: either the tile's different
fp16 accumulation order, or the extra rounding the packed path takes by forming the zero on device (measured offline at
0.0128 step versus 0.0085 for the offline-precomputed zero). The reference build's rowC number settles it in one read.

### THE BUG THAT COST THREE ROUNDS, and the signature that should have caught it in one

`moe_grouped_ppu.cuh:352` selects the interleaved B layout with `n % 256 == 0 && k % 256 == 0`. The fixture was N=128, so
the driver took plain ColumnMajor while the harness had preprocessed B with interleave-256: an interleaved buffer read as
if it were not one.

The signature was there from the first run and I did not read it: the output was **invariant under the quant mode AND the
tile** -- rowA (ScaleOnly, zeros=nullptr), rowB (affine) and rowC (a completely different tile) were bit-identical. Output
that does not move when you change the quant mode or the tile is not a scale bug; it is upstream of both. Instead I read
"1991/2048 bad" as numerics and spent three rounds on the scale channel, the zero channel and B's nibble packing.

Reverse-engineering in python did rule out the relabelling explanations before the real cause was found: no combination of
A as [M][K] or [K][M], q as [N][K] or [K][N], and the scale plane as [scale_k][N] or [N][scale_k], with or without the
zero, reproduces the observed D[0,:4]. The harness now refuses N % 256 != 0 and prints why.

### PACKED PERF, ROUND 2: the register decode bought 9.4 us and left +43%

Same config, decode band L=64 top-k=8 N=K=2048 gs=32 SK_QUANT=2, macro verified on the compile command both times:

    16x64:256 w16x16 s4 S=4    baseline 33.88    packed 57.83 (byte-pointer decode)    48.40 (register decode)
    16x64:256 w16x16 s4 S=8    baseline 47.90    packed 78.05                          67.02
    winner splitk=1            baseline 20.22    packed's best is a TK=64 row at 27.36
    control 16x32:64 S=1       baseline 25.51    packed 27.36   (+1.85, the unconditional construction, not yet fixed)

So byte-addressing the register array was real and worth 9.4 us on that row, and something else still costs +43%.

**The next suspect, and it is not to be settled by reasoning:** `PackedLaneState` is passed BY REFERENCE through two or
three calls. If it is not fully inlined and SROA'd, the struct itself lives in local memory and every `st.u[S][w]` is a
local access again -- i.e. removing the byte pointer may only have moved the local traffic from the extraction to the
state. acu answers that directly (local load/store counts, registers per thread, actual blocks/CU), so the same-config
A/B against the profiler comes before any further edit.

A timing-only ablation is in place as a cross-check: `PPU_PACKED_SCALE_NOP=1` makes the decode return constants and skips
the unit read entirely, so the numbers are deliberately wrong and only the clock matters. Back to baseline means the cost
is the state and the decode; still slow means it is the unconditional tensor construction, the byte-element g2s
TiledCopy, or the larger Params, and looking at the decode again would be wasted.

If it IS the state, the structural fix is to decode all Scale_TileK groups ONCE per k-tile into a fragment-shaped
register array indexed by a compile-time group -- about 64 extra registers per lane on top of ~140, which is a real
trade-off to price, not an obvious win.

## F IS CORRECT ON HARDWARE, AND COSTS 24.17 vs 20.2 -- the remaining gap is one exposed LDG

rowA/rowB/rowC all MATCH in the packed build with the native plane, so the gguf's own scale/zero form runs end to end
through a decode-at-load-time design. rowA and rowB are the control: they keep the fp16 path even in the packed build and
still match, which is what says the read side really is untouched.

    decode band winner, same config both sides (16x128:256 w16x16 s2 S=1)
      baseline            20.12 / 20.22 / 20.36 us   (three runs)
      E, register decode  26.89   -- and TK=256 fell off the board entirely
      F, loader decode    24.17   -- TK=256 back at the top, i.e. the inner loop is clean

E -> F recovered 2.7 us and the inner loop is now byte-identical to fp16, so the remaining +19% is almost entirely the
LDG that F exposes: the loader does load -> wait -> decode -> store in one thread, at the point where a cp.async used to
be ISSUED. cp.async returns immediately; a plain load does not.

TWO INDEX BUGS ON THE WAY, both the same shape -- one quantity, two derivations, only one of them checked:
  * the store used SmemCopyLayoutScale's flattened (n, 1, stage*SK+g) while the tensor it got was SmemLayoutScale's
    (n, group, stage). Two functions build a tensor called `sS` with different layouts. The hardware caught it as
    "Exception TSM out of range", which is the good failure: a fault, not a wrong number.
  * `scale_load_k` is already a TILE index -- partition_S leaves the last mode selecting which block of Scale_TileK
    groups a call loads -- so dividing it by Scale_TileK folded all 19 superblocks onto 0..2. The k bound written a few
    lines earlier (scale_residue_k = nsb * Scale_TileK) already encoded the correct reading, so the file disagreed with
    itself and only the bound was right.

### F' (chosen): cp.async the raw bytes into staging, decode at the barrier that already exists

The gmem side goes back to async and the decode stays amortised per column. The decode slot is the pipeline's own
`cp_async_wait` + `__syncthreads` at the last k_block: after the wait the staged bytes have landed, before the sync the
planes are private, and the sync publishes them -- no new barrier. mma() already holds shared_tensors and thread_idx, so
this needs no tuple plumbing at all.

Cost, stated because it is a real trade and the register route was rejected on exactly this basis: staging is
TN * 16 * Stages bytes, which at Scale_TileK == 8 equals ONE scale tile, so the channel goes from two smem tiles to
three. At TN=128/Stages=2 that is 4 KB against A's 49 KB and B's 12 KB.

### F' REFUTES THE LATENCY HYPOTHESIS

    decode band winner, SK_QUANT=2       baseline 20.12-20.36    F 24.17    F' 25.00 / 24.79

Restoring async on the scale channel did not help, and the 4 KB of staging it costs made it marginally worse. So the
exposed LDG in F was NOT the main term -- which was my hypothesis and it is now refuted.

What both versions share is the decode itself, and the part I had not priced is the STORES: per CTA per k-tile the loader
writes 64 columns x 8 groups x 2 planes = 1024 STS.16, where the fp16 path has cp.async write the same two tiles with
ZERO instructions. Plus 512 decodes x ~11 instructions. Roughly 208 added warp-instructions per CTA per k-tile, ~426K
over the kernel -- an order less than E's ~3M, consistent with F being much better than E, and still real.

So the trade is not "ALU for LDS" as in E; it is "explicit stores plus decode ALU for a free cp.async". With LSU at 6%
busy that is again the wrong direction, and the honest reading is that this optimisation does not pay in the decode band
at gs=32. What it buys -- gmem halved, the zero tile gone, no offline pre-multiplication, bank conflicts gone -- only
becomes visible where smem capacity or occupancy is the binding constraint.

acu next, same config both sides, in this order: smem/block and measured blocks/CU (F' adds 4 KB per CTA and 4 blocks/CU
means 16 KB, which could cross a threshold and alone explain 4-5 us), then tsm.st, then the instruction totals, then the
stall mix. If occupancy is the cause, staging only needs ONE stage rather than Stages -- the decode runs immediately
after the wait -- which is a small change worth trying.

---

## The occupancy line of attack is closed, three ladders deep (measured)

`64 8 2048 2048 32 3`, `SK_QUANT=2`, one run so the rows are comparable. Reference: `16x64:256 s2 S=1` = **20.52 us**
(the previous run put `16x128:256 s2 S=1` at 20.11 -- within the ~13% same-config spread, so TN=64 and TN=128 are
TIED, and my earlier "fewest blocks wins" was reading noise).

**TileK = 128 lost, so the prologue hypothesis is refuted by its own pre-committed falsifier.** The prediction written
into CMakeLists before the run: fill share is `Stages/(K/TK + Stages)`, 20% at TK=256 against 11% at TK=128, so if the
pipeline fill were the term these rows should win ~9% (≈18.7 us).

| row | S=1 | S=2 | S=4 | S=8 |
|---|---|---|---|---|
| `16x128:128 s2` | 24.16 | 24.55 | 27.87 | 29.41 |
| `16x128:128 s4` | **22.06** | 23.63 | 27.35 | 31.51 |
| `16x64:128 s2` | 22.94 | 22.48 | 26.64 | 27.79 |
| `16x64:128 s4` | **21.85** | 27.11 | 28.27 | 33.34 |

Best TK=128 row is 21.85, i.e. **6.5% WORSE** than 20.52. And the data refutes the hypothesis a second time from the
inside: **s4 beats s2 at both TileN** (22.06 vs 24.16; 21.85 vs 22.94). s4 has MORE prologue (20% against 11%), so if
fill were a cost term s4 would lose. It wins by 8-9%. Fill is not a cost; depth is a benefit.

**Therefore the persistent kernel is off the table for this band**, on the falsifier recorded next to those config
rows. It would not have bought the usual thing either -- 128 tiles against 128 CTAs is exactly one wave, no tail.

**Split-K lost everywhere, and occupancy is monotonically harmful past 28 warps/CU:**

| warps/CU | 14.2 | 28.4 | 56.9 | 113.8 |
|---|---|---|---|---|
| `16x64:128 s2` | 22.94 | 22.48 | 26.64 | 27.79 |

The bench's own line: `speedup from split-K: 0.938x`. And **HBM sits at 31-35% in every single row** regardless of
concurrency -- 8x the warps bought no bandwidth at all. So the kernel is neither bandwidth-bound nor
memory-latency-bound, and Little's Law was right about the headroom and wrong about the lever. Three independent
ladders now say so: TN (32/64/128), TK (64/128/256), S (1/2/4/8).

## What it actually is: the dequant pipeline, at 14.5 instructions per mma

From the pinned base acu, `v.lop3.i 546,816 + v.bfi.i 270,336 + v.fma.f16 745,472 + v.add.f16 335,872 = 1,898,496`
against `v.mma.f32.f16.m16n16k16 = 131,072`. That is **14.5 instructions per mma instruction, and 43% of the whole
kernel**.

CALL IT THE DEQUANT PIPELINE, NOT THE CONVERTER, because I first wrote the latter and it does not survive its own
arithmetic. `v.fma.f16` is shared: per warp per k-tile there are `8*(WN/16)*(TK/16)` = 128 B elements = 64 half2, and
each half2 takes ONE fma for the converter's magic correction and ONE for the scale/zero application. So the 745,472 is
split roughly half and half (the per-element model predicts 1,048,576, i.e. it overcounts 1.4x, so "roughly" is the
strongest word available). The consequence is a size, not a quibble: **TODO #18 attacks only the scale-application
half of the fma, so it reaches at most about a quarter of the 43%** -- call it ~10% of instructions -- while `WM`
attacks the element COUNT and therefore both halves plus the lop3 and bfi.

It lands on the closed form already in fold_derivation/README.md:
`cvt elems per mma = 128/WM` -- WN and TK cancel EXACTLY, only WM survives. At WM=16 that is 8 elements per mma at ~2
instructions each.

This explains every measurement above at once: the chain is per-warp ALU, so more warps cannot shorten it (splitk),
more CTAs cannot shorten it (TN), and more k-tiles cannot shorten it (TK). And `moe_ok` requires `WM <= TM`, so
**decode's TM=16 pins WM at 16, the worst value of the only surviving variable.**

Three ways in, and only one is cheap:
1. `WM = 32` -- halves it, and needs `TM >= 32`, i.e. 31/32 of the A tile padding at batch 1. Worth measuring against
   the padding cost rather than assumed dead.
2. **Fewer instructions per converted element** -- TODO #18, fold `(-1024, zero-point, 2^-b)` into one `(s', b')` pair
   so the dequant is a single `hfma2`. This is the one that attacks the 43% directly and does not fight `moe_ok`.
3. Don't convert in this band at all: the CUDA-core GEMV is already 22.27 us against the tensor-core 20.74 us, i.e.
   the same place. Consistent with "GEMV is ALU-bound, not bandwidth-bound" -- both kernels are ALU-bound and both land
   at 20-22 us, which is what a 43%-converter kernel and a no-tensor-core kernel converging looks like.

---

## rowC's regression: subnormal d, and the fix that costs no mainloop instructions

**The bisect.** `PPU_PACKED_PAIR=0` (scalar `group_of_words`, everything else identical) makes rowC MATCH again. So
the native transport, the shared stores and the loop are clean and the defect is inside `group_pair_of_words`, whose
only device-specific content is two inline-asm instructions with zero local coverage -- the host gate l96 compiles
under nvcc, where `CUTLASS_GGUF_PACKED_F16X2_ASM` selects the scalar fallback.

**The measurement, on `real_weight/q4k_packed.bin` (blk.11.ffn_down.weight, L=1 N=256 K=4864 gs=32, nsb=19):**

| quantity | range | subnormal |
|---|---|---|
| `d` | 1.585e-05 .. 9.484e-04 | **3914 / 4864 = 80.5%** |
| `dmin` | 1.096e-04 .. 1.117e-02 | 0 |
| `d*sc` (what the fp16-plane offline stores) | 5.65e-04 .. 5.98e-02 | **0 / 38912** |
| `dmin*mn` | 0 .. 0.704 | 0 / 38912 |

fp16's smallest normal is 6.104e-05. So **Q4_K's `d` is subnormal for most superblocks, `dmin` never is, and neither
product ever is.** That last row is why the shipped path has never met this: the offline forms `d*sc` in fp32 and
stores only the normal product, so no subnormal fp16 has ever reached an instruction. The packed decode is the first
thing on this path to multiply BY `d` on the device.

**AND IT IS NOT THE CAUSE -- I refuted my own hypothesis by simulating it, which I should have done before writing it
down.** Feed the fixture through the affine model twice, once whole and once with every subnormal-d superblock losing
its scale term, and count outputs past the harness's own tolerance (`|d-g| > 2e-2 + 6e-2|g|`):

    FTZ-on-d simulated:  bad = 3626 / 4096 = 88.5%          observed on hardware: 724 / 4096 = 17.7%

Five times too much damage, and it could not be otherwise: 80.5% of superblocks losing their scale cannot leave 82% of
outputs inside tolerance. **Subnormal d is a real latent hazard for any device path that multiplies by d -- it is not
what broke rowC.** The `.noftz` probe stays because the exposure is real, but it is no longer the leading explanation.

**The leading explanation now is the inline-asm CONSTRAINT.** Both wrappers used `"=r"`, which permits the destination
to alias any input. In `packed_decode_stage` the multiplier `m2` is live across all EIGHT unrolled groups and dies at
the last one, so an allocator may choose `dest == m2` for the FINAL fma only -- corrupting some groups and not others,
and differing between builds. That is exactly the observed shape (bad=128 in one build, 724 in another), and it is the
only candidate on the list that predicts a partial, build-sensitive failure; `volatile` prevents removal and CSE but
says nothing about operand aliasing. `"=&r"` is already in the tree, so **the next build may simply pass**, and
`test_ppu_f16x2_probe` section (4) asks the question directly by running each op with the destination tied to each
input in turn and comparing against the non-aliased form -- no hardcoded expected values, so it cannot be wrong about
the arithmetic.

**Why that hypothesis fits and the other candidates do not.** Operand order matches the reference uses; the
`+ 0x64806480` fold cannot carry between lanes (`0x6480 + 63 = 0x64BF`); every decode index is compile-time; the two
straddles (group 1's min at bit 62, group 6's scale at bit 92) are handled by the same condition the long-standing
scalar path uses; and the call site passes `ZMul=8`, `ScaleBias=0`, `HasMin=true` correctly.

**The subnormal exposure, kept for the record.** This ISA carries an explicit `.noftz` qualifier on an f16x2 op
(`cutlass/functional.h:830`, `ppu.atom.gpu.global.add.noftz.f16x2`), which only makes sense if the DEFAULT flushes. A
flushed subnormal multiplier zeroes the SCALE lane while the zero lane -- whose `dmin` is normal -- survives, and each
output sums over 19 superblocks so only the subnormal ones are lost. That predicts a PARTIAL failure dominated by
scale, which is what rowC shows (bad=128 then 724 of 4096, not everything). It is a hypothesis until the probe runs.

**The gate that settles it**: `test_ppu_f16x2_probe` -- no GEMM, no shared memory, no mma, so a failure cannot be
confused with tile plumbing. It compares each asm op against the scalar op it claims to equal, with a subnormal
multiplier in one lane and a normal one in the other (the asymmetry is the signature), and reports the per-group
decode split by group, because Q4_K has exactly two straddling fields -- group 1's min at bit 62 and group 6's scale
at bit 92 -- so FTZ and a straddle bug produce different tables.

**The decision tree.**
1. `PPU_F16X2_NOFTZ=1` assembles AND the probe goes clean -> keep the packed decode, done.
2. The mnemonic is rejected -> the fallback below.
3. It assembles and does NOT fix it -> the hypothesis is wrong; the probe's per-group table says whether it is the
   straddle instead.

**The fallback, and it costs ZERO mainloop instructions.** Multiply `d` and `dmin` by one per-tensor `2^k` offline and
undo it once in the epilogue: `moe_grouped_ppu.cuh` already uses `LinearCombination` with a scalar `alpha` (its fusion
args default to alpha=1, beta=0), so the inverse is a launch parameter, and `2^-k` is a power of two, so it introduces
no rounding at all. The headroom on this tensor, both ends computed rather than assumed:

* lower bound `min(d) * 2^k >= 6.104e-05` gives **k >= 2**;
* upper bound comes from the dequantised B value `scale*q + zero`, at most ~1.6 today against fp16's 65504, giving
  **k <= 14**;
* **k = 8** sits in the middle: `min d` becomes 4.06e-03 (66x above the threshold), `max dmin` 2.86, `max B` ~410.

`k` must be computed per tensor from that pair of bounds and carried in the fixture header, not hardcoded -- the
margin is a property of the weights, and on this tensor k=2 clears the floor by only 4%.

**What this says about the native format generally.** The obstacle is not the implementation: it is that the gguf's
`d` is routinely subnormal in fp16, so ANY device path that multiplies by it is one FTZ qualifier away from silent
zeros. The load-time conversion kernel of the handoff's section 10 avoids it by construction, since it forms the
product in fp32 and stores the normal result. That is now a correctness argument for that design, not only a
performance one.

### The probe ran, and it clears the decode entirely

`test_ppu_f16x2_probe` on ppu001, all four sections:

    (3) CUTLASS_GGUF_PACKED_F16X2_ASM = 1        the device build really does execute the asm, not the fallback
    (1) sub.f16x2   29 cases -> 0 disagree       incl. subnormal multipliers, signed zeros, the decode's constants
    (1) fma.f16x2   29 cases -> 0 disagree
    (2) packed vs scalar decode on device: 0 of 38912 (unit, group) disagree, per group g0..g7 all 0
    (4) destination/input aliasing changes the answer in 0 of 5 aliased forms

**So `group_pair_of_words` is bit-for-bit correct on the hardware, over the real fixture, in every group including the
two that straddle a word.** Three hypotheses die at once: FTZ (already refuted by simulation, now also by measurement),
the `"=r"` aliasing one, and any suspicion of operand order or the straddle. Section (3) also retires a doubt worth
naming: the asm really is what runs, so the timing conclusions drawn from the packed path are about the thing they
claim to be about.

**Which makes the bisect result the interesting fact, not the answer.** `PPU_PACKED_PAIR` selects between two
functions now proven to produce identical bits, so it cannot change the result through arithmetic -- yet it does. What
it also changes is instruction count, register pressure and scheduling inside a mainloop that has a `cp_async_wait`, a
`__syncthreads` and 256 threads running a 128-thread copy. A defect that is invisible in isolation and appears only
under that context is a race or an out-of-range partition, not a formula.

**Next, in order.** First rerun rowC on the current tree: `"=&r"` is in place, and an aliasing hazard can be real in
the mainloop (where the allocator is under pressure) while invisible in a probe with three live values. If it still
fails, the first suspect is the one flagged in review and never resolved: `GmemTiledCopyScalePacked`'s thread layout is
`Layout<Shape<Int<Scale_TileN>,_1>>` -- 128 threads at Scale_TileN=128 -- while `get_slice(thread_idx)` is called by
all `size(TiledMma)` = 256 threads, so threads 128..255 partition outside their own stage of `smem_scale_raw`. That is
independent of `PPU_PACKED_PAIR`, which is an argument against it, but it is the only unexamined write into that
buffer and it must be ruled in or out before anything else is tried.

---

## Corrections from an adversarial perf review. Six, and two of them void conclusions I committed here

**1. TileN is NOT an occupancy axis, and neither is TileK. Warps per CTA is `(TM/WM)*(TN/WN)` = TN/16 here, and CTAs
is `8 * 2048/TN`, so the product is 1024 launch warps for EVERY row of the TileN ladder:**

    16x32:256   512 CTAs x 2 warps = 1024        16x64:256  256 x 4 = 1024        16x128:256  128 x 8 = 1024

TileN redistributes the same work into fewer, wider CTAs; it does not change concurrency. TileK is not in the grid at
all. **So "occupancy refuted three ways" is wrong: only split-K varies concurrency, and the three ladders are one.**
And `lowbit_moe_bench.hpp:250` ALREADY SAYS "TileN and TileK CANCEL" -- the repo had derived this and I wrote the
opposite into CMakeLists and into this file. Same defect as everywhere else in this task, in a file I was quoting from.

**2. The part is 72 CUs, not 64** (`lowbit_moe_bench.hpp:193`, confirmed independently by acu's wave arithmetic,
1024 - 3*288 = 160; the bench divides by 72 at `moe_splitk_bench_common.hpp:209`). Consequences: 128 CTAs is **1.78
waves**, not one, so **"exactly one wave, no tail to reclaim" -- my argument against persistence -- is void**; a tail
exists and the second wave fills 56 of 72 CUs. The bench's `warps/CU` column is launch-warps / 72, NOT achieved
occupancy, so 14.2 -> 113.8 is not an occupancy measurement and I read it as one. Every blocks/CU figure I wrote used
64 and is off by 1.125x.

**3. The s4-versus-s2 result is 8.7% on one row and 4.75% on the other, not "8-9%" on both** (24.16 -> 22.06 and
22.94 -> 21.85). Against a ~13% same-config spread the second is not usable on its own. The prologue conclusion is
weakened accordingly: what the data disfavours is a LARGE fill-only bottleneck, not a 9% fill term. The fill model was
wrong anyway -- the mainloop issues `Stages - 1` stages, not `Stages`, and has a drain loop, so
`Stages/(K/TK + Stages)` was never the code's fill fraction.

**4. "+12.9% fully accounted" is too strong.** The decomposition it rests on never ran: the E-era
`PPU_PACKED_SCALE_NOP` was deleted by the F/F' rewrites and only reintroduced in `b5567944`. No baseline/NOP/full
triple exists yet. Instruction share is also not cycle share -- those instructions can issue while memory operations
are outstanding.

**5. The floors, computed properly.** Compulsory traffic 21,037,056 B at 2766 GB/s = **7.606 us**, so 20.11 us is
**2.64x** off it. MMA 131,072 x 2 x 16^3 = 1.074 GFLOP at the repo's 500 TF/s = **2.147 us**, i.e. **9.36x** off. The
traffic floor is much the closer of the two, which points at a memory-dependency path rather than at issue.

**6. "the +70% was local memory" overstates its own commit.** Byte pointer 57.83 us, register extraction 48.40,
baseline 33.88: the fix recovered 9.43 of 23.95 us of excess, i.e. 39%, and left the kernel 42.9% slow. Local memory
was one large subproblem, not the whole gap.

### What the review says is actually the limiter, and the experiment that would settle it

Not peak bandwidth: a **memory-dependency / request-latency path** in the mainloop, with dequant/issue work second.
The in-tree acu pair at `TODO.md:1256` is better evidence for this than anything I produced -- achieved warps really
rose 14.02 -> 26.84 while time went 20.18 -> 20.96 us and **Memory Dependency rose 0.98 -> 1.772**. That is a measured
stall reason moving with concurrency, which is what "adding warps does not help this path" looks like.

**The missing ablation is not the one I built.** `PPU_PACKED_SCALE_NOP` removes the ADDED packed decoder. What would
test "dequant is the limiter" is a `PPU_B_DEQUANT_NOP` that keeps B loads, scale loads and the MMA count but removes
the int4 conversion and affine chain from the BASELINE. That does not exist.

### And what to stop doing

Calling TileN and TileK occupancy experiments. Using launch-warps/CU as achieved occupancy. Inferring "not
latency-bound" from low HBM utilisation. Inferring a bottleneck from opcode share without stall or pipe evidence.
Comparing sub-13% timings across runs or mixed shapes. Spending rounds on 0.6-4% packed micro-optimisations before the
NOP decomposition runs.

### l97's premise was wrong: the caller already wraps. Four hypotheses, four refutations, cause still unknown

`mma()` calls `partition_extra_inputs(..., thread_idx % (Scale_GmemCopyThrLayoutH * Scale_GmemCopyThrLayoutW))` at
line 715 -- **the thread index is already in [0, 128) before the packed slice ever sees it.** l97 measured
`get_slice(t)` for t up to 255, a call that does not exist in the tree, and the modulo I then added is a no-op. Found
by grepping the caller, which takes thirty seconds and which I did not do before committing a "fix".

So rowC's intermittency has now survived four hypotheses: FTZ (refuted by simulating it -- 88.5% predicted damage
against 17.7% observed), `"=r"` aliasing (refuted by rebuilding with `PPU_F16X2_EARLYCLOBBER=0` and still passing),
out-of-range partition (refuted above), and the paired store race (real, but removed in fce4fa41 while bad went 128 ->
724). **The cause is unknown and the current pass is not evidence of a fix.** Treat rowC as flaky until it has passed
N times; keep `PPU_PACKED_PAIR` for bisecting when it returns.

### Where the bank conflicts actually are, and they are not mine

The decomposition is almost exact:

    441,344 baseline shared loads  -  168,960 A/B tsm.ld.swzl  =  272,384 scalar scale/zero loads
    278,528 conflicts / 272,384 loads = 1.0226

**Essentially every scalar scale/zero load is conflicted**, and they are the FINE per-group reload pair in
`transform_B_kblock` -- the shipped path, identical in both builds, so they explain NONE of the 2.59 us delta.

The map, derived from the real TV layout rather than guessed: `h(t) = 256*(t mod 4) + floor(t/4) + C`. A 256-half
thread stride is 128 bank words, exactly four 128-byte bank periods, so it vanishes mod 32 and the lanes land on banks
`0,0,1,1,2,2,3,3` with **four distinct words on each of four banks** -- confirming l94's "4-way on 4 banks (16 addrs)"
(l94 and l95 were both recompiled and rerun to check). Transactions reconcile too: 272,384 x 4 = 1,089,536 plus
819,200 / 168,960 = 4.85 per swizzled load = 1,908,736 against the measured 1,945,600.

**The packed raw read is clean**: +4,608 loads, +36,864 transactions = exactly 8 per load, **+0 conflicts**. The old
claim that the native form "removes the conflicts" was true of the read-side register decode, not of this staged
decode -- F' decodes into the same fp16 planes, so the consumers keep the old conflicts.

**The packed STORE conflicts count out exactly**: `4 decoder warps x 8 groups x 2 planes x 9 passes x 128 CTAs =
73,728`, the observed increment. Cause: 32 lanes store 32 adjacent 2-byte values, occupying 16 bank words, two
halfword stores per bank, and stores cannot broadcast.

**The ownership-safe store fix is not column pairing.** It is to store the same column's `(scale, zero)` as one 32-bit
word in an INTERLEAVED decoded tile and give the consumers even/odd halfword views: every thread derives both halves
from its own unit, and 32 consecutive 4-byte stores hit all 32 banks once. It changes the physical layout, so the read
bank probe must be rerun -- not a free patch.

**Worth attacking?** 278,528 / (128 CTAs x 8 warps) = 272 conflict events per warp against ~266 scalar loads;
removing them saves roughly 272-798 bank-service cycles per warp path against 34,191, i.e. **0.8-2.3% (0.16-0.47 us)**
-- second order, and zero help for the packed delta. Caveat: acu's conflict counter is 1.02 per instruction, which
suggests it counts a conflicted instruction or replay rather than every excess word; its exact definition is unknown.

### The experiment that is actually next, and it is cheap

Threads `n` and `n+128` are **duplicate owners of the same column**: both issue the cp.async for it and each waits on
its own copy, so both may legally read that unit. So split the groups instead of the columns -- threads 0-127 decode
groups 0-3 of column `t`, threads 128-255 decode groups 4-7 of column `t-128`. **All eight warps decode four groups
each instead of four warps decoding eight**, with the same number of decode operations, the same stores, the same
conflicts and the same ownership. Cost: one extra 16 B shared read and header per column pair.

If it improves materially, the slowest-warp/barrier placement is what costs; if it does not, aggregate issue demand
is. That is a cleaner placement test than moving the wait, and unlike the 0.6% store fix it is also a real
optimisation. Run it together with the NOP triple.

---

## The full batch ran. Three of my four ideas are negative, and the swizzle is BROKEN

One run, one binary set, six splitk binaries verified distinct. **base is 23.67 us here against 20.11 in the earlier
pinned acu run -- a 17% gap between two runs that both claim the same pinned configuration, larger than the ~13%
spread on record. Only WITHIN-run comparisons below are usable, and nothing here may be compared to earlier numbers.**

| variant | us | GB/s | vs base |
|---|---:|---:|---:|
| base | 23.67 | 888.9 | -- |
| swz | 25.33 | 830.5 | **+7.0%** |
| bdqnop | **21.05** | 999.2 | **-11.1%** |
| pack | 24.23 | 868.3 | +2.4% |
| packnop | 24.78 | 848.9 | +4.7% |
| packsplit | 26.92 | 781.4 | **+13.7%** |

### 1. The swizzle is wrong, not just slow

`test_moe_grouped_verify` with `PPU_SCALE_SWIZZLE=1` alone dies with a **device-side assert** -- `Assertion \`false\`
failed`, block [0,10,7] thread [48,0,0], at line 104, inside `copy_B_and_extra_info`. So a view of that buffer does
NOT carry the swizzle, or a copy's vectorisation contract is violated, in a configuration `test_q4k_packed_gemm` does
not exercise. Both my local enumeration and the review concluded "every live address is swizzled"; **the hardware
says otherwise**, and the local checks were all on the TN=128/Stages=2 shape while the verifier sweeps others.

It is also 7% SLOWER where it does run. Two independent reasons to leave it off; the flag stays for diagnosis but the
idea does not ship in this form. Fixing it means reproducing the assert on a named shape first -- `MOEG_*` narrows the
verifier -- and only then hunting the view.

### 2. The baseline dequant is 11.1%, not "43%"

`base - bdqnop = 2.62 us = 11.1%`. That is the whole int4->fp16 pipeline: the conversion, the affine application, and
(because the ablation leaves only one fragment element live) most of the scale/zero loads. So the 43%-of-dynamic-
instructions figure is **11% of time** -- the instructions really do issue largely in the shadow of memory, exactly as
the review argued and against what I wrote. **This is the honest ceiling for TODO #18 and for every dequant idea**, and
it is still the largest single term anyone has measured on this kernel.

### 3. The placement hypothesis is dead

`packsplit` is **11% WORSE than pack** (26.92 vs 24.23). Eight warps decoding four groups each, with identical decode
count, stores, conflicts and ownership, loses to four warps decoding eight. So the publication barrier's critical path
is not what the packed path pays for; the extra 16 B unit read per column and the duplicated per-column setup cost
more than the shortened chain saves. That also retires the persistent-kernel line for this band, which rested on the
same "the critical path is the problem" reasoning.

### 4. The packed decode's arithmetic costs nothing

`packnop` (24.78) is **SLOWER** than `pack` (24.23). Removing the decode arithmetic while keeping the transport and
the stores did not help. Within noise either way, but it certainly is not the term -- so `pack - base`, whatever it is
in a given run, is transport and stores, not the f16x2 arithmetic I spent the day optimising.

### What survives

Only one thing here is both large and real: **the dequant pipeline at 11.1%**. Everything else measured today --
swizzle, group split, packed arithmetic -- is zero or negative. And `pack - base` is +2.4% in this run against +12.9%
in the pinned acu run, which says the native-format tax itself is not well determined and needs repeated interleaved
runs before any deployment decision rests on it.
