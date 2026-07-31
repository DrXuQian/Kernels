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
