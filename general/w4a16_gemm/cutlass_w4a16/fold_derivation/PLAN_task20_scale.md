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

## Feasibility: the native layouts ARE cute-expressible (this is the load-bearing result)

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

**`d` is CONSTANT over a K-tile.** The GGUF block tiles K in chunks of 256 per column, and every TileK we run (32/64/128)
divides 256 with aligned tiles, so `d` (and `dmin`) is one value per `(n, k-tile)` — a `[TileN]` vector per tile, not a
per-group quantity. The per-group work is therefore ONLY the 4+2 field extraction; the `d` multiply is a per-tile broadcast.
**Verify this before relying on it** (it is exactly the kind of relation that must be read off the tensor, not asserted).

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

### Phase 2 — `ElementScale` becomes a packed type, decoded through the Layouts

* `SmemLayoutScale` carries **bytes** instead of `half_t`; the g2s copy shrinks by the ratio in the table.
* `make_scale_fragment` gains a decode step driven by the Phase-0 Layouts — the same relationship `MixGemmEmit` has to the
  B converter. No hand-written shift arithmetic anywhere; that is the whole point of the cute constraint.
* `d` / `dmin` ride along as a `[TileN]` per-tile vector (pending the Phase-0 verification that they are tile-constant).
* `ElementScale = half_t` remains the default specialization, untouched, for GPTQ/AWQ.

### Phase 3 — the offline stops widening

`real_weight/dump_real_weights.py` emits the native scale bytes plus `d`/`dmin` instead of the widened fp16 plane. **Open
question to settle by reading, not assuming**: whether the scale needs any per-tile permutation. The B operand does
(`place_derived`), but the scale is indexed by `(n, k/gs)` and is broadcast along M, and the current path builds it with a
stride-0 broadcast view rather than a permutation — so probably only N-tile blocking is needed. Confirm against the actual
`make_scale_fragment` before writing a packer.

### Phase 4 — measure, in the places where it can show

Not the MoE band first. In order: **gs=16 dense** (largest channel, and where the smem tile is 12 KB today at
TN=128/TK=128/s3), then **decode/GEMV**, then the MoE band with `MOE_ONLY` on the current winners to confirm no regression.
Report the smem delta explicitly — if occupancy moves, that is the mechanism, and if it does not, the traffic saving is real
but invisible here and should be claimed for decode only.

## Risks, and the two things to verify before writing kernel code

1. **`d` tile-constancy** — asserted above from block geometry, not yet read off the tensor. If a K-tile can straddle a
   superblock, the `[TileN]` simplification collapses and `d` becomes per-group.
2. **Decode cost vs the 2.6% ceiling.** #14 bounds the entire scale path at 2.6% (gs=16, APG=1), so the decode must stay
   at a couple of ops per scale register. Decode **once per group into the fragment**, amortised over the `APG` mma atoms
   that share it — never per mma atom.
3. A stale claim to re-check while in this file: `moe_grouped_ppu.cuh` says gs=16 "Needs TK=128 (SK=8=TK/16 cap)", but the
   constraint is `SK <= TK/16` with `SK = ceil(TK/gs) = TK/16` at gs=16, which holds for **any** TK. That comment was
   probably true of an older cap. The last comment in this file asserting a TileK limit ("TK=32 still won't compile") was
   wrong, so this one gets tested rather than believed.
