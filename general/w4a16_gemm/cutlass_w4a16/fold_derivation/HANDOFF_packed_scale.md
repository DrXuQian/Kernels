# HANDOFF — the GGUF-native scale/zero channel (plan #20)

Written at the end of the session that built it. Everything below is either **measured** (and says where) or **derived and
gated** (and says by which probe). Where something is an assumption, it says so.

---

## 1. What this was for

Feed the mixed-input grouped GEMM the scale/zero **in the form the gguf actually stores** — 6-bit codes plus one fp16 `d`
(and `dmin`) per 256-element superblock — instead of two fp16 planes that the offline pre-multiplies. Two motivations:

* **the ask**: the device should hold the gguf's bytes, not a widened copy (device memory is the scarce thing);
* **the hoped-for win**: gmem traffic halved, the zero channel's smem gone, and the bank conflicts gone.

## 2. What is DONE and verified on hardware (ppu001)

| thing | evidence |
|---|---|
| **Q3_K / Q6_K drop the zero channel entirely** | `test_q3_bconcat_real` rungs 6–7 MATCH incl. `last rung vs native Q3_K golden bad=0/32768`; `test_q65_bconcat_real` two Q6 `ScaleOnly bias 32` rows + one Q5 `bias 16` row MATCH, `0 failing configuration(s)` |
| **the gguf's native scale/zero runs end to end** | `test_q4k_packed_gemm` rowA/rowB/**rowC** all MATCH with `PPU_PACKED_SCALE=1`, on real `blk.11.ffn_down.weight` |
| **the single-plane int4 AFFINE path is correct** | `test_q4k_packed_gemm` rowB — the FIRST external-golden check of it; everything validated before was ScaleOnly (`bench_cutlass_w4a16::xcheck_grouped` passes `zeros=nullptr`) |
| **the bit maps** | `l91` over 4096 real superblocks × 8 groups vs `get_scale_min_k4`, 0 bad |
| **the decode's arithmetic** | `l93`: 32768 real Q4_K groups, `(scale,zero)` **bit-identical** to the host's fp32-then-round |
| **the converter bias mechanism** | `l92`: `x*mul + add == c − Bias` over the full concatenated range for Q3 (W=3, bias 4), Q6 (W=6, 32), Q5 (W=5, 16), 0 bad; defaults unchanged |
| **the offline** | `dump_packed_scale.py` 4 gates green on real weights, incl. `f16(f32(d)*f32(sc)) == f16(d)*f16(sc)` bit-for-bit over 136192 groups |

## 3. What is DONE and NOT paying off

Same-config A/B, decode band `L=64 top-k=8 N=K=2048 gs=32`, `SK_QUANT=2` (ScaleZero), winner `16x128:256 w16x16 s2 S=1`:

| build | winner |
|---|---|
| baseline (fp16 planes) | **20.12 / 20.22 / 20.36 us** (three runs; same-config spread is ~13%) |
| **E** — decode per lane in registers | 26.89 (and TK=256 fell off the board) |
| **F** — decode in a synchronous loader, into the fp16 planes | 24.17 (TK=256 back on top → inner loop is clean) |
| **F′** — cp.async raw bytes to staging, decode at the existing barrier | 25.00 / 24.79 |

**Read this as a refutation, not as work in progress.** F′ restored async and got *worse*, so the exposed load latency was
never the main term. What F and F′ share is the decode: per CTA per k-tile, 512 decodes (~11 instructions each) **and 1024
`STS.16`** — where the fp16 path has cp.async write the same two tiles with **zero instructions**. With acu showing LSU at
6% busy and IALU/FALU at 14%, trading a free async copy for explicit stores plus ALU is the wrong direction in this band.

**Recommendation: stop pushing this in the decode band.** Keep it macro-gated and OFF; the wins it does have (gmem halved,
one stream, the zero tile removable, no offline pre-multiplication, bank conflicts gone) only become visible where **smem
capacity or occupancy** is the binding constraint — prefill/dense with large tiles, or configs where the scale+zero tiles
cost a block per CU. That is where to measure next.

## 4. Ruled out — do not retry these

Each of these cost real time; the reason matters more than the verdict.

1. **The 16 B unit as one `uint128_t` element.** Would have kept the gmem tensor scalar and every residue line unchanged.
   cute rejects it: `Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<uint128_t>, uint128_t>` with a `(_1,_1)` value layout fails
   `copy_traits.hpp`'s *"dst failed to vectorize into registers"*. The element must be `uint8_t` with `(_1,_16)`.
2. **Decoding all groups into a fragment-shaped register array** (once per k-tile, no state passed around): ~**+64
   registers** per lane on top of ~140. Refused for prefill's sake.
3. **Interleaving `(scale, zero)` in N so one read gets both.** `tCrS` is `make_fragment_like(partition_fragment_B(...))`,
   so each slot is bound to a specific `n` of the mma B operand; interleaving gives a lane 4 scales and 4 zeros of
   *different* n.
4. **Read-side (per-lane) decode — E.** Per lane per k-tile it needs 2 columns × 8 groups = 16 `(scale,zero)` pairs at ~11
   instructions each (Q4_K is affine: two fields, two conversions, two products, one ZMul fma) → ~1.4M extra ALU to save
   ~0.27M shared loads. And the same `(column, group)` is decoded by all 4 lanes sharing that column (`l94 (7)`), which no
   tuning removes.
5. **"The exposed LDG is the remaining cost."** Refuted by F′ (see §3).
6. **A template-parameter gate around a generic lambda** (`packed_or_empty<On>([&](auto){...})`) to keep an off unit from
   instantiating a body: EDG instantiates the lambda body anyway, so empty stand-ins reach `partition_S`.
7. **`if constexpr (kPackedScaleOn)` as a discard mechanism.** `kPackedScaleOn` is a *class constant*, not
   value-dependent, so **both branches are instantiated**. Only a condition that depends on a template parameter of the
   enclosing function discards. This one bit twice.

## 5. Current code state

* actlize branch `ppu-w4a16-dev`, Kernels branch `ppu_dev`; the Kernels gitlink must be bumped with every actlize change
  (the box builds from committed state — one bump was missed this session and produced a "my change did nothing" round).
* `PPU_PACKED_SCALE=1` turns the channel on. **Default off ⇒ byte-identical**: the staging engine is last in
  `SharedStorage` and sized 0, and the read side (s2r, fragments, all four transform arms) is the fp16 path unchanged.
* `kPackedScaleOn = (Scale_TileK == 8)` — per **unit**, not per binary, because the splitk bench compiles 11 shapes into
  one binary and a `static_assert` there fails the whole build. At gs=32 that is `TileK == 256`, which is also the
  one-superblock-per-k-tile condition, so one test covers both.
* `PPU_PACKED_SCALE_NOP=1` is a **timing-only** ablation: the decode returns constants, so results are deliberately wrong.

New/changed files worth knowing:

| file | what |
|---|---|
| `actlize/include/cutlass/gguf_packed_scale.h` | the 16 B unit's bit map and decode: `PackBits`, `bit_of`, `code_of`, `code_from_words<Bit>`, `put_code`, `int_to_half_small`, `head_of_words`, `group_of_words<G,ScaleBias,HasMin,ZMul>`. Lives in actlize because the mainloop needs it; the harness header re-exports it so the map exists **once**. |
| `actlize/.../ppu_mma_aiu_multistage_mixed_input.hpp` | `SmemLayoutScaleRawStaged`, `GmemTiledCopyScalePacked`, `packed_decode_stage`, `kPackedScaleOn`, `kPackedZMul` |
| `actlize/include/cutlass/detail/collective.hpp` | `detail::NoZero` + `strip_no_zero_t` — how the 2-plane tuple says "no zero" while keeping the second plane at index 3 |
| `gguf_scale_layout.hpp` | per-format `Traits`: `kScaleBits/kMinBits/kGroups/kBlockBytes/kHasMin/kSigned/**kScaleBias**` |
| `gguf_scale_decode.hpp` | harness-side re-export + `Superblock<KType>` |
| `rwmoep_loader.hpp` | reads the `RWMOEP` fixture; refuses on magic, geometry, truncation **and trailing bytes** |
| `test_q4k_packed_gemm.cu` | the end-to-end gate, three attributable rows |
| `test_q4k_native_scale.cu` | device decode vs host reference, no GEMM |
| `real_weight/dump_packed_scale.py` | the offline: int8 `sc/mn` + fp16 `d/dmin`, magic `RWMOEP` |
| `real_weight/q4k_packed.bin` | the fixture, **N=256** (1506352 bytes) |
| `fold_derivation/l91..l95` | the local gates (see §2) |

## 6. How to run everything

```bash
cd .../Kernels/general/w4a16_gemm/cutlass_w4a16
EX=.../third_party/actlize/build_w4a16_compare/examples/99_kernels_w4a16_compare
```

**build.sh `rm -rf`s the same build dir every time**, so build a target and run it **immediately** — the previous
binary is gone. `PPU_DEFS` is space-separated and build.sh prints `PPU_DEFS verified on <target>'s compile command`;
if that line is missing the macro is NOT in the binary and any A/B is a binary compared with itself.

```bash
# correctness, both builds; three rows, each with its own verdict
TARGET=test_q4k_packed_gemm ./build.sh >/dev/null 2>&1 && $EX/test_q4k_packed_gemm real_weight/q4k_packed.bin
PPU_DEFS=PPU_PACKED_SCALE=1 TARGET=test_q4k_packed_gemm ./build.sh 2>&1 | grep -E "PPU_DEFS|WARNING"
  $EX/test_q4k_packed_gemm real_weight/q4k_packed.bin

# Q3/Q6 zero removal
TARGET=test_q3_bconcat_real  ./build.sh >/dev/null 2>&1 && $EX/test_q3_bconcat_real real_weight/real_q3k_concat.bin
TARGET=test_q65_bconcat_real ./build.sh >/dev/null 2>&1 && $EX/test_q65_bconcat_real

# decode band A/B (same config both sides; do NOT compare winners across runs)
PPU_DEFS="SK_QUANT=2"                        TARGET=test_moe_splitk_bench ./build.sh && $EX/test_moe_splitk_bench 64 8 2048 2048 32 3
PPU_DEFS="SK_QUANT=2 PPU_PACKED_SCALE=1"     TARGET=test_moe_splitk_bench ./build.sh && $EX/test_moe_splitk_bench 64 8 2048 2048 32 3

# acu, one config, both sides
SPLITK_ONLY="16x128:256" SPLITK_S=1 SPLITK_ACU=1 acu -o X.report --set full -f $EX/test_moe_splitk_bench 64 8 2048 2048 32 3

# local gates (no box)
A=../../../../third_party/actlize/include
nvcc -std=c++17 -x cu -arch=sm_80 -w -I stub_inc -I $A -I .. -I . -o /tmp/l94 fold_derivation/l94_native_scale_path.cu && /tmp/l94
nvcc -std=c++17 -x cu -arch=sm_80 -w -D__HGGCCC__ --expt-relaxed-constexpr -I stub_inc -I $A -I .. -I . -o /tmp/l95 fold_derivation/l95_stub_vs_real.cu   # compiling IS the pass
./fold_derivation/syntax_check.sh <file.cu>                    # and with EXTRA_DEFS=-DPPU_PACKED_SCALE=1
```

## 7. Facts that are not obvious from the code and cost time to rediscover

* **`moe_grouped_ppu.cuh:352`**: `n % 256 == 0 && k % 256 == 0` selects the **interleaved** B layout. An N that is not a
  multiple of 256 hands an interleave-256 buffer to a column-major reader → deterministic garbage that is **invariant
  under the quant mode and the tile**. That invariance is the signature; it is upstream of the scale channel by
  definition. (Cost: three rounds looking at the scale channel.)
* **`scale_load_k` is a TILE index**, not a group index — `partition_S` leaves the last mode selecting which block of
  `Scale_TileK` groups a call loads.
* **`sS` means two different layouts** in two functions: `partition_extra_inputs` builds it with `SmemLayoutScale`
  `(n, group, stage)`; `partition_extra_mma_info` with `SmemCopyLayoutScale` `(n, 1, stage*SK+g)`. Using the wrong one
  faults as `Exception TSM out of range` (a fault, not a wrong number — the good failure).
* **Q3_K's scale is `d*(sc6 − 32)`**, not `d*sc6`. It is the only non-zero `kScaleBias`; Q6_K's codes are signed int8.
* **`kSymBias2Plane = 1 << (W−1)`** — Q3_K 4, Q6_K 32 — is what lets a symmetric k-quant drop its zero channel entirely.
* **`ZMul = 8`** cancels the int4 converter's own −8. The alternative (converter `Bias = 0`) is accuracy-equivalent
  (0.0128 vs 0.0148 step against fp64 truth) but touches the shipped converter.
* **build.sh only descends into `_overlay_dirs=(gemv_lowbit)`** when overlaying sources, so a compiled header under
  `real_weight/` is absent at box build time while the local front-end check resolves it against the real tree and passes.
  Compiled headers live flat next to the harnesses.
* **`syntax_check.sh` compiles with `-DPPU_FORCE_INSTANTIATE=1`**, so it sees units the main `.cu` never builds — several
  errors only appear there. Record a **flag-on baseline** for each variant you care about; never baseline a `fatal error`.
* **Same-config run-to-run spread is ~13%.** A/B only within one run or back-to-back with identical filtering, and pin the
  config — the "winner" row can change config between runs.
* **`MOEG_SMEM=1` prints `GemmKernel::SharedStorageSize`**, but only for harnesses that go through
  `moe_grouped_ppu::filter_and_run` — and it cannot show a zero-tile saving in a **ScaleOnly** harness, because
  `elements_per_smem_zero()` is already 0 there.

## 8. The failure mode that produced most of the defects

**One relation, two derivations, only one of them checked.** Every significant bug this session was that shape:

* `PPU_A_CUBE_H` patched `DefaultGemm_AIU_Operand` while mixed-input builds its atom in `MixGemm_AIU_Operand` → inert;
* `Cvt2Plane` biased at the chunked site while the unchunked `Cvt2Plane::convert` did real arithmetic;
* the smem pitch as two literals (builder 16, collective 64) → `invalid VA`;
* `sS` (§7) and `scale_load_k` (§7) — and for `scale_load_k` the file **already contained the correct reading** in the k
  bound written a few lines above;
* the plan's own gate was wrong: it named `test_lowbit_grouped`, whose oracle is the same kernel at L=1 and is therefore
  structurally blind to a wrong dequant constant;
* B's nibble packing "corrected" against the convention validated three lines away in another harness.

What actually caught things: **grep every call site** rather than the one just edited; **name a shared quantity once** and
quote it; ask **what a number is invariant to** before reading it as arithmetic; and prefer a gate whose oracle is
*independent* of the kernel.

## 9. If you pick this up

**The open question is not "how do I make the decode cheaper".** It is **"where does the smem/traffic saving buy
occupancy"**, because in the decode band it demonstrably does not pay. Concretely:

1. Take the acu pair from §6 on `16x128:256` and check `smem/block` and measured `blocks/CU` first — F′ adds 4 KB/CTA
   and 4 blocks/CU means 16 KB, which could cross a threshold and alone explain 4–5 us. If it does, staging only needs
   **one** stage (the decode runs right after the wait), which is a small change.
2. Then measure where smem is the binding constraint: **gs=16 dense** (the scale tile is 12 KB at TN=128/TK=128/s3) and
   prefill shapes, with `MOEG_SMEM=1` reporting the delta. Report the smem delta explicitly — if occupancy moves, that is
   the mechanism; if it does not, the traffic saving is real but invisible there.
3. Q3_K/Q6_K are the better candidates for the whole idea than Q4_K: they have **no min channel at all** after plan #20
   phase 1, so their packed form is one int8 plus `d` and the zero tile is gone rather than merely re-formed.
4. Still owed: `unpack_q6k_expert` in `dump_real_weights.py` (there is none), so Q6_K's `−32` centre has never been
   checked against real Q6_K weights — only against the format definition and a synthetic golden I wrote.

## 10. The option this work did not price: a SEPARATE conversion kernel

Asked at the end of the session, and it reframes everything above. Note first that **the baseline already is "another
kernel"** — the conversion exists, it just runs on the host at fixture-build time. Moving it to a device kernel that runs
**once at load time** gets the gguf-native *ingestion* (read the mmapped native bytes directly, no host pre-multiplication
step, compatible with llama.cpp's loader) at **zero runtime cost**, because the GEMM then sees the untouched fp16 planes.

What it gives up is exactly what motivated the in-GEMM decode: device memory. Per 256 weights of Q4_K:

| form | bytes / 256 weights | vs native | in-GEMM decode cost |
|---|---|---|---|
| native (q 128 + d/dmin/scales 16) | **144** | — | measured **+20%** on the decode band (20.2 -> 24.2) |
| **int8 sc/mn + fp16 d/dmin** | **148** | **+2.8%** | codes are BYTE-addressable: no 6-bit extraction, no word straddle, ~4-5 instructions per pair instead of ~11 |
| fp16 planes, pre-multiplied (baseline) | **160** | +11% | 0 |

So keeping the native form saves **11%** of weight bytes and costs +20% of this band's GEMM time. On a 4.4 GB Q4_K_M
model that 11% is ~480 MB. Which side wins is a deployment question, not a kernel question — but it should be decided with
both numbers in view, which it was not when this work started.

**The middle point deserves a second look.** `int8 sc/mn + fp16 d/dmin` is exactly what `dump_packed_scale.py` already
emits (`RWMOEP`), and it was rejected early in this session *for widening the codes* — judged against native's 144. Its
real competitor is the 160 that production actually uses, and against that it is a 7.5% memory **saving** with an in-GEMM
decode several times cheaper than the 6-bit one, because most of those ~11 instructions per pair were bit extraction and
straddle handling that byte-aligned codes do not need.

Pieces that already exist for either route: `packed_decode_stage`'s body is the conversion kernel almost verbatim,
`rwmoep_loader.hpp` reads the fixture, `dump_packed_scale.py` emits the int8+d form, and `test_q4k_native_scale.cu` is
already a device-side decode-vs-host-reference gate — i.e. a standalone conversion kernel would inherit its test.
