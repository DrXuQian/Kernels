# Multi-format GGUF MoE W4A16/W8A16 — status & box-validation plan

**Written this session (static-checked; NOT yet run on PPU — no SDK on the dev box):** support for GGUF
`Q2_K / Q3_K / Q5_K / Q6_K / Q8_0` in addition to the already-validated `Q4_K` and GPTQ-Int4.

## Design (recap)
All formats offline-unpack to a uniform intermediate `(q_signed intN + fp16 per-gs scale [+ fp16 zero])`, then
feed the SAME grouped mixed-input collective (kernel/epilogue/scheduler/FINE-scale all shared). Bit width picks
the slot:

| fmt  | block | gs | q_signed range | zero? | slot | collective |
|------|-------|----|----------------|-------|------|------------|
| Q8_0 | 34B   | 32 | [-127,127]     | no    | int8 | W8A16 |
| Q6_K | 210B  | 16 | [-32,31]       | no    | int8 | W8A16 |
| Q5_K | 176B  | 32 | [-16,15]       | yes(affine) | int8 | W8A16 |
| Q2_K | 84B   | 16 | [0,3]          | yes(affine) | int4 | W4A16 |
| Q3_K | 110B  | 16 | [-4,3]         | no    | int4 | W4A16 |
| Q4_K | 144B  | 32 | [-8,7]         | yes   | int4 | W4A16 (validated) |

- Unpackers: `real_weight/gguf_dequant.py` — ported from llama.cpp `dequantize_row_qX_K`, **verified locally**
  (vectorized vs independent element-by-element loop reference on synthetic blocks → all MATCH, fp16 rounding).
- Collective: `moe_grouped_ppu.cuh` `launch<>/filter_and_run<>` templatized on `ElementB` (int4b_t / int8_t);
  `gs=16` reuses the `Gs32` tag (gs<=TK ⇒ load reload_factor=1 for any StaticGroupSize; the real 16-grouping
  comes from `Scale_TileK=ceil(TK/16)` + the FINE per-mma-atom apply, `APG=1`).
- Driver: `test_moe_grouped_real.cu` reads the `.bin` `bits` field → `run_case<int8_t|int4_t>`.

## Box-validation order (isolate the NEW int8/gs=16 collective paths before real weights)

Build once: `TARGET=test_moe_grouped_real PPU_SDK=<SDK> ./build.sh`
`BIN=$(find ../../../third_party/actlize/build_w4a16_compare -name test_moe_grouped_real -type f | head -1)`

1. **int8 collective works at all?** (W8A16, synthetic)
   `python3 real_weight/dump_real_weights.py synth --n 512 --k 2048 --gs 32 --experts 4 --m 128 --mode 0 --bits 8 --out real_weight/s8.bin && $BIN real_weight/s8.bin`
   → if MISMATCH/compile-fail, the int8 mixed-input path (AIU int8 swzl / interleave-256 / transpose orientation)
     is the culprit — NOT the format unpackers. Debug here first.
2. **int8 + gs=16 + affine (Q6/Q5-like):**
   `... synth --gs 16 --mode 1 --bits 8 ... && $BIN`
3. **int4 + gs=16 (Q2/Q3-like):** `... synth --gs 16 --mode 1 --bits 4 ...`
4. **Real weights** (Qwen3.5-35B-*-GGUF; the dumper auto-detects ggml_type per tensor):
   - Q6_K is typically `ffn_down_exps`; Q4_K is `ffn_gate/up_exps`. Pick a model/tensor of the target type:
   `python3 real_weight/dump_real_weights.py gguf <model.gguf> --layer 0 --experts 0-7 --proj down --m 128 --out real_weight/q6k.bin && $BIN real_weight/q6k.bin`
   - The dumper prints `[gguf] fmt=q6_k gs=16 slot=int8 mode=0` + golden nonzero% (guards trivial 0==0 match).

## Known box-validation RISKS (in priority order)
1. **int8 (W8A16) AIU path** — the AIU bulk-load + `ldmatrix.swzl` were tuned/verified for int4/b16. int8 uses the
   same `ColumnMajorInterleaved<256>` + `MixGemm_AIU_Operand<int8_t>`; the builder asserts a16w8 support but it is
   UNPROVEN on ppu001 here. If step 1 fails: try the non-interleaved path (force `AiuInterleaved=false`, i.e. feed
   n or k not %256, or add an `il=false` override) — plain cp.async int8 may work where the swzl doesn't.
2. **int8 transpose orientation** — int4 needed the dumped q transposed to [N][K] (kernel reads B buffer [N][K]).
   Assumed identical for int8; if step-1 synth MISMATCHes with a whole-tensor-scramble pattern, re-run the driver's
   (removed) 4-way host-golden probe to re-localize for int8.
3. **gs=16 FINE (APG=1)** — reload scale every mma atom. Compile-time paths exist; unproven perf/corner. Step 2/3 cover it.
4. **Q3_K 6-bit sub-scale unpack** (the aux bit-manipulation) — verified vs the loop reference locally, but the
   loop reference is my own transcription; a llama.cpp-inference cross-check (compare dumped W to `llama.cpp`'s
   `dequantize_row_q3_K`) would catch a shared transcription error. Same caveat for element-ORDER of all K-quants:
   the kernel-vs-golden MATCH does NOT prove fidelity to llama.cpp's element order (both use my unpack) — a real
   end-to-end model check is still needed for inference correctness.

## Not done
- Q4_0 / Q4_1 / Q5_0 / Q5_1 legacy (non-K) — Q5_0/1 are 5-bit like Q5_K but per-32 with a single fp16 d[+m], easy
  to add to gguf_dequant if a model uses them.
- Perf (this is correctness-first). int8 W8A16 perf, and Q2/Q3 in int4 slots waste HBM vs their native 2/3-bit.
- Real-inference fidelity cross-check vs llama.cpp dequant (see risk 4).
