# GGUF low-bit MoE support — status (int8/W8A16 dropped)

**Scope decision:** the int8 / W8A16 collective path is **NOT built** (dropped). So `Q8_0 / Q6_K / Q5_K` (which
need an int8 slot) are **unsupported**. Only **int4 (W4A16)** formats: `Q4_K` (validated), `Q2_K`, `Q3_K`, GPTQ-Int4.

## Q2_K / Q3_K — padded into the int4 slot (NO collective change)
| fmt  | native bits | gs | q_signed | stored as | zero |
|------|-------------|----|----------|-----------|------|
| Q2_K | 2           | 16 | [0,3]    | int4      | affine (min) |
| Q3_K | 3           | 16 | [-4,3]   | int4      | symmetric |

- Both fit the int4 range, so they ride the EXISTING W4A16 collective + the FINE `gs=16` apply — **zero collective
  change**. gs=16 reuses the `Gs32` tag (gs<=TK ⇒ load reload_factor=1 regardless of StaticGroupSize; the real
  16-grouping is `Scale_TileK=ceil(TK/16)` + the FINE per-mma-atom scale).
- Unpackers `real_weight/gguf_dequant.py::unpack_q2k/unpack_q3k`, verified locally (vectorized vs an independent
  element-by-element loop reference on synthetic blocks → MATCH).

### ⚠️ Padding is "make a given Q2/Q3 checkpoint RUN correctly", NOT efficient low-bit
Storing 2/3-bit weights in a 4-bit slot **wastes HBM** (Q2: 2×, Q3: 1.33× vs native) — and a Q2_K checkpoint padded
to int4 costs the same HBM as Q4_K but with worse accuracy, so you'd normally just use Q4_K. Padding only makes
sense if you are handed a Q2_K/Q3_K checkpoint you must load as-is. **True memory-efficient Q2/Q3 needs collective
changes** (native 2/3-bit HBM packing + a per-format dequant-to-fp16 prologue, since the PPU has no int2/int3 MMA
operand — only int4/int8). That is deferred / open.

## Box-validation (int4 only)
Build: `TARGET=test_moe_grouped_real PPU_SDK=<SDK> ./build.sh`
1. synth int4 + gs=16 (Q2/Q3-shaped):
   `python3 real_weight/dump_real_weights.py synth --gs 16 --mode 1 --n 512 --k 2048 --experts 4 --m 128 --out s.bin && $BIN s.bin`
   → validates the gs=16 FINE path (APG=1) that Q2/Q3 rely on.
2. real Q2_K / Q3_K checkpoint (dumper auto-detects ggml_type; falls back with a clear error for int8-slot types):
   `python3 real_weight/dump_real_weights.py gguf <q2k_or_q3k.gguf> --layer 0 --experts 0-7 --proj gate --m 128 --out r.bin && $BIN r.bin`

Risk: gs=16 FINE (APG=1, reload scale every mma atom) is compile-checked only. And kernel-vs-golden MATCH does not
prove fidelity to llama.cpp's element order (both use my unpack) — a real-inference cross-check is still needed.

## Deleted this round (per "int8 不需要做")
- collective launcher ElementB templatization (reverted to hardcoded int4b_t); driver int8 `run_case`; gguf_dequant
  `unpack_q8_0/unpack_q6k/unpack_q5k`; synth `--bits 8`.
