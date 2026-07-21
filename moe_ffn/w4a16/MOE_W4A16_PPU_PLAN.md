# Plan: MoE FFN W4A16 on PPU (ppu001)

Goal: add W4A16 MoE FFN support for the PPU, referencing the NVIDIA implementations already vendored under
`moe_ffn/w4a16/` (TensorRT-LLM SM80 grouped path, vLLM Machete SM90). This is the MoE analogue of the dense
W4A16 work in `general/w4a16_gemm/` (cutlass_w4a16 = actlize mixed-input; marlin_ppu = hand-written).

## What MoE FFN W4A16 needs (from the trtllm reference structure)

A MoE FFN forward is: **route → permute → FC1 grouped GEMM → activation(GLU) → FC2 grouped GEMM → unpermute+combine**.
The trtllm standalone breaks into exactly these pieces:

| trtllm component | role | PPU status |
|---|---|---|
| `auxiliary/custom_moe_routing.cu` | top-k gate → expert ids + weights | port (plain CUDA) |
| `auxiliary/moe_align.cu`, `moe_expert_map.cu` | permute tokens into per-expert contiguous rows; build `total_tokens_including_expert` | port (plain CUDA); a simple version already exists in `gemv_w4a16_ppu.cu` (row_expert/row_token) |
| `moe_gemm_template_dispatch_tma_ws_mixed_dtype.h` + `splitk_gemm_grouped.h` | the grouped **mixed-input** (W4A16) GEMM over experts | **the core gap** — see below |
| `moe_cuda_core_gemv.cu` | decode GEMV, `m_per_expert=1`, same int4/scale layout | **have** `gemv_w4a16_ppu.cu::moe_gemv_rows` (validated MATCH) |
| activation (SiLU+mul / GLU) | FC1 output → FC2 input | port (elementwise); marlin/ggml GLU exists |
| `MoeGemmRunner` + profiled tactic | pick config per shape | have the fpA_intB/machete tactic skeleton to mirror |

The two GEMMs per layer (qwen35moe, Q4_K_M, 256 experts, topk 8):
- **FC1** (gate+up fused): `N=1024, K=2048` per expert, weight int4.
- **FC2** (down): `N=2048, K=512` per expert.
Rows per GEMM = `tokens * topk` (expanded), grouped by expert.

## The core gap: grouped mixed-input GEMM

Two routes, and we already own most of route B.

### Route A — cutlass/actlize (mirror trtllm structure)
Combine actlize's **mixed-input mainloop** (`ppu_mma_aiu_multistage_mixed_input.hpp`, the FinegrainedGs
schedules we drove to 61% dense) with its **array/group kernel** (`ppu_aiu_gemm_array_group.hpp`,
`KernelAiuMultistageBatchArray`). These are NOT combined in v1.0.0 — the port is: make the array/group kernel
accept a mixed-input mainloop schedule (they currently key on different schedule bases), add the per-expert
`ptr_array` / `total_tokens_including_expert` plumbing to the mixed-input collective's arguments, and thread
the per-expert scale/zero pointers. Reuses the dense tuning; gs is 64/128 only (finegrained), so this fits
GPTQ/AWQ but not Q4_K's gs=32 without a new Gs32 specialization.

### Route B — our hand-written AIU (already works)
- **Prefill**: `marlin_moe_aiu_ppu.cuh` — Q4_K MoE grouped GEMM on the AIU, persistent scheduler, validated
  MATCH, 52% peak (memory ppu-moe-q4k-aiu). This already IS a grouped mixed-input GEMM, at gs=32.
- **Decode**: `gemv_w4a16_ppu.cu::moe_gemv_rows` — validated, ~10 us.
Route B covers Q4_K gs=32 (what qwen35moe_Q4_K_M actually is) and is proven on ppu001; route A does not do
gs=32 yet.

**Recommendation:** build the MoE FFN harness in the trtllm-mirrored structure, wiring route B (our validated
AIU kernels) as the default backend, and add route A (actlize grouped mixed-input) as a second backend/tactic
candidate for gs=64/128 (GPTQ/AWQ). This reuses proven kernels, matches the reference structure, and keeps the
tactic/config-selection layer we built for dense.

## Staged approach

1. **Harness + routing/aux** (port trtllm auxiliary): topk routing, permute to per-expert rows,
   `total_tokens_including_expert`, unpermute+combine. Verify against a CPU reference on random gates.
2. **FC1/FC2 via route B**: wire `marlin_moe_aiu` (prefill) and `moe_gemv_rows` (decode) as the grouped GEMM,
   with the GLU activation between. End-to-end MoE FFN, correctness-gated.
3. **Tactic selection**: per (m_per_expert, N, K) pick prefill-GEMM vs decode-GEMV (m_per_expert==1 → GEMV),
   like trtllm's moe_cuda_core_gemv fallback; profile + shape-keyed cache.
4. **Route A (optional, gs=64/128)**: port the actlize grouped mixed-input GEMM as a second backend and add it
   to the tactic candidates; compare against route B.
5. **Offline reorder**: per-expert weight repack (Q4_K unpack → our AIU swzl int4 layout for route B, or
   interleave-256 for route A) done once at load, replacing the weight in HBM (one copy — 不能两份在显存).

## Open questions to settle before coding
- Which format first: Q4_K gs=32 (route B, matches the real GGUF model) or GPTQ/AWQ gs=128 (route A, matches
  the NVIDIA reference)? Q4_K is the actual target, so route B first.
- Reuse the existing routing in `gemv_w4a16_ppu.cu` vs porting trtllm's `custom_moe_routing.cu` (more complete:
  handles renormalization, expert bias). Port trtllm's for completeness.
- Activation: qwen uses SiLU-GLU (gate*silu(up) or up*silu(gate)); confirm the exact GLU order against ggml.
