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

## THE cutlass mixed-precision grouped GEMM (ragged, trtllm-style) -- the real target

Reference is trtllm's `MoeFCGemm` (`moe_cutlass_kernel.h`), NOT machete's batched bench. It is RAGGED: a
`GemmMoeProblemVisitor` reads `total_tokens_including_expert` and maps each threadblock tile to (expert,
m-offset); A is one contiguous `[sum_tokens][K]` (tokens permuted by expert), B/scales are `[experts][...]`
uniform-stride. Variable tokens per expert. The mixed precision is the weight-only Mma; the grouping is the
problem visitor. `MoeFCGemm<mixed-input Mma>`.

**actlize already has the entire ragged machinery -- but only wired to the PLAIN mainloop:**
- `GroupProblemShape` (group_array_problem_shape.hpp): array of per-expert `[M_e, N, K]`.
- BatchArray collective (`ppu_mma_aiu_multistage_batch_array.hpp`): `ElementA const** ptr_A; ElementB const**
  ptr_B;` per-expert pointer arrays; `IsGroupedGemmKernel = !is_same<InternalStrideA, StrideA>` (StrideA a
  pointer type => grouped).
- `ppu_aiu_gemm_array_group.hpp`: GroupScheduler iterates tiles over the ragged GroupProblemShape; per tile
  uses `ptr_A[l_coord]`, `get_problem_shape(l_coord)`.

**The gap = the mixed-input collective lacks this grouped ptr-array form** (it is single base + L-slice). The
port, mirroring exactly how BatchArray got grouped:
1. Add a grouped-args variant to the mixed-input collective (`ppu_mma_aiu_multistage_mixed_input.hpp`):
   `ptr_A**`, `ptr_B**`, `ptr_S**`, `ptr_Z**` per-expert arrays; `StrideA/B/Scale` as pointer types to trip
   `IsGroupedGemmKernel`; `load_init` indexes `ptr_*[l_coord]` instead of L-slicing a single base.
2. A grouped kernel (copy `ppu_aiu_gemm_array_group.hpp`) whose enable_if accepts the mixed-input schedule and
   whose `operator()` also builds `gS`(+`gZ`) from `ptr_S[l_coord]` + `group_size` and drives the mixed-input
   collective's scale-aware interface (not BatchArray's gA/gB-only one).
3. Host: build `GroupProblemShape` (per-expert token counts from routing) + the ptr arrays.

This is the actlize analogue of `MoeFCGemm<mixed-input Mma>`. It is a real cutlass3 collective+kernel port and
needs on-box compile iteration (private SDK; cannot compile here).

### Step-2 recipe (MINIMAL ragged: only A is ragged; B/S/Z stay L-strided)

Key simplification found by reading the collectives: in a MoE grouped GEMM only **A** is ragged (variable
tokens/expert). **B, scale, zero are uniform per-expert stride** (`[experts][N][K]`, `[experts][N][scale_k]`),
which the mixed-input collective ALREADY addresses via its `l_coord` L-slice (verified: lines 367/380/391).
So we do NOT need to grouped-ize B/S/Z pointers -- only A. That is much smaller than fully mirroring
BatchArray (which arrays all of ptr_A/ptr_B).

Delivered as an actlize patch (like actlize_ppu001.patch; submodule stays pinned) applied by build.sh:

1. **Mixed-input collective** (`ppu_mma_aiu_multistage_mixed_input.hpp`): add a per-expert A base to its
   Params/Arguments -- `ElementA const* const* ptr_A_array` + a per-expert row offset (or pass
   `total_tokens_including_expert`). In `load_init`, when grouped, build `mA` from
   `ptr_A_array[l_coord]` with shape `(M_e, K)` instead of `mA_mkl(_,_,l_coord)`. B/scale/zero paths unchanged
   (keep the l_coord L-slice). `M_e` comes from the group problem shape.

2. **Grouped kernel** (copy `ppu_aiu_gemm_mixed_input.hpp`, NOT array_group -- the mixed-input kernel already
   drives the mixed-input collective's `load_init(blk_coord)`/scale-aware `operator()`): add the GroupScheduler
   + `GroupProblemShape` (per-expert `[M_e,N,K]`) from `ppu_aiu_gemm_array_group.hpp`, so tiles map to
   (expert, m-offset) over ragged M. `l_coord` from the scheduler selects B/S/Z (L-slice) and indexes the A
   base array. Relax nothing about scales -- the mixed-input collective already carries them.

3. **Host**: routing -> per-expert token counts -> `GroupProblemShape` + `ptr_A_array[e] = A + cumtokens[e]*K`.
   A is one contiguous `[sum_tokens][K]` (tokens permuted by expert), exactly like trtllm.

This keeps the entire scale/dequant path (the hard, tuned part) untouched and adds only ragged-A addressing +
the group tile scheduler -- the smallest change that yields a true ragged mixed-precision grouped GEMM.

### Uniform-m fast path (stepping stone, NOT the goal)  -- DONE, VALIDATED
Step 1 result on ppu001 (moe_gemm_ppu.cuh + test_moe_gemm_ppu.cu), qwen35moe FC1 n=1024 k=2048, 8 experts,
uniform m_per_expert, winner tile 64x64x128/s3 (same as dense): m=128 43.6% MFU, m=512 49.6%, m=2048 55.6%;
FC2 (n=2048 k=512) m=512 36.9%. So rank-4 mixed-input compiles/runs, the per-expert GEMM works, and the dense
tuning transfers. Step 2 (ragged) changes ONLY the addressing; the collective math is identical.

`general/w4a16_gemm/cutlass_w4a16/moe_gemm_ppu.cuh` drives the EXISTING batched mixed-input kernel (rank-4
[M,N,K,L], `l_coord=blockIdx.z`) -- works when every expert has the same m_per_expert (what the machete bench
assumes). Useful to validate the mixed-input-per-expert GEMM math + tuning before the ragged scheduler, but it
pads/wastes on real ragged routing, so it is not the deliverable.

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

## STATUS (ppu001)
- Step 1 (batched uniform-m mixed-input): DONE, 55.6% MFU @ m=2048 FC1.
- Step 2a (grouped via GroupScheduler, uniform): DONE, compiles+runs, 20.0% MFU (grouped path 2.4x slower
  than batched -- a tuning gap, not correctness).
- Step 2b (RAGGED, variable tokens/expert): RUNS. actlize_moe_ragged.patch gives the mixed-input collective a
  per-expert A base (ptr_A + group_row_offsets[l_coord]*K); ragged case (tokens 64/128/192/256) launches, no
  OOB, 87.3 TFLOP/s scaled to the ragged total. This is the cutlass mixed-precision RAGGED grouped GEMM asked
  for. REMAINING: (a) CPU-reference correctness verify (timing can't prove ragged reads the right rows);
  (b) D contiguous [sum_tokens][N] via an epilogue offset (2c; 2b uses padded [L][M_max][N]); (c) the grouped
  MFU/tuning gap (17-20% vs batched 49-55%).
