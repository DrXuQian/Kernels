# FlashInfer GDN Prefill Performance Analysis

This note explains why `linear_prefill_flashinfer_gdn` has low utilization for
the Qwen3.5-122B-A10B prefill shape:

```text
total_seqlen=3823, num_seqs=1, q_heads=16, k_heads=16, v_heads=64, head_dim=128
```

## Summary

The largest issue is not the stage-count choice. The production-style build uses
separable compilation (`-dc`) for this SM90 warp-specialized CUTLASS/FlashInfer
kernel. In that mode ptxas reports:

```text
Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility across compilation units.
Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources
```

This directly explains the low tensor utilization: the kernel relies on
warp-group register reallocation (`warpgroup_reg_alloc/dealloc`) to run WGMMA
efficiently. When `setmaxnreg` is ignored, WGMMA issue is serialized.

An isolated single-translation-unit build, with the same source-level kernel
configuration, cuts H800 single-kernel time by about 2x:

| Binary | Build style | Variant | CUDA-event median | `nsys` single-kernel time |
|---|---|---|---:|---:|
| `linear_attn/bench_gdn_prefill` | separable compilation | 2/3/2 stages | 0.5232 ms | 522.285 us |
| `bench_gdn_tile_study` | separable compilation | 2/2/2 stages | 0.5092 ms | 515.494 us |
| `bench_gdn_tile_study_single_tu` | single translation unit | 2/3/2 stages | 0.2617 ms | 262.810 us |

## Launch-Limited Parallelism

Even after fixing the build style, this kernel is structurally hard to drive to
high whole-GPU utilization on H800.

`nsys` launch/resource facts for the production shape:

| Build | Grid | Block | Registers/thread | Dynamic SMEM | Executed shared memory |
|---|---:|---:|---:|---:|---:|
| production separable | 64 CTAs | 512 threads | 128 | 186,368 B | 200,704 B |
| study k2 separable | 64 CTAs | 512 threads | 128 | 169,984 B | 200,704 B |
| study single-TU default | 64 CTAs | 512 threads | 128 | 186,368 B | 200,704 B |

The scheduler maps GVA work as:

```text
grid.x = num_seqs * num_v_heads = 1 * 64 = 64 CTAs
```

The block has four warp groups:

```text
1 load/store WG + 2 state-math WGs + 1 auxiliary-math WG = 4 WGs = 512 threads
```

Dynamic shared memory is about 170-186 KiB, so only one CTA can reside on an SM.
On an H800 with 132 SMs this caps launch-level active SMs at:

```text
64 / 132 = 48.5%
```

Each resident CTA has 16 warps. Against 64 max resident warps/SM, device-wide
resident warp occupancy is at most:

```text
64 CTAs * 16 warps / (132 SMs * 64 warps/SM) = 12.1%
```

This is not an NCU counter; it is the launch/resource upper bound from the actual
kernel launch. It explains why whole-GPU utilization can look low even when each
active CTA is doing useful work.

## Serial Work Inside Each CTA

For `seqlen=3823` and tile `64`, each CTA processes:

```text
ceil(3823 / 64) = 60 chunks
```

These chunks are not independent. The GDN/DeltaRule state update is recurrent, so
the CTA iterates through the chunks in sequence. The main loop has dependent
stages:

1. load Q/K/V/alpha/beta
2. auxiliary WG computes `K @ K^T` and `Q @ K^T`
3. auxiliary WG applies alpha/beta/masks and inverts the 64x64 triangular system
4. state WGs consume QK/KK results to update state and output
5. store output and final state

The source contains multiple pipeline waits, named barriers, and WGMMA waits
inside each tile iteration. That is expected for this algorithm, but it means the
kernel is not equivalent to one large dense GEMM.

## Non-Tensor-Core Work

A significant fraction of each tile is non-WGMMA:

- alpha cumulative product/log handling
- `exp2f`-based alpha scaling
- beta scaling
- lower-triangular masking
- 64x64 triangular inverse in shared memory
- shared-memory stores/loads for QK/KK handoff
- named barriers between the auxiliary and state warp groups

This work is CUDA-core / shared-memory / synchronization heavy. On a platform with
weaker CUDA-core throughput than H800, this part can dominate even if BF16 Tensor
Core peak is similar.

## Why Stage Tuning Was Small

The stage-count sweep changed only the depth of Q/K/V TMA pipelines:

| Variant | Stages Q/K/V | Separable median |
|---|---:|---:|
| default | 2/3/2 | 0.5179 ms |
| k2 | 2/2/2 | 0.5092 ms |
| q3 | 3/3/2 | 0.5476 ms |
| v3 | 2/3/3 | 0.5439 ms |

This moves performance by only a few percent because it does not address the
larger problem: WGMMA serialization from the separable build.

After the single-TU build fixes that issue, the original stage config is again
best among tested variants:

| Variant | Stages Q/K/V | Single-TU median |
|---|---:|---:|
| default | 2/3/2 | 0.2617 ms |
| k2 | 2/2/2 | 0.2653 ms |
| q3 | 3/3/2 | 0.2734 ms |
| v3 | 2/3/3 | 0.2730 ms |

## FlashQLA-Style Block-DV Study

FlashQLA also increases CTA count by choosing `block_DV=64/32` when the number
of V heads is too small to fill the GPU. A study-only CUDA prototype was added
under `src/block_dv/` that keeps the Q/K dimension at 128 and slices the V/O
compute path to `block_DV=64`.

The prototype is functional and memcheck-clean:

```bash
make blockdv_single_tu -j
compute-sanitizer --tool memcheck --print-limit 1 \
  ./bench_gdn_blockdv_study_single_tu 3823 16 64 128 1 --tile 64 --block-dv 64 --variant default
```

However it is slower on H800:

| Kernel | CTAs | H800 single-TU kernel time |
|---|---:|---:|
| original FlashInfer GDN | 64 | 259 us nsys / 0.257 ms CUDA event |
| block-DV=64 prototype | 128 | 512 us nsys / 0.505 ms CUDA event |

The likely reason is structural: this CUTLASS collective duplicates the QK/KK
and alpha/beta auxiliary path for every V slice. The V/state work is split, but
the auxiliary work is not shared across the two `DV=64` CTAs. So for
`T=3823,Hqk=16,Hv=64,D=128`, extra occupancy is outweighed by duplicated work.
The prototype is intended for performance diagnosis of the fused prefill kernel;
it should stay isolated and should not replace the default benchmark.

Trying to push the same idea further to `block_DV=32` is blocked by the current
CUTLASS GMMA shape constraints. The state/O GMMA tile's M dimension becomes 32,
and SM90 GMMA rejects it at compile time because tile M must be a multiple of 64.
So increasing grid beyond 128 CTAs requires a different decomposition, such as
sequence/context splitting with state correction or splitting auxiliary/state
phases, not just a smaller V tile.

## Checkpointed Split-Sequence Study

The next decomposition uses existing FlashInfer semantics instead of resetting
state. The first pass runs the built-in checkpointing mode and writes recurrent
state at segment boundaries. The second pass treats each segment as an
independent logical sequence with `InitStateFromInput=true`, using those boundary
states as initial state.

This raises only the timed second pass from `1 * 64 = 64` CTAs to
`segments * 64` CTAs. For the target `T=3823,Hqk=16,Hv=64,D=128`, the best
measured split was `segment_tokens=768`, which produces 5 segments and 320 CTAs:

| Path | CTAs | H800 CUDA-event time |
|---|---:|---:|
| original single-TU GDN | 64 | 0.2575 ms |
| split-seq second pass, 768-token segments | 320 | 0.2074 ms |
| checkpoint pass only | 64 | 0.2706 ms |
| checkpoint + pack + split pass | 64 + 4096 + 320 | 0.5097 ms |
| state-only checkpoint pass | 64 | 0.1774 ms |
| state-only checkpoint + pack + split pass | 64 + 4096 + 320 | 0.4102 ms |
| scan transition pass | 320 | 0.1174 ms |
| scan transition + prefix + split pass | 320 + prefix + 320 | 0.3773 ms |

The split second pass is about 20% faster than the original single kernel, and
the target-shape output matches the full-sequence checkpoint output exactly in
bf16 storage (`max_abs=0`). nsys confirms the intended grid shape:

| Kernel | gridX | Duration |
|---|---:|---:|
| checkpoint GDN | 64 | 273.476 us |
| `pack_segment_input_states` | 4096 | 25.792 us |
| split GDN | 320 | 211.458 us |

The state-only checkpoint variant removes Q load, QK, Q@state, QK@V, and O store
from the checkpoint pass. It remains a study-only copied collective under
`src/state_only/` and does not change the production FlashInfer extraction. It is
correct on the target shape:

```text
state_split --check: max_abs=0 max_rel=0 elements=31318016
compute-sanitizer: ERROR SUMMARY: 0 errors
```

Its nsys breakdown for `state_both` is:

| Kernel | gridX | Duration |
|---|---:|---:|
| state-only checkpoint GDN | 64 | 176.194 us |
| `pack_segment_input_states` | 4096 | 23.104 us |
| split GDN | 320 | 210.051 us |

The scan-style variant computes each segment's state transition from zero in
parallel, computes segment decay coefficients, composes segment input states,
then runs the same split output pass. It keeps both GDN passes at 320 CTAs and
is correct and memcheck-clean on the target shape (`max_abs=0`,
`compute-sanitizer: ERROR SUMMARY: 0 errors`). Its nsys breakdown is:

| Kernel | gridX | Duration |
|---|---:|---:|
| state-only per-segment transition | 320 | 122.562 us |
| `compute_segment_coeffs` | 320 | 4.896 us |
| `compose_segment_input_states` | 4096 | 37.952 us |
| split GDN | 320 | 211.106 us |

So sequence splitting is theoretically viable and is the first tested path that
both increases grid count and preserves recurrent state semantics. It is not yet
a usable replacement: even after removing Q/O from checkpointing, the correct
multi-pass flow is still `0.3773 ms`, slower than the original `0.2575 ms`. A
production-quality variant would need to fuse transition, prefix, and output, or
use a cooperative kernel with an in-kernel prefix phase, rather than launching
separate transition and output kernels.

## Practical Next Steps

1. Build this FlashInfer GDN prefill kernel as one translation unit, or otherwise
   avoid separable compilation for the TU containing `cutlass::device_kernel`.
   Keep this as an isolated study target unless the default benchmark semantics
   are intentionally changed.
2. Check production `linear_attn/bench_gdn_prefill` for the same ptxas warning.
   If present, it is the primary reason the current production standalone is slow.
3. Keep tile `64x64x128`; the naive `128x128x128` GVA DeltaRule instantiation fails
   compile-time MMA layout assertions in the extracted FlashInfer collective.
4. Plain V-dimension blocking is not enough for this CUTLASS extraction because
   it duplicates the auxiliary path. A useful next attempt would need to share
   QK/KK/alpha-beta work across V slices or split auxiliary/state phases.
5. The checkpointed split-sequence prototype shows a viable direction for the
   output pass. State-only and scan-style prepasses reduce the prefix cost but do
   not make the multi-pass design faster overall. The next useful implementation
   would need to fuse the phases or use a cooperative in-kernel state prefix.
