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

## Practical Next Steps

1. Build this FlashInfer GDN prefill kernel as one translation unit, or otherwise
   avoid separable compilation for the TU containing `cutlass::device_kernel`.
   Keep this as an isolated study target unless the default benchmark semantics
   are intentionally changed.
2. Check production `linear_attn/bench_gdn_prefill` for the same ptxas warning.
   If present, it is the primary reason the current production standalone is slow.
3. Keep tile `64x64x128`; the naive `128x128x128` GVA DeltaRule instantiation fails
   compile-time MMA layout assertions in the extracted FlashInfer collective.
4. If more optimization is needed, it likely requires algorithmic restructuring:
   more independent CTAs, persistent scheduling across heads/chunks, or splitting
   auxiliary/state phases. Stage-count tuning alone is not enough.
