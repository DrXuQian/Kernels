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

### First-Principles Split Boundary

FlashQLA's context-parallel path is not a single-kernel implementation of the
whole recurrence. Its forward wrapper first launches:

1. `chunk_local_cumsum` for the gate prefix data;
2. `kkt_solve` for the per-chunk triangular system;
3. optional CP preprocessing (`get_warmup_chunks`, `fused_gdr_h`,
   `correct_initial_states`) when the heuristic enables CP;
4. `fused_chunk_gdr_fwd` for the main output/state pass.

The upstream CP heuristic disables CP when the effective batch/head grid is
already large enough:

```text
use_cp = Be * H <= 40 or (Be * H <= 56 and max_seq_chunks >= 128)
```

For the repo target shape (`B=1,Hv=64,T~4k`), `Be * H = 64`, so upstream
FlashQLA does not enable sequence CP by default. This matches the measured
`auto_cp=True` and `auto_cp=False` timings being effectively identical.

The recurrence is the limiting dependency:

```text
state_i = F_i(state_{i-1}, K_i, V_i, alpha_i, beta_i)
out_i   = G_i(state_{i-1}, Q_i, K_i, V_i, alpha_i, beta_i)
```

Splitting the sequence creates more CTAs only if each segment also receives the
correct prefix state. A single-kernel split can therefore win only if it provides
one of these mechanisms:

- a legal in-kernel prefix among segment CTAs, without requiring all split CTAs
  to be resident at a grid-wide barrier;
- a cluster-local prefix formulation that also computes output in the same
  fused cluster work;
- or a compact correction formula that applies the prefix state's contribution
  without running another full GDN-like traversal.

Simply launching more CTAs is not sufficient. A small head-count sweep with the
single-TU baseline shows the wave-quantization effect:

```bash
for hv in 32 64 96 128; do
  studies/linear_prefill_flashinfer_gdn/bench_gdn_tile_study_single_tu \
    3823 16 $hv 128 1 --tile 64 --variant default --bench 5 20
done
```

| `Hv` | Grid CTAs | Median (ms) | Interpretation |
|---:|---:|---:|---|
| 32 | 32 | 0.2608 | one underfilled wave |
| 64 | 64 | 0.2613 | still one underfilled wave |
| 96 | 96 | 0.2723 | closer to filling local H800 SMs |
| 128 | 128 | 0.5362 | spills into a second wave on the 114-SM local H800 |

This confirms the core issue: below one resident wave, whole-GPU utilization is
poor, but latency is still dominated by the serial per-CTA recurrent loop. Once
the grid exceeds one resident wave, latency can jump instead of falling. A
winning decomposition must both fill SMs and reduce or reuse the serial
per-segment work.

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
| scan transition + prefix + split pass, 768-token segments | 320 + prefix + 320 | 0.3773 ms |
| scan transition + prefix + split pass, 1280-token segments | 192 + prefix + 192 | 0.3660 ms |

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

Sweeping segment sizes did not find an end-to-end win. The best measured point
was `segment_tokens=1280`, with 192 CTAs for both GDN passes:

| Kernel | gridX | Duration |
|---|---:|---:|
| state-only per-segment transition | 192 | 126.017 us |
| `compute_segment_coeffs` | 192 | 5.408 us |
| `compose_segment_input_states` | 4096 | 20.864 us |
| split GDN | 192 | 210.883 us |

The sweep result:

| Segment tokens | Segments | GDN CTAs | Median (ms) |
|---:|---:|---:|---:|
| 2048 | 2 | 128 | 0.4753 |
| 1536 | 3 | 192 | 0.4292 |
| 1280 | 3 | 192 | 0.3660 |
| 1024 | 4 | 256 | 0.4072 |
| 896 | 5 | 320 | 0.4225 |
| 768 | 5 | 320 | 0.3793 |
| 704 | 6 | 384 | 0.4181 |
| 640 | 6 | 384 | 0.4350 |
| 512 | 8 | 512 | 0.4510 |
| 384 | 10 | 640 | 0.4957 |
| 256 | 15 | 960 | 0.6183 |

So sequence splitting is theoretically viable and is the first tested path that
both increases grid count and preserves recurrent state semantics. It is not yet
a usable replacement: even after removing Q/O from checkpointing, the correct
multi-pass flow is still `0.3660 ms`, slower than the original `0.2575 ms`. A
production-quality variant would need to fuse transition, prefix, and output, or
use a cooperative kernel with an in-kernel prefix phase, rather than launching
separate transition and output kernels.

Nsight Systems was used as a profiler fallback because this local machine blocks
Nsight Compute counters with `ERR_NVGPUCTRPERM`. It confirms that the split
variants really do fix the launch-grid underfill. On the local H800 PCIe,
`sms=114` and the GDN kernel uses `186368` bytes dynamic shared memory, which
keeps occupancy at one active CTA per SM:

| Case | Kernel | gridX | blockX | dyn smem | nsys duration |
|---|---|---:|---:|---:|---:|
| original single-TU | GDN | 64 | 512 | 186368 B | 264.7 us |
| zero-split, 768 tokens | GDN | 320 | 512 | 186368 B | 195.3 us |
| scan-both, 1280 tokens | state transition GDN | 192 | 512 | 186368 B | 127.7 us |
| scan-both, 1280 tokens | compose prefix | 4096 | 256 | 0 B | 21.2 us |
| scan-both, 1280 tokens | split output GDN | 192 | 512 | 186368 B | 213.6 us |

So the objective's mechanical part is met for sub-passes: grid count can be made
larger than SM count. The performance part is not met end-to-end because the
extra state transition, prefix composition, and second output pass cost more
than the occupancy gain saves.

For systems with NCU counters enabled, the direct metric command is:

```bash
ncu --csv --page raw --print-units base --kernel-name-base demangled \
  -k regex:.*FlatKernelTmaWarpSpecializedDeltaRule.* --launch-count 1 \
  ./bench_gdn_tile_study_single_tu 3823 16 64 128 --tile 64 --variant default
```

### Zero-State Output Plus Prefix Correction

Another possible decomposition is:

1. Run per-segment GDN from zero state. This produces each segment's local output
   and local final state with a large grid.
2. Compose prefix states across segments.
3. Add the exact prefix-state contribution to the output.

The key question is whether step 3 is cheap. For correctness, the prefix
contribution is not just a simple `Q @ prefix_state` GEMM: within a segment, the
prefix state is transformed by the same K/beta/alpha homogeneous recurrence that
the full GDN kernel uses. A diagnostic `correction_full` mode was added to time
the exact correction by running the init-state GDN path with `V=0`.

Measured CUDA-event times:

| Mode | Segment tokens | CTAs | Median (ms) | Meaning |
|---|---:|---:|---:|---|
| zero-split | 768 | 320 | 0.1888 | zero-state local output + local transition |
| zero-split | 1280 | 192 | 0.1979 | zero-state local output + local transition |
| correction-full | 768 | 320 | 0.2064 | exact prefix correction, `V=0` |
| correction-full | 1280 | 192 | 0.2072 | exact prefix correction, `V=0` |
| zero-v-correction-full | 768 | 320 | 0.2049 | exact correction, skips V TMA/load path |
| zero-v-correction-full | 1280 | 192 | 0.2084 | exact correction, skips V TMA/load path |

For the best prefix-compose point from the scan sweep (`segment_tokens=1280`),
the lower-bound total is:

```text
zero_split 0.1979 ms + prefix compose ~0.026 ms + correction_full 0.2072 ms
  ~= 0.431 ms
```

This is slower than the current `scan_both` path and much slower than the
original single kernel. Therefore an exact correction pass does not rescue the
multi-pass split-sequence approach unless the correction itself is rewritten into
a substantially cheaper specialized kernel. The current FlashInfer/CUTLASS GDN
structure keeps too much of the original Q/K auxiliary and homogeneous state
work even when `V=0`.

## Cooperative In-Kernel Prefix Feasibility

The natural next question is whether transition, prefix, and output can be fused
into one CUDA cooperative kernel with `cooperative_groups::this_grid().sync()`.
`bench_gdn_coop_probe` was added to answer the mechanical launch constraint
before attempting a much larger CUTLASS rewrite.

The probe instantiates the real FlashInfer/CUTLASS GDN kernels and reports their
resource usage:

```bash
make coop_probe -j
./bench_gdn_coop_probe
./bench_gdn_coop_probe --launch-dummy
```

Observed on the local H800 PCIe:

| Kernel variant | Stages Q/K/V | Threads/block | Shared storage | Active blocks/SM | Max resident cooperative grid |
|---|---:|---:|---:|---:|---:|
| full GDN original | 2/3/2 | 512 | 186368 B | 1 | 114 |
| full GDN `k2` | 2/2/2 | 512 | 169984 B | 1 | 114 |
| full GDN minimal-stage probe | 1/1/1 | 512 | 120832 B | 1 | 114 |
| full GDN checkpoint | 2/3/2 | 512 | 186368 B | 1 | 114 |
| full GDN init-state split | 2/3/2 | 512 | 186368 B | 1 | 114 |
| state-only transition | 2/3/2 | 512 | 186368 B | 1 | 114 |
| state-only checkpoint | 2/3/2 | 512 | 186368 B | 1 | 114 |

The dummy cooperative launch uses the same block size and dynamic shared-memory
request. It succeeds at 64 CTAs and fails at 128 CTAs and above:

```text
launch grid=64   ok
launch grid=128  failed: too many blocks in cooperative launch
launch grid=192  failed: too many blocks in cooperative launch
launch grid=320  failed: too many blocks in cooperative launch
```

This blocks the simple fused cooperative-grid-sync approach. The split output
phase becomes interesting at 192/256/320 CTAs, but a cooperative kernel with the
current GDN shared-memory footprint can only resident-launch one CTA per SM.
Reducing K stages to the already-tested `k2` variant saves only 16 KB. Even the
diagnostic `1/1/1` pipeline-stage probe is still 120832 B, above the
approximately half-SMEM threshold needed for two resident blocks on this device.
On a larger H800 the exact number changes with SM count, but the constraint
remains: `max_grid = SM_count` while active blocks/SM is 1.

Therefore the next viable fused design cannot simply add a global grid sync to
the existing FlashInfer GDN kernel. It would need one of:

- a materially smaller-SMEM kernel shape so at least two blocks per SM can be
  resident under cooperative launch;
- a persistent/work-queue algorithm that never requires all split CTAs to be
  resident at the same barrier;
- or a different prefix/state formulation that avoids a grid-wide sync inside
  the heavy GDN kernel.

### Thread-Block Cluster Probe

Hopper thread-block clusters have different residency semantics from a full-grid
cooperative launch. A cluster only requires the blocks inside one cluster to be
co-resident, so it could match the natural decomposition here: one cluster per
V/output head, with one CTA per sequence segment inside that cluster.

`bench_gdn_coop_probe --launch-cluster-dummy` launches a tiny
`cooperative_groups::this_cluster().sync()` kernel using the same block size and
dynamic shared-memory request as the GDN kernels. On the local H800 PCIe, it
succeeds for all segment counts tested:

| Cluster size | Grid | Result |
|---:|---:|---|
| 1 | 64 | ok |
| 2 | 128 | ok |
| 3 | 192 | ok |
| 4 | 256 | ok |
| 5 | 320 | ok |
| 8 | 512 | ok |

`bench_gdn_coop_probe --bench-cluster-prefix` then times the prefix-state
exchange pattern that a cluster implementation would need. The microbenchmark
writes one `128x128` float state per segment/head to global memory, runs
`this_cluster().sync()`, and composes each segment's prefix state inside the same
cluster-launched kernel:

| Segments | Grid | Median (ms) | Notes |
|---:|---:|---:|---|
| 2 | 128 | 0.0212 | write + cluster sync + prefix compose |
| 3 | 192 | 0.0384 | write + cluster sync + prefix compose |
| 5 | 320 | 0.0586 | write + cluster sync + prefix compose |
| 8 | 512 | 0.1260 | write + cluster sync + prefix compose |

This is only a launch feasibility test; it does not prove that a full GDN
cluster implementation is easy. The state exchanged between segment CTAs is
large (`128x128` float per output/state head), and the current GDN CTA already
uses 186 KB of shared memory, so the realistic communication path is global
scratch plus cluster synchronization, not keeping all prefix state in distributed
shared memory. The microbenchmark shows that this prefix exchange is plausible,
but not free.

This was then connected to the real split-sequence dataflow as
`cluster_scan_split` / `cluster_scan_both`. The cluster kernel computes each
segment/head decay coefficient, uses `this_cluster().sync()` across segment CTAs
for one head, and composes the prefix input states consumed by the existing
init-state GDN split pass. Correctness on the target shape is exact:

```text
./bench_gdn_splitseq_study_single_tu 3823 16 64 128 \
  --segment-tokens 768 --mode cluster_scan_both --check
check: max_abs=0 max_rel=0 elements=31318016
```

The measured path is still slower than the existing non-cluster scan path:

| Mode | Segment tokens | Grid shape | Median (ms) | Notes |
|---|---:|---|---:|---|
| original single-TU | full sequence | 64 | 0.2603 | same rebuilt local binary |
| scan-both | 1280 | 192 + prefix + 192 | 0.3658 | best current multi-pass point |
| cluster-scan-both | 768 | 320 + cluster-prefix + 320 | 0.3932 | real cluster compose |
| cluster-scan-both | 1280 | 192 + cluster-prefix + 192 | 0.4027 | real cluster compose |

So thread-block clusters solve the mechanical synchronization constraint, but a
standalone cluster-prefix pass is not enough. To beat the original single GDN
kernel, prefix handling needs to be fused into the GDN CTA/cluster work so the
algorithm does not pay for separate transition and output/correction passes.

## Fused Cluster Scope

The current FlashInfer GDN kernel cannot become the desired fused cluster kernel
through a launch-only change:

- `FlatBuilderDeltaRule` always uses `IndividualTileScheduler`, and that
  scheduler maps work only over `(seq, head)`. There is no segment dimension in
  the production work descriptor.
- `FlatMainloopTmaWarpSpecializedDeltaRule` currently has
  `ClusterShape = Shape<_1, _1, _1>`. The existing GDN kernel therefore does not
  have a segment cluster baked into its CUTLASS pipeline/barrier layout.
- The mainloop computes each block's output from the current recurrent state and
  then updates that state. If a long sequence is split into independent segment
  CTAs, the correct prefix state for segment `i` is not known until earlier
  segment transitions are complete. Starting from zero gives the fast
  `zero_split` path, but the exact correction still needs another GDN-like pass
  (`correction_full`).

This is why the implemented cluster compose kernel is correct but not faster:
it only moves the prefix composition into a cluster launch; it does not remove
the need for a separate state-transition pass and a separate output/correction
pass. A real winning implementation would need a new collective/scheduler that
either:

1. computes segment transition summaries and output correction summaries in one
   pass, then applies prefix state without re-reading the full Q/K/V stream; or
2. derives an output formula that lets the zero-state output be corrected from a
   compact per-segment summary instead of a second GDN-like traversal.

Without one of those algorithmic changes, increasing the grid improves SM
occupancy for a sub-pass but does not improve end-to-end latency.

### Zero-V Correction Probe

`correction_full` already passes a zero V tensor, but the generic GDN collective
still instantiates the V TMA/load pipeline. A dedicated `zero_v_correction_full`
study collective was added to test whether that was the remaining easy win. It
skips V descriptor prefetch, V TMA loads, and the V pipeline wait/release, while
preserving Q/K, QK/KK, `S@K`, `NewV`, output, and state-update math.

The path is exact against the generic correction path:

```text
./bench_gdn_splitseq_study_single_tu 3823 16 64 128 \
  --segment-tokens 1280 --mode zero_v_correction_full --check
check: max_abs=0 max_rel=0 elements=31318016

compute-sanitizer --tool memcheck --print-limit 1 \
  ./bench_gdn_splitseq_study_single_tu 3823 16 64 128 \
  --segment-tokens 1280 --mode zero_v_correction_full
ERROR SUMMARY: 0 errors
```

The measured benefit is negligible:

| Mode | Segment tokens | Median (ms) | Avg (ms) |
|---|---:|---:|---:|
| correction-full | 768 | 0.2065 | 0.2064 |
| zero-v-correction-full | 768 | 0.2049 | 0.2050 |
| correction-full | 1280 | 0.2072 | 0.2070 |
| zero-v-correction-full | 1280 | 0.2084 | 0.2092 |

This rules out the zero V load as the dominant correction cost. The correction
pass is dominated by the structural Q/K and state-transform work, and the custom
zero-v collective also triggers a ptxas WGMMA serialization warning.

## Stream-Pipelined Chunk Overlap

A different way to use the idle SMs is to keep GDN itself underfilled but overlap
later chunk-local work with the next GDN segment. This respects the recurrence:
segment `i+1` starts only after segment `i` writes its final state, but post work
for segment `i` can run on a separate stream while segment `i+1` is running.

The study mode is exact for GDN output:

```text
./bench_gdn_splitseq_study_single_tu 3823 16 64 128 \
  --segment-tokens 1280 --post-rounds 32 \
  --mode stream_segments_post_overlap --check
check: max_abs=0 max_rel=0 elements=31318016
```

Measured CUDA-event timings on the local H800:

| Mode | Segment tokens | Grid shape | Median (ms) | Meaning |
|---|---:|---|---:|---|
| stream-segments | 1280 | 3 sequential 64-CTA GDN launches | 0.3130 | exact segmented GDN only |
| stream-segments + post serial | 1280 | GDN segment + synthetic post on same stream | 0.4641 | no overlap, `post_rounds=32` |
| stream-segments + post overlap | 1280 | GDN stream + synthetic post stream | 0.4192 | overlaps post for completed chunks |

The nsys timeline confirms real kernel overlap:

| Kernel | gridX | Duration |
|---|---:|---:|
| segment 0 GDN | 64 | 103.1 us |
| segment 0 post + segment 1 GDN | 4096 + 64 | 108.2 us + 111.4 us, overlapped |
| segment 1 post + segment 2 GDN | 4096 + 64 | 108.9 us + 104.8 us, overlapped |
| segment 2 post | 4096 | 47.3 us |

This does not make one GDN kernel fill the GPU, and the synthetic post kernel is
only a scheduling proxy. It does show a viable first-principles path: pipeline
the linear-attention block at sequence-segment granularity so chunk-local
downstream operations consume SMs that the 64-CTA GDN segment does not occupy.
The next useful experiment would replace the synthetic post kernel with the real
chunk-local downstream work, such as output gating/projection or residual add,
and measure whether the extra segment launches are repaid by overlap.

That next check was done with the real extracted fused RMSNorm+sigmoid-gate
kernel used by the linear-attention block. The gate is chunk-local and has shape
`(segment_tokens * num_v_heads, head_dim)`.

```text
./bench_gdn_splitseq_study_single_tu 3823 16 64 128 \
  --segment-tokens 1280 --mode stream_segments_rms_gate_overlap --check
check: max_abs=0 max_rel=0 elements=31318016
```

Measured CUDA-event timings:

| Mode | Segment tokens | Median (ms) | Meaning |
|---|---:|---:|---|
| stream-segments | 1280 | 0.3108 | exact segmented GDN only |
| stream-segments + RMS gate serial | 1280 | 0.7516 | GDN and real gate on one stream |
| stream-segments + RMS gate overlap | 1280 | 0.6720 | real gate overlaps with following GDN segment |

The nsys timeline confirms the real gate overlaps with GDN:

| Kernel | gridX | Duration |
|---|---:|---:|
| segment 0 GDN | 64 | 103.5 us |
| segment 1 GDN + segment 0 RMS gate | 64 + 81920 | 102.1 us + 216.7 us, overlapped |
| segment 2 GDN + segment 1 RMS gate | 64 + 81920 | 98.7 us + 210.5 us, overlapped |
| segment 2 RMS gate | 80832 | 149.7 us |

This validates the scheduling idea with a real downstream operator. The hidden
time is modest, about 80 us in this configuration, because the RMS gate itself
launches a very large grid and dominates once it starts. Still, it is the first
correct path that converts GDN's unused-SM slack into useful work from the same
linear-attention block.

Still, this is the first synchronization mechanism tested that is mechanically
compatible with both requirements:

- large grid count (`segments * num_v_heads`, e.g. 192 or 320 CTAs);
- local in-kernel ordering among the segments of one head.

The remaining concrete implementation direction is therefore a fused
cluster-per-head GDN prototype:

1. change the tile scheduler so blocks are ordered as
   `(head, segment_in_cluster)`, with `clusterDim.x = segments`;
2. run the per-segment state transition from zero in each CTA;
3. store each segment's state and decay coefficient to a per-cluster/global
   scratch buffer;
4. use `this_cluster().sync()`;
5. compose prefix states inside the cluster;
6. compute the segment output with the composed prefix state.

This would be a larger rewrite than the current real cluster-compose probe, but
it is no longer blocked by CUDA launch constraints. It also has a clear
performance caveat: a cluster version that still performs separate
state-transition and full output/correction GDN work remains close to the
current multi-pass cost. For the cluster path to beat the original single
kernel, it must reuse work within the CTA/cluster or otherwise avoid a second
full GDN-like pass.

## Completion Audit

Objective: use a FlashQLA-like or otherwise valid decomposition to raise the GDN
prefill grid enough to fill the GPU SMs.

| Requirement | Evidence | Status |
|---|---|---|
| Keep experiments isolated from production benchmarks | All new GDN experiments live under `studies/linear_prefill_flashinfer_gdn`; `bench_all.sh` is not modified | Done |
| Try FlashQLA-style V blocking | `block_DV=64` prototype reaches 128 CTAs and is memcheck-clean | Done, not faster |
| Understand why smaller V blocking cannot continue | `block_DV=32` fails CUTLASS SM90 GMMA static assertion: tile M must be multiple of 64 | Done |
| Try sequence splitting without breaking recurrent state | Checkpointed split-seq path uses `EnableCheckpointing` + `InitStateFromInput`; target-shape `max_abs=0` | Done |
| Increase grid beyond H800 SM count | split/scan paths reach 192, 256, 320, 384, 512, 640, and 960 CTA cases depending on segment size; nsys confirms `gridX=320` for `zero_split` and `gridX=192` for both GDN passes in `scan_both` | Done |
| Check whether SM underfill is mechanically fixed | GDN occupancy is one CTA/SM due to 186368 B dynamic SMEM; original grid 64 underfills local 114-SM H800, while 192/320-CTA split passes can fill it | Done for grid coverage; direct NCU counters blocked locally by `ERR_NVGPUCTRPERM` |
| Verify correctness of the best semantic paths | `scan_split --check: max_abs=0`; compute-sanitizer reports `ERROR SUMMARY: 0 errors` | Done |
| Check cooperative in-kernel prefix feasibility | Actual GDN kernels use 512 threads and 186368 B SMEM, so cooperative resident grid is only one CTA per SM; dummy cooperative launch fails at 128+ CTAs on local H800 PCIe | Done, blocked |
| Check zero-state output plus exact correction decomposition | `zero_split` is 0.1979 ms and exact `V=0` correction is 0.2072 ms at 1280-token segments, giving a lower-bound total around 0.431 ms with prefix compose | Done, not faster |
| Check thread-block cluster feasibility | Dummy cluster launch with GDN-sized 512-thread/186368-byte CTA succeeds for cluster sizes 2/3/5/8; cluster prefix-state exchange costs 0.0384 ms for 3 segments and 0.0586 ms for 5 segments | Done, feasible but not sufficient |
| Check real thread-block cluster compose in split path | `cluster_scan_both --check` reports `max_abs=0`; timing is 0.3932 ms at 768-token segments and 0.4027 ms at 1280-token segments | Done, correct but slower |
| Check whether zero V loading dominates correction cost | `zero_v_correction_full --check` reports `max_abs=0`; timing is 0.2049 ms vs 0.2065 ms at 768-token segments and 0.2084 ms vs 0.2072 ms at 1280-token segments | Done, not the bottleneck |
| Achieve an end-to-end faster replacement | Best correct multi-pass path is 0.3658 ms vs original 0.2603 ms on the rebuilt local binary | Not achieved |

Conclusion: grid count can be made large and correctness can be preserved, but
the tested multi-pass decompositions do not improve end-to-end latency. The
remaining viable direction is not more standalone passes; it is a fused or
cooperative implementation that performs transition, prefix, and output in one
kernel or one cooperative launch.

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
   would need to fuse the phases, but a direct cooperative-grid-sync version is
   blocked by the current 186 KB shared-memory footprint. Reducing SMEM enough to
   allow at least two resident CTAs per SM is a prerequisite for that path.
