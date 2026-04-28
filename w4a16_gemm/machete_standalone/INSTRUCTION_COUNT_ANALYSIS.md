# CUTLASS55 vs Machete Instruction Count Analysis

This note is based primarily on the instruction-count table from the PPU
profiler for:

- CUTLASS example-55 style kernel
- Machete `128x128_1x1x1_TmaMI_TmaCoop_streamK`

The key point is that the Tensor Core work is effectively the same, but the
CUDA-core instruction mix is not. The difference is mostly layout/scheduler
overhead, not different GEMM math.

## Input Table

### CUTLASS55

| Instruction | Count |
|---|---:|
| `v.lop3.b32` | 8,403,328 |
| `v.mul.f16x2` | 8,388,608 |
| `s.wait` | 7,795,801 |
| `v.mov.b32` | 7,451,248 |
| `s.mov.b32` | 7,229,981 |
| `v.mov.v2s` | 7,093,579 |
| `v.madl.i32` | 6,886,016 |
| `v.add.f16x2` | 4,194,304 |
| `v.fma.f16x2` | 4,194,304 |
| `v.shll.b32` | 3,214,016 |
| `v.reg.dchk` | 2,751,680 |
| `v.shrl.b32` | 2,124,962 |
| `v.mma.g.arv` | 2,097,152 |
| `v.mma.g.commit.grp` | 2,097,152 |
| `v.mma.g.f32.f16.m64n128k16.scale_d` | 2,088,960 |

### Machete `128x128_1x1x1_TmaMI_TmaCoop_streamK`

| Instruction | Count |
|---|---:|
| `s.wait` | 8,728,288 |
| `v.lop3.b32` | 8,394,048 |
| `v.mul.f16x2` | 8,388,608 |
| `s.add.i32` | 8,044,072 |
| `s.shll.b32` | 4,211,960 |
| `v.add.f16x2` | 4,194,304 |
| `v.fma.f16x2` | 4,194,304 |
| `v.mov.b32` | 3,480,352 |
| `s.mov.b32` | 2,957,928 |
| `v.reg.dchk` | 2,736,536 |
| `v.shrl.b32` | 2,103,680 |
| `tsm.ld.b32` | 2,097,152 |
| `v.mma.g.arv` | 2,097,152 |
| `v.mma.g.commit.grp` | 2,097,152 |
| `s.shrl.b64` | 2,096,120 |
| `v.mma.g.f32.f16.m64n128k16.scale_d` | 2,088,960 |
| `v.bfi.b32` | 2,064,384 |
| `s.cnvt.i64.i32` | 2,048,000 |
| `s.csel.b32` | 2,023,616 |
| `s.cbr.nz` | 2,014,368 |
| `s.nop` | 1,861,592 |

Absence from the top table should not be read as exactly zero. It means the
instruction was not in the reported top set or was below the displayed
threshold.

## Direct Facts From The Table

### Tensor Core issue counts are essentially identical

| Instruction | CUTLASS55 | Machete | Meaning |
|---|---:|---:|---|
| `v.mma.g.arv` | 2,097,152 | 2,097,152 | Same MMA arrival count |
| `v.mma.g.commit.grp` | 2,097,152 | 2,097,152 | Same MMA commit count |
| `v.mma.g.f32.f16.m64n128k16.scale_d` | 2,088,960 | 2,088,960 | Same GMMA math count |

So the useful WGMMA/Tensor Core work is not the source of the difference.
Both kernels are doing the same dense GEMM math for this shape.

### The core dequant math is also very similar

| Instruction | CUTLASS55 | Machete | Difference |
|---|---:|---:|---:|
| `v.mul.f16x2` | 8,388,608 | 8,388,608 | 0 |
| `v.add.f16x2` | 4,194,304 | 4,194,304 | 0 |
| `v.fma.f16x2` | 4,194,304 | 4,194,304 | 0 |
| `v.lop3.b32` | 8,403,328 | 8,394,048 | +9,280 CUTLASS55 |
| `v.shrl.b32` | 2,124,962 | 2,103,680 | +21,282 CUTLASS55 |
| `v.reg.dchk` | 2,751,680 | 2,736,536 | +15,144 CUTLASS55 |

The big gap is not because one kernel performs much more dequant arithmetic.
The main nibble extraction / scale math / FP16 arithmetic counts are either
identical or very close.

### CUTLASS55 spends more visible instructions on vector moves and index math

| Instruction | CUTLASS55 | Machete | Difference |
|---|---:|---:|---:|
| `v.mov.b32` | 7,451,248 | 3,480,352 | +3,970,896 CUTLASS55 |
| `s.mov.b32` | 7,229,981 | 2,957,928 | +4,272,053 CUTLASS55 |
| `v.mov.v2s` | 7,093,579 | not in top list | CUTLASS55-heavy |
| `v.madl.i32` | 6,886,016 | not in top list | CUTLASS55-heavy |
| `v.shll.b32` | 3,214,016 | not in top list | CUTLASS55-heavy |

This is the biggest difference visible in the CUTLASS55 table. CUTLASS55 has
substantially more vector register movement and vector integer address/index
arithmetic.

### Machete spends more visible instructions on scalar control and scheduling

| Instruction | CUTLASS55 | Machete | Interpretation |
|---|---:|---:|---|
| `s.add.i32` | not in top list | 8,044,072 | scalar tile/index arithmetic |
| `s.shll.b32` | not in top list | 4,211,960 | scalar shifts for layout/index math |
| `s.shrl.b64` | not in top list | 2,096,120 | scalar 64-bit address/index manipulation |
| `s.cnvt.i64.i32` | not in top list | 2,048,000 | scalar index conversion |
| `s.csel.b32` | not in top list | 2,023,616 | predicated scalar control |
| `s.cbr.nz` | not in top list | 2,014,368 | branch/control overhead |
| `s.nop` | not in top list | 1,861,592 | scheduling/alignment filler |
| `tsm.ld.b32` | not in top list | 2,097,152 | shared/TMA-side load pattern |

Machete does not eliminate overhead. It changes its shape: less visible vector
move/address work, more visible scalar control/scheduler work.

## Why The Mix Is Different

### 1. B layout is not the same

CUTLASS55 uses CUTLASS's generic mixed-input path:

- `compute_memory_reordering_atom`
- `LayoutB_Reordered`
- `ValueShuffle = Layout<Shape<_2, _4>, Stride<_4, _1>>`
- generic `CollectiveBuilder`

Machete uses vLLM's custom path:

- `PrepackedLayoutBTemplate`
- `MacheteCollectiveMma`
- `MacheteKernelTag`

The profiler table is consistent with this difference.

Inference from the table: CUTLASS55 still performs more per-fragment movement
and index calculation inside the kernel, which shows up as high `v.mov.*`,
`s.mov.b32`, `v.madl.i32`, and `v.shll.b32`.

Inference from the code: Machete's offline prepack layout is closer to the
custom mainloop's expected GMMA register-load order. That reduces some visible
vector movement, but the custom traversal introduces more scalar index/control
instructions.

### 2. Scheduler is different

Current CUTLASS55 backend uses `GemmUniversal` without an explicit
`StreamKScheduler`:

```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;
```

Machete uses StreamK:

```cpp
using TileScheduler = cutlass::gemm::StreamKScheduler;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler>;
```

This matches the instruction table:

- Machete has much more scalar control/indexing: `s.add.i32`, `s.csel.b32`,
  `s.cbr.nz`, `s.cnvt.i64.i32`, `s.shrl.b64`.
- CUTLASS55 has more vector-side register movement and address arithmetic.

So if a profiler reports "CUDA core instruction count", Machete may look high
because StreamK and custom traversal add scalar instructions even when Tensor
Core work is unchanged.

### 3. Same GMMA count does not imply same CUDA-core count

The table shows identical GMMA issue/commit counts. That only says the Tensor
Core math is the same.

CUDA-core instructions include:

- int4 extraction / bit manipulation
- scale application
- register movement
- address calculation
- scheduler and tile-control logic
- predicate/branch/select logic

Those are exactly where the two kernels differ.

## What The Table Does Not Prove

The table does not prove that one kernel has less total work in all cases:

- It is a top-instruction table, not a complete instruction dump.
- Missing instructions are not necessarily zero.
- Some scalar instructions may be overlapped with Tensor Core latency.
- `s.wait` is high in both kernels, so latency hiding and pipeline occupancy
  matter as much as raw instruction count.

It also does not prove that prepack/reorder is in the timed region. For that,
use `nsys` or the platform profiler capture range to verify the region contains
only the target GEMM kernel.

## Practical Reading Of This Table

For this specific comparison:

1. The Tensor Core workload is the same.
2. The core dequant arithmetic is almost the same.
3. CUTLASS55 has more vector move/address instructions.
4. Machete has more scalar scheduler/control/index instructions.
5. The difference is mostly from layout/mainloop/scheduler design, not from
   different GEMM math.

If the question is "why do CUDA-core instruction counts differ so much", the
short answer is:

> CUTLASS55 pays more in generic mixed-input layout/register movement, while
> Machete pays more in StreamK/custom-prepacked-layout scalar control. Both
> issue essentially the same Tensor Core MMA work.

## Follow-up Experiments

To isolate the sources more cleanly:

1. Compare CUTLASS55 and Machete with the same tile shape and same capture
   region, using `--warmup=0 --iters=10 --profile_gemm_only` under `nsys`.
2. Add or test a non-StreamK Machete variant. If the scalar `s.*` control
   instructions drop sharply, the StreamK scheduler is a major contributor.
3. Add a StreamK CUTLASS55 variant. If CUTLASS55 gains similar `s.csel`,
   `s.cbr`, and `s.add` counts, the scheduler explains much of the gap.
4. Compare Machete prepacked layout against CUTLASS55 `LayoutB_Reordered`
   using the same scheduler. That would isolate layout from scheduling.
