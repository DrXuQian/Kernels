# Decode GEMV Bandwidth Scaling vs Pure Memory Copy

This note explains why decode GEMV bandwidth depends strongly on `(M,N,K)`,
using the local H800 PCIe measurements as a concrete reference. The main target
is Qwen3.5 decode with `M=1`, `K=3072`, and variable `N`.

## What Is Being Compared

There are three different bandwidth notions in the measurements:

| Case | Work | Traffic counted |
|---|---|---|
| Pure copy | `dst = src` | read + write bytes |
| Residual add | `out = a + b` | 2 reads + 1 write |
| LM head GEMV | `logits[N] = hidden[K] * W[N,K]` | `W` read + logits write |

For decode GEMV, the hidden vector is only `K * sizeof(fp16)` bytes. With
`K=3072`, that is only 6 KiB, so it is negligible for large `N` and can be
cached or staged. The dominant mandatory traffic is the weight matrix:

```text
weight_bytes = N * K * sizeof(fp16)
output_bytes = N * sizeof(fp32)
mandatory_bytes ~= weight_bytes + output_bytes
```

## Pure Copy Roofline

Measured with `studies/linear_residual_add_bw`:

| Size | cudaMemcpy D2D | SM copy kernel |
|---:|---:|---:|
| 64 MiB | 1.681 TB/s | 1.661 TB/s |
| 128 MiB | 1.761 TB/s | 1.715 TB/s |
| 512 MiB | 1.829 TB/s | 1.762 TB/s |
| 2048 MiB | 1.857 TB/s | 1.773 TB/s |
| 4096 MiB | 1.860 TB/s | 1.775 TB/s |

The useful local roofline is therefore:

```text
runtime D2D copy: ~1.86 TB/s
SM copy kernel:   ~1.77 TB/s
```

The difference is expected: `cudaMemcpyAsync` can use the best internal copy
path, while an SM copy kernel consumes scheduler/warp issue resources.

### How The Pure Copy Is Implemented

There are two copy implementations in the study.

The first is CUDA runtime D2D copy:

```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice);
```

This is not compiled from our CUDA kernel source. It calls the CUDA runtime /
driver copy path, and is the closest simple proxy for device copy throughput.

The second is an explicit SM kernel:

```cpp
__global__ void copy_u8_kernel(
    ulonglong4* __restrict__ dst,
    ulonglong4 const* __restrict__ src,
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride)
    {
        dst[i] = src[i];
    }
}
```

`ulonglong4` is 32 bytes, so each loop iteration is a wide vector load plus a
wide vector store. This is intentionally a streaming memory kernel with minimal
arithmetic.

This kernel should not be optimized away for the following reasons:

- `src` and `dst` are runtime device pointers allocated by `cudaMalloc`.
- The kernel has a global-memory store to `dst`; CUDA device global-memory
  stores are externally visible side effects.
- The compiler cannot assume `dst` is dead after the kernel, because kernel
  launches are externally visible calls and device memory may be observed by
  later kernels or host copies.
- The benchmark records CUDA events around the launch and synchronizes the stop
  event, so the kernel must complete.

That said, the current copy study is a performance microbenchmark, not a
correctness test. It does not copy `dst` back or checksum it after every timed
iteration because that would add non-kernel overhead. If we want to remove even
the appearance of possible dead-store concerns, the right variant is:

```text
copy_u8_kernel <<<...>>> (dst, src)
checksum_kernel <<<1, ...>>> (dst, one scalar output)
```

and only time `copy_u8_kernel`, while running the checksum once after the timed
loop. That keeps the measured kernel clean but forces the copied data to be
observed. For normal CUDA compilation, the explicit global-memory store is
already enough to keep the copy kernel from being eliminated.

## Residual Add As A Sanity Check

Residual add is closer to a model kernel than `cudaMemcpyAsync` because it is an
SM kernel and performs one fp16 add per element.

| Kernel | Shape | Effective BW |
|---|---:|---:|
| scalar residual add | `(3823,3072)` | ~1.36 TB/s |
| scalar residual add | large shapes | ~1.43 TB/s |
| vectorized half8 residual add | `(3823,3072)` | ~1.68 TB/s |
| vectorized half8 residual add | large shapes | ~1.84 TB/s |

The scalar path is slower because it does:

```cpp
half -> float -> add -> half
```

The vectorized study path uses `half2` and 16-byte load/store groups. That gets
very close to the copy roofline. This validates that the machine can reach
copy-like bandwidth from a custom SM kernel if the kernel issues wide,
coalesced memory operations and has enough work.

## cuBLAS Decode GEMV Scaling

Measured with:

```bash
general/bench_cublas_gemm \
  --m=1 --n=N --k=K \
  --dtype=fp16 --out-dtype=fp32 \
  --bench 50 100
```

For `M=1, K=3072`:

| N | Mandatory bytes | cuBLAS median | Effective BW |
|---:|---:|---:|---:|
| 1 | 0.010 MB | 16.8 us | ~0.0006 TB/s |
| 64 | 0.394 MB | 15.0 us | ~0.026 TB/s |
| 256 | 1.574 MB | 15.8 us | ~0.100 TB/s |
| 512 | 3.148 MB | 15.6 us | ~0.202 TB/s |
| 2048 | 12.591 MB | 16.9 us | ~0.745 TB/s |
| 248320 | 1526.671 MB | 790.3 us | ~1.932 TB/s |

The important observation is that the latency for small and medium `N` is almost
flat around `12-17 us`. Effective bandwidth grows mainly because the byte count
increases while the fixed latency is amortized.

## Why Size Changes Effective Bandwidth

Use this decomposition:

```text
total_time =
    launch / library fixed cost
  + scheduler and CTA setup cost
  + reduction cost
  + memory pipeline fill/drain time
  + steady-state memory time
  + compute time
```

Effective bandwidth is:

```text
effective_BW = mandatory_bytes / total_time
```

For small `N`, `mandatory_bytes` is tiny, but the non-steady-state terms are not
tiny. Therefore `mandatory_bytes / total_time` is small.

For large `N`, the steady-state memory term dominates, and the fixed terms are
amortized. Then effective bandwidth approaches the copy roofline.

## Parallelism Limit

For `M=1` GEMV, independent work mostly comes from `N`, not `M`.

If each output column or row maps to a warp/CTA, small `N` gives too few
independent work units:

| N | Work units intuition |
|---:|---|
| 1 | one dot product, cannot fill the GPU |
| 64 | at most dozens of independent dot products |
| 256 | still small for an H800-class GPU |
| 2048 | enough to improve utilization, but kernel is still short |
| 248320 | massive parallelism, enough to keep HBM saturated |

The local H800 has 114 SMs. To saturate HBM, the kernel needs many active warps
and enough outstanding memory requests across SMs. Small `N` decode GEMV ends
before the memory subsystem reaches a long steady state.

## Per-Row Work Is Also Small

Each output element computes a dot product of length `K=3072`.

For fp16 weights:

```text
weight per output = 3072 * 2 B = 6 KiB
```

That is not enough traffic per output row to amortize a lot of control and
reduction overhead. When `N` is small, both conditions are bad:

```text
few output rows * small traffic per row
```

When `N=248320`, the per-row work is still 6 KiB, but there are enough rows to
run many independent warps and generate enough outstanding memory traffic.

## LM Head Custom Kernel

The study kernel in this directory assumes row-major `weight[N,K]`. One warp
computes one vocab row and loads K coalesced as `half2`.

Local H800 result for `N=248320, K=3072`:

| Kernel | Median | Effective BW |
|---|---:|---:|
| custom `shared`, 8 warps/block | 0.7927 ms | 1.926 TB/s |
| custom `global`, 8 warps/block | 0.7897 ms | 1.933 TB/s |
| custom `ptx_global`, 8 warps/block | 0.7868 ms | 1.940 TB/s |
| custom `ptx_u4`, 8 warps/block, `-O3` | 0.7887 ms | 1.936 TB/s |
| custom `ptx_r2u4`, 8 warps/block, `-O3` | 0.7888 ms | 1.935 TB/s |
| custom `ptx_r2u8`, 8 warps/block, `-O3` | 0.7900 ms | 1.932 TB/s |
| custom `ptx_chunk4`, 8 warps/block, `-O3` | 0.7807 ms | 1.955 TB/s |
| custom `ptx_r2_chunk4`, 8 warps/block, `-O3` | 0.7863 ms | 1.942 TB/s |
| custom `ptx_r2_chunk4u2`, 8 warps/block, `-O3` | 0.7868 ms | 1.940 TB/s |
| custom `ptx_r2_chunk4u4`, 8 warps/block, `-O3` | 0.7862 ms | 1.942 TB/s |
| custom `ptx_r2_chunk4u2_smem`, 8 warps/block, `-O3` | 0.7871 ms | 1.940 TB/s |
| custom `ptx_r4_chunk4`, 8 warps/block, `-O3` | 0.7885 ms | 1.936 TB/s |
| custom `ptx_u4`, 8 warps/block, `-O1` | 0.7883 ms | 1.937 TB/s |
| custom `ptx_r2u4`, 8 warps/block, `-O1` | 0.7886 ms | 1.936 TB/s |
| cuBLAS | 0.7903 ms | ~1.932 TB/s |

For LM head, cuBLAS and the dedicated kernel are both near the copy roofline.
This is because the problem has enough `N` and enough mandatory weight traffic.

The `ptx_global` variant keeps the same warp-per-row mapping, but fixes the
critical memory and arithmetic instructions with inline PTX:

```text
ld.global.u32   // one packed half2 load from hidden / weight
fma.rn.f32      // explicit fp32 FMA
st.global.f32   // logits write
```

This is not meant to beat cuBLAS materially. It exists to reduce differences
from CUDA C++ frontend optimization when comparing compiler behavior.

Two more aggressive PTX variants are included for non-NVIDIA backend studies:

| Variant | Idea |
|---|---|
| `ptx_u4` | 4-way K unroll; issue multiple packed hidden/weight loads before using them; 4 independent accumulators per lane. |
| `ptx_r2u4` | Same 4-way K unroll, but each warp computes 2 vocab rows and reuses the hidden load across two independent weight streams. |
| `ptx_r2u8` | 2 rows/warp, 8-way K unroll, all 24 packed loads of an iteration issued before any FMA. Widens the outstanding-load window (~24 loads in flight vs ~8 for `ptx_r2u4`) for latency-bound backends idle behind `s.wait`/`vmcnt` load waits. |
| `ptx_chunk4` | Each lane loads 4 consecutive `half2` values with `ld.global.v4.u32`, reducing load instruction count and improving contiguous memory issue. |
| `ptx_r2_chunk4` | 2 rows/warp plus `chunk4`; tests whether more independent weight streams help after load vectorization. |
| `ptx_r2_chunk4u2` | 2 rows/warp, `chunk4` v4 loads, 2-tile unroll with all 6 v4 loads issued before any FMA. Wide-load version of `ptx_r2u8`: same ~96 bytes/lane outstanding, 6 wide loads instead of 24 narrow ones. |
| `ptx_r2_chunk4u4` | 2 rows/warp, `chunk4` v4 loads, 4-tile unroll with all 12 v4 loads issued before any FMA. Doubles the `ptx_r2_chunk4u2` window to 192 bytes/lane; 54 registers on H800, no spills. |
| `ptx_r2_chunk4u2_smem` | `ptx_r2_chunk4u2` with the hidden vector staged into shared memory once per block. Diagnostic: the gap vs `ptx_r2_chunk4u2` measures how much redundant hidden re-read traffic a backend pushes to DRAM. |
| `ptx_r4_chunk4` | 4 rows/warp plus `chunk4`; higher memory-level parallelism but much higher register pressure. |
| `ptx_ru` | Configurable `rows_per_warp={1,2,4,8}` and `k_unroll={4,8,16}` path for memory-outstanding sweeps. |

On the local H800, `ptx_chunk4` is the best variant. This suggests load
instruction count / contiguous issue still matters slightly even when the simple
PTX loop is already near cuBLAS. The row-multiplexed variants (`r2`/`r4`) do not
win on H800, likely because their additional registers and arithmetic scheduling
cost offset the extra independent memory streams. They are still useful on
platforms where profiling shows the simple GEMV loop stalling on memory
dependency instead of vmem pipe pressure.

`ptx_r2u8` is the explicit outstanding-load experiment. Its inner loop issues
all 24 packed loads (8 hidden + 8 weight row0 + 8 weight row1) before any
conversion or FMA, versus ~8 loads ahead of the first FMA in `ptx_r2u4`. On
H800 it is a no-op: `ptxas` owns instruction scheduling and re-interleaves loads
with FMAs regardless of source order, so the final SASS keeps only a ~9-`LDG`
contiguous batch and `ptx_r2u8` measures the same as `ptx_r2u4` (both at the
bandwidth roofline, both 40 registers, no spills). It is built for latency-bound
backends whose compiler is faithful to source load grouping — the signature is a
block timeline where the SM is idle ~50% of the time and the per-instruction
profile is dominated by `s.wait vldcnt(N)` / `vmcnt` waits after a batch of N
loads. There the deeper source-level load batch translates into a deeper
hardware request queue, so more DRAM latency is overlapped and the load-wait
stalls shrink. The value to verify after switching to `ptx_r2u8` is the sum of
`s.wait`/`vmcnt` stall cycles and the timeline idle fraction, not raw FLOPs.

`ptx_r2_chunk4u2` applies the same deep-window idea on top of 128-bit
`ld.global.v4.u32` loads. Running `ptx_r2_chunk4` (wide, shallow window),
`ptx_r2u8` (narrow, deep window) and `ptx_r2_chunk4u2` (wide, deep window) on the
target backend separates the two levers: whether the gain comes from wider
memory transactions or from a deeper outstanding-request queue.

`ptx_r2_chunk4u4` extends the wide-and-deep variant to 4 chunk4 tiles (12 v4
loads, 192 bytes/lane in flight). Each v4 load consumes one `vldcnt`/`vmcnt`
slot but carries 128 bits, so the wide-load variants reach a given byte window
with 4x fewer outstanding-load counter slots than the scalar `ptx_r2u8` path --
which matters when that counter, not the register file, is the limit. The
practical depth ceiling is register pressure: `ptx_r2_chunk4u4` is 54 registers
on H800, and going deeper risks spills that are fatal for a bandwidth kernel.

The outstanding-window variants all read the hidden vector from global memory,
once per warp. In a GEMV the weight matrix is read exactly once, but the hidden
vector is needed by every output row, so each of the ~N/2 warps re-reads the
whole 6 KB hidden vector — ~763 MB of hidden traffic for `N=248320`, versus
1526 MB of weight. The benchmark's mandatory-traffic figure counts only the
weight, which is correct when a backend caches hidden (H800's 50 MB L2 does, so
`ptx_r2_chunk4u2_smem` ties `ptx_r2_chunk4u2`). On a backend that re-reads
hidden from DRAM, the real traffic is ~1.5x the counted figure, which alone caps
the reported bandwidth utilization near 67%. `ptx_r2_chunk4u2_smem` stages the
hidden vector into shared memory once per block to remove that redundancy; the
gap between it and `ptx_r2_chunk4u2` isolates how large the effect is on a given
backend.

For SASS / final-ISA inspection, use:

```bash
cd studies/lm_head_gemv_bw
make clean && make ARCH=-arch=sm_90a
cuobjdump --dump-sass ./bench_lm_head_gemv > /tmp/lm_head.sass
c++filt < /tmp/lm_head.sass | grep -n "lm_head_gemv_ptx"
```

The key checks are whether `ptx_chunk4` remains a wide load sequence in final
ISA, whether the backend introduces spills, and how many independent
instructions separate load issue from FMA use.

The in-directory cuBLAS bandwidth example is:

```bash
cd studies/lm_head_gemv_bw
./bench_lm_head_gemv --op=cublas --n=248320 --k=3072 --warmup=100 --iters=200
```

## Implication For Decode Kernels

For decode dense projections:

| Operator | Shape | Expected behavior |
|---|---:|---|
| shared expert gate | `(1,1,3072)` | latency-bound |
| linear in_proj_a/b | `(1,64,3072)` | latency-bound |
| router gate | `(1,256,3072)` | latency-bound |
| full-attn K/V projection | `(1,512,3072)` | latency-bound / under-filled |
| LM head | `(1,248320,3072)` | bandwidth-bound |

Therefore:

- cuBLAS is appropriate for LM head.
- cuBLAS is not a good latency/bandwidth model for small decode GEMV.
- Small decode GEMV needs specialized kernels or fusion with adjacent ops.
- A specialized kernel can reduce fixed overhead, but it still cannot reach copy
  roofline unless the shape exposes enough parallelism and traffic.

## Why Pure Copy Is Not A Sufficient Predictor

Pure copy measures the hardware's ability to move bytes in a long streaming
kernel. Decode GEMV adds:

- one reduction per output element,
- limited parallelism when `N` is small,
- warp/CTA scheduling overhead,
- library dispatch and heuristic overhead for cuBLAS,
- memory pipeline fill/drain effects for short kernels.

Only large-`N` GEMV resembles pure copy strongly enough for bandwidth roofline
reasoning to hold directly.
