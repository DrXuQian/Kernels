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
| cuBLAS | 0.7903 ms | ~1.932 TB/s |

For LM head, cuBLAS and the dedicated kernel are both near the copy roofline.
This is because the problem has enough `N` and enough mandatory weight traffic.

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
