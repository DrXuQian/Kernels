# LM Head GEMV Bandwidth Study

This is a standalone study for Qwen3.5 LM head decode:

```text
logits[1, vocab] = hidden[1, hidden] x lm_head_weight[vocab, hidden]^T
vocab = 248320
hidden = 3072
dtype = fp16 input/weight, fp32 output
```

It is intentionally not wired into the main `compile.sh` or `bench_all.sh`.

The optimized kernel assumes row-major weight layout `weight[N, K]`, so each
warp computes one vocab row and loads the K dimension coalesced as `half2`.
The activation vector is staged into shared memory once per CTA.

## Build

```bash
cd studies/lm_head_gemv_bw
make clean && make ARCH=-arch=sm_90a
```

With an explicit toolkit:

```bash
CUDA_ROOT=/path/to/cuda make clean all ARCH=-arch=sm_90a
```

## Run

Run the optimized kernel and copy roofline:

```bash
./bench_lm_head_gemv --op=all --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200
```

Only cuBLAS bandwidth baseline:

```bash
./bench_lm_head_gemv --op=cublas --n=248320 --k=3072 --warmup=100 --iters=200
```

Only LM head GEMV:

```bash
./bench_lm_head_gemv --op=shared --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200
```

Only the inline-PTX GEMV path:

```bash
./bench_lm_head_gemv --op=ptx --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200
```

More aggressive inline-PTX variants:

```bash
# 4-way K unroll, 4 independent accumulators per lane.
./bench_lm_head_gemv --op=ptx_u4 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200

# 2 output rows per warp + 4-way K unroll. This is aimed at platforms where
# the single-row GEMV path shows memory_dependency stalls.
./bench_lm_head_gemv --op=ptx_r2u4 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200
```

Compile with `-O1` if you want to reduce backend scheduling aggressiveness for
compiler-sensitivity checks:

```bash
make clean && make ARCH=-arch=sm_90a CFLAGS='-O1 -std=c++17 --expt-relaxed-constexpr'
```

Only copy roofline with the same weight bytes:

```bash
./bench_lm_head_gemv --op=copy --n=248320 --k=3072 --warmup=100 --iters=200
```

Try different CTA shapes:

```bash
for w in 4 8 16; do
  ./bench_lm_head_gemv --op=shared --n=248320 --k=3072 --warps-per-block=$w --warmup=100 --iters=200
done
```

The same cuBLAS baseline can also be run from the generic repo benchmark:

```bash
../../general/bench_cublas_gemm \
  --m=1 --n=248320 --k=3072 \
  --dtype=fp16 --out-dtype=fp32 \
  --bench 50 100
```

Bandwidth accounting:

- `weight+out BW`: counts mandatory LM head traffic: `N*K*sizeof(fp16) + N*sizeof(fp32)`.
- `copy BW`: counts read + write bytes for a vectorized copy of the weight-sized buffer.

## H800 Reference

Local H800 PCIe, `N=248320`, `K=3072`, fp16 input/weight and fp32 logits:

| case | median | effective bandwidth |
|---|---:|---:|
| `shared`, 4 warps/block | 0.7964 ms | 1.917 TB/s |
| `shared`, 8 warps/block | 0.7927 ms | 1.926 TB/s |
| `shared`, 16 warps/block | 0.7928 ms | 1.926 TB/s |
| `global`, 8 warps/block | 0.7897 ms | 1.933 TB/s |
| `ptx_global`, 8 warps/block | 0.7868 ms | 1.940 TB/s |
| `ptx_u4`, 8 warps/block, `-O3` | 0.7887 ms | 1.936 TB/s |
| `ptx_r2u4`, 8 warps/block, `-O3` | 0.7888 ms | 1.935 TB/s |
| `ptx_u4`, 8 warps/block, `-O1` | 0.7883 ms | 1.937 TB/s |
| `ptx_r2u4`, 8 warps/block, `-O1` | 0.7886 ms | 1.936 TB/s |
| cuBLAS baseline | 0.7903 ms | ~1.932 TB/s |
| `copy_u8` weight-sized copy | 1.7238 ms | 1.770 TB/s |

The GEMV bandwidth is computed using mandatory LM head traffic
`weight_bytes + logits_bytes`, not read+write copy traffic. For this shape that
is `1526.671 MB`. The dedicated row-major kernel is effectively matching cuBLAS
for this decode LM head case.

The `ptx_global` path fixes the critical load/FMA/store sequence with inline PTX
(`ld.global.u32`, `fma.rn.f32`, `st.global.f32`) to reduce sensitivity to CUDA
C++ frontend optimization differences.

On H800, the more aggressive PTX variants do not beat `ptx_global`; they are
included mainly for other backends where the simple loop may still show
`memory_dependency` stalls.
