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

# 4 consecutive half2 per lane using ld.global.v4.u32. This reduces load
# instruction count and is currently the best H800 variant.
./bench_lm_head_gemv --op=ptx_chunk4 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200

# 2 output rows per warp + 4-way K unroll. This is aimed at platforms where
# the single-row GEMV path shows memory_dependency stalls.
./bench_lm_head_gemv --op=ptx_r2u4 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200

# 2 output rows per warp + 8-way K unroll, every packed load issued before any
# FMA. Keeps ~24 vector loads in flight per warp (vs ~8 for ptx_r2u4) to widen
# the outstanding-load window on backends whose block timeline shows the SM idle
# behind s.wait / vmcnt load waits. Profile this and compare against ptx_r2u4.
./bench_lm_head_gemv --op=ptx_r2u8 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200

# 2 output rows per warp + consecutive chunk4 loads.
./bench_lm_head_gemv --op=ptx_r2_chunk4 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200

# 2 rows/warp + chunk4 v4 loads + 2-tile unroll: all 6 v4 loads issued before
# any FMA. Combines the wide loads of ptx_r2_chunk4 with the deep outstanding
# window of ptx_r2u8 (same ~96 bytes/lane in flight as ptx_r2u8, but via 6 wide
# loads instead of 24 narrow ones).
./bench_lm_head_gemv --op=ptx_r2_chunk4u2 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200

# 2 rows/warp + chunk4 v4 loads + 4-tile unroll: 12 v4 loads (192 bytes/lane)
# issued before any FMA. Doubles the outstanding window of ptx_r2_chunk4u2 for
# backends that can sustain a deeper in-flight load queue.
./bench_lm_head_gemv --op=ptx_r2_chunk4u4 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200

# ptx_r2_chunk4u2 with the hidden vector staged into shared memory once per
# block (only the 4 weight v4 loads hit global). Diagnostic: the gap vs
# ptx_r2_chunk4u2 measures how much hidden re-read traffic a backend pushes to
# DRAM. Ties ptx_r2_chunk4u2 on backends that cache the hidden vector.
./bench_lm_head_gemv --op=ptx_r2_chunk4u2_smem --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200

# Weight streamed global -> shared memory via cp.async (sm_80+), so in-flight
# load data lands in smem instead of registers. Decouples the outstanding-load
# window from register pressure: 32 registers, Stages-deep software pipeline.
# --k-unroll selects the pipeline depth Stages (4|8|16); sweep it x warps-per-block.
./bench_lm_head_gemv --op=ptx_cpasync --n=248320 --k=3072 --warps-per-block=8 --k-unroll=16 --warmup=100 --iters=200

# 4 output rows per warp + consecutive chunk4 loads. Useful to test whether
# more independent weight streams help a backend, but it has higher register pressure.
./bench_lm_head_gemv --op=ptx_r4_chunk4 --n=248320 --k=3072 --warps-per-block=8 --warmup=100 --iters=200
```

Configurable outstanding experiment:

```bash
# rows_per_warp={1,2,4,8}, k_unroll={4,8,16}
for r in 1 2 4 8; do
  for u in 4 8 16; do
    ./bench_lm_head_gemv \
      --op=ptx_ru \
      --rows-per-warp=$r \
      --k-unroll=$u \
      --n=248320 --k=3072 \
      --warps-per-block=8 \
      --warmup=100 --iters=200
  done
done
```

Single-kernel NCU capture for the same sweep:

```bash
METRICS='gpu__time_duration.avg,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct'

for r in 1 2 4 8; do
  for u in 4 8 16; do
    ncu --target-processes all --kernel-name-base demangled --page raw --csv \
      --metrics "$METRICS" \
      ./bench_lm_head_gemv \
        --op=ptx_ru \
        --rows-per-warp=$r \
        --k-unroll=$u \
        --n=248320 --k=3072 \
        --warps-per-block=8 \
        --warmup=0 --iters=1 \
      > /tmp/gemv_r${r}_u${u}.csv
  done
done
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
| `ptx_r2u8`, 8 warps/block, `-O3` | 0.7900 ms | 1.932 TB/s |
| `ptx_chunk4`, 8 warps/block, `-O3` | 0.7807 ms | 1.955 TB/s |
| `ptx_r2_chunk4`, 8 warps/block, `-O3` | 0.7863 ms | 1.942 TB/s |
| `ptx_r2_chunk4u2`, 8 warps/block, `-O3` | 0.7868 ms | 1.940 TB/s |
| `ptx_r2_chunk4u4`, 8 warps/block, `-O3` | 0.7862 ms | 1.942 TB/s |
| `ptx_r2_chunk4u2_smem`, 8 warps/block, `-O3` | 0.7871 ms | 1.940 TB/s |
| `ptx_cpasync`, 8 warps/block, `-O3` | 0.7820 ms | 1.952 TB/s |
| `ptx_r4_chunk4`, 8 warps/block, `-O3` | 0.7885 ms | 1.936 TB/s |
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

On H800, `ptx_chunk4` is the best variant. The multi-row variants are included
mainly for other backends where the single-row loop may still show
`memory_dependency` stalls; on NVIDIA, their extra register pressure does not
pay off.

`ptx_r2u8` widens the outstanding-load window in source: all 24 packed loads of
an iteration (8 hidden + 8 weight row0 + 8 weight row1) are issued before any
conversion or FMA, versus ~8 for `ptx_r2u4`. On H800 this does not change
bandwidth: `ptxas` reschedules the loop and interleaves loads with FMAs
regardless of source order, so the final SASS keeps only a ~9-`LDG` contiguous
batch, and the kernel is already at the bandwidth roofline. `ptx_r2u8` is meant
for latency-bound backends whose compiler keeps the source load grouping (a
profile that shows N loads followed by one `s.wait vldcnt(N-1)` / `vmcnt`
instruction). There, issuing 24 loads up front gives the memory subsystem a
deeper request queue to hide DRAM latency, which should shrink the load-wait
stalls and the idle gaps in the block timeline.

`ptx_r2_chunk4u2` is the wide-load version of the same idea: it keeps the same
~96 bytes/lane outstanding as `ptx_r2u8`, but issues them as 6 `ld.global.v4.u32`
(128-bit) loads instead of 24 `ld.global.u32`. Running `ptx_r2_chunk4` (wide,
shallow), `ptx_r2u8` (narrow, deep) and `ptx_r2_chunk4u2` (wide, deep) on a
backend separates the load-width lever from the load-count lever.

`ptx_r2_chunk4u4` doubles that window again: 4 chunk4 tiles, 12 `v4` loads,
192 bytes/lane in flight. On H800 ptxas still schedules it to the roofline at 54
registers with no spills; on a latency-bound backend it is the deepest of the
wide-load variants before register pressure becomes the limit.

`ptx_r2_chunk4u2_smem` is a diagnostic, not a depth variant: it is
`ptx_r2_chunk4u2` with the hidden vector staged into shared memory once per
block, so the inner loop reads hidden from smem and only weight hits global. In
a GEMV the weight is read once, but the hidden vector is needed by every output
row — each warp re-reads the whole 6 KB hidden vector, ~763 MB total for this
shape. A backend with enough cache (H800: 50 MB L2) absorbs those re-reads, so
`ptx_r2_chunk4u2_smem` ties `ptx_r2_chunk4u2` on H800. A backend that re-reads
hidden from DRAM does not — the gap between the two variants measures that
otherwise-uncounted redundant traffic.

`ptx_cpasync` attacks the register ceiling directly. Every other variant
receives weight into registers, so the outstanding-load window is bounded by the
register file — and registers spent on a deeper window are registers not spent
on occupancy. `ptx_cpasync` streams weight global→shared with `cp.async`: the
in-flight data lands in shared memory, the kernel stays at 32 registers
regardless of pipeline depth, and the `Stages`-deep window is bounded by shared
memory instead. On H800 it ties the roofline (1.952 TB/s, and the case
self-checks correctness against the simple GEMV); it is built for a
latency-bound backend where register-resident windows could not add in-flight
loads without losing occupancy.

The pipeline depth is `--k-unroll` (Stages 4/8/16). The per-warp in-flight
window is `(Stages-1)` tiles * 512 B, so Stages=16 holds ~7.5 KB/warp in shared
memory — about 2.5x the register-resident `chunk4` window — still at 32
registers. By Little's law the bandwidth-delay product sets how much must be in
flight chip-wide; the register file caps that for every other variant, and only
`cp.async` can grow it by spending shared memory instead. On H800 (already at
the roofline) deeper Stages still nudges up: 1.952 → 1.960 → 1.965 TB/s at
Stages 4 → 8 → 16.
