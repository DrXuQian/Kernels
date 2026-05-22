# Linear Residual Add Bandwidth Study

This directory is intentionally outside the main `compile.sh` and `bench_all.sh`
flows. It is a small study for comparing:

- baseline scalar residual add: `fp16 -> fp32 add -> fp16`
- optimized residual add: vectorized `Half8` load/store with `half2` add
- pure memory copy: `cudaMemcpyAsync` device-to-device
- pure memory copy: SM copy kernels using `uint4` and `ulonglong4`
- pure memory copy: SM `ulonglong4` kernels with load/store split unroll

The target operator is Qwen3.5 linear/full-attention residual add:

```text
out = residual + update
shape = (tokens, 3072)
dtype = fp16
traffic = tokens * 3072 * 3 * sizeof(fp16)
```

For copy kernels, bandwidth counts both read and write traffic:

```text
traffic = 2 * bytes
```

## Build

```bash
cd studies/linear_residual_add_bw
make clean && make ARCH=-arch=sm_90a
```

With an explicit toolkit:

```bash
CUDA_ROOT=/path/to/cuda make clean all ARCH=-arch=sm_90a
```

## Run

Run all cases for the Qwen3.5 prefill shape:

```bash
./bench_residual_add_bw --op=all --tokens=3823 --hidden=3072 --warmup=100 --iters=200
```

Run larger sequence lengths:

```bash
for t in 3823 8192 16384 32768 65536 131072; do
  ./bench_residual_add_bw --op=all --tokens=$t --hidden=3072 --warmup=100 --iters=200
done
```

Run only copy roofline with a fixed allocation size:

```bash
./bench_residual_add_bw --op=copy --mib=2048 --warmup=200 --iters=300
```

Run the copy outstanding experiment. These variants use the same `ulonglong4`
vector type, but each thread preloads `1/2/4/8/16` vectors before storing them:

```bash
./bench_residual_add_bw --op=copy_unroll --mib=4096 --warmup=100 --iters=200

for u in 1 2 4 8 16; do
  ./bench_residual_add_bw --op=copy_u8_unroll${u} --mib=4096 --warmup=100 --iters=200
done
```

Single-kernel NCU capture:

```bash
METRICS='gpu__time_duration.avg,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct'

for u in 1 2 4 8 16; do
  ncu --target-processes all --kernel-name-base demangled --page raw --csv \
    --metrics "$METRICS" \
    ./bench_residual_add_bw --op=copy_u8_unroll${u} --mib=4096 --warmup=0 --iters=1 \
    > /tmp/copy_u${u}.csv
done
```

Run only the vectorized residual add:

```bash
./bench_residual_add_bw --op=half8 --tokens=65536 --hidden=3072 --warmup=100 --iters=200
```

## H800 Reference

On the local H800 PCIe, the main observations were:

| case | effective bandwidth |
|---|---:|
| scalar residual add, Qwen3.5 shape `(3823,3072)` | ~1.36 TB/s |
| scalar residual add, large shape | ~1.43 TB/s |
| vectorized half8 residual add, Qwen3.5 shape `(3823,3072)` | ~1.68 TB/s |
| vectorized half8 residual add, large shape | ~1.84 TB/s |
| SM copy kernel large shape | ~1.77 TB/s |
| `cudaMemcpyAsync` D2D large shape | ~1.86 TB/s |

Interpretation: the scalar production kernel is not a pure memory-copy roofline
case because it converts through fp32 per element. The half8 study path removes
that overhead and reaches roughly the device-copy roofline on H800.
