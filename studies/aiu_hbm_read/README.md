# PPU 1.0 AIU HBM read benchmark

Standalone, no PyTorch/CUTLASS/Quactlize dependency. It moves raw bytes from
global memory to shared memory with native AIU `.bulk...linear.2d.b8` loads.
There is no GEMM, dequantization, transpose, or full-size output write. Only
an eight-byte completion sentinel per CTA is written to global memory.

## Build and run

Use the PPU 1.0 SDK environment. Do **not** pass `-arch=sm_XX`: that selects a
different target on this SDK. The source rejects non-PPU1.0 device compilation.

```bash
cd Kernels
source /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK/envsetup.sh
make -C studies/aiu_hbm_read
studies/aiu_hbm_read/bench_aiu_read --self-test

# 64 MiB is the operator-provided ZW810 cache capacity, not a guessed SDK value.
# Change it if your device has a different verified L2/LLC capacity.
CUDA_VISIBLE_DEVICES=0 studies/aiu_hbm_read/bench_aiu_read \
  --mib 2048 --l2-mib 64 --peak-gbps 2700 | tee aiu-read.log
```

The default sweep has 48 small configurations: 4/8/16 KiB per bulk copy,
1/2/4/8 outstanding groups, and 1/2/4/8 CTAs per reported compute unit.
Resource-ineligible configurations are reported, not mislabeled as measured.
To repeat one configuration:

```bash
CUDA_VISIBLE_DEVICES=0 studies/aiu_hbm_read/bench_aiu_read \
  --mib 2048 --l2-mib 64 --tile-kib 8 --stages 4 --blocks-per-sm 4 \
  --warmups 2 --samples 31 --peak-gbps 2700
```

`AIU_READ_RESULT` includes raw samples, median/min/max microseconds, decimal
`read_GBs`, shared memory and occupancy capacity. `AIU_READ_BEST` picks the
highest median throughput. `peak_pct` is relative to the **supplied** decimal
GB/s peak, not an SDK measurement or a claim that this is a universal ceiling.
An error returns nonzero; never accept a partial log without `AIU_READ_BEST`.

## Measurement and validation

- Each launch reads every input tile once. Tiles have contiguous 128-byte
  rows and aligned bases; adjacent CTAs cover adjacent complete tiles.
- A ring of shared slots overlaps AIU groups. Before reusing slot `i`,
  `wait_group<stages-1>` retires its previous transfer. `wait_group<0>` drains
  all remaining reads before kernel completion.
- The untimed check specialization consumes **every word of every tile**,
  using address-dependent integer tags, and checks pipeline wrap/drain.
  The timed specialization checks only final per-CTA sentinels afterward.
  Volatile assembly prevents discarded earlier loads; inspect native ISA too.
- Initialization, full validation, cache eviction and warmups are excluded.
  Event samples include the complete read kernel and its final drain/sentinel.
- Default source is 2 GiB; require at least 16 times verified cache capacity.
  A separate 4-times-cache allocation is read before **every** launch outside
  its timed interval. Randomized tags avoid an all-zero/compressible fixture.
- Run with no concurrent GPU workload. Keep the full identity line and repeat
  the winner; a single noisy maximum is not proof of the hardware limit.

This measures the **HBM-to-shared AIU path**, including AIU issue/completion
and shared writes. Requested bytes / event time is effective read bandwidth.
Actual HBM traffic and cache hit rate still need ACU counters; the benchmark
does not label these estimated-byte numbers as measured DRAM transactions.

## Inspect and profile

```bash
"$PPU_SDK/bin/hgobjdump" --list-elf studies/aiu_hbm_read/bench_aiu_read
"$PPU_SDK/bin/hgobjdump" --dump-isa studies/aiu_hbm_read/bench_aiu_read > aiu-read.isa
"$PPU_SDK/bin/hgobjdump" --dump-resource-usage=all studies/aiu_hbm_read/bench_aiu_read
```

For ACU, select `aiu_read<..., false>` (not initialization, eviction or the
`true` validation specialization); skip its two warmup launches and capture
one measured launch. Use the installed ACU's kernel-name/filter syntax from
`acu --help`. Inspect actual DRAM read bytes/throughput, L2/LLC hits and AIU
waits. Profiler replay timing and normal event timing are different evidence.
Retain the `.acurep`, SDK release receipt, source commit and binary SHA256.

Local compilation/ISA inspection and a host schedule test are not substitutes
for a PPU run. Device results are initially pending.
