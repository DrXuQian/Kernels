# linear_prefill_flashinfer_gdn study

This is an isolated optimization study for `linear_prefill_flashinfer_gdn`.
It is intentionally not wired into the top-level compile or benchmark scripts.

See [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) for the deeper root-cause
analysis. See [FLASHQLA_UPSTREAM_BENCH.md](FLASHQLA_UPSTREAM_BENCH.md) for the
upstream FlashQLA speed check on the target shape.

Optional helper scripts:

- `bench_flashqla_upstream.py`: run the upstream FlashQLA Python/TileLang benchmark.
- `sweep_flashinfer_cp_proxy.sh`: sweep a local CUDA split-sequence performance proxy.

The production kernel uses the extracted FlashInfer DeltaNet/GDN prefill path with
Qwen3.5-122B-A10B shape:

```text
total_seqlen=3823, num_q_heads=16, num_k_heads=16, num_v_heads=64, head_dim=128, num_seqs=1
```

## Hypothesis

The current SM90 FlashInfer path hardcodes `TileShape = 64x64x128`. A naive
`128x128x128` token tile is not a legal instantiation for this extracted GVA DeltaRule
collective: it fails compile-time static assertions around the auxiliary QK/KK MMA
thread layout. This study therefore keeps the legal `64x64x128` tile and tunes the
pipeline stage counts:

- `--variant default`: `StagesQ/K/V = 2/3/2`, matching production
- `--variant k2`: `StagesQ/K/V = 2/2/2`
- `--variant q3`: `StagesQ/K/V = 3/3/2`
- `--variant v3`: `StagesQ/K/V = 2/3/3`

## Build

```bash
cd studies/linear_prefill_flashinfer_gdn
make clean
make -j
```

When using a non-default CUDA toolkit:

```bash
CUDA_ROOT=/path/to/cuda \
GPU_ARCH=sm_90a \
make clean all -j
```

Build the single-translation-unit diagnostic target:

```bash
make single_tu -j
```

## Run

Single launch:

```bash
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant default
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant k2
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant q3
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant v3
```

CUDA-event timing:

```bash
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant default --bench 20 100
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant k2 --bench 20 100
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant q3 --bench 20 100
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant v3 --bench 20 100
```

Single-TU timing:

```bash
./bench_gdn_tile_study_single_tu 3823 16 64 128 1 --tile 64 --variant default --bench 20 200
```

Nsight Systems single-kernel check:

```bash
nsys profile -t cuda --force-overwrite=true -o gdn_tile64 \
  ./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant default

nsys profile -t cuda --force-overwrite=true -o gdn_single_tu_default \
  ./bench_gdn_tile_study_single_tu 3823 16 64 128 1 --tile 64 --variant default
```

## Notes

Only the common bf16 GVA path is instantiated:

- `IsGVA=true`
- `NeedsAlpha=true`
- `NeedsBeta=true`
- `InitStateFromInput=false`
- `EnableCheckpointing=false`

This keeps compile time lower and avoids changing the production FlashInfer GDN extraction.

## H800 Results

Local H800 CUDA-event timing, shape `3823 16 64 128 1`, command:

```bash
./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant <variant> --bench 20 200
```

| Variant | Stages Q/K/V | Median (ms) | Avg (ms) | Min (ms) | Max (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| production `linear_attn/bench_gdn_prefill` | 2/3/2 | 0.5232 | 0.5190 | 0.5105 | 0.5271 | production binary |
| study `default` | 2/3/2 | 0.5179 | 0.5160 | 0.5095 | 0.5215 | same stage config as production |
| study `k2` | 2/2/2 | 0.5092 | 0.5123 | 0.5052 | 0.5219 | best measured variant |

Shorter sweep with `--bench 10 50`:

| Variant | Stages Q/K/V | Median (ms) | Avg (ms) |
|---|---:|---:|---:|
| default | 2/3/2 | 0.5251 | 0.5250 |
| k2 | 2/2/2 | 0.5107 | 0.5115 |
| q3 | 3/3/2 | 0.5476 | 0.5476 |
| v3 | 2/3/3 | 0.5439 | 0.5439 |

`k2` is the best of the tested separable-compilation variants, but the gain is
small: about 1.5% versus the production binary in the longer repeat.

The bigger result is the single-translation-unit build:

| Build | Variant | Median (ms) | Avg (ms) | Notes |
|---|---|---:|---:|---|
| separable | default | 0.5179 | 0.5160 | mirrors production-style `-dc` build |
| separable | k2 | 0.5092 | 0.5123 | best separable variant |
| single TU | default | 0.2617 | 0.2617 | avoids `setmaxnreg` loss |

The larger tile idea does not apply cleanly to this extracted GVA DeltaRule
collective because `128x128x128` fails compile-time MMA layout assertions.

Single-launch `nsys` checks:

```bash
nsys profile -t cuda --force-overwrite=true -o /tmp/gdn_prefill_k2_single \
  ./bench_gdn_tile_study 3823 16 64 128 1 --tile 64 --variant k2

nsys profile -t cuda --force-overwrite=true -o /tmp/gdn_prefill_single_tu_default \
  ./bench_gdn_tile_study_single_tu 3823 16 64 128 1 --tile 64 --variant default

nsys stats --report cuda_gpu_kern_sum --format csv /tmp/gdn_prefill_k2_single.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/gdn_prefill_single_tu_default.nsys-rep
```

Observed CUDA kernel summary:

| Build | Variant | CUDA kernels | Kernel time |
|---|---|---:|---:|
| separable | k2 | 1 | 515.494 us |
| single TU | default | 1 | 262.810 us |
