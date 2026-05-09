# Upstream FlashQLA Speed Check

Source checked: <https://github.com/QwenLM/FlashQLA>, commit `6ef4858`.

This is an upstream Python/TileLang benchmark. It is intentionally not part of
the default `compile.sh` or `bench_all.sh` flow.

## Setup

```bash
git clone --depth 1 https://github.com/QwenLM/FlashQLA.git /tmp/FlashQLA
pip install -v /tmp/FlashQLA
```

Verified local environment:

```text
GPU: NVIDIA H800 PCIe, SM90
CUDA: 12.8
PyTorch: 2.8.0+cu128
TileLang: 0.1.8
FlashInfer Python: 0.6.9
```

## Commands

Qwen3.5-122B-A10B TP1-like shape used by this repo:

```bash
python3 studies/linear_prefill_flashinfer_gdn/bench_flashqla_upstream.py \
  --seqlen 3823 --h-qk 16 --h-v 64 --warmup 10 --repeats 50
```

Low-head upstream benchmark shape where FlashQLA's auto-CP path is expected to help:

```bash
python3 studies/linear_prefill_flashinfer_gdn/bench_flashqla_upstream.py \
  --seqlen 4096 --h-qk 2 --h-v 8 --warmup 10 --repeats 50
```

Current standalone FlashInfer single-translation-unit reference:

```bash
studies/linear_prefill_flashinfer_gdn/bench_gdn_tile_study_single_tu \
  3823 16 64 128 1 --bench 10 50
```

## H800 Results

| Shape | Case | Time |
|---|---|---:|
| `T=3823,Hqk=16,Hv=64` | FlashQLA `auto_cp=True` | 0.455 ms |
| `T=3823,Hqk=16,Hv=64` | FlashQLA `auto_cp=False` | 0.464 ms |
| `T=3823,Hqk=16,Hv=64` | FlashInfer Python | 0.272 ms |
| `T=3823,Hqk=16,Hv=64` | Local FlashInfer single-TU standalone | 0.263 ms |
| `T=4096,Hqk=16,Hv=64` | FlashQLA `auto_cp=True` | 0.455 ms |
| `T=4096,Hqk=16,Hv=64` | FlashQLA `auto_cp=False` | 0.455 ms |
| `T=4096,Hqk=16,Hv=64` | FlashInfer Python | 0.283 ms |
| `T=4096,Hqk=2,Hv=8` | FlashQLA `auto_cp=True` | 0.133 ms |
| `T=4096,Hqk=2,Hv=8` | FlashQLA `auto_cp=False` | 0.191 ms |
| `T=4096,Hqk=2,Hv=8` | FlashInfer Python | 0.343 ms |

## Interpretation

For the repo's target TP1-like shape (`Hqk=16,Hv=64,T≈4k`), upstream FlashQLA is
slower than FlashInfer. `auto_cp=True` and `auto_cp=False` are effectively the
same because FlashQLA's heuristic does not enable intra-card CP at `Hv=64`.

For the low-head TP8-like shape (`Hqk=2,Hv=8,T=4096`), FlashQLA is much faster
than FlashInfer. In that regime the auto-CP path launches additional warmup and
state-correction kernels and improves SM utilization.

So the upstream claim is not universally false, but it does not hold for our
primary `Hqk=16,Hv=64` usecase. For this repo, the useful ideas to borrow are:

1. keep the current FlashInfer GDN kernel in a single translation unit;
2. consider sequence/context splitting only if we implement the warmup/correct
   state path, because naive sequence splitting changes the recurrence;
3. consider `DV` blocking like FlashQLA's `block_DV=64/32` only if it can be
   added without losing the current CUTLASS kernel's lower fused-kernel time.

## Nsight Snapshot

For `T=3823,Hqk=16,Hv=64`, one captured call showed:

| Implementation | Kernel breakdown |
|---|---|
| FlashQLA | `chunk_local_cumsum` 6.6 us + `kkt_solve` 80 us + `fused_chunk_gdr_fwd` 294 us |
| FlashInfer Python | one CUTLASS GDN kernel, about 221 us |

The Python event time is higher than pure kernel time because the high-level API
also allocates temporary tensors and launches multiple kernels.
