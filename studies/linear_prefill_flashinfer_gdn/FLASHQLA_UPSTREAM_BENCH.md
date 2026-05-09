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

Correctness-relaxed CP proxy using the local CUDA/FlashInfer kernel:

```bash
studies/linear_prefill_flashinfer_gdn/sweep_flashinfer_cp_proxy.sh
```

Forced upstream FlashQLA sequence-CP diagnostic:

```bash
python3 studies/linear_prefill_flashinfer_gdn/bench_flashqla_force_cp.py \
  --seqlen 3823 --h-qk 16 --h-v 64 --warmup 10 --repeats 50
```

## H800 Results

| Shape | Case | Time |
|---|---|---:|
| `T=3823,Hqk=16,Hv=64` | FlashQLA `auto_cp=True` | 0.455 ms |
| `T=3823,Hqk=16,Hv=64` | FlashQLA `auto_cp=False` | 0.464 ms |
| `T=3823,Hqk=16,Hv=64` | FlashQLA forced sequence-CP | 0.505 ms |
| `T=3823,Hqk=16,Hv=64` | FlashInfer Python | 0.272 ms |
| `T=3823,Hqk=16,Hv=64` | Local FlashInfer single-TU standalone | 0.263 ms |
| `T=4096,Hqk=16,Hv=64` | FlashQLA `auto_cp=True` | 0.455 ms |
| `T=4096,Hqk=16,Hv=64` | FlashQLA `auto_cp=False` | 0.455 ms |
| `T=4096,Hqk=16,Hv=64` | FlashInfer Python | 0.283 ms |
| `T=4096,Hqk=2,Hv=8` | FlashQLA `auto_cp=True` | 0.133 ms |
| `T=4096,Hqk=2,Hv=8` | FlashQLA `auto_cp=False` | 0.191 ms |
| `T=4096,Hqk=2,Hv=8` | FlashInfer Python | 0.343 ms |

## Local CUDA CP Proxy

This proxy splits the total `3823` tokens into multiple independent `cu_seqlens`
entries and runs the same local single-TU FlashInfer GDN kernel. This is not
mathematically equivalent to one recurrent sequence because the state is reset at
each segment. It only measures the optimistic performance upper bound of more
CTA-level parallelism before adding FlashQLA-style warmup/correction kernels.

Command:

```bash
studies/linear_prefill_flashinfer_gdn/sweep_flashinfer_cp_proxy.sh
```

Observed H800 results:

| Segments | Median time |
|---:|---:|
| 1 | 0.2607 ms |
| 2 | 0.2870 ms |
| 4 | 0.2284 ms |
| 8 | 0.2206 ms |
| 16 | 0.2373 ms |
| 32 | 0.2942 ms |

The best relaxed split is `8` segments at `0.2206 ms`, only about 15% faster
than the correct single-sequence local CUDA baseline. Since a correct CP port
would need extra warmup and state-correction kernels, this suggests that a full
FlashQLA-style CP CUDA port is unlikely to deliver a large gain for the target
`Hqk=16,Hv=64,T≈4k` shape.

The forced upstream sequence-CP run used FlashQLA's real warmup/correction path
with `cp_cu_seqlens=[0,1024,2048,3072,3823]`. It measured `0.505 ms`, slower
than FlashQLA's default path. This confirms that the sequence-CP part of
FlashQLA is not the first mechanism to port for this repo's target shape.

## Interpretation

For the repo's target TP1-like shape (`Hqk=16,Hv=64,T≈4k`), upstream FlashQLA is
slower than FlashInfer. `auto_cp=True` and `auto_cp=False` are effectively the
same because FlashQLA's heuristic does not enable sequence-CP at `Hv=64`.

For the low-head TP8-like shape (`Hqk=2,Hv=8,T=4096`), FlashQLA is much faster
than FlashInfer. In that regime the auto-CP path launches additional warmup and
state-correction kernels and improves SM utilization.

So the upstream claim is not universally false, but it does not hold if copied
as sequence-CP for our primary `Hqk=16,Hv=64` usecase. The more useful idea to
borrow is FlashQLA's V-dimension blocking heuristic: for `Hv=64` on H800, it
chooses `block_DV=64`, so each V head is split into two CTAs and the launch has
about `128` CTAs instead of `64`. That can improve SM occupancy without adding
the recurrent state-correction kernels required by sequence-CP.

For this repo, the useful ideas to borrow are:

1. keep evaluating the FlashInfer GDN single-translation-unit build inside this study;
2. implement a study-only `block_DV=64` CUDA path that slices the output/state V
   dimension while keeping the Q/K dimension at 128;
3. consider sequence/context splitting only if we implement the warmup/correct
   state path, because naive sequence splitting changes the recurrence;
4. keep the production benchmark unchanged until the study path is both faster
   and validated.

## Nsight Snapshot

For `T=3823,Hqk=16,Hv=64`, one captured call showed:

| Implementation | Kernel breakdown |
|---|---|
| FlashQLA | `chunk_local_cumsum` 6.6 us + `kkt_solve` 80 us + `fused_chunk_gdr_fwd` 294 us |
| FlashInfer Python | one CUTLASS GDN kernel, about 221 us |

The Python event time is higher than pure kernel time because the high-level API
also allocates temporary tensors and launches multiple kernels.
