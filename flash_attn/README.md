# Flash-Attn

Standalone full-attention benchmark entry points.

## Layout

| Path | Contents |
|---|---|
| `bench_flash_attn.py` | FlashAttention Python inference benchmark |
| `bench_flash_infer.py` | FlashInfer Python attention benchmark |
| `bench_rmsnorm` | Category-local build of `general/bench_rmsnorm.cu` |

## Covered Block Pieces

The repo-level `bench_all.sh` covers the non-Python full-attention block pieces:

- hidden RMSNorm
- W4A16 q/k/v/o projections
- q/k RMSNorm via `bench_rmsnorm` with `batch=tokens*heads`, `embed=256`
- FlashAttention core via `bench_flash_attn.py`
- residual add

Qwen3-Next full attention has an output gate after attention:
`attn_output = sigmoid(gate) * attn_output`, with shape `(tokens,8192)`.
In upstream vLLM and TensorRT-LLM `_torch` paths this is expressed as Torch
elementwise ops, and no standalone CUDA implementation has been found in the
local extracted sources.

MRoPE / rotary embedding is not currently extracted as a standalone CUDA kernel
in this repo. Upstream vLLM has a Triton MRoPE kernel; no local TensorRT-LLM
standalone CUDA MRoPE implementation has been found in the extracted sources.

## Build

The Python attention benchmarks are not compiled by this repo. The Makefile
builds only shared CUDA kernels needed by the Flash-Attn category.

```bash
make -C flash_attn

# Or from the repo root:
./compile.sh build flash_attn
```

## Run

```bash
./bench_all.sh --case flash_attn_prefill_full_attn
./bench_all.sh --case flash_attn_decode_full_attn
```
