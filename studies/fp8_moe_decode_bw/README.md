# FP8 MoE Decode Bandwidth Study

This study tracks MiniMax-M2.7 TP1 decode-one-token FP8 MoE GEMM bandwidth on
H800 PCIe. It is intentionally separate from the default compile and benchmark
flows.

## Current Best Path

Default benchmark path:

```bash
MOE_GEMM_BACKEND=fp8_trtllm ./bench_MiniMax-M2.7_TP1.sh decode --case moe_gate_up_decode_fp8_trtllm
MOE_GEMM_BACKEND=fp8_trtllm ./bench_MiniMax-M2.7_TP1.sh decode --case moe_down_decode_fp8_trtllm
```

The underlying binary is:

```bash
moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release/bench_moe_fp8_blockscale_gemm
```

The kernel is TensorRT-LLM/CUTLASS block-FP8 grouped GEMM with static
TMA/WGMMA. DeepGEMM JIT is not used unless `--deep-gemm` is explicitly passed.

## Tactic Search

The standalone binary supports a small static tile search:

```text
tile_m = 64 or 128
tile_n = 64 or 128
tile_k = 128
```

Search command:

```bash
moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release/bench_moe_fp8_blockscale_gemm \
  --experts=8 --m_per_expert=1 --n=3072 --k=3072 \
  --sweep_configs \
  --tactic=moe_ffn/fp8/trtllm_cutlass_standalone/tactics_h800_minimax.cache \
  --bench 1 7
```

Load command:

```bash
moe_ffn/fp8/trtllm_cutlass_standalone/build_cmake_release/bench_moe_fp8_blockscale_gemm \
  --experts=8 --m_per_expert=1 --n=3072 --k=3072 \
  --tactic=moe_ffn/fp8/trtllm_cutlass_standalone/tactics_h800_minimax.cache \
  --bench 1 5
```

Use at least one warmup for direct benchmark timing. `--bench 0 1` is still
valid for profiler capture, but the binary's CUDA-event timer includes cold
start noise in that mode. Nsight Systems kernel duration is the authoritative
single-kernel latency.

The no-tactic static fallback now uses the recorded best MiniMax heuristic:
decode-like `m_per_expert <= 64` defaults to `64x128x128`, while prefill-like
`m_per_expert > 64` defaults to `128x64x128`.

## H800 PCIe Results

Nsight Systems kernel duration with `bench_h800_bandwidth.sh`, using 2.0 TB/s as
the practical H800 PCIe bandwidth reference:

| case | tile | duration | estimated bytes | effective bandwidth | peak pct |
|---|---|---:|---:|---:|---:|
| gate/up `(experts=8,m=1,n=3072,k=3072)` | `64x128x128` | 51.552 us | 75,614,280 | 1.467 TB/s | 73.3% |
| down `(experts=8,m=1,n=3072,k=1536)` | `64x128x128` | 20.479 us | 37,831,752 | 1.847 TB/s | 92.4% |

The previous static fallback tile was `128x64x128`:

| case | old latency | new latency | improvement |
|---|---:|---:|---:|
| gate/up | 60.992 us | 51.552 us | 1.18x |
| down | 30.783 us | 20.479 us | 1.50x |

## Rejected Candidate

`tile_n=32` was tested as an extra grouped-GEMM decode candidate. It compiled
successfully but was slower:

| candidate | gate/up median | down median |
|---|---:|---:|
| `64x32x128` | 66.4 us | 40.9 us |
| `64x128x128` | 51.8 us | 24.2 us |

Therefore `tile_n=32` is omitted from the committed tactic search space.

## CUDA-Core Direction

For TP1 down, the current static TMA/WGMMA kernel already reaches roughly 92% of
the H800 PCIe bandwidth reference, so a CUDA-core replacement is unlikely to
improve it.

Gate/up still has headroom at roughly 73% of the same reference. A CUDA-core
study kernel is only worth pursuing for `m_per_expert == 1` if it can stream the
FP8 expert weights with fewer synchronization and scheduling overheads than the
current grouped WGMMA kernel. The risk is high: a CUDA-core implementation must
decode FP8 values, apply per-128 activation scales and 128x128 weight scales,
and accumulate in FP32 without tensor cores. That extra scalar work can easily
turn a bandwidth-bound kernel into a compute/convert-bound kernel.

Recommended next experiment, if more work is needed:

1. Keep the current TMA/WGMMA tactic as the default.
2. Add a separate study-only CUDA-core kernel for `m_per_expert == 1`.
3. Support only the TP1 gate/up and down shapes initially.
4. Compare only Nsight Systems kernel duration and derived effective bandwidth.
5. Do not switch the model benchmark default unless the study kernel beats
   `64x128x128` on gate/up and does not regress down.
