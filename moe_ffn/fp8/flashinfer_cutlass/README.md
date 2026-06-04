# FlashInfer CUTLASS FP8 MoE

MiniMax-M2.7 uses ordinary block-wise FP8 E4M3 weights with
`weight_block_size=[128,128]`. This is not MXFP8. On H800/SM90 the matching
vLLM/FlashInfer path is:

```python
flashinfer.fused_moe.cutlass_fused_moe(..., use_deepseek_fp8_block_scale=True)
```

This directory keeps the Python harness separate from the default build because
the CUDA kernels come from the installed FlashInfer package. It is only a
reference path for the SM90 CUTLASS backend; the final target for this repo is a
fully extracted C++/CUDA standalone binary.

The default MiniMax benchmark wrappers do not use this path. The standalone
extraction track lives in `../trtllm_cutlass_standalone/`, which uses the
TensorRT-LLM/CUTLASS block-FP8 grouped GEMM kernel layer directly. This
FlashInfer directory remains a reference harness for comparing against the
package-selected fused path.

Generate or update the FlashInfer autotuner tactic cache explicitly:

```bash
python3 moe_ffn/fp8/flashinfer_cutlass/bench_flashinfer_cutlass_fp8_moe.py \
  --tokens 3823 --experts 8 --topk 8 --hidden 3072 --intermediate 1536 \
  --tactic-cache moe_ffn/fp8/flashinfer_cutlass/tactics_h800_minimax_tp1.json \
  --tune --warmup 0 --iters 1
```

Run with an existing cache and no hidden tuning:

```bash
python3 moe_ffn/fp8/flashinfer_cutlass/bench_flashinfer_cutlass_fp8_moe.py \
  --tokens 3823 --experts 8 --topk 8 --hidden 3072 --intermediate 1536 \
  --tactic-cache moe_ffn/fp8/flashinfer_cutlass/tactics_h800_minimax_tp1.json \
  --warmup 0 --iters 1
```

For local machines without NCU permission, use `--nsys-latency` from the model
runner to get kernel duration only. Bandwidth utilization still requires NCU
DRAM counters.
