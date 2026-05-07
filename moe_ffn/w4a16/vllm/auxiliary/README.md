# MoE Auxiliary Kernels

MoE FFN pipeline 中 GEMM 以外的辅助 kernel，从 vLLM 提取。

For Qwen3.5-122B decode, use `num_experts=256`, `topk=8`, and
`block_size=16`. With 256 experts, vLLM's `moe_align` uses the generic
two-kernel path, not the small-batch 64-expert path.

## Pipeline

```
topk_gating → moe_align → [GEMM gate_up] → silu_and_mul → [GEMM down] → moe_sum
```

## 编译

```bash
make
```

## 运行

```bash
./bench_topk_gating   [num_tokens] [num_experts] [topk]              # default: 1 256 8
./bench_moe_align     [num_tokens] [num_experts] [topk] [block_size] # default: 1 256 8 16
./bench_silu_and_mul  [num_tokens] [top_k] [hidden_size]             # default: 1 8 3072
./bench_moe_sum       [num_tokens] [topk] [hidden_size]              # default: 1 8 3072
```

## ncu Profile

```bash
ncu --set full --kernel-name "topkGating"    -o topk  ./bench_topk_gating 1 256 8
ncu --set full --kernel-name "moe_align"     -o align ./bench_moe_align 1 256 8 16
ncu --set full --kernel-name "silu_and_mul"  -o silu  ./bench_silu_and_mul 1 8 3072
ncu --set full --kernel-name "moe_sum"       -o sum   ./bench_moe_sum 1 8 3072
```
