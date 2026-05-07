# Marlin MoE W4A16 GEMM

从 vLLM 提取的 Marlin MoE kernel：FP16 activation + INT4 weight + group_size=128。

For Qwen3.5-122B decode, `num_experts=256` and `top_k=8`; the active routed
rows are represented by `M * top_k` inside the standalone benchmark.

## 编译

```bash
make
```

## 运行

```bash
./bench_marlin_moe [M] [num_experts] [top_k] [K] [N]   # default binary values are a smoke shape
```

## ncu Profile

```bash
ncu --set full --kernel-name "Marlin" -o marlin_decode_gate_up  ./bench_marlin_moe 1 256 8 2048 3072 --balanced --no-topk-weights --bench 0 1
ncu --set full --kernel-name "Marlin" -o marlin_decode_down     ./bench_marlin_moe 1 256 8 3072 1024 --balanced --bench 0 1
```
