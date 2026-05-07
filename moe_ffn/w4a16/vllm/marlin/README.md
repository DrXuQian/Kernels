# Marlin MoE W4A16 GEMM

从 vLLM 提取的 Marlin MoE kernel：FP16 activation + INT4 weight + group_size=128。

## 编译

```bash
make
```

## 运行

```bash
./bench_marlin_moe [M] [num_experts] [top_k] [K] [N]   # default: 1 64 8 2048 5632
```

## ncu Profile

```bash
ncu --set full --kernel-name "Marlin" -o marlin_decode  ./bench_marlin_moe 1 64 8 2048 5632
ncu --set full --kernel-name "Marlin" -o marlin_prefill ./bench_marlin_moe 128 64 8 2048 5632
```
