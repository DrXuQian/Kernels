# MoE W4A16 Kernels

MoE FFN layer 的 standalone kernel bench，FP16 activation + INT4 weight。

## 目录

```
marlin/          Marlin MoE GEMM (standalone C++, 从 vLLM 提取)
auxiliary/       辅助 kernel: topk, align, silu_and_mul, sum (从 vLLM 提取)
bench_python/    Python bench: Marlin vs moe_wna16 vs BF16 (需 PyTorch)
```

## Pipeline

```
topk_gating → moe_align → Marlin GEMM (gate_up) → silu_and_mul → Marlin GEMM (down) → moe_sum
[auxiliary]    [auxiliary]   [marlin]                [auxiliary]      [marlin]             [auxiliary]
```

## 编译

```bash
cd marlin    && make    # Marlin GEMM
cd auxiliary && make    # 辅助 kernel
```
