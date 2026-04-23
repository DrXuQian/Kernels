#!/usr/bin/env python3
"""
Bench: Marlin W4A16 GEMV for DeltaNet projections (non-MoE)
Usage: python3 bench_marlin_gemv.py [M] [K] [N] [group_size]

nsys: nsys profile --trace=cuda python3 bench_marlin_gemv.py 1 3072 12288
"""
import sys, os
sys.path.insert(0, '/root/autodl-tmp/marlin')
import torch
import numpy as np
import marlin_cuda

M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12288
G = int(sys.argv[4]) if len(sys.argv) > 4 else 128

WARMUP, REPEAT = 20, 100
dev = torch.device('cuda:0')

print(f"bench Marlin GEMV (W4A16): M={M} K={K} N={N} group={G}")

A = torch.randn(M, K, dtype=torch.half, device=dev)
B = torch.randint(-2**31, 2**31, (K * N // 8,), dtype=torch.int32, device=dev)
C = torch.zeros(M, N, dtype=torch.half, device=dev)
s = torch.rand(K // G, N, dtype=torch.half, device=dev) * 0.1 + 0.01
ws = torch.zeros(N // 128 * 16, dtype=torch.int32, device=dev)

# Warmup
for _ in range(WARMUP):
    marlin_cuda.mul(A, B, C, s, ws, -1, -1, -1, 16)
torch.cuda.synchronize()

# Bench
starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
for i in range(REPEAT):
    starts[i].record()
    marlin_cuda.mul(A, B, C, s, ws, -1, -1, -1, 16)
    ends[i].record()
torch.cuda.synchronize()

times = np.array([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
print(f"  CUDA event: median={np.median(times):.1f} μs, min={np.min(times):.1f} μs")
print("Done.")
