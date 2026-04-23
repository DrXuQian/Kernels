#!/usr/bin/env python3
"""
Bench: Marlin W4A16 GEMV for DeltaNet projections (non-MoE)
Usage: python3 bench_marlin_gemv.py [M] [K] [N] [group_size] [--bench W I]
Default: single run.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, '/root/autodl-tmp/marlin')
import marlin_cuda

args = sys.argv[1:]
bench_mode = False; warmup = 0; repeat = 1; clean = []
i = 0
while i < len(args):
    if args[i] == '--bench':
        bench_mode = True
        warmup = int(args[i+1]) if i+1 < len(args) else 20
        repeat = int(args[i+2]) if i+2 < len(args) else 100
        i += 3
    else: clean.append(args[i]); i += 1

M = int(clean[0]) if len(clean) > 0 else 1
K = int(clean[1]) if len(clean) > 1 else 3072
N = int(clean[2]) if len(clean) > 2 else 12288
G = int(clean[3]) if len(clean) > 3 else 128
print(f"bench Marlin GEMV (W4A16): M={M} K={K} N={N} group={G}")

A = torch.randn(M, K, dtype=torch.half).cuda()
B = torch.randint(-2**31, 2**31, (K*N//8,), dtype=torch.int32).cuda()
C = torch.zeros(M, N, dtype=torch.half).cuda()
s = (torch.rand(K//G, N, dtype=torch.half) * 0.1 + 0.01).cuda()
ws = torch.zeros(N//128*16, dtype=torch.int32).cuda()
torch.cuda.synchronize()

fn = lambda: marlin_cuda.mul(A, B, C, s, ws, -1, -1, -1, 16)
if not bench_mode:
    fn(); torch.cuda.synchronize()
else:
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat): ss[i].record(); fn(); ee[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e)*1000 for s,e in zip(ss, ee)])
    print(f"  Kernel time: median={np.median(times):.1f} μs, min={np.min(times):.1f} μs (warmup={warmup}, iters={repeat})")
print("Done.")
