#!/usr/bin/env python3
"""
Bench: fused_recurrent_gated_delta_rule decode (fla Triton kernel)
This is the SAME kernel vLLM uses for GDN decode.

Usage: python3 bench_gdn_decode.py [num_heads] [head_dim]
Default: Qwen3.5-122B (h=64, d=128) / 35B (h=32, d=128)

nsys: nsys profile --trace=cuda python3 bench_gdn_decode.py
"""
import sys
import torch
import numpy as np
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

WARMUP = 20
REPEAT = 100
dev = torch.device('cuda:0')

num_heads = int(sys.argv[1]) if len(sys.argv) > 1 else 64
head_dim  = int(sys.argv[2]) if len(sys.argv) > 2 else 128
B, T = 1, 1  # decode: single token

print(f"bench gdn_decode (fla Triton): B={B} T={T} H={num_heads} D={head_dim}")

q = torch.randn(B, T, num_heads, head_dim, dtype=torch.bfloat16, device=dev)
k = torch.randn(B, T, num_heads, head_dim, dtype=torch.bfloat16, device=dev)
v = torch.randn(B, T, num_heads, head_dim, dtype=torch.bfloat16, device=dev)
g = torch.randn(B, T, num_heads, dtype=torch.float32, device=dev).sigmoid().log()
beta = torch.randn(B, T, num_heads, head_dim, dtype=torch.bfloat16, device=dev).sigmoid()
state = torch.randn(B, num_heads, head_dim, head_dim, dtype=torch.float32, device=dev) * 0.01

# Warmup
for _ in range(WARMUP):
    fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, scale=1.0/head_dim**0.5,
                                      initial_state=state, output_final_state=True)
torch.cuda.synchronize()

# Bench
starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
for i in range(REPEAT):
    starts[i].record()
    fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, scale=1.0/head_dim**0.5,
                                      initial_state=state, output_final_state=True)
    ends[i].record()
torch.cuda.synchronize()

times = np.array([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])  # μs
print(f"  CUDA event: median={np.median(times):.1f} μs, min={np.min(times):.1f} μs")
print("Done.")
