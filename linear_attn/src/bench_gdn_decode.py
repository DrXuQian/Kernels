#!/usr/bin/env python3
"""
Bench: fused_recurrent_gated_delta_rule decode (fla Triton kernel)
Usage: python3 bench_gdn_decode.py [num_heads] [head_dim] [--bench W I]
Default: single run. Add --bench for timing.
"""
import sys, torch, numpy as np
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

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

H = int(clean[0]) if len(clean) > 0 else 64
D = int(clean[1]) if len(clean) > 1 else 128
dev = torch.device('cuda:0')
print(f"bench gdn_decode (fla Triton): B=1 T=1 H={H} D={D}")

q = torch.randn(1, 1, H, D, dtype=torch.bfloat16).cuda()
k = torch.randn(1, 1, H, D, dtype=torch.bfloat16).cuda()
v = torch.randn(1, 1, H, D, dtype=torch.bfloat16).cuda()
g = torch.randn(1, 1, H, dtype=torch.float32).sigmoid().log().cuda()
beta = torch.randn(1, 1, H, D, dtype=torch.bfloat16).sigmoid().cuda()
state = (torch.randn(1, H, D, D, dtype=torch.float32) * 0.01).cuda()
torch.cuda.synchronize()

fn = lambda: fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, scale=1.0/D**0.5,
                                               initial_state=state, output_final_state=True)
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
