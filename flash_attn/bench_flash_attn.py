#!/usr/bin/env python3
"""
Bench: FlashAttention v3 decode + prefill (GQA)
Qwen3.5-122B: num_heads=32, num_kv_heads=2, head_dim=256

Usage:
  python3 bench_flash_attn.py decode [ctx_len] [num_heads] [num_kv_heads] [head_dim]
  python3 bench_flash_attn.py prefill [seq_len] [num_heads] [num_kv_heads] [head_dim] [--ctx ctx_len]
  python3 bench_flash_attn.py decode 3823 --bench 20 100
  python3 bench_flash_attn.py prefill 1024 --ctx 8192 --bench 20 100

--ctx sets the K/V (context) length independently of seq_len, modelling
chunked prefill / prefill against an existing KV cache: seq_len new query
tokens attend to ctx_len total tokens (ctx_len >= seq_len). flash-attn's
causal mask is bottom-right aligned, so the new tokens see all prior context.
Without --ctx, ctx_len defaults to seq_len (full prefill).

Default: single run (for ncu/nsys). Add --bench W I for timing.
All data allocated on CPU, only FlashAttn kernel runs on GPU.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [
    path for path in sys.path
    if Path(path or ".").resolve() != REPO_ROOT
]

import torch
import numpy as np
from flash_attn import flash_attn_func

# Parse --bench and --ctx flags
bench_mode = False
warmup = 0
repeat = 1
context_len = None
args = sys.argv[1:]
clean_args = []
i = 0
while i < len(args):
    if args[i] == '--bench':
        bench_mode = True
        warmup = int(args[i+1]) if i+1 < len(args) else 20
        repeat = int(args[i+2]) if i+2 < len(args) else 100
        i += 3
    elif args[i] in ('--ctx', '--context-len'):
        context_len = int(args[i+1])
        i += 2
    else:
        clean_args.append(args[i])
        i += 1

mode        = clean_args[0] if len(clean_args) > 0 else "decode"
seq_len     = int(clean_args[1]) if len(clean_args) > 1 else 3823
NUM_HEADS   = int(clean_args[2]) if len(clean_args) > 2 else 32
NUM_KV_HEADS= int(clean_args[3]) if len(clean_args) > 3 else 2
HEAD_DIM    = int(clean_args[4]) if len(clean_args) > 4 else 256

# context_len is the K/V (context) length. For prefill it may exceed seq_len
# (chunked prefill / prefill against an existing KV cache); it defaults to
# seq_len, which reproduces the original full-attention shapes.
if context_len is None:
    context_len = seq_len
if mode != "decode" and context_len < seq_len:
    print(f"WARNING: context_len ({context_len}) < seq_len ({seq_len}); causal "
          f"prefill requires context_len >= seq_len. Using context_len = seq_len.")
    context_len = seq_len

print(f"bench flash_attn {mode}: heads={NUM_HEADS} kv_heads={NUM_KV_HEADS} dim={HEAD_DIM} seq={seq_len} ctx={context_len}")

# ── Allocate on CPU, copy to GPU ──
if mode == "decode":
    q_cpu = torch.randn(1, 1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    k_cpu = torch.randn(1, context_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v_cpu = torch.randn(1, context_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
else:
    q_cpu = torch.randn(1, seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    k_cpu = torch.randn(1, context_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v_cpu = torch.randn(1, context_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)

q = q_cpu.cuda()
k = k_cpu.cuda()
v = v_cpu.cuda()
torch.cuda.synchronize()

# ── Kernel ──
if mode == "decode":
    kernel = lambda: flash_attn_func(q, k, v)
else:
    kernel = lambda: flash_attn_func(q, k, v, causal=True)

def profiler_start():
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        pass

def profiler_stop():
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        pass

if not bench_mode:
    # Single run for ncu/nsys
    profiler_start()
    kernel()
    torch.cuda.synchronize()
    profiler_stop()
else:
    # Warmup + timed iterations
    for _ in range(warmup):
        kernel()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    profiler_start()
    for i in range(repeat):
        starts[i].record()
        kernel()
        ends[i].record()
    torch.cuda.synchronize()
    profiler_stop()

    times = np.array([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
    print(f"  Kernel time: median={np.median(times):.1f} μs, min={np.min(times):.1f} μs (warmup={warmup}, iters={repeat})")

print("Done.")
