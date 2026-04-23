#!/usr/bin/env python3
"""
Bench: FlashAttention v3 decode + prefill for Full Attention layers
Qwen3.5-122B: num_heads=32, num_kv_heads=2, head_dim=256

Usage:
  python3 bench_flash_attn.py decode [ctx_len] [num_heads] [num_kv_heads] [head_dim]
  python3 bench_flash_attn.py prefill [seq_len] [num_heads] [num_kv_heads] [head_dim]

All data allocation on CPU, only FlashAttn kernel runs on GPU.
For clean nsys trace: nsys profile --trace=cuda python3 bench_flash_attn.py decode 3823
"""
import sys
import torch
import numpy as np

mode        = sys.argv[1] if len(sys.argv) > 1 else "decode"
seq_len     = int(sys.argv[2]) if len(sys.argv) > 2 else 3823
NUM_HEADS   = int(sys.argv[3]) if len(sys.argv) > 3 else 32
NUM_KV_HEADS= int(sys.argv[4]) if len(sys.argv) > 4 else 2
HEAD_DIM    = int(sys.argv[5]) if len(sys.argv) > 5 else 256
WARMUP, REPEAT = 20, 100

try:
    from flash_attn import flash_attn_func
    FA = "flash_attn"
except ImportError:
    FA = "sdpa"

print(f"bench {FA} {mode}: heads={NUM_HEADS} kv_heads={NUM_KV_HEADS} dim={HEAD_DIM} seq={seq_len}")

# ── Allocate on CPU, then copy to GPU (no GPU RNG kernel) ──
if mode == "decode":
    q_cpu = torch.randn(1, 1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    k_cpu = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v_cpu = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
else:
    q_cpu = torch.randn(1, seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    k_cpu = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v_cpu = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)

q = q_cpu.cuda()
k = k_cpu.cuda()
v = v_cpu.cuda()
torch.cuda.synchronize()  # ensure copy done before timing

# ── Build kernel function (no extra GPU ops in the lambda) ──
if FA == "flash_attn":
    if mode == "decode":
        kernel = lambda: flash_attn_func(q, k, v)
    else:
        kernel = lambda: flash_attn_func(q, k, v, causal=True)
else:
    # SDPA fallback: pre-expand KV on GPU (one-time), then only SDPA in loop
    rep = NUM_HEADS // NUM_KV_HEADS
    k_exp = k.repeat_interleave(rep, dim=2)
    v_exp = v.repeat_interleave(rep, dim=2)
    qt = q.transpose(1, 2)
    kt = k_exp.transpose(1, 2)
    vt = v_exp.transpose(1, 2)
    is_causal = (mode == "prefill")
    torch.cuda.synchronize()
    kernel = lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=is_causal)

# ── Warmup ──
for _ in range(WARMUP):
    kernel()
torch.cuda.synchronize()

# ── Benchmark ──
starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
for i in range(REPEAT):
    starts[i].record()
    kernel()
    ends[i].record()
torch.cuda.synchronize()

times = np.array([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
print(f"  {FA} {mode}: median={np.median(times):.1f} μs, min={np.min(times):.1f} μs")
print("Done.")
