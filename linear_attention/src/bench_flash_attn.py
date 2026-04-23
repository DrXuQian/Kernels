#!/usr/bin/env python3
"""
Bench: FlashAttention v3 decode + prefill for Full Attention layers
Qwen3.5-122B: num_heads=32, num_kv_heads=2, head_dim=256
Usage: python3 bench_flash_attn.py [mode] [seq_len]
  mode: decode (default) or prefill
  seq_len: context length for decode, or prefill length
"""
import sys
import torch
import numpy as np

try:
    from flash_attn import flash_attn_func
    FA_VERSION = "flash_attn"
except ImportError:
    FA_VERSION = None

mode = sys.argv[1] if len(sys.argv) > 1 else "decode"
seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 3823

# 122B Full Attention config
NUM_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 256
WARMUP, REPEAT = 20, 100
dev = torch.device('cuda:0')

print(f"bench flash_attn {mode}: heads={NUM_HEADS} kv_heads={NUM_KV_HEADS} dim={HEAD_DIM} seq={seq_len}")

if FA_VERSION is None:
    # Use torch SDPA as fallback
    print("  flash_attn not installed, using torch.nn.functional.scaled_dot_product_attention")
    FA_VERSION = "sdpa"

def benchmark_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
    for i in range(REPEAT):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return np.median([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])

if mode == "decode":
    # Decode: Q is (1,1,num_heads,head_dim), KV cache is (1,seq_len,num_kv_heads,head_dim)
    q = torch.randn(1, 1, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    k = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)

    if FA_VERSION == "flash_attn":
        t = benchmark_fn(lambda: flash_attn_func(q, k, v))
    else:
        # Expand KV for GQA
        rep = NUM_HEADS // NUM_KV_HEADS
        k_exp = k.repeat_interleave(rep, dim=2)
        v_exp = v.repeat_interleave(rep, dim=2)
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_exp.transpose(1, 2)
        v_sdpa = v_exp.transpose(1, 2)
        t = benchmark_fn(lambda: torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa))

    print(f"  {FA_VERSION} decode: median={t:.1f} μs (ctx={seq_len})")

elif mode == "prefill":
    # Prefill: Q,K,V all (1,seq_len,heads,head_dim)
    q = torch.randn(1, seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    k = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)

    if FA_VERSION == "flash_attn":
        t = benchmark_fn(lambda: flash_attn_func(q, k, v, causal=True))
    else:
        rep = NUM_HEADS // NUM_KV_HEADS
        k_exp = k.repeat_interleave(rep, dim=2)
        v_exp = v.repeat_interleave(rep, dim=2)
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_exp.transpose(1, 2)
        v_sdpa = v_exp.transpose(1, 2)
        t = benchmark_fn(lambda: torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True))

    print(f"  {FA_VERSION} prefill: median={t:.1f} μs (seq={seq_len})")

print("Done.")
