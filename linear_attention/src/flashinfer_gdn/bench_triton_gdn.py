"""
Benchmark: Triton chunk_gated_delta_rule (from flash-linear-attention)
Compare with FlashInfer CUTLASS SM90 and KDA.
Qwen3.5-122B DeltaNet GVA: q_heads=16, k_heads=16, v_heads=64, dim=128
"""
import torch
import numpy as np

# flash-linear-attention (fla) triton kernel
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gdn

WARMUP = 20
REPEAT = 100
dev = torch.device('cuda:0')


def benchmark_fn(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return np.median(times), np.min(times), np.max(times)


print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Qwen3.5-122B DeltaNet GVA config
num_q_heads = 16   # = num_k_heads
num_v_heads = 64
head_dim = 128
batch = 1

print(f"Config: q_heads={num_q_heads}, v_heads={num_v_heads}, dim={head_dim}")
print(f"{'seqlen':<8} {'Triton fla (ms)':>16} {'min':>10} {'max':>10}")
print("-" * 50)

for total_seqlen in [256, 1024, 2048, 3823, 8192]:
    # fla expects: q [B, H, T, D], k [B, H, T, D], v [B, Hv, T, D]
    # with GQA/GVA: q/k have fewer heads, v has more
    q = torch.randn(batch, num_q_heads, total_seqlen, head_dim,
                    dtype=torch.bfloat16, device=dev)
    k = torch.randn(batch, num_q_heads, total_seqlen, head_dim,
                    dtype=torch.bfloat16, device=dev)
    v = torch.randn(batch, num_v_heads, total_seqlen, head_dim,
                    dtype=torch.bfloat16, device=dev)
    # gate (alpha): [B, H_sab, T, D] where H_sab = max(q_heads, v_heads)
    # For fla, gate shape depends on the implementation
    # Try matching the larger head count
    g = torch.randn(batch, num_v_heads, total_seqlen, head_dim,
                    dtype=torch.bfloat16, device=dev).sigmoid()
    beta = torch.randn(batch, num_v_heads, total_seqlen, head_dim,
                       dtype=torch.bfloat16, device=dev).sigmoid()

    try:
        # fla signature: (q, k, v, g, beta, scale=...)
        # g and beta: [B, H, T, D] for gated delta rule
        out = fla_chunk_gdn(q, k, v, g, beta)
        torch.cuda.synchronize()

        med, mn, mx = benchmark_fn(
            lambda: fla_chunk_gdn(q, k, v, g, beta))
        print(f"{total_seqlen:<8} {med:>14.4f} ms {mn:>10.4f} {mx:>10.4f}")
    except Exception as e:
        print(f"{total_seqlen:<8} GVA failed: {e}")
        # Fallback: all 64 heads
        try:
            q2 = torch.randn(batch, num_v_heads, total_seqlen, head_dim,
                             dtype=torch.bfloat16, device=dev)
            k2 = torch.randn(batch, num_v_heads, total_seqlen, head_dim,
                             dtype=torch.bfloat16, device=dev)
            out = fla_chunk_gdn(q2, k2, v, g, beta)
            torch.cuda.synchronize()
            med, mn, mx = benchmark_fn(
                lambda: fla_chunk_gdn(q2, k2, v, g, beta))
            print(f"{'':8} (h=64 fallback) {med:>10.4f} ms {mn:>10.4f} {mx:>10.4f}")
        except Exception as e2:
            print(f"{'':8} fallback also failed: {e2}")
