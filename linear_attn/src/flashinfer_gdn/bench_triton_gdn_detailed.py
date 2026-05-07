"""
Benchmark individual Triton sub-kernels of fla chunk_gated_delta_rule.
Qwen3.5-122B: q_heads=16, v_heads=64, dim=128

Pipeline (from nsys):
  l2norm(Q) → l2norm(K) → cumsum(gate) → intra(KKᵀ+inverse) →
  recompute_w_u → chunk_gated_delta_rule_fwd_h → chunk_fwd_o
"""
import torch
import numpy as np
import math
from fla.ops.gated_delta_rule.chunk import (
    l2norm_fwd,
    chunk_local_cumsum,
    chunk_gated_delta_rule_fwd_intra,
    chunk_gated_delta_rule_fwd_h,
    chunk_fwd_o,
    chunk_gated_delta_rule_fwd,
    recompute_w_u_fwd,
)

WARMUP = 20
REPEAT = 100
dev = torch.device('cuda:0')
BT = 64  # chunk size

def bench(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return np.median(times) * 1000  # μs


print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

B = 1
H_qk = 16
H_v = 64
H = H_v     # sab_heads = max(q, v)
D = 128
scale = 1.0 / math.sqrt(D)

for T in [256, 1024, 2048, 3823, 8192]:
    print(f"═══ seqlen={T}, B={B}, q_heads={H_qk}, v_heads={H_v}, dim={D} ═══")

    # fla layout: [B, H, T, D]
    q = torch.randn(B, H_qk, T, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, H_qk, T, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, H_v, T, D, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=dev).sigmoid()
    beta = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=dev).sigmoid()

    # ── 1. Total e2e ──
    try:
        out = chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale,
                                          initial_state=None,
                                          output_final_state=False)
        torch.cuda.synchronize()
        t_e2e = bench(lambda: chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, initial_state=None, output_final_state=False))
        print(f"  {'TOTAL (e2e)':.<45} {t_e2e:>8.1f} μs")
    except Exception as e:
        print(f"  TOTAL e2e FAILED: {e}")
        t_e2e = float('nan')

    # ── 2. l2norm(Q) ──
    try:
        t_l2q = bench(lambda: l2norm_fwd(q))
        print(f"  {'l2norm_fwd(Q)':.<45} {t_l2q:>8.1f} μs")
    except Exception as e:
        print(f"  l2norm(Q): {e}")

    # ── 3. l2norm(K) ──
    try:
        t_l2k = bench(lambda: l2norm_fwd(k))
        print(f"  {'l2norm_fwd(K)':.<45} {t_l2k:>8.1f} μs")
    except Exception as e:
        print(f"  l2norm(K): {e}")

    # ── 4. chunk_local_cumsum(gate) ──
    try:
        t_cum = bench(lambda: chunk_local_cumsum(g, chunk_size=BT))
        print(f"  {'chunk_local_cumsum(gate)':.<45} {t_cum:>8.1f} μs")
    except Exception as e:
        print(f"  cumsum: {e}")

    # ── 5. intra-chunk (KKᵀ, inverse, etc.) ──
    try:
        qn = l2norm_fwd(q)[0] * scale
        kn = l2norm_fwd(k)[0]
        gk = chunk_local_cumsum(g, chunk_size=BT)
        torch.cuda.synchronize()
        # intra returns A (attention weights within chunk)
        t_intra = bench(lambda: chunk_gated_delta_rule_fwd_intra(qn, kn, beta, gk, BT))
        print(f"  {'chunk_gated_delta_rule_fwd_intra':.<45} {t_intra:>8.1f} μs")

        # ── 6. recompute_w_u ──
        A = chunk_gated_delta_rule_fwd_intra(qn, kn, beta, gk, BT)
        torch.cuda.synchronize()
        t_wu = bench(lambda: recompute_w_u_fwd(kn, v, beta, A))
        print(f"  {'recompute_w_u_fwd':.<45} {t_wu:>8.1f} μs")

        # ── 7. chunk_gated_delta_rule_fwd_h (⭐ state update) ──
        w, u = recompute_w_u_fwd(kn, v, beta, A)
        torch.cuda.synchronize()
        t_h = bench(lambda: chunk_gated_delta_rule_fwd_h(
            kn, w, u, g=gk, initial_state=None, output_final_state=False, chunk_size=BT))
        print(f"  {'chunk_gated_delta_rule_fwd_h ⭐':.<45} {t_h:>8.1f} μs")

        # ── 8. chunk_fwd_o (output) ──
        h, _, _ = chunk_gated_delta_rule_fwd_h(
            kn, w, u, g=gk, initial_state=None, output_final_state=False, chunk_size=BT)
        torch.cuda.synchronize()
        t_o = bench(lambda: chunk_fwd_o(qn, kn, v, h, g=gk, scale=1.0, chunk_size=BT))
        print(f"  {'chunk_fwd_o':.<45} {t_o:>8.1f} μs")

        # Sum
        t_sum = t_l2q + t_l2k + t_cum + t_intra + t_wu + t_h + t_o
        print(f"  {'Sum of sub-kernels':.<45} {t_sum:>8.1f} μs")

    except Exception as e:
        import traceback; traceback.print_exc()

    print()
