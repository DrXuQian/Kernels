"""
MoE GEMM Benchmark — larger sizes to expose real compute scaling
Sweep: increase tokens AND N/K to realistic LLM shapes
"""
import sys
import os
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'kernels', 'moe_wna16'))
sys.path.insert(0, os.path.join(_dir, 'kernels', 'marlin_moe'))

import torch
import numpy as np
import moe_wna16_cuda
import marlin_moe_cuda

dev = torch.device('cuda:0')
TOP_K = 2
GROUP_SIZE = 128
BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 128, 128
MARLIN_TILE = 16
WARMUP, REPEAT = 30, 100


def benchmark_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
    for i in range(REPEAT):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return np.median([s.elapsed_time(e) for s, e in zip(starts, ends)])


def make_routing(total_tokens, top_k, num_experts, block_size_m):
    total_exp = total_tokens * top_k
    tpe = max(total_exp // num_experts, 1)
    tpe_pad = ((tpe + block_size_m - 1) // block_size_m) * block_size_m
    total_pad = tpe_pad * num_experts
    sorted_ids = torch.full((total_pad,), total_exp * top_k, dtype=torch.int32, device=dev)
    expert_ids = torch.zeros(total_pad // block_size_m, dtype=torch.int32, device=dev)
    for e in range(num_experts):
        s = e * tpe_pad
        for t in range(min(tpe, tpe_pad)):
            sorted_ids[s + t] = min(e * tpe + t, total_exp - 1)
        for b in range(tpe_pad // block_size_m):
            expert_ids[e * (tpe_pad // block_size_m) + b] = e
    num_post_pad = torch.tensor([total_pad], dtype=torch.int32, device=dev)
    topk_w = torch.ones(total_exp, dtype=torch.float32, device=dev) / top_k
    return sorted_ids, expert_ids, num_post_pad, topk_w, tpe, total_exp


print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ─── Sweep 1: Fix N=K=4096 (realistic hidden dim), vary tokens ───
NUM_EXPERTS = 8
N, K = 4096, 4096

print(f"═══ Sweep 1: N={N}, K={K}, {NUM_EXPERTS} experts, top_k={TOP_K}, group={GROUP_SIZE} ═══")
print(f"{'Tokens':<8} {'M/exp':<7} {'Total GFLOP':>12} "
      f"{'moe_wna16':>11} {'MarlinMoE':>11} {'BF16_x8':>11} | "
      f"{'wna16/BF16':>10} {'Marlin/BF16':>11}")
print("-" * 110)

for total_tokens in [1, 4, 16, 64, 256, 1024, 4096]:
    sid, eid, npp, tw, tpe, texp = make_routing(total_tokens, TOP_K, NUM_EXPERTS, BLOCK_SIZE_M)
    sid_m, eid_m, npp_m, tw_m, _, _ = make_routing(total_tokens, TOP_K, NUM_EXPERTS, 64)

    flops = 2.0 * texp * N * K
    gflop = flops / 1e9

    inp = torch.randn(total_tokens, K, dtype=torch.half, device=dev)

    # 1. moe_wna16
    out = torch.zeros(texp, N, dtype=torch.half, device=dev)
    qw = torch.randint(0, 256, (NUM_EXPERTS, N, K // 2), dtype=torch.uint8, device=dev)
    sc = torch.rand(NUM_EXPERTS, N, K // GROUP_SIZE, dtype=torch.half, device=dev) * 0.1
    ms_wna16 = benchmark_fn(lambda: moe_wna16_cuda.moe_wna16_gemm(
        inp, out.zero_(), qw, sc, None, tw, sid, eid, npp, TOP_K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, 4))

    # 2. Marlin MoE
    mbq = torch.randint(0, 2**31, (NUM_EXPERTS, K // MARLIN_TILE, N * MARLIN_TILE // 8),
                         dtype=torch.int32, device=dev)
    msc = torch.rand(NUM_EXPERTS, K // GROUP_SIZE, N, dtype=torch.half, device=dev) * 0.1
    # workspace needs to be at least sms (number of SMs)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    mws = torch.zeros(max(N // 64 * 16, sms + 16), dtype=torch.int32, device=dev)
    try:
        marlin_moe_cuda.moe_wna16_marlin_gemm(
            inp, None, mbq, None, msc, None, None, None, None, None, mws,
            sid_m, eid_m, npp_m, tw_m, 64, TOP_K, False,
            marlin_moe_cuda.U4B8_TYPE_ID, total_tokens, N, K,
            True, False, False, False, -1, -1, -1)
        torch.cuda.synchronize()
        ms_marlin = benchmark_fn(lambda: marlin_moe_cuda.moe_wna16_marlin_gemm(
            inp, None, mbq, None, msc, None, None, None, None, None, mws,
            sid_m, eid_m, npp_m, tw_m, 64, TOP_K, False,
            marlin_moe_cuda.U4B8_TYPE_ID, total_tokens, N, K,
            True, False, False, False, -1, -1, -1))
    except Exception as e:
        ms_marlin = float('nan')

    # 3. BF16 per-expert sequential
    bA = [torch.randn(max(tpe, 1), K, dtype=torch.bfloat16, device=dev) for _ in range(NUM_EXPERTS)]
    bB = [torch.randn(K, N, dtype=torch.bfloat16, device=dev) for _ in range(NUM_EXPERTS)]
    def bf16_moe():
        for e in range(NUM_EXPERTS):
            torch.mm(bA[e], bB[e])
    ms_bf16 = benchmark_fn(bf16_moe)

    r1 = ms_wna16 / ms_bf16
    r2 = ms_marlin / ms_bf16 if not np.isnan(ms_marlin) else float('nan')
    ms_m = f"{ms_marlin:>9.4f}ms" if not np.isnan(ms_marlin) else "       N/A"
    r2s = f"{r2:>10.2f}x" if not np.isnan(r2) else "        N/A"

    print(f"{total_tokens:<8} {tpe:<7} {gflop:>10.1f}G "
          f"{ms_wna16:>9.4f}ms {ms_m} {ms_bf16:>9.4f}ms | "
          f"{r1:>10.2f}x {r2s}")

print()

# ─── Sweep 2: Realistic model shapes (DeepSeek-V2 / Mixtral) ───
print(f"═══ Sweep 2: Realistic model shapes ═══")
print(f"{'Config':<35} {'moe_wna16':>11} {'MarlinMoE':>11} {'BF16_x8':>11} | "
      f"{'wna16/BF16':>10} {'Marlin/BF16':>11}")
print("-" * 100)

configs = [
    # (name, num_experts, total_tokens, N, K)
    ("Mixtral-8x7B gate (bs=1)",      8,    1, 14336, 4096),
    ("Mixtral-8x7B gate (bs=16)",     8,   16, 14336, 4096),
    ("Mixtral-8x7B gate (bs=128)",    8,  128, 14336, 4096),
    ("Mixtral-8x7B gate (bs=512)",    8,  512, 14336, 4096),
    ("Mixtral-8x7B down (bs=16)",     8,   16,  4096, 14336),
    ("Mixtral-8x7B down (bs=128)",    8,  128,  4096, 14336),
    ("DeepSeek-V2 (bs=1)",           64,    1,  1536, 2048),
    ("DeepSeek-V2 (bs=16)",          64,   16,  1536, 2048),
    ("DeepSeek-V2 (bs=128)",         64,  128,  1536, 2048),
]

for name, ne, tokens, n, k in configs:
    sid, eid, npp, tw, tpe, texp = make_routing(tokens, TOP_K, ne, BLOCK_SIZE_M)
    sid_m, eid_m, npp_m, tw_m, _, _ = make_routing(tokens, TOP_K, ne, 64)

    inp = torch.randn(tokens, k, dtype=torch.half, device=dev)

    # moe_wna16
    out = torch.zeros(texp, n, dtype=torch.half, device=dev)
    qw = torch.randint(0, 256, (ne, n, k // 2), dtype=torch.uint8, device=dev)
    sc = torch.rand(ne, n, k // GROUP_SIZE, dtype=torch.half, device=dev) * 0.1
    ms_wna16 = benchmark_fn(lambda: moe_wna16_cuda.moe_wna16_gemm(
        inp, out.zero_(), qw, sc, None, tw, sid, eid, npp, TOP_K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, 4))

    # Marlin MoE
    try:
        mbq = torch.randint(0, 2**31, (ne, k // MARLIN_TILE, n * MARLIN_TILE // 8),
                             dtype=torch.int32, device=dev)
        msc = torch.rand(ne, k // GROUP_SIZE, n, dtype=torch.half, device=dev) * 0.1
        mws = torch.zeros(max(n // 64 * 16, sms + 16), dtype=torch.int32, device=dev)
        marlin_moe_cuda.moe_wna16_marlin_gemm(
            inp, None, mbq, None, msc, None, None, None, None, None, mws,
            sid_m, eid_m, npp_m, tw_m, 64, TOP_K, False,
            marlin_moe_cuda.U4B8_TYPE_ID, tokens, n, k,
            True, False, False, False, -1, -1, -1)
        torch.cuda.synchronize()
        ms_marlin = benchmark_fn(lambda: marlin_moe_cuda.moe_wna16_marlin_gemm(
            inp, None, mbq, None, msc, None, None, None, None, None, mws,
            sid_m, eid_m, npp_m, tw_m, 64, TOP_K, False,
            marlin_moe_cuda.U4B8_TYPE_ID, tokens, n, k,
            True, False, False, False, -1, -1, -1))
    except Exception as e:
        ms_marlin = float('nan')

    # BF16 per-expert
    bA = [torch.randn(max(tpe, 1), k, dtype=torch.bfloat16, device=dev) for _ in range(ne)]
    bB = [torch.randn(k, n, dtype=torch.bfloat16, device=dev) for _ in range(ne)]
    def bf16_moe():
        for e in range(ne):
            torch.mm(bA[e], bB[e])
    ms_bf16 = benchmark_fn(bf16_moe)

    r1 = ms_wna16 / ms_bf16
    r2 = ms_marlin / ms_bf16 if not np.isnan(ms_marlin) else float('nan')
    ms_m = f"{ms_marlin:>9.4f}ms" if not np.isnan(ms_marlin) else "       N/A"
    r2s = f"{r2:>10.2f}x" if not np.isnan(r2) else "        N/A"

    print(f"{name:<35} "
          f"{ms_wna16:>9.4f}ms {ms_m} {ms_bf16:>9.4f}ms | "
          f"{r1:>10.2f}x {r2s}")

print("-" * 100)
print()
print("Notes:")
print("  Ratio < 1.0 = faster than BF16 sequential")
print("  BF16_x8 = sequential cuBLAS (not grouped GEMM)")
