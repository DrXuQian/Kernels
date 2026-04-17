"""
MoE GEMM Benchmark (Production kernels from vLLM):
  1. moe_wna16 CUDA (W4A16) — vLLM production for sparse tokens
  2. marlin_moe_wna16 (W4A16) — vLLM Marlin MoE
  3. BF16 cuBLAS per-expert (baseline, no grouped GEMM available w/o flashinfer)

MoE: 8 experts, top_k=2, N=1024, K=1024
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

NUM_EXPERTS = 8
TOP_K = 2
N = 1024
K = 1024
GROUP_SIZE = 128
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128
WARMUP = 30
REPEAT = 100
dev = torch.device('cuda:0')

# Marlin uses tile_size=16 for weight packing
MARLIN_TILE = 16


def benchmark_fn(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return np.median([s.elapsed_time(e) for s, e in zip(starts, ends)])


def make_moe_routing(total_tokens, top_k, num_experts, block_size_m):
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
print(f"MoE: {NUM_EXPERTS} experts, top_k={TOP_K}, N={N}, K={K}, group={GROUP_SIZE}")
print(f"Marlin b_type_id (U4B8): {marlin_moe_cuda.U4B8_TYPE_ID}")
print()

header = (f"{'Tokens':<8} {'M/exp':<7} "
          f"{'moe_wna16':>11} {'MarlinMoE':>11} {'BF16_x8':>11} | "
          f"{'wna16/BF16':>10} {'Marlin/BF16':>11}")
print(header)
print("-" * len(header))

for total_tokens in [8, 16, 32, 64, 128, 256, 512, 1024]:
    sorted_ids, expert_ids, num_post_pad, topk_w, tpe, total_exp = \
        make_moe_routing(total_tokens, TOP_K, NUM_EXPERTS, BLOCK_SIZE_M)

    # === Shared data ===
    inp = torch.randn(total_tokens, K, dtype=torch.half, device=dev)

    # === 1. moe_wna16 ===
    out_wna16 = torch.zeros(total_exp, N, dtype=torch.half, device=dev)
    qw = torch.randint(0, 256, (NUM_EXPERTS, N, K // 2), dtype=torch.uint8, device=dev)
    sc = torch.rand(NUM_EXPERTS, N, K // GROUP_SIZE, dtype=torch.half, device=dev) * 0.1

    ms_wna16 = benchmark_fn(lambda: moe_wna16_cuda.moe_wna16_gemm(
        inp, out_wna16.zero_(), qw, sc, None, topk_w,
        sorted_ids, expert_ids, num_post_pad, TOP_K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, 4))

    # === 2. Marlin MoE ===
    # Marlin weight: (num_experts, K//tile, N*tile//pack_factor) packed
    # For u4b8 with tile=16, pack_factor=8: b_q_weight shape = (num_experts, K//16, N*16//8) = (num_experts, K//16, N*2)
    marlin_b_q = torch.randint(0, 2**31, (NUM_EXPERTS, K // MARLIN_TILE, N * MARLIN_TILE // 8),
                                dtype=torch.int32, device=dev)
    # scales: (num_experts, K // group_size, N)
    marlin_scales = torch.rand(NUM_EXPERTS, K // GROUP_SIZE, N, dtype=torch.half, device=dev) * 0.1
    marlin_workspace = torch.zeros(N // 64 * 16, dtype=torch.int32, device=dev)

    # Make routing for Marlin (uses moe_block_size = 128 typically)
    MOE_BLOCK_SIZE = 64
    sorted_ids_m, expert_ids_m, num_post_pad_m, topk_w_m, _, _ = \
        make_moe_routing(total_tokens, TOP_K, NUM_EXPERTS, MOE_BLOCK_SIZE)

    try:
        out_marlin = marlin_moe_cuda.moe_wna16_marlin_gemm(
            inp, None, marlin_b_q, None, marlin_scales,
            None, None, None, None, None,
            marlin_workspace,
            sorted_ids_m, expert_ids_m, num_post_pad_m, topk_w_m,
            MOE_BLOCK_SIZE, TOP_K, False,
            marlin_moe_cuda.U4B8_TYPE_ID,
            total_tokens, N, K,
            True, False, False, False, -1, -1, -1)
        torch.cuda.synchronize()

        ms_marlin = benchmark_fn(lambda: marlin_moe_cuda.moe_wna16_marlin_gemm(
            inp, None, marlin_b_q, None, marlin_scales,
            None, None, None, None, None,
            marlin_workspace,
            sorted_ids_m, expert_ids_m, num_post_pad_m, topk_w_m,
            MOE_BLOCK_SIZE, TOP_K, False,
            marlin_moe_cuda.U4B8_TYPE_ID,
            total_tokens, N, K,
            True, False, False, False, -1, -1, -1))
    except Exception as e:
        ms_marlin = float('nan')
        if total_tokens <= 16:
            pass  # Might fail for very small M
        else:
            print(f"  [WARN] Marlin MoE failed at tokens={total_tokens}: {e}")

    # === 3. BF16 per-expert (sequential cuBLAS) ===
    bA = [torch.randn(max(tpe, 1), K, dtype=torch.bfloat16, device=dev) for _ in range(NUM_EXPERTS)]
    bB = [torch.randn(K, N, dtype=torch.bfloat16, device=dev) for _ in range(NUM_EXPERTS)]

    def bf16_moe():
        for e in range(NUM_EXPERTS):
            torch.mm(bA[e], bB[e])
    ms_bf16 = benchmark_fn(bf16_moe)

    r1 = ms_wna16 / ms_bf16 if ms_bf16 > 0 else 0
    r2 = ms_marlin / ms_bf16 if ms_bf16 > 0 and not np.isnan(ms_marlin) else float('nan')

    ms_m_str = f"{ms_marlin:>9.4f}ms" if not np.isnan(ms_marlin) else "       N/A"
    r2_str = f"{r2:>10.2f}x" if not np.isnan(r2) else "        N/A"

    print(f"{total_tokens:<8} {tpe:<7} "
          f"{ms_wna16:>9.4f}ms {ms_m_str} {ms_bf16:>9.4f}ms | "
          f"{r1:>10.2f}x {r2_str}")

print("-" * len(header))
print()
print("Notes:")
print("  moe_wna16   = vLLM production CUDA kernel (fused, single launch)")
print("  MarlinMoE   = vLLM Marlin MoE kernel (tensor-core based, fused)")
print("  BF16_x8     = 8 sequential cuBLAS calls (NOT grouped GEMM)")
print("  BF16 production MoE on Hopper = FlashInfer CUTLASS grouped GEMM (not benchmarked, needs flashinfer)")
print("  Ratio < 1.0 = faster than BF16 sequential baseline")
