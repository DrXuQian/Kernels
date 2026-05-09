#!/usr/bin/env python3
"""Force upstream FlashQLA CP for the target GDN shape.

FlashQLA's default heuristic does not enable intra-card CP for the repo's
Hqk=16/Hv=64 shape. This optional study monkey-patches the heuristic so the
same upstream FlashQLA kernels run with CP enabled.
"""

import argparse
import gc
import math
import sys
import time
import traceback

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Force upstream FlashQLA CP")
    parser.add_argument("--seqlen", type=int, default=3823)
    parser.add_argument("--h-qk", type=int, default=16)
    parser.add_argument("--h-v", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def cleanup_cuda() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    try:
        import tilelang
        from flash_qla import chunk_gated_delta_rule_fwd as qla_fwd
        from flash_qla.utils import l2norm
        import flash_qla.ops.gated_delta_rule.chunk.cp_context as cp
    except Exception:
        traceback.print_exc()
        print(
            "Install upstream FlashQLA first:\n"
            "  git clone --depth 1 https://github.com/QwenLM/FlashQLA.git /tmp/FlashQLA\n"
            "  pip install -v /tmp/FlashQLA",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1

    cp_plan = {}

    def forced_calc_cp_seqs(raw_cu_seqlens, chunk_size, num_v_heads):
        device = raw_cu_seqlens.device
        seqlen_dtype = raw_cu_seqlens.dtype
        raw = raw_cu_seqlens.tolist()
        seqlens = [raw[i + 1] - raw[i] for i in range(len(raw) - 1)]
        num_chunks = [(x + chunk_size - 1) // chunk_size for x in seqlens]

        max_local_chunks = 2 ** round(
            math.log2(math.sqrt(num_v_heads * sum(num_chunks) / cp.MULTI_PROCESSOR_COUNT) * 3)
        )
        max_local_chunks = max(max_local_chunks, 4)
        max_local_tokens = max_local_chunks * chunk_size

        cp_cu_seqlens = []
        ht_mask = []
        seq_map_c2r = []
        seq_map_r2c = [0]
        for i, chunks in enumerate(num_chunks):
            start = raw[i]
            end = raw[i + 1]
            if chunks > max_local_chunks:
                while start < end:
                    cp_cu_seqlens.append(start)
                    ht_mask.append(False)
                    seq_map_c2r.append(i)
                    start += max_local_tokens
                ht_mask[-1] = True
            else:
                cp_cu_seqlens.append(start)
                ht_mask.append(True)
                seq_map_c2r.append(i)
            seq_map_r2c.append(len(cp_cu_seqlens))
        cp_cu_seqlens.append(raw[-1])

        cp_plan["max_local_chunks"] = max_local_chunks
        cp_plan["cp_cu_seqlens"] = list(cp_cu_seqlens)

        return (
            True,
            torch.tensor(cp_cu_seqlens, dtype=seqlen_dtype, device=device, requires_grad=False),
            torch.tensor(seq_map_r2c, dtype=seqlen_dtype, device=device, requires_grad=False),
            torch.tensor(seq_map_c2r, dtype=seqlen_dtype, device=device),
            torch.tensor(ht_mask, dtype=torch.bool, device=device, requires_grad=False),
        )

    cp._calc_cp_seqs = forced_calc_cp_seqs

    batch = 1
    total_seqlen = args.seqlen
    h_qk = args.h_qk
    h_v = args.h_v
    head_dim = args.head_dim
    scale = head_dim ** -0.5

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"tilelang={getattr(tilelang, '__version__', 'unknown')}")
    print(f"gpu={torch.cuda.get_device_name()} sm={torch.cuda.get_device_capability()}")
    print(f"shape: B={batch} T={total_seqlen} H_QK={h_qk} H_V={h_v} D={head_dim} dtype=bf16")

    cu_seqlens = torch.tensor([0, total_seqlen], dtype=torch.int32, device="cuda")
    q = l2norm(torch.randn(batch, total_seqlen, h_qk, head_dim, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(batch, total_seqlen, h_qk, head_dim, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(batch, total_seqlen, h_v, head_dim, device="cuda", dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(batch, total_seqlen, h_v, device="cuda", dtype=torch.float32)) / 16
    beta = torch.randn(batch, total_seqlen, h_v, device="cuda", dtype=torch.float32).sigmoid()
    h0 = torch.randn(batch, h_v, head_dim, head_dim, device="cuda", dtype=torch.float32)

    def call_qla():
        return qla_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            output_h=False,
            cu_seqlens=cu_seqlens,
            auto_cp=True,
        )

    cleanup_cuda()
    start = time.time()
    call_qla()
    torch.cuda.synchronize()
    first_call_ms = (time.time() - start) * 1000.0
    ms = tilelang.profiler.do_bench(call_qla, warmup=args.warmup, rep=args.repeats)

    print(f"forced_cp_max_local_chunks={cp_plan.get('max_local_chunks')}")
    print(f"forced_cp_cu_seqlens={cp_plan.get('cp_cu_seqlens')}")
    print(f"first_call_wall_ms={first_call_ms:.3f}")
    print(f"event_ms={float(ms):.6f} warmup={args.warmup} repeats={args.repeats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
