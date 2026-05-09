#!/usr/bin/env python3
"""Benchmark upstream FlashQLA GDN prefill for selected Qwen3.5 shapes.

This optional study script depends on upstream FlashQLA and TileLang. It is not
part of the default compile or bench_all flow.
"""

import argparse
import gc
import sys
import time
import traceback

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark upstream FlashQLA GDN prefill")
    parser.add_argument("--seqlen", type=int, default=3823)
    parser.add_argument("--h-qk", type=int, default=16)
    parser.add_argument("--h-v", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--auto-cp", choices=("true", "false", "both"), default="both")
    parser.add_argument("--skip-fi", action="store_true", help="Skip FlashInfer Python baseline")
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
    except Exception:
        traceback.print_exc()
        print(
            "Install upstream FlashQLA first:\n"
            "  git clone --depth 1 https://github.com/QwenLM/FlashQLA.git /tmp/FlashQLA\n"
            "  pip install -v /tmp/FlashQLA",
            file=sys.stderr,
        )
        return 1

    fi_fwd = None
    if not args.skip_fi:
        try:
            from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_fwd
        except Exception as exc:
            print(f"[warn] FlashInfer Python baseline unavailable: {exc}", file=sys.stderr)

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1

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

    def bench(name, fn):
        cleanup_cuda()
        print(f"\n--- {name} ---")
        start = time.time()
        fn()
        torch.cuda.synchronize()
        first_call_ms = (time.time() - start) * 1000.0
        ms = tilelang.profiler.do_bench(fn, warmup=args.warmup, rep=args.repeats)
        print(f"first_call_wall_ms={first_call_ms:.3f}")
        print(f"event_ms={ms:.6f} warmup={args.warmup} repeats={args.repeats}")
        return float(ms)

    results = []
    modes = [True, False] if args.auto_cp == "both" else [args.auto_cp == "true"]
    for auto_cp in modes:

        def call_qla(auto_cp=auto_cp):
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
                auto_cp=auto_cp,
            )

        results.append((f"flash_qla_auto_cp_{str(auto_cp).lower()}", bench(f"FlashQLA auto_cp={auto_cp}", call_qla)))

    if fi_fwd is not None:

        def call_fi():
            return fi_fwd(
                q=q.view(-1, h_qk, head_dim),
                k=k.view(-1, h_qk, head_dim),
                v=v.view(-1, h_v, head_dim),
                g=g.view(-1, h_v),
                beta=beta.view(-1, h_v),
                scale=scale,
                initial_state=h0,
                cu_seqlens=cu_seqlens,
                output_final_state=True,
            )

        results.append(("flashinfer_python", bench("FlashInfer Python", call_fi)))

    print("\nsummary_csv")
    print("case,seqlen,h_qk,h_v,event_ms")
    for name, ms in results:
        print(f"{name},{total_seqlen},{h_qk},{h_v},{ms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
