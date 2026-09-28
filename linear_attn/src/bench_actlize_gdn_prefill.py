#!/usr/bin/env python3
"""Benchmark GDN-QSA-sm80's cuLA-derived fused_sm90 forward.

Usage: bench_actlize_gdn_prefill.py [seq_len] [q_heads] [v_heads] --bench 0 1
Set GDN_QSA_ROOT to the source checkout (or install its Python package), and
GDN_QSA_SM90_EXTENSION to the already-built extension. No compilation occurs
in this runner. Inputs are prepared on CPU before the measured forward call.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import sys


CONFIGURATIONS = ("control", "value64", "value64-local-inverse", "value128-paired")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seq_len", nargs="?", type=int, default=2048)
    parser.add_argument("q_heads", nargs="?", type=int, default=16)
    parser.add_argument("v_heads", nargs="?", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16",
                        help="fused_sm90 accepts BF16 Q/K/V/beta only")
    parser.add_argument("--bench", type=int, nargs=2, metavar=("WARMUP", "ITERS"))
    parser.add_argument("--backend", choices=("cuda_sm90", "ppu17"),
                        default=os.environ.get("GDN_QSA_SM90_BACKEND", "cuda_sm90"))
    parser.add_argument("--configuration", choices=CONFIGURATIONS,
                        default=os.environ.get("GDN_QSA_SM90_CONFIGURATION", "control"))
    parser.add_argument("--source-check", action="store_true",
                        default=os.environ.get("GDN_QSA_SM90_SOURCE_CHECK") == "1",
                        help="PPU-fork CUDA simulation input; one call, no timing")
    parser.add_argument("--repo", default=os.environ.get("GDN_QSA_ROOT"),
                        help="GDN-QSA-sm80 checkout containing gdn_qsa_sm80/")
    parser.add_argument("--extension", default=os.environ.get("GDN_QSA_SM90_EXTENSION"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the call contract without importing Torch or launching")
    args = parser.parse_args()
    if min(args.seq_len, args.q_heads, args.v_heads, args.num_seqs) <= 0:
        parser.error("sequence length, batch and head counts must be positive")
    if args.head_dim != 128 or args.v_heads % args.q_heads:
        parser.error("fused_sm90 requires head_dim=128 and v_heads divisible by q_heads")
    if max(args.num_seqs * args.seq_len, args.num_seqs * args.v_heads,
           args.v_heads * args.head_dim) > 2**31 - 1:
        parser.error("shape exceeds the fused_sm90 int32 extent/grid limits")
    if args.backend not in ("cuda_sm90", "ppu17") or args.configuration not in CONFIGURATIONS:
        parser.error("invalid GDN_QSA_SM90_BACKEND or GDN_QSA_SM90_CONFIGURATION")
    warmup, iters = args.bench if args.bench is not None else (0, 1)
    if warmup < 0 or iters <= 0:
        parser.error("--bench requires warmup >= 0 and iters > 0")
    if args.source_check and (args.backend != "ppu17" or (warmup, iters) != (0, 1)):
        parser.error("--source-check requires --backend ppu17 and exactly one call (--bench 0 1)")
    return args


def load_forward(args):
    # A sibling checkout is convenient, but never hardcode a developer's path.
    repo = Path(args.repo).expanduser() if args.repo else Path(__file__).resolve().parents[3] / "GDN-QSA-sm80"
    if args.repo or repo.is_dir():
        if not (repo / "gdn_qsa_sm80" / "gdn_sm90_interface.py").is_file():
            raise RuntimeError(f"missing cuLA fused_sm90 interface under {repo}; set GDN_QSA_ROOT")
        sys.path.insert(0, str(repo.resolve()))
    if not args.extension or not Path(args.extension).expanduser().is_file():
        raise RuntimeError("set GDN_QSA_SM90_EXTENSION (or --extension) to the built SM90 extension")
    extension = str(Path(args.extension).expanduser().resolve())
    os.environ["GDN_QSA_SM90_EXTENSION"] = extension
    try:
        interface = importlib.import_module("gdn_qsa_sm80.gdn_sm90_interface")
        loading = importlib.import_module("gdn_qsa_sm80.backends.loading")
    except ImportError as exc:
        raise RuntimeError("cannot import GDN-QSA-sm80; set GDN_QSA_ROOT or install its package") from exc
    # Preload the DSO without a warmup kernel, so module loading is outside timing.
    module = loading._load("_gdn_fused_sm90", extension)
    target = args.backend + ("-source-check" if args.source_check else "")
    if (module.target != target or module.math_contract != interface.MATH_CONTRACT
            or getattr(module, "configuration", None) != args.configuration):
        raise RuntimeError("SM90 extension target, math contract or configuration does not match the request")
    return interface.gdn_chunk_sm90


def main():
    args = parse_args()
    warmup, iters = args.bench if args.bench is not None else (0, 1)
    contract = dict(algorithm="fused_sm90", backend=args.backend,
                    configuration=args.configuration, source_check=args.source_check,
                    batch=args.num_seqs, seq_len=args.seq_len, q_heads=args.q_heads,
                    v_heads=args.v_heads, head_dim=args.head_dim, dtype=args.dtype,
                    gate="fp32 natural-log increments", state="fp32 [B,Hv,K,V]",
                    output_final_state=True, warmup=warmup, iters=iters)
    print("bench actlize_gdn_prefill (cuLA): " + json.dumps(contract), flush=True)
    if args.dry_run:
        return 0

    import torch
    forward = load_forward(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible device is not available")
    torch.set_num_threads(1)
    rng = torch.Generator(device="cpu").manual_seed(1234)
    q_shape = (args.num_seqs, args.seq_len, args.q_heads, args.head_dim)
    v_shape = (args.num_seqs, args.seq_len, args.v_heads, args.head_dim)
    gate_shape = v_shape[:-1]
    q_cpu = torch.nn.functional.normalize(torch.randn(q_shape, generator=rng, device="cpu"), dim=-1).bfloat16()
    k_cpu = torch.nn.functional.normalize(torch.randn(q_shape, generator=rng, device="cpu"), dim=-1).bfloat16()
    v_cpu = (torch.randn(v_shape, generator=rng, device="cpu") * 0.1).bfloat16()
    # g is log(decay), not the positive decay consumed by the old FlashInfer bench.
    g_cpu = torch.empty(gate_shape, dtype=torch.float32, device="cpu").uniform_(-0.8, -0.2, generator=rng)
    beta_cpu = torch.empty(gate_shape, dtype=torch.float32, device="cpu").uniform_(0.25, 0.75, generator=rng).bfloat16()
    inputs = tuple(t.cuda() for t in (q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu))
    torch.cuda.synchronize()

    def run_once():
        return forward(*inputs, initial_state=None, output_final_state=True,
                       backend=args.backend, source_check=args.source_check,
                       configuration=args.configuration)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    timed = args.bench is not None and not args.source_check
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)] if timed else []
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)] if timed else []
    torch.cuda.profiler.start()
    try:
        for i in range(iters):
            if timed:
                starts[i].record()
            output, state = run_once()
            if timed:
                ends[i].record()
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    if timed:
        times = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
        print(f"  Kernel time: median={statistics.median(times):.1f} us, "
              f"min={min(times):.1f} us, avg={statistics.mean(times):.1f} us "
              f"(warmup={warmup}, iters={iters})")
    print(f"output shape: {tuple(output.shape)}, final state shape: {tuple(state.shape)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        sys.exit(f"actlize_gdn_prefill: {exc}")
