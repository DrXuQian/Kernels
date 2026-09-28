#!/usr/bin/env python3
"""Benchmark actlizeLA's installed cuLA-derived SM90 forward.

Usage: bench_actlize_gdn_prefill.py [seq_len] [q_heads] [v_heads] --bench 0 1
Uses `from actlize_la import gdn_forward` and the bundle registered by
actlizeLA's tools/install_sm90.sh. No extension path is needed for normal use.
No compilation occurs here. Inputs are prepared on CPU before measurement.
"""

import argparse
from dataclasses import asdict
from functools import partial
import importlib
import json
import os
from pathlib import Path
import statistics
import sys


CONFIGURATIONS = ("auto", "control", "value64", "value64-local-inverse", "value128-paired")


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
    parser.add_argument("--mode", choices=("device", "perfmodel"), default="device",
                        help="perfmodel uses configured metadata and exactly one untimed forward")
    parser.add_argument("--sm-count", type=int,
                        help="explicit model SM count (e.g. 20); requires --mode perfmodel")
    parser.add_argument("--backend", choices=("cuda_sm90", "ppu17"),
                        default=os.environ.get("ACTLIZE_LA_BACKEND", "cuda_sm90"))
    parser.add_argument("--configuration", choices=CONFIGURATIONS,
                        default=os.environ.get("ACTLIZE_LA_SM90_CONFIGURATION", "auto"),
                        help="auto uses actlizeLA's registered shape-dispatched bundle")
    parser.add_argument("--source-check", action="store_true",
                        default=os.environ.get("ACTLIZE_LA_SOURCE_CHECK") == "1",
                        help="PPU-fork CUDA simulation input; one call, no timing")
    parser.add_argument("--repo", default=os.environ.get("ACTLIZE_LA_ROOT"),
                        help="optional actlizeLA checkout; normally use the installed package")
    parser.add_argument("--extension", help="explicit diagnostic binary; requires a fixed --configuration")
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
    if args.backend not in ("cuda_sm90", "ppu17"):
        parser.error("ACTLIZE_LA_BACKEND must be cuda_sm90 or ppu17")
    if args.configuration not in CONFIGURATIONS:
        parser.error("invalid ACTLIZE_LA_SM90_CONFIGURATION")
    if args.configuration == "auto" and (args.backend != "cuda_sm90" or args.source_check or args.extension):
        parser.error("auto uses the installed CUDA SM90 bundle; diagnostics/PPU require a fixed --configuration")
    warmup, iters = args.bench if args.bench is not None else (0, 1)
    if warmup < 0 or iters <= 0:
        parser.error("--bench requires warmup >= 0 and iters > 0")
    if args.mode == "perfmodel":
        if args.sm_count is None or args.sm_count <= 0:
            parser.error("--mode perfmodel requires an explicit positive --sm-count")
        if (warmup, iters) != (0, 1):
            parser.error("--mode perfmodel requires exactly one call (--bench 0 1)")
    elif args.sm_count is not None:
        parser.error("--sm-count is a model input; it requires --mode perfmodel")
    if args.source_check and (args.backend != "ppu17" or (warmup, iters) != (0, 1)):
        parser.error("--source-check requires --backend ppu17 and exactly one call (--bench 0 1)")
    return args


def load_forward(args):
    if args.repo:
        repo = Path(args.repo).expanduser()
        if not (repo / "actlize_la" / "gdn_interface.py").is_file():
            raise RuntimeError(f"missing actlize_la under {repo}; check ACTLIZE_LA_ROOT")
        sys.path.insert(0, str(repo.resolve()))
    try:
        from actlize_la import gdn_forward, load_sm90
    except ImportError as exc:
        raise RuntimeError(
            f"cannot import actlize_la with {sys.executable}; run the benchmark with the Python "
            "used by tools/install_sm90.sh (set PYTHON=/path/to/that/python in the model .sh)"
        ) from exc
    profile_options = {}
    if args.mode == "perfmodel":
        try:
            from actlize_la import get_device_profile
        except ImportError as exc:
            raise RuntimeError(
                "perfmodel requires actlizeLA's updated Python frontend (6f6a2b7 or later); "
                "update the frontend in this Python environment"
            ) from exc
        profile = get_device_profile(mode=args.mode, backend=args.backend, sm_count=args.sm_count)
        print("actlizeLA device profile: " + json.dumps(asdict(profile)), flush=True)
        profile_options = dict(mode=args.mode, sm_count=args.sm_count)
    if args.configuration == "auto":
        # Preload/verify the registered bundle without a warmup kernel. The
        # public gdn_forward call then reuses this cached bundle during timing.
        automatic = load_sm90()
        return (partial(gdn_forward, algorithm="auto", backend="cuda_sm90", **profile_options),
                partial(automatic.select, **profile_options))

    # Fixed configurations and PPU are explicit diagnostics. Old GDN variables
    # must not override the normal installed actlizeLA automatic path.
    supplied = args.extension or os.environ.get("GDN_QSA_SM90_EXTENSION")
    if not supplied or not Path(supplied).expanduser().is_file():
        raise RuntimeError("fixed --configuration requires --extension pointing to its built SM90 binary")
    extension = str(Path(supplied).expanduser().resolve())
    os.environ["GDN_QSA_SM90_EXTENSION"] = extension
    interface = importlib.import_module("actlize_la.gdn_sm90_interface")
    loading = importlib.import_module("actlize_la.backends.loading")
    # Preload the DSO without a warmup kernel, so module loading is outside timing.
    module = loading._load("_gdn_fused_sm90", extension)
    target = args.backend + ("-source-check" if args.source_check else "")
    if (module.target != target or module.math_contract != interface.MATH_CONTRACT
            or getattr(module, "configuration", None) != args.configuration):
        raise RuntimeError("SM90 extension target, math contract or configuration does not match the request")
    return partial(interface.gdn_chunk_sm90, backend=args.backend,
                   source_check=args.source_check, configuration=args.configuration), None


def main():
    args = parse_args()
    warmup, iters = args.bench if args.bench is not None else (0, 1)
    contract = dict(provider="actlize_la", algorithm="fused_sm90", backend=args.backend,
                    configuration=args.configuration, source_check=args.source_check,
                    mode=args.mode, sm_count=args.sm_count,
                    batch=args.num_seqs, seq_len=args.seq_len, q_heads=args.q_heads,
                    v_heads=args.v_heads, head_dim=args.head_dim, dtype=args.dtype,
                    gate="fp32 natural-log increments", state="fp32 [B,Hv,K,V]",
                    output_final_state=True, warmup=warmup, iters=iters)
    print("bench actlize_gdn_prefill (cuLA): " + json.dumps(contract), flush=True)
    if args.dry_run:
        return 0

    import torch
    forward, select = load_forward(args)
    if args.mode == "device" and not torch.cuda.is_available():
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
    if select is not None:
        choice = select(inputs[0], inputs[2])
        print(f"actlizeLA SM90 selection: configuration={choice.configuration} "
              f"basis={choice.basis} policy={choice.policy_id}", flush=True)

    def run_once():
        return forward(*inputs, initial_state=None, output_final_state=True)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    physical = args.mode == "device" and not args.source_check
    timed = args.bench is not None and physical
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)] if timed else []
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)] if timed else []
    if physical:
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
        if physical:
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
