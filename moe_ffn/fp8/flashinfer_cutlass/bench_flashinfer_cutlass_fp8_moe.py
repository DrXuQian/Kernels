#!/usr/bin/env python3
"""MiniMax-style FP8 block-scale MoE benchmark using FlashInfer CUTLASS SM90.

This covers the CUDA/CUTLASS backend selected by vLLM for static 128x128
block-wise FP8 MoE on Hopper.  It is intentionally separate from the default
compile path because it depends on the installed FlashInfer Python package.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import sys
from pathlib import Path

import torch


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", "--m", dest="tokens", type=positive_int, default=3823)
    parser.add_argument("--hidden", type=positive_int, default=3072)
    parser.add_argument("--intermediate", type=positive_int, default=1536)
    parser.add_argument("--experts", type=positive_int, default=8)
    parser.add_argument("--topk", type=positive_int, default=8)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=positive_int, default=1)
    parser.add_argument(
        "--bench",
        nargs=2,
        type=int,
        metavar=("WARMUP", "ITERS"),
        help="Compatibility alias for --warmup WARMUP --iters ITERS.",
    )
    parser.add_argument(
        "--tactic-cache",
        type=Path,
        help="FlashInfer autotuner JSON cache. Required unless --tune or --allow-fallback is set.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Profile tactics and save them into --tactic-cache before timed execution.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow FlashInfer fallback tactic when no tactic cache is provided.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip finite output check after the timed loop.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def make_fp8_weight(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    cpu = torch.randn(shape, dtype=torch.float32).clamp_(-1.0, 1.0)
    return cpu.to(torch.float8_e4m3fn).to(device=device).contiguous()


def make_inputs(args: argparse.Namespace, device: torch.device):
    dtype = dtype_from_name(args.dtype)
    if args.hidden % 128 != 0:
        raise ValueError("--hidden must be divisible by 128 for DeepSeek FP8 block scaling")
    if args.intermediate % 128 != 0:
        raise ValueError("--intermediate must be divisible by 128 for DeepSeek FP8 block scaling")
    if args.topk > args.experts:
        raise ValueError("--topk must be <= --experts for this synthetic balanced benchmark")

    # Create tensors on CPU first, then transfer before any timed region.
    x = torch.randn((args.tokens, args.hidden), dtype=torch.float32).to(dtype).to(device).contiguous()
    w1 = make_fp8_weight((args.experts, 2 * args.intermediate, args.hidden), device)
    w2 = make_fp8_weight((args.experts, args.hidden, args.intermediate), device)

    scale1 = torch.ones(
        (args.experts, (2 * args.intermediate) // 128, args.hidden // 128),
        dtype=torch.float32,
    ).to(device)
    scale2 = torch.ones(
        (args.experts, args.hidden // 128, args.intermediate // 128),
        dtype=torch.float32,
    ).to(device)

    ids = torch.empty((args.tokens, args.topk), dtype=torch.int32)
    pattern = torch.arange(args.topk, dtype=torch.int32).view(1, args.topk)
    ids.copy_(pattern.expand(args.tokens, args.topk))
    ids = ids.to(device).contiguous()
    weights = torch.full((args.tokens, args.topk), 1.0 / args.topk, dtype=torch.float32).to(device)
    out = torch.empty((args.tokens, args.hidden), dtype=dtype, device=device)
    return x, ids, weights, w1, w2, scale1, scale2, out


def run_once(cutlass_fused_moe, tensors, dtype: torch.dtype, tune_max_num_tokens: int):
    x, ids, weights, w1, w2, scale1, scale2, out = tensors
    result = cutlass_fused_moe(
        input=x,
        token_selected_experts=ids,
        token_final_scales=weights,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output_dtype=dtype,
        output=out,
        quant_scales=[scale1, scale2],
        use_deepseek_fp8_block_scale=True,
        activation_type=3,  # Swiglu
        tune_max_num_tokens=tune_max_num_tokens,
    )
    if isinstance(result, list):
        return result[0]
    return result


def cache_context(args: argparse.Namespace):
    from flashinfer.autotuner import autotune

    if args.tactic_cache is None:
        if args.allow_fallback:
            return contextlib.nullcontext()
        raise SystemExit("--tactic-cache is required unless --tune or --allow-fallback is set")

    if args.tune:
        args.tactic_cache.parent.mkdir(parents=True, exist_ok=True)
        return autotune(
            True,
            cache=str(args.tactic_cache),
            tuning_buckets=(args.tokens,),
            round_up=False,
        )

    if not args.tactic_cache.is_file():
        raise SystemExit(f"missing tactic cache: {args.tactic_cache}; run once with --tune")
    return autotune(
        False,
        cache=str(args.tactic_cache),
        tuning_buckets=(args.tokens,),
        round_up=False,
    )


def main() -> int:
    args = parse_args()
    if args.bench is not None:
        args.warmup, args.iters = args.bench
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise SystemExit(f"FlashInfer DeepSeek FP8 block-scale CUTLASS path requires SM90, got sm_{major}{minor}")

    from flashinfer.fused_moe import cutlass_fused_moe

    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    tensors = make_inputs(args, device)
    torch.cuda.synchronize()

    print(
        "flashinfer_cutlass_fp8_moe:"
        f" tokens={args.tokens} experts={args.experts} topk={args.topk}"
        f" hidden={args.hidden} intermediate={args.intermediate} dtype={args.dtype}"
        f" cache={args.tactic_cache if args.tactic_cache else 'fallback'} tune={int(args.tune)}"
    )

    with cache_context(args):
        # In tune mode this profiles tactics and writes the cache before timing.
        # In cache-hit mode do not issue an extra warmup unless requested; nsys
        # captures process-wide CUDA kernels and should see only the selected
        # warmup/timed calls.
        if args.tune:
            run_once(cutlass_fused_moe, tensors, dtype, args.tokens)
            torch.cuda.synchronize()

        for _ in range(args.warmup):
            run_once(cutlass_fused_moe, tensors, dtype, args.tokens)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            out = run_once(cutlass_fused_moe, tensors, dtype, args.tokens)
            stop.record()
            stop.synchronize()
            times_ms.append(start.elapsed_time(stop))

    if not args.skip_check:
        if not torch.isfinite(out).all().item():
            raise SystemExit("output contains non-finite values")

    median_ms = statistics.median(times_ms)
    avg_ms = statistics.fmean(times_ms)
    print(
        "  Kernel time:"
        f" median={median_ms:.6f} ms, avg={avg_ms:.6f} ms,"
        f" min={min(times_ms):.6f} ms, max={max(times_ms):.6f} ms"
        f"  (warmup={args.warmup}, iters={args.iters})"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("FLASHINFER_LOGGING_LEVEL", "WARNING")
    sys.exit(main())
