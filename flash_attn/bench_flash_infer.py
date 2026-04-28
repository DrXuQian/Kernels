#!/usr/bin/env python3
"""
Bench: FlashInfer single-request FlashAttention decode + prefill (GQA).

Default Qwen3.5-122B attention shape:
  decode:  Q [32, 256],         K/V [ctx_len, 2, 256]
  prefill: Q [seq_len, 32, 256], K/V [seq_len, 2, 256]

Usage:
  python3 flash_attn/bench_flash_infer.py decode [ctx_len] [num_heads] [num_kv_heads] [head_dim]
  python3 flash_attn/bench_flash_infer.py prefill [seq_len] [num_heads] [num_kv_heads] [head_dim]
  python3 flash_attn/bench_flash_infer.py decode 3823 --bench 20 100

Default single-run mode is intended for nsys/ncu captures. Add
`--bench W I` for CUDA-event timing. With nsys, use
`--capture-range=cudaProfilerApi --capture-range-end=stop` to capture only
the FlashInfer call.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from typing import Any, Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashInfer FlashAttention benchmark")
    parser.add_argument("mode", nargs="?", choices=("decode", "prefill"), default="decode")
    parser.add_argument("seq_len", nargs="?", type=int, default=3823)
    parser.add_argument("num_heads", nargs="?", type=int, default=32)
    parser.add_argument("num_kv_heads", nargs="?", type=int, default=2)
    parser.add_argument("head_dim", nargs="?", type=int, default=256)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--backend", default=None, help="FlashInfer backend for prefill APIs, for example: fa2/fa3/auto")
    parser.add_argument("--kv-layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument("--logits-soft-cap", type=float, default=None)
    parser.add_argument("--window-left", type=int, default=None)
    parser.add_argument("--use-tensor-cores", action="store_true", help="Pass use_tensor_cores=True to decode API")
    parser.add_argument("--no-causal", action="store_true", help="Use non-causal prefill attention")
    parser.add_argument("--bench", nargs="*", type=int, metavar=("WARMUP", "ITERS"))
    args = parser.parse_args()

    if args.bench is None:
        args.warmup = 0
        args.iters = 1
        args.bench_mode = False
    else:
        args.warmup = args.bench[0] if len(args.bench) >= 1 else 20
        args.iters = args.bench[1] if len(args.bench) >= 2 else 100
        args.bench_mode = True
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--bench requires warmup >= 0 and iters > 0")
    return args


def load_flashinfer() -> Any:
    try:
        import flashinfer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "flashinfer is not installed. Install FlashInfer first, for example: "
            "pip install flashinfer-python flashinfer-cubin"
        ) from exc
    return flashinfer


def resolve_flashinfer_func(flashinfer: Any, submodule_name: str, func_name: str) -> Callable[..., torch.Tensor]:
    if hasattr(flashinfer, func_name):
        return getattr(flashinfer, func_name)
    submodule = getattr(flashinfer, submodule_name, None)
    if submodule is not None and hasattr(submodule, func_name):
        return getattr(submodule, func_name)
    raise SystemExit(f"FlashInfer API not found: {submodule_name}.{func_name}")


def call_with_supported_kwargs(fn: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        sig = inspect.signature(fn)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        pass
    return fn(*args, **kwargs)


def profiler_start() -> None:
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        pass


def profiler_stop() -> None:
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        pass


def make_tensors(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)

    if args.mode == "decode":
        q = torch.randn(args.num_heads, args.head_dim, dtype=dtype, device=device, generator=gen)
    else:
        q = torch.randn(args.seq_len, args.num_heads, args.head_dim, dtype=dtype, device=device, generator=gen)

    if args.kv_layout == "NHD":
        kv_shape = (args.seq_len, args.num_kv_heads, args.head_dim)
    else:
        kv_shape = (args.num_kv_heads, args.seq_len, args.head_dim)
    k = torch.randn(*kv_shape, dtype=dtype, device=device, generator=gen)
    v = torch.randn(*kv_shape, dtype=dtype, device=device, generator=gen)
    torch.cuda.synchronize()
    return q, k, v


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment")

    flashinfer = load_flashinfer()
    q, k, v = make_tensors(args)

    print(
        f"bench flashinfer {args.mode}: seq={args.seq_len} heads={args.num_heads} "
        f"kv_heads={args.num_kv_heads} dim={args.head_dim} dtype={args.dtype} "
        f"kv_layout={args.kv_layout}"
    )

    if args.mode == "decode":
        fn = resolve_flashinfer_func(flashinfer, "decode", "single_decode_with_kv_cache")

        def kernel() -> torch.Tensor:
            return call_with_supported_kwargs(
                fn,
                q,
                k,
                v,
                kv_layout=args.kv_layout,
                use_tensor_cores=True if args.use_tensor_cores else None,
                sm_scale=args.sm_scale,
                window_left=args.window_left,
                logits_soft_cap=args.logits_soft_cap,
            )

    else:
        fn = resolve_flashinfer_func(flashinfer, "prefill", "single_prefill_with_kv_cache")

        def kernel() -> torch.Tensor:
            return call_with_supported_kwargs(
                fn,
                q,
                k,
                v,
                causal=not args.no_causal,
                kv_layout=args.kv_layout,
                sm_scale=args.sm_scale,
                window_left=args.window_left,
                logits_soft_cap=args.logits_soft_cap,
                backend=args.backend,
            )

    # Single dry call in timing mode is intentionally avoided; --bench warmup
    # controls all untimed launches.
    if not args.bench_mode:
        profiler_start()
        out = kernel()
        torch.cuda.synchronize()
        profiler_stop()
        print(f"output shape: {tuple(out.shape)}")
        print("Done.")
        return 0

    for _ in range(args.warmup):
        kernel()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    profiler_start()
    for _ in range(args.iters):
        start.record()
        kernel()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    profiler_stop()

    t = torch.tensor(times_us, device="cpu")
    print(
        f"  Kernel time: median={t.median().item():.1f} us, "
        f"avg={t.mean().item():.1f} us, min={t.min().item():.1f} us, "
        f"max={t.max().item():.1f} us (warmup={args.warmup}, iters={args.iters})"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
