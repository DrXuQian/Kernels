#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Bench Gated Delta Net gate preparation:

  g    = -exp(A_log) * softplus(a + dt_bias)
  beta = sigmoid(b)

This matches the TensorRT-LLM/vLLM Triton path.  No CUDA implementation was
found in TensorRT-LLM or in the extracted standalone CUDA kernels; CUDA GDN
benches consume already prepared g/beta.

Default Qwen3.5-122B shape:
  a,b:     [3823, 64]
  A_log:   [64]
  dt_bias: [64]

Usage:
  python3 linear_attn/src/bench_gdn_gate_prep.py
  python3 linear_attn/src/bench_gdn_gate_prep.py --tokens 1
  python3 linear_attn/src/bench_gdn_gate_prep.py --bench 20 1000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_gate_prep_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    total_elems: tl.constexpr,
    heads: tl.constexpr,
    softplus_beta: tl.constexpr,
    softplus_threshold: tl.constexpr,
    output_g_exp: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * block_size + tl.arange(0, block_size)
    mask = offs < total_elems
    h = offs % heads

    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + h, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h, mask=mask, other=0.0).to(tl.float32)

    x = a + dt_bias
    beta_x = softplus_beta * x
    sp = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    g = -tl.exp(A_log) * sp
    if output_g_exp:
        g = tl.exp(g)
    beta = 1.0 / (1.0 + tl.exp(-b))

    tl.store(g_ptr + offs, g, mask=mask)
    tl.store(beta_ptr + offs, beta, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=3823)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--bench", nargs=2, metavar=("WARMUP", "ITERS"), type=int)
    parser.add_argument("--output-g-exp", action="store_true")
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.tokens <= 0 or args.heads <= 0:
        raise SystemExit("--tokens and --heads must be positive")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    total_elems = args.tokens * args.heads

    print(
        "bench gdn_gate_prep (Triton): "
        f"tokens={args.tokens} heads={args.heads} dtype={args.dtype} "
        f"output_g_exp={args.output_g_exp}"
    )

    torch.manual_seed(1234)
    a_cpu = torch.empty(args.tokens, args.heads, dtype=dtype).uniform_(-0.25, 0.25)
    b_cpu = torch.empty(args.tokens, args.heads, dtype=dtype).uniform_(-0.25, 0.25)
    A_log_cpu = torch.empty(args.heads, dtype=torch.float32).uniform_(-2.0, 0.5)
    dt_bias_cpu = torch.empty(args.heads, dtype=torch.float32).uniform_(-0.25, 0.25)

    a = a_cpu.cuda()
    b = b_cpu.cuda()
    A_log = A_log_cpu.cuda()
    dt_bias = dt_bias_cpu.cuda()
    g = torch.empty(args.tokens, args.heads, dtype=torch.float32, device="cuda")
    beta = torch.empty_like(g)

    grid = (triton.cdiv(total_elems, args.block_size),)

    def run_once():
        _gdn_gate_prep_kernel[grid](
            a,
            b,
            A_log,
            dt_bias,
            g,
            beta,
            total_elems,
            args.heads,
            1.0,
            20.0,
            args.output_g_exp,
            args.block_size,
        )
        return g, beta

    torch.cuda.synchronize()
    if args.bench is None:
        profiler_start()
        out_g, out_beta = run_once()
        torch.cuda.synchronize()
        profiler_stop()
        print(f"g shape: {tuple(out_g.shape)}, beta shape: {tuple(out_beta.shape)}")
        print("Done.")
        return 0

    warmup, iters = args.bench
    if warmup < 0 or iters <= 0:
        raise SystemExit("--bench requires warmup >= 0 and iters > 0")

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    profiler_start()
    for i in range(iters):
        starts[i].record()
        run_once()
        ends[i].record()
    torch.cuda.synchronize()
    profiler_stop()

    times = np.array([s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)])
    print(
        f"  Kernel time: median={np.median(times):.3f} us, "
        f"min={np.min(times):.3f} us, avg={np.mean(times):.3f} us "
        f"(warmup={warmup}, iters={iters})"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
