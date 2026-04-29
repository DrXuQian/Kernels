#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Bench vLLM's packed Triton GDN decode kernel.

This matches vLLM's non-spec decode fast path after causal_conv1d_update:
  fused_recurrent_gated_delta_rule_packed_decode

Default Qwen3.5-122B linear-attention decode shape:
  mixed_qkv [1, 12288], a/b [1, 64], state [2, 64, 128, 128]

Usage:
  python3 linear_attention/src/bench_vllm_triton_gdn_decode.py
  python3 linear_attention/src/bench_vllm_triton_gdn_decode.py 1 16 64 128
  python3 linear_attention/src/bench_vllm_triton_gdn_decode.py --bench 100 1000
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from vllm_triton_gdn.ops import fused_recurrent_gated_delta_rule_packed_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs="?", type=int, default=1)
    parser.add_argument("q_heads", nargs="?", type=int, default=16)
    parser.add_argument("v_heads", nargs="?", type=int, default=64)
    parser.add_argument("head_dim", nargs="?", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--bench", nargs=2, metavar=("WARMUP", "ITERS"), type=int)
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

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    qkv_dim = 2 * args.q_heads * args.head_dim + args.v_heads * args.head_dim
    scale = 1.0 / math.sqrt(args.head_dim)

    print(
        "bench vllm_triton_gdn_decode: "
        f"B={args.batch} q_heads={args.q_heads} v_heads={args.v_heads} "
        f"dim={args.head_dim} dtype={args.dtype}"
    )

    torch.manual_seed(1234)
    mixed_cpu = torch.randn(args.batch, qkv_dim, dtype=dtype) * 0.1
    a_cpu = torch.empty(args.batch, args.v_heads, dtype=dtype).uniform_(-0.25, 0.25)
    b_cpu = torch.empty(args.batch, args.v_heads, dtype=dtype).uniform_(-0.25, 0.25)
    A_log_cpu = torch.empty(args.v_heads, dtype=torch.float32).uniform_(-2.0, 0.5)
    dt_bias_cpu = torch.empty(args.v_heads, dtype=torch.float32).uniform_(-0.25, 0.25)
    state_cpu = torch.randn(
        args.batch + 1,
        args.v_heads,
        args.head_dim,
        args.head_dim,
        dtype=torch.float32,
    ) * 0.01
    indices_cpu = torch.arange(1, args.batch + 1, dtype=torch.int64)
    out_cpu = torch.empty(args.batch, 1, args.v_heads, args.head_dim, dtype=dtype)

    mixed_qkv = mixed_cpu.cuda()
    a = a_cpu.cuda()
    b = b_cpu.cuda()
    A_log = A_log_cpu.cuda()
    dt_bias = dt_bias_cpu.cuda()
    state = state_cpu.cuda()
    state_indices = indices_cpu.cuda()
    out = out_cpu.cuda()

    torch.cuda.synchronize()

    def run_once():
        return fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=state,
            out=out,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    if args.bench is None:
        profiler_start()
        run_once()
        torch.cuda.synchronize()
        profiler_stop()
        print(f"output shape: {tuple(out.shape)}")
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
        f"  Kernel time: median={np.median(times):.1f} us, "
        f"min={np.min(times):.1f} us, avg={np.mean(times):.1f} us "
        f"(warmup={warmup}, iters={iters})"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
