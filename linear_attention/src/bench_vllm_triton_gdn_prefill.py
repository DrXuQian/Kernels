#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Bench vLLM's Triton/FLA GDN prefill path.

Default Qwen3.5-122B linear-attention shape:
  mixed_qkv [3823, 12288], a/b [3823, 64]
  q/k heads=16, value heads=64, head_dim=128

The default path matches vLLM prefill after causal_conv1d:
  fused_post_conv_prep -> chunk_gated_delta_rule

Usage:
  python3 linear_attention/src/bench_vllm_triton_gdn_prefill.py
  python3 linear_attention/src/bench_vllm_triton_gdn_prefill.py 3823 16 64 128 1
  python3 linear_attention/src/bench_vllm_triton_gdn_prefill.py --bench 10 100
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from vllm_triton_gdn.ops import chunk_gated_delta_rule, fused_post_conv_prep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seq_len", nargs="?", type=int, default=3823)
    parser.add_argument("q_heads", nargs="?", type=int, default=16)
    parser.add_argument("v_heads", nargs="?", type=int, default=64)
    parser.add_argument("head_dim", nargs="?", type=int, default=128)
    parser.add_argument("num_seqs", nargs="?", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--bench", nargs=2, metavar=("WARMUP", "ITERS"), type=int)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Skip fused_post_conv_prep and benchmark only chunk_gated_delta_rule.",
    )
    parser.add_argument(
        "--no-final-state",
        action="store_true",
        help="Do not write final recurrent state.",
    )
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
    total_tokens = args.seq_len * args.num_seqs
    qkv_dim = 2 * args.q_heads * args.head_dim + args.v_heads * args.head_dim
    scale = 1.0 / math.sqrt(args.head_dim)
    output_final_state = not args.no_final_state

    print(
        "bench vllm_triton_gdn_prefill: "
        f"seq={args.seq_len} seqs={args.num_seqs} "
        f"q_heads={args.q_heads} v_heads={args.v_heads} dim={args.head_dim} "
        f"dtype={args.dtype} core_only={args.core_only}"
    )

    torch.manual_seed(1234)
    mixed_cpu = torch.randn(total_tokens, qkv_dim, dtype=dtype) * 0.1
    a_cpu = torch.empty(total_tokens, args.v_heads, dtype=dtype).uniform_(-0.25, 0.25)
    b_cpu = torch.empty(total_tokens, args.v_heads, dtype=dtype).uniform_(-0.25, 0.25)
    A_log_cpu = torch.empty(args.v_heads, dtype=torch.float32).uniform_(-2.0, 0.5)
    dt_bias_cpu = torch.empty(args.v_heads, dtype=torch.float32).uniform_(-0.25, 0.25)
    state_cpu = torch.zeros(
        args.num_seqs,
        args.v_heads,
        args.head_dim,
        args.head_dim,
        dtype=torch.float32,
    )
    cu_cpu = torch.arange(0, total_tokens + 1, args.seq_len, dtype=torch.int32)

    mixed_qkv = mixed_cpu.cuda()
    a = a_cpu.cuda()
    b = b_cpu.cuda()
    A_log = A_log_cpu.cuda()
    dt_bias = dt_bias_cpu.cuda()
    initial_state = state_cpu.cuda()
    cu_seqlens = cu_cpu.cuda()

    if args.core_only:
        q_cpu = torch.randn(total_tokens, args.q_heads, args.head_dim, dtype=dtype) * 0.1
        k_cpu = torch.randn(total_tokens, args.q_heads, args.head_dim, dtype=dtype) * 0.1
        v_cpu = torch.randn(total_tokens, args.v_heads, args.head_dim, dtype=dtype) * 0.1
        g_cpu = torch.empty(total_tokens, args.v_heads, dtype=torch.float32).uniform_(
            -0.8, -0.2
        )
        beta_cpu = torch.empty(total_tokens, args.v_heads, dtype=torch.float32).uniform_(
            0.25, 0.75
        )
        q = q_cpu.cuda().unsqueeze(0)
        k = k_cpu.cuda().unsqueeze(0)
        v = v_cpu.cuda().unsqueeze(0)
        g = g_cpu.cuda().unsqueeze(0)
        beta = beta_cpu.cuda().unsqueeze(0)
        torch.cuda.synchronize()

    def run_once():
        if args.core_only:
            q_in, k_in, v_in, g_in, beta_in = q, k, v, g, beta
        else:
            q0, k0, v0, g0, beta0 = fused_post_conv_prep(
                conv_output=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                num_k_heads=args.q_heads,
                head_k_dim=args.head_dim,
                head_v_dim=args.head_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            q_in = q0.unsqueeze(0)
            k_in = k0.unsqueeze(0)
            v_in = v0.unsqueeze(0)
            g_in = g0.unsqueeze(0)
            beta_in = beta0.unsqueeze(0)

        return chunk_gated_delta_rule(
            q=q_in,
            k=k_in,
            v=v_in,
            g=g_in,
            beta=beta_in,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )

    torch.cuda.synchronize()
    if args.bench is None:
        profiler_start()
        result = run_once()
        torch.cuda.synchronize()
        profiler_stop()
        out = result[0] if isinstance(result, tuple) else result
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
