#!/usr/bin/env python3
"""
Bench: FlashInfer Python chunk_gated_delta_rule prefill.

Default Qwen3.5-122B DeltaNet GVA shape:
  Q/K [3823, 16, 128], V/O [3823, 64, 128], gate/beta [3823, 64]

Usage:
  python3 linear_attention/src/bench_flashinfer_gdn_prefill.py
  python3 linear_attention/src/bench_flashinfer_gdn_prefill.py 3823 16 64 128 1
  python3 linear_attention/src/bench_flashinfer_gdn_prefill.py 3823 --bench 20 100

Default single-run mode is intended for nsys/ncu captures. Add --bench W I for
CUDA-event timing. All input data is created on CPU and copied to GPU before the
measured FlashInfer call.
"""

import math
import sys

import numpy as np
import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule


def parse_args():
    bench_mode = False
    warmup = 0
    repeat = 1
    dtype_name = "bf16"
    output_final_state = False
    use_qk_l2norm = False

    args = sys.argv[1:]
    clean_args = []
    i = 0
    while i < len(args):
        if args[i] == "--bench":
            bench_mode = True
            warmup = int(args[i + 1]) if i + 1 < len(args) else 20
            repeat = int(args[i + 2]) if i + 2 < len(args) else 100
            i += 3
        elif args[i] == "--dtype":
            dtype_name = args[i + 1]
            i += 2
        elif args[i] == "--output-final-state":
            output_final_state = True
            i += 1
        elif args[i] == "--qk-l2norm":
            use_qk_l2norm = True
            i += 1
        else:
            clean_args.append(args[i])
            i += 1

    seq_len = int(clean_args[0]) if len(clean_args) > 0 else 3823
    q_heads = int(clean_args[1]) if len(clean_args) > 1 else 16
    v_heads = int(clean_args[2]) if len(clean_args) > 2 else 64
    head_dim = int(clean_args[3]) if len(clean_args) > 3 else 128
    num_seqs = int(clean_args[4]) if len(clean_args) > 4 else 1

    if dtype_name not in ("fp16", "bf16"):
        raise SystemExit("--dtype must be fp16 or bf16")
    if warmup < 0 or repeat <= 0:
        raise SystemExit("--bench requires warmup >= 0 and iters > 0")

    return (
        seq_len,
        q_heads,
        v_heads,
        head_dim,
        num_seqs,
        dtype_name,
        bench_mode,
        warmup,
        repeat,
        output_final_state,
        use_qk_l2norm,
    )


def profiler_start():
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        pass


def profiler_stop():
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        pass


def main():
    (
        seq_len,
        q_heads,
        v_heads,
        head_dim,
        num_seqs,
        dtype_name,
        bench_mode,
        warmup,
        repeat,
        output_final_state,
        use_qk_l2norm,
    ) = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    total_tokens = seq_len * num_seqs
    k_heads = q_heads
    out_heads = max(q_heads, v_heads)
    gate_heads = out_heads
    scale = 1.0 / math.sqrt(head_dim)

    print(
        "bench flashinfer_gdn_prefill: "
        f"seq={seq_len} seqs={num_seqs} q_heads={q_heads} "
        f"k_heads={k_heads} v_heads={v_heads} dim={head_dim} dtype={dtype_name}"
    )

    torch.manual_seed(1234)

    # Allocate and initialize all payload tensors on CPU. The H2D copies are
    # completed before timing/profiling begins, matching bench_flash_attn.py.
    q_cpu = torch.randn(total_tokens, q_heads, head_dim, dtype=dtype) * 0.1
    k_cpu = torch.randn(total_tokens, k_heads, head_dim, dtype=dtype) * 0.1
    v_cpu = torch.randn(total_tokens, v_heads, head_dim, dtype=dtype) * 0.1
    g_cpu = torch.empty(total_tokens, gate_heads, dtype=torch.float32).uniform_(0.35, 0.65)
    beta_cpu = torch.empty(total_tokens, gate_heads, dtype=torch.float32).uniform_(0.35, 0.65)
    out_cpu = torch.empty(total_tokens, out_heads, head_dim, dtype=dtype)
    cu_cpu = torch.arange(0, total_tokens + 1, seq_len, dtype=torch.int64)

    q = q_cpu.cuda()
    k = k_cpu.cuda()
    v = v_cpu.cuda()
    g = g_cpu.cuda()
    beta = beta_cpu.cuda()
    out = out_cpu.cuda()
    cu_seqlens = cu_cpu.cuda()

    out_state = None
    if output_final_state:
        out_state_cpu = torch.empty(num_seqs, gate_heads, head_dim, head_dim, dtype=torch.float32)
        out_state = out_state_cpu.cuda()

    torch.cuda.synchronize()

    def kernel():
        return chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            output=out,
            output_state=out_state,
        )

    if not bench_mode:
        profiler_start()
        result = kernel()
        torch.cuda.synchronize()
        profiler_stop()
        if isinstance(result, tuple):
            result = result[0]
        print(f"output shape: {tuple(result.shape)}")
        print("Done.")
        return 0

    for _ in range(warmup):
        kernel()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    profiler_start()
    for i in range(repeat):
        starts[i].record()
        kernel()
        ends[i].record()
    torch.cuda.synchronize()
    profiler_stop()

    times = np.array([s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)])
    print(
        f"  Kernel time: median={np.median(times):.1f} us, "
        f"min={np.min(times):.1f} us, avg={np.mean(times):.1f} us "
        f"(warmup={warmup}, iters={repeat})"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
