#!/usr/bin/env python3
"""FA3 non-causal BF16 prefill: B=1, Q/KV=1024, Hq/Hkv=56, Dq/Dv=128.

Default: one forward call, no warmup or CUDA-event timing.
Use --bench WARMUP ITERS to measure latency on a physical GPU.
"""

import argparse
import statistics


def load_fa3():
    try:
        import flash_attn_3.flash_attn_interface as interface
    except ModuleNotFoundError as exc:
        if exc.name not in ("flash_attn_3", "flash_attn_3.flash_attn_interface"):
            raise
        # Older FA3 releases and flash-attention-for-sail use this layout.
        try:
            import flash_attn_interface as interface
        except ModuleNotFoundError as legacy_exc:
            if legacy_exc.name != "flash_attn_interface":
                raise
            raise SystemExit(
                "FlashAttention-3 is required. Install the hopper/ package "
                "or your platform's FA3 fork into the active Python environment."
            ) from legacy_exc
    return interface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench", nargs=2, type=int, metavar=("WARMUP", "ITERS"),
        help="enable CUDA-event timing; default is one untimed forward call",
    )
    args = parser.parse_args()
    warmup, iters = args.bench if args.bench is not None else (0, 1)
    if warmup < 0 or iters <= 0:
        parser.error("--bench requires WARMUP >= 0 and ITERS > 0")

    import torch

    fa3 = load_fa3()
    batch = 1
    seqlen_q = 1024
    seqlen_k = 1024
    num_heads = 56
    num_heads_k = 56
    head_dim = 128
    head_dim_value = 128
    dtype = torch.bfloat16

    print(f"FlashAttention backend: FA3 interface={fa3.__name__} path={fa3.__file__}")
    print(
        f"bench flash_attn3 prefill: batch={batch} seqlen_q={seqlen_q} "
        f"seqlen_k={seqlen_k} num_heads={num_heads} num_heads_k={num_heads_k} "
        f"head_dim={head_dim} head_dim_value={head_dim_value} "
        f"dtype=bf16 causal=False warmup={warmup} iters={iters}",
        flush=True,
    )

    # Seed and initialize on CPU so random-number kernels are not profiled.
    generator = torch.Generator(device="cpu").manual_seed(0)
    q_cpu = torch.randn(
        (batch, seqlen_q, num_heads, head_dim),
        device="cpu", dtype=dtype, generator=generator,
    )
    k_cpu = torch.randn(
        (batch, seqlen_k, num_heads_k, head_dim),
        device="cpu", dtype=dtype, generator=generator,
    )
    v_cpu = torch.randn(
        (batch, seqlen_k, num_heads_k, head_dim_value),
        device="cpu", dtype=dtype, generator=generator,
    )
    q, k, v = q_cpu.cuda(), k_cpu.cuda(), v_cpu.cuda()
    torch.cuda.synchronize()

    def forward():
        return fa3.flash_attn_func(q, k, v, causal=False)

    with torch.inference_mode():
        if args.bench is None:
            out = forward()
            torch.cuda.synchronize()
        else:
            for _ in range(warmup):
                forward()
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for start, end in zip(starts, ends):
                start.record()
                out = forward()
                end.record()
            torch.cuda.synchronize()
            times_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
            print(f"  FA3 time: median={statistics.median(times_us):.1f} us, "
                  f"min={min(times_us):.1f} us (warmup={warmup}, iters={iters})")

    print(f"Done. output_shape={tuple(out.shape)} output_dtype={out.dtype}")


if __name__ == "__main__":
    main()
