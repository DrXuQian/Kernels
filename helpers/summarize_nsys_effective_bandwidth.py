#!/usr/bin/env python3
"""Estimate effective bandwidth from nsys kernel duration and benchmark shapes.

This is a fallback for environments where NCU DRAM counters are blocked. It
uses the timed kernel duration from nsys and computes mandatory benchmark
traffic from the standalone benchmark command line. The result is effective
traffic bandwidth, not a hardware DRAM-counter measurement.
"""

import argparse
import math
import re
import shlex
from pathlib import Path


DTYPE_BYTES = {
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp16": 2,
    "bf16": 2,
    "float16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float": 4,
}


def parse_number(value):
    value = str(value).strip().replace(",", "")
    if not value or value in {"-", "—"}:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def fmt(value, digits=3):
    if value is None or math.isnan(value):
        return ""
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def dtype_nbytes(dtype):
    return DTYPE_BYTES.get(str(dtype).lower(), math.nan)


def parse_args_from_command(command):
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "", {}

    if not tokens:
        return "", {}
    if tokens[0] == "cd" and "&&" in tokens:
        tokens = tokens[tokens.index("&&") + 1 :]
    if not tokens:
        return "", {}

    exe = Path(tokens[0]).name
    opts = {}
    positional = []
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            positional.extend(tokens[i + 1 :])
            break
        if token.startswith("--"):
            if "=" in token:
                key, value = token[2:].split("=", 1)
                opts[key] = value
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                opts[token[2:]] = tokens[i + 1]
                i += 1
            else:
                opts[token[2:]] = "1"
        else:
            positional.append(token)
        i += 1
    opts["_positional"] = positional
    return exe, opts


def int_opt(opts, name, default=None):
    value = opts.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def str_opt(opts, name, default=None):
    return opts.get(name, default)


def gemm_bytes(opts, allow_fp8=False):
    m = int_opt(opts, "m")
    n = int_opt(opts, "n")
    k = int_opt(opts, "k")
    dtype = str_opt(opts, "dtype", "fp16")
    out_dtype = str_opt(opts, "out-dtype", str_opt(opts, "output-dtype", "same"))
    if out_dtype == "same":
        out_dtype = "fp16" if dtype.startswith("fp8") else dtype
    in_b = dtype_nbytes(dtype)
    out_b = dtype_nbytes(out_dtype)
    if None in (m, n, k) or math.isnan(in_b) or math.isnan(out_b):
        return math.nan, ""
    if dtype.startswith("fp8") and not allow_fp8:
        return math.nan, "fp8 cuBLAS traffic is not modeled"
    total = m * k * in_b + k * n * in_b + m * n * out_b
    return total, f"gemm({m},{n},{k},{dtype}->{out_dtype})"


def block_fp8_bytes(opts):
    m = int_opt(opts, "m")
    n = int_opt(opts, "n")
    k = int_opt(opts, "k")
    out_dtype = str_opt(opts, "out-dtype", str_opt(opts, "output-dtype", "fp16"))
    out_b = dtype_nbytes(out_dtype)
    if None in (m, n, k) or math.isnan(out_b):
        return math.nan, ""
    kernel_m = round_up(m, 4)
    k_blocks = round_up(k, 128) // 128
    n_blocks = round_up(n, 128) // 128
    total = (
        kernel_m * k  # A fp8
        + n * k  # B fp8
        + m * n * out_b  # D
        + kernel_m * k_blocks * 4  # scale_a fp32
        + n_blocks * k_blocks * 4  # scale_b fp32
    )
    return total, f"block_fp8_gemm(m={m},kernel_m={kernel_m},n={n},k={k},out={out_dtype})"


def moe_fp8_bytes(opts):
    experts = int_opt(opts, "experts", 8)
    m_per_expert = int_opt(opts, "m_per_expert")
    n = int_opt(opts, "n")
    k = int_opt(opts, "k")
    if None in (experts, m_per_expert, n, k):
        return math.nan, ""
    total_m = experts * m_per_expert
    k_blocks = round_up(k, 128) // 128
    n_blocks = round_up(n, 128) // 128
    padded_rows = (total_m + experts * 31) // 32 * 32
    total = (
        total_m * k  # A fp8
        + experts * n * k  # B fp8
        + total_m * n * 2  # D bf16
        + padded_rows * k_blocks * 4  # scale_a fp32
        + experts * n_blocks * k_blocks * 4  # scale_b fp32
        + (experts + 1) * 8  # offsets
    )
    return total, f"moe_fp8(experts={experts},m_per={m_per_expert},n={n},k={k})"


def rmsnorm_bytes(opts, log_text):
    batch = int_opt(opts, "batch")
    embed = int_opt(opts, "embed")
    dtype = str_opt(opts, "dtype", "fp16")
    if batch is None or embed is None:
        match = re.search(r"bench rmsnorm: batch=(\d+) embed=(\d+) dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            batch = int(match.group(1))
            embed = int(match.group(2))
            dtype = match.group(3)
    b = dtype_nbytes(dtype)
    if None in (batch, embed) or math.isnan(b):
        return math.nan, ""
    total = batch * embed * b + embed * b + batch * embed * b
    return total, f"rmsnorm(batch={batch},embed={embed},dtype={dtype})"


def linear_ops_bytes(opts, log_text):
    op = str_opt(opts, "op", "")
    tokens = int_opt(opts, "tokens")
    hidden = int_opt(opts, "hidden")
    dtype = str_opt(opts, "dtype", "fp16")
    if tokens is None or hidden is None:
        match = re.search(r"linear ops bench: op=([A-Za-z0-9_]+) tokens=(\d+) hidden=(\d+).* dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            op = match.group(1)
            tokens = int(match.group(2))
            hidden = int(match.group(3))
            dtype = match.group(4)
    b = dtype_nbytes(dtype)
    if op != "residual_add" or None in (tokens, hidden) or math.isnan(b):
        return math.nan, ""
    total = 3 * tokens * hidden * b
    return total, f"residual_add(tokens={tokens},hidden={hidden},dtype={dtype})"


def gated_activation_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = topk = inter = None
    dtype = "fp16"
    if len(pos) >= 4:
        tokens, topk, inter, dtype = int(pos[0]), int(pos[1]), int(pos[2]), pos[3]
    else:
        match = re.search(r"gated_activation: tokens=(\d+) topk=(\d+) rows=\d+ inter=(\d+) dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            tokens = int(match.group(1))
            topk = int(match.group(2))
            inter = int(match.group(3))
            dtype = match.group(4)
    b = dtype_nbytes(dtype)
    if None in (tokens, topk, inter) or math.isnan(b):
        return math.nan, ""
    total = 3 * tokens * topk * inter * b
    return total, f"gated_activation(tokens={tokens},topk={topk},inter={inter},dtype={dtype})"


def estimate_bytes(exe, opts, log_text):
    if exe in {"bench_cublas_gemm", "bench_cuda_core_gemv"}:
        return gemm_bytes(opts, allow_fp8=True)
    if exe == "bench_cutlass_block_fp8_gemm":
        return block_fp8_bytes(opts)
    if exe == "bench_moe_fp8_blockscale_gemm":
        return moe_fp8_bytes(opts)
    if exe == "bench_rmsnorm":
        return rmsnorm_bytes(opts, log_text)
    if exe == "bench_linear_ops":
        return linear_ops_bytes(opts, log_text)
    if exe == "bench_gated_activation":
        return gated_activation_bytes(opts, log_text)
    return math.nan, f"unsupported executable: {exe}"


def parse_nsys_aggregate(summary_path):
    rows = {}
    if not summary_path.exists():
        return rows
    for line in summary_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 5:
            continue
        case = parts[0].strip("`")
        duration_ns = parse_number(parts[3])
        latency_us = parse_number(parts[4])
        if not math.isnan(duration_ns):
            rows[case] = {"duration_ns": duration_ns, "latency_us": latency_us}
    return rows


def read_command_from_log(log_path):
    if not log_path.exists():
        return "", ""
    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        if line.startswith("command:"):
            return line[len("command:") :].strip(), text
    return "", text


def main():
    parser = argparse.ArgumentParser(description="Estimate effective bandwidth from nsys benchmark output.")
    parser.add_argument("out_dir", type=Path, help="Benchmark OUT_DIR containing nsys_latency_summary.md and case logs.")
    parser.add_argument("--peak-gbps", type=float, default=3350.0)
    args = parser.parse_args()

    aggregate = parse_nsys_aggregate(args.out_dir / "nsys_latency_summary.md")
    if not aggregate:
        raise SystemExit(f"no nsys aggregate rows found under {args.out_dir}")

    supported = []
    unsupported = []
    for case, timing in sorted(aggregate.items()):
        command, log_text = read_command_from_log(args.out_dir / f"{case}.log")
        exe, opts = parse_args_from_command(command)
        est_bytes, estimator = estimate_bytes(exe, opts, log_text)
        duration_ns = timing["duration_ns"]
        if math.isnan(est_bytes) or duration_ns <= 0:
            unsupported.append((case, estimator or "missing command/shape metadata"))
            continue
        gbps = est_bytes / duration_ns
        pct = 100.0 * gbps / args.peak_gbps if args.peak_gbps > 0 else math.nan
        supported.append((case, duration_ns, est_bytes, gbps, pct, estimator))

    print("## Nsight Systems Effective Bandwidth Estimate")
    print()
    print(
        "Effective bandwidth is computed from theoretical benchmark traffic divided by nsys CUDA kernel duration. "
        "It is a fallback when NCU DRAM counters are unavailable."
    )
    print()
    print("| case | duration_ns | estimated_bytes | effective_GBps | peak_pct | estimator |")
    print("|---|---:|---:|---:|---:|---|")
    for case, duration_ns, est_bytes, gbps, pct, estimator in supported:
        print(f"| `{case}` | {fmt(duration_ns)} | {fmt(est_bytes)} | {fmt(gbps)} | {fmt(pct)} | `{estimator}` |")

    if unsupported:
        print()
        print("## Unsupported Cases")
        print("| case | reason |")
        print("|---|---|")
        for case, reason in unsupported:
            print(f"| `{case}` | {reason} |")


if __name__ == "__main__":
    main()
