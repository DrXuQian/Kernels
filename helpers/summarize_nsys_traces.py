#!/usr/bin/env python3
"""Summarize Nsight Systems cuda_gpu_trace CSV files by bench_all case."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from model_latency_summary import write_model_latency_summary


FILTERED_KERNEL_SUBSTRINGS = (
    "init_",
    "initialize_tensor",
    "fill_half_kernel",
)

BENCH_122_ROW_RE = re.compile(
    r"^\| `(?P<case>[^`]+)` "
    r"\| (?P<h800_kernels>[^|]+) "
    r"\| (?P<h800_cycles_avg>[^|]+) "
    r"\| (?P<h800_cycles_max>[^|]+) "
    r"\| (?P<h800_duration_ns>[^|]+) "
    r"\| (?P<h800_latency_us>[^|]+) "
    r"\| (?P<ppu_cycles>[^|]+) "
    r"\| (?P<ppu_latency_us>[^|]+) \|$"
)


MODEL_CONFIG_ARGS = (
    ("model_layers", "model_layers"),
    ("full_attn_layers", "full_attn_layers"),
    ("linear_attn_layers", "linear_attn_layers"),
    ("moe_ffn_layers", "moe_ffn_layers"),
    ("sampling_prefill_count", "sampling_prefill_count"),
    ("sampling_decode_count", "sampling_decode_count"),
)


def add_model_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-layers", dest="model_layers", type=int)
    parser.add_argument("--full-attn-layers", dest="full_attn_layers", type=int)
    parser.add_argument("--linear-attn-layers", dest="linear_attn_layers", type=int)
    parser.add_argument("--moe-ffn-layers", dest="moe_ffn_layers", type=int)
    parser.add_argument("--sampling-prefill-count", dest="sampling_prefill_count", type=int)
    parser.add_argument("--sampling-decode-count", dest="sampling_decode_count", type=int)


def model_config_from_args(args: argparse.Namespace) -> dict[str, int] | None:
    config: dict[str, int] = {}
    for attr, key in MODEL_CONFIG_ARGS:
        value = getattr(args, attr)
        if value is not None:
            config[key] = value
    return config or None


def parse_number(value: str) -> float:
    value = value.strip().replace(",", "")
    if not value or value == "-":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def case_from_trace_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_trace_cuda_gpu_trace", "_cuda_gpu_trace"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def should_filter_kernel(name: str) -> bool:
    stripped = name.strip()
    if not stripped or stripped.startswith("["):
        return True
    lower = stripped.lower()
    return any(pattern in lower for pattern in FILTERED_KERNEL_SUBSTRINGS)


def parse_trace(path: Path) -> dict[str, object] | None:
    try:
        with path.open(newline="", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None

    kernels = []
    for row in rows:
        name = row.get("Name", "")
        if should_filter_kernel(name):
            continue
        duration_ns = parse_number(row.get("Duration (ns)", ""))
        if math.isnan(duration_ns):
            continue
        kernels.append(
            {
                "name": name,
                "duration_ns": duration_ns,
            }
        )

    if not kernels:
        return None

    total_ns = sum(float(row["duration_ns"]) for row in kernels)
    slowest = max(kernels, key=lambda row: float(row["duration_ns"]))
    return {
        "case": case_from_trace_path(path),
        "kernels": len(kernels),
        "duration_ns": total_ns,
        "latency_us": total_ns / 1000.0,
        "slowest_kernel": slowest["name"],
        "source": str(path),
    }


def parse_ncu_h800_latencies(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        match = BENCH_122_ROW_RE.match(line.strip())
        if not match:
            continue
        latency = parse_number(match.group("h800_latency_us"))
        if not math.isnan(latency):
            out[match.group("case")] = latency
    return out


def fmt(value: float) -> str:
    if math.isnan(value):
        return "-"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}"


def sanitize_kernel(name: object, max_len: int = 96) -> str:
    text = str(name).replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def markdown_table(rows: list[dict[str, object]], ncu: dict[str, float]) -> str:
    lines = [
        "| case | nsys kernels | nsys latency us | NCU H800 latency us | delta us | nsys/NCU | slowest kernel |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        case = str(row["case"])
        nsys_us = float(row["latency_us"])
        ncu_us = ncu.get(case, math.nan)
        delta = nsys_us - ncu_us if not math.isnan(ncu_us) else math.nan
        ratio = nsys_us / ncu_us if not math.isnan(ncu_us) and ncu_us > 0 else math.nan
        lines.append(
            f"| `{case}` | {row['kernels']} | {fmt(nsys_us)} | {fmt(ncu_us)} | "
            f"{fmt(delta)} | {fmt(ratio)} | `{sanitize_kernel(row['slowest_kernel'])}` |"
        )
    return "\n".join(lines)


def top_deltas(rows: list[dict[str, object]], ncu: dict[str, float], limit: int) -> str:
    deltas = []
    for row in rows:
        case = str(row["case"])
        ncu_us = ncu.get(case, math.nan)
        if math.isnan(ncu_us):
            continue
        nsys_us = float(row["latency_us"])
        deltas.append((abs(nsys_us - ncu_us), case, nsys_us, ncu_us))
    deltas.sort(reverse=True)

    lines = [
        "| case | nsys latency us | NCU H800 latency us | delta us | nsys/NCU |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, case, nsys_us, ncu_us in deltas[:limit]:
        lines.append(f"| `{case}` | {fmt(nsys_us)} | {fmt(ncu_us)} | {fmt(nsys_us - ncu_us)} | {fmt(nsys_us / ncu_us)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Nsight Systems cuda_gpu_trace CSV files.")
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--compare-md", type=Path, default=Path("bench_122B.md"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--model-summary-dir", type=Path)
    parser.add_argument("--bench-out-dir", type=Path)
    add_model_config_args(parser)
    args = parser.parse_args()

    rows = []
    missing = []
    for path in sorted(args.trace_dir.glob("*cuda_gpu_trace.csv")):
        row = parse_trace(path)
        if row is None:
            missing.append(path.name)
        else:
            rows.append(row)

    if not rows:
        raise SystemExit(f"no kernel rows found under {args.trace_dir}")

    rows.sort(key=lambda row: str(row["case"]))
    ncu = parse_ncu_h800_latencies(args.compare_md)

    lines = [
        "# Nsight Systems H800 Kernel Summary",
        "",
        f"Trace directory: `{args.trace_dir}`.",
        "",
        "Only CUDA kernel rows are summed. CUDA memcpy/memset rows and known initialization kernels are filtered.",
        "",
    ]
    if missing:
        lines.extend(["Files without kernel rows:", ""])
        lines.extend(f"- `{name}`" for name in missing)
        lines.append("")

    if ncu:
        compared = sum(1 for row in rows if str(row["case"]) in ncu)
        lines.extend(
            [
                f"Compared against H800 NCU latency values parsed from `{args.compare_md}`.",
                f"Matched cases: {compared}/{len(rows)}.",
                "",
                "## Largest Nsys vs NCU Deltas",
                "",
                top_deltas(rows, ncu, args.top),
                "",
            ]
        )

    lines.extend(["## Per-Case Aggregate", "", markdown_table(rows, ncu), ""])
    text = "\n".join(lines).rstrip() + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)

    if args.model_summary_dir:
        model_rows = [
            {
                "case": str(row["case"]),
                "latency_us": float(row["latency_us"]),
                "duration_ns": float(row["duration_ns"]),
                "source": str(row["source"]),
            }
            for row in rows
        ]
        report_path, summary_text = write_model_latency_summary(
            model_rows,
            args.model_summary_dir,
            title="Nsight Systems Model Latency Summary",
            source_name="nsys",
            bench_out_dir=args.bench_out_dir,
            model_config=model_config_from_args(args),
        )
        print()
        print(summary_text)
        print()
        print(f"Model latency summary: {report_path}")


if __name__ == "__main__":
    main()
