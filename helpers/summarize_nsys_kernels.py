#!/usr/bin/env python3
"""Summarize Nsight Systems CUDA kernel duration reports."""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

from model_latency_summary import write_model_latency_summary


FILTERED_KERNEL_SUBSTRINGS = (
    "init_",
    "initialize_tensor",
    "fill_half_kernel",
)

MODEL_CONFIG_ARGS = (
    ("model_layers", "model_layers"),
    ("full_attn_layers", "full_attn_layers"),
    ("linear_attn_layers", "linear_attn_layers"),
    ("dense_ffn_layers", "dense_ffn_layers"),
    ("moe_ffn_layers", "moe_ffn_layers"),
    ("sampling_prefill_count", "sampling_prefill_count"),
    ("sampling_decode_count", "sampling_decode_count"),
)


def add_model_config_args(parser):
    parser.add_argument("--model-layers", dest="model_layers", type=int, help="Total transformer layers.")
    parser.add_argument("--full-attn-layers", dest="full_attn_layers", type=int, help="Full-attention layer count.")
    parser.add_argument("--linear-attn-layers", dest="linear_attn_layers", type=int, help="Linear-attention layer count.")
    parser.add_argument("--dense-ffn-layers", dest="dense_ffn_layers", type=int, help="Dense-FFN layer count.")
    parser.add_argument("--moe-ffn-layers", dest="moe_ffn_layers", type=int, help="MoE-FFN layer count.")
    parser.add_argument("--sampling-prefill-count", dest="sampling_prefill_count", type=int, help="Prefill sampling count.")
    parser.add_argument("--sampling-decode-count", dest="sampling_decode_count", type=int, help="Decode sampling count.")


def model_config_from_args(args):
    config = {}
    for attr, key in MODEL_CONFIG_ARGS:
        value = getattr(args, attr)
        if value is not None:
            config[key] = value
    return config or None


def parse_number(value):
    value = value.strip().replace(",", "")
    if not value or value.lower() in {"n/a", "nan"}:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def should_filter_kernel(kernel_name):
    lower_name = kernel_name.lower()
    return any(pattern in lower_name for pattern in FILTERED_KERNEL_SUBSTRINGS)


def sanitize_kernel(name, max_len=96):
    name = name.replace("\n", " ")
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def fmt_number(value):
    if value is None or math.isnan(value):
        return ""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.3f}"


def nsys_stats_csv(path):
    cmd = [
        "nsys",
        "stats",
        "--force-export=true",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        str(path),
    ]
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def parse_case(path):
    result = nsys_stats_csv(path)
    if result.returncode != 0:
        return [], result.stdout.strip()

    rows = []
    header = None
    for parsed in csv.reader(result.stdout.splitlines()):
        if not parsed:
            continue
        if "Total Time (ns)" in parsed and "Name" in parsed:
            header = parsed
            continue
        if header is None or len(parsed) < len(header):
            continue

        record = {header[i]: parsed[i] for i in range(min(len(header), len(parsed)))}
        kernel_name = record.get("Name", "")
        if not kernel_name or should_filter_kernel(kernel_name):
            continue

        total_ns = parse_number(record.get("Total Time (ns)", ""))
        instances = parse_number(record.get("Instances", ""))
        avg_ns = parse_number(record.get("Avg (ns)", ""))
        med_ns = parse_number(record.get("Med (ns)", ""))
        min_ns = parse_number(record.get("Min (ns)", ""))
        max_ns = parse_number(record.get("Max (ns)", ""))
        if math.isnan(total_ns) and math.isnan(avg_ns):
            continue

        rows.append(
            {
                "case": path.stem,
                "kernel_name": kernel_name,
                "instances": instances,
                "total_ns": total_ns,
                "avg_ns": avg_ns,
                "med_ns": med_ns,
                "min_ns": min_ns,
                "max_ns": max_ns,
            }
        )
    return rows, ""


def print_missing_rows(missing):
    if not missing:
        return
    print("## Nsight Systems Files Without Kernel Rows")
    print("| case | diagnostic |")
    print("|---|---|")
    for path, diagnostic in missing:
        text = diagnostic.replace("\n", " | ")
        if len(text) > 180:
            text = text[:177] + "..."
        print(f"| `{path.stem}` | {text or 'no CUDA kernel rows found'} |")


def print_detail(rows):
    print("## Nsight Systems Kernel Detail")
    print("| case | instances | total_duration_ns | avg_ns | med_ns | min_ns | max_ns | kernel |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| `{row['case']}` | {fmt_number(row['instances'])} | {fmt_number(row['total_ns'])} | "
            f"{fmt_number(row['avg_ns'])} | {fmt_number(row['med_ns'])} | "
            f"{fmt_number(row['min_ns'])} | {fmt_number(row['max_ns'])} | "
            f"`{sanitize_kernel(row['kernel_name'])}` |"
        )


def print_aggregate(aggregate):
    print("## Nsight Systems Case Aggregate")
    print("| case | kernels | kernel_instances | sum_duration_ns | latency_us |")
    print("|---|---:|---:|---:|---:|")
    for row in aggregate:
        print(
            f"| `{row['case']}` | {row['kernels']} | {fmt_number(row['kernel_instances'])} | "
            f"{fmt_number(row['sum_duration_ns'])} | {fmt_number(row['latency_us'])} |"
        )


def nan_sum(values):
    total = 0.0
    saw_value = False
    for value in values:
        if not math.isnan(value):
            total += value
            saw_value = True
    return total if saw_value else math.nan


def main():
    parser = argparse.ArgumentParser(description="Summarize Nsight Systems CUDA kernel duration reports.")
    parser.add_argument("nsys_dir", type=Path)
    parser.add_argument("--detail", action="store_true", help="Print every kernel row.")
    parser.add_argument("--model-summary-dir", type=Path, help="Write model-level latency tables and SVG charts here.")
    parser.add_argument(
        "--bench-out-dir",
        type=Path,
        help="Benchmark output directory used to expand deduped logical cases. Defaults to the nsys directory parent.",
    )
    add_model_config_args(parser)
    args = parser.parse_args()

    all_rows = []
    missing = []
    rep_files = sorted(args.nsys_dir.glob("*.nsys-rep"))
    for path in rep_files:
        rows, diagnostic = parse_case(path)
        if rows:
            all_rows.extend(rows)
        else:
            missing.append((path, diagnostic))

    if not all_rows:
        if rep_files:
            print_missing_rows(missing)
        else:
            print("## Nsight Systems Files Without Kernel Rows")
            print("| case | diagnostic |")
            print("|---|---|")
            print(f"| `(none)` | no `.nsys-rep` files found under `{args.nsys_dir}` |")
        raise SystemExit(f"no Nsight Systems CUDA kernel rows found under {args.nsys_dir}")

    aggregate = []
    model_rows = []
    for case in sorted({row["case"] for row in all_rows}):
        case_rows = [row for row in all_rows if row["case"] == case]
        sum_duration_ns = nan_sum(row["total_ns"] for row in case_rows)
        kernel_instances = nan_sum(row["instances"] for row in case_rows)
        latency_us = sum_duration_ns / 1000.0 if not math.isnan(sum_duration_ns) else math.nan
        aggregate.append(
            {
                "case": case,
                "kernels": len(case_rows),
                "kernel_instances": kernel_instances,
                "sum_duration_ns": sum_duration_ns,
                "latency_us": latency_us,
            }
        )
        if not math.isnan(latency_us):
            model_rows.append(
                {
                    "case": case,
                    "latency_us": latency_us,
                    "duration_ns": sum_duration_ns,
                    "source": str(args.nsys_dir / f"{case}.nsys-rep"),
                }
            )

    print_aggregate(aggregate)
    if args.detail:
        print()
        print_detail(all_rows)
        if missing:
            print()
            print_missing_rows(missing)

    if args.model_summary_dir:
        report_path, summary_text = write_model_latency_summary(
            model_rows,
            args.model_summary_dir,
            title="Nsight Systems Model Latency Summary",
            source_name="nsys",
            bench_out_dir=args.bench_out_dir or args.nsys_dir.parent,
            model_config=model_config_from_args(args),
        )
        print()
        print(summary_text)
        print()
        print(f"Model latency summary: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
