#!/usr/bin/env python3
"""Summarize Nsight Compute DRAM bandwidth CSV logs."""

import argparse
import csv
import math
from pathlib import Path


KERNEL_NAME_COLUMNS = ("Kernel Name", "kernel_name", "launch_kernel_name", "Name", "name")
KERNEL_ID_COLUMNS = ("ID", "id", "Kernel ID", "kernel_id", "launch_id")
DURATION_COLUMNS = (
    "gpu_time_duration.avg",
    "gpu__time_duration.avg",
    "gpu__time_duration.sum",
    "gpu_time_duration.sum",
)
READ_BYTES_COLUMNS = ("dram__bytes_read.sum", "dram_bytes_read.sum")
WRITE_BYTES_COLUMNS = ("dram__bytes_write.sum", "dram_bytes_write.sum")
TOTAL_BYTES_COLUMNS = ("dram__bytes.sum", "dram_bytes.sum")
DRAM_PCT_COLUMNS = (
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_active",
)
WIDE_COLUMNS = set(
    KERNEL_NAME_COLUMNS
    + KERNEL_ID_COLUMNS
    + DURATION_COLUMNS
    + READ_BYTES_COLUMNS
    + WRITE_BYTES_COLUMNS
    + TOTAL_BYTES_COLUMNS
    + DRAM_PCT_COLUMNS
)
LONG_METRICS = set(DURATION_COLUMNS + READ_BYTES_COLUMNS + WRITE_BYTES_COLUMNS + TOTAL_BYTES_COLUMNS + DRAM_PCT_COLUMNS)
FILTERED_KERNEL_SUBSTRINGS = (
    "init_",
    "initialize_tensor",
    "fill_half_kernel",
)


def parse_number(value):
    value = str(value).strip().replace(",", "")
    if not value or value.lower() in {"n/a", "nan"}:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def first_value(record, names, default=""):
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return default


def first_number(record, names):
    for name in names:
        if name not in record:
            continue
        value = parse_number(record[name])
        if not math.isnan(value):
            return value
    return math.nan


def should_filter_kernel(kernel_name):
    lower_name = kernel_name.lower()
    return any(pattern in lower_name for pattern in FILTERED_KERNEL_SUBSTRINGS)


def empty_row(path, kernel_id, kernel_name):
    return {
        "case": path.stem,
        "kernel_id": kernel_id,
        "kernel_name": kernel_name,
        "duration_ns": math.nan,
        "read_bytes": math.nan,
        "write_bytes": math.nan,
        "total_bytes": math.nan,
        "dram_pct": math.nan,
    }


def parse_wide_csv(path, parsed_rows):
    rows = []
    header = None
    for parsed in parsed_rows:
        if not parsed:
            continue
        if any(column in parsed for column in WIDE_COLUMNS):
            header = parsed
            continue
        if header is None or len(parsed) < len(header):
            continue

        record = {header[i]: parsed[i] for i in range(min(len(header), len(parsed)))}
        duration_ns = first_number(record, DURATION_COLUMNS)
        read_bytes = first_number(record, READ_BYTES_COLUMNS)
        write_bytes = first_number(record, WRITE_BYTES_COLUMNS)
        total_bytes = first_number(record, TOTAL_BYTES_COLUMNS)
        dram_pct = first_number(record, DRAM_PCT_COLUMNS)
        if all(math.isnan(v) for v in (duration_ns, read_bytes, write_bytes, total_bytes, dram_pct)):
            continue

        kernel = first_value(record, KERNEL_NAME_COLUMNS)
        if should_filter_kernel(kernel):
            continue
        row = empty_row(path, first_value(record, KERNEL_ID_COLUMNS), kernel)
        row.update(
            {
                "duration_ns": duration_ns,
                "read_bytes": read_bytes,
                "write_bytes": write_bytes,
                "total_bytes": total_bytes,
                "dram_pct": dram_pct,
            }
        )
        rows.append(row)
    return rows


def parse_long_csv(path, parsed_rows):
    rows = []
    header = None
    by_key = {}
    for parsed in parsed_rows:
        if not parsed:
            continue
        if "Metric Name" in parsed and "Metric Value" in parsed:
            header = parsed
            continue
        if header is None or len(parsed) < len(header):
            continue

        record = {header[i]: parsed[i] for i in range(min(len(header), len(parsed)))}
        metric = record.get("Metric Name", "")
        if metric not in LONG_METRICS:
            continue

        kernel = first_value(record, KERNEL_NAME_COLUMNS)
        if should_filter_kernel(kernel):
            continue
        kernel_id = first_value(record, KERNEL_ID_COLUMNS)
        key = (kernel_id, kernel)
        row = by_key.get(key)
        if row is None:
            row = empty_row(path, kernel_id, kernel)
            by_key[key] = row
            rows.append(row)

        value = parse_number(record.get("Metric Value", ""))
        if math.isnan(value):
            continue
        if metric in DURATION_COLUMNS:
            row["duration_ns"] = value
        elif metric in READ_BYTES_COLUMNS:
            row["read_bytes"] = value
        elif metric in WRITE_BYTES_COLUMNS:
            row["write_bytes"] = value
        elif metric in TOTAL_BYTES_COLUMNS:
            row["total_bytes"] = value
        elif metric in DRAM_PCT_COLUMNS:
            row["dram_pct"] = value
    return rows


def parse_case(path):
    try:
        parsed_rows = list(csv.reader(path.read_text(errors="replace").splitlines()))
    except OSError:
        return []
    rows = parse_long_csv(path, parsed_rows)
    rows.extend(parse_wide_csv(path, parsed_rows))
    return rows


def nan_sum(values):
    total = 0.0
    saw_value = False
    for value in values:
        if not math.isnan(value):
            total += value
            saw_value = True
    return total if saw_value else math.nan


def fmt(value, digits=3):
    if value is None or math.isnan(value):
        return ""
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def sanitize_kernel(name, max_len=96):
    name = name.replace("\n", " ")
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def weighted_dram_pct(rows):
    numerator = 0.0
    denominator = 0.0
    plain = []
    for row in rows:
        pct = row["dram_pct"]
        if math.isnan(pct):
            continue
        duration = row["duration_ns"]
        if not math.isnan(duration) and duration > 0:
            numerator += pct * duration
            denominator += duration
        else:
            plain.append(pct)
    if denominator > 0:
        return numerator / denominator
    if plain:
        return sum(plain) / len(plain)
    return math.nan


def aggregate_case(case, rows, peak_gbps):
    duration_ns = nan_sum(row["duration_ns"] for row in rows)
    read_bytes = nan_sum(row["read_bytes"] for row in rows)
    write_bytes = nan_sum(row["write_bytes"] for row in rows)
    total_bytes = nan_sum(row["total_bytes"] for row in rows)
    if math.isnan(total_bytes):
        total_bytes = nan_sum(v for v in (read_bytes, write_bytes))
    achieved_gbps = math.nan
    if not math.isnan(total_bytes) and not math.isnan(duration_ns) and duration_ns > 0:
        achieved_gbps = total_bytes / duration_ns
    computed_util = math.nan
    if peak_gbps and not math.isnan(achieved_gbps):
        computed_util = achieved_gbps / peak_gbps * 100.0
    return {
        "case": case,
        "kernels": len(rows),
        "duration_ns": duration_ns,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total_bytes": total_bytes,
        "achieved_gbps": achieved_gbps,
        "dram_pct": weighted_dram_pct(rows),
        "computed_util": computed_util,
    }


def print_aggregate(aggregate):
    print("## Nsight Compute Bandwidth Aggregate")
    print("| case | kernels | duration_ns | read_bytes | write_bytes | total_bytes | achieved_GBps | dram_peak_pct | computed_peak_pct |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in aggregate:
        print(
            f"| `{row['case']}` | {row['kernels']} | {fmt(row['duration_ns'])} | {fmt(row['read_bytes'])} | "
            f"{fmt(row['write_bytes'])} | {fmt(row['total_bytes'])} | {fmt(row['achieved_gbps'])} | "
            f"{fmt(row['dram_pct'])} | {fmt(row['computed_util'])} |"
        )


def print_detail(rows):
    print("## Nsight Compute Bandwidth Detail")
    print("| case | kernel_id | duration_ns | read_bytes | write_bytes | total_bytes | achieved_GBps | dram_peak_pct | kernel |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        total_bytes = row["total_bytes"]
        if math.isnan(total_bytes):
            total_bytes = nan_sum((row["read_bytes"], row["write_bytes"]))
        achieved = math.nan
        if not math.isnan(total_bytes) and not math.isnan(row["duration_ns"]) and row["duration_ns"] > 0:
            achieved = total_bytes / row["duration_ns"]
        print(
            f"| `{row['case']}` | {row['kernel_id']} | {fmt(row['duration_ns'])} | {fmt(row['read_bytes'])} | "
            f"{fmt(row['write_bytes'])} | {fmt(total_bytes)} | {fmt(achieved)} | {fmt(row['dram_pct'])} | "
            f"`{sanitize_kernel(row['kernel_name'])}` |"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize Nsight Compute DRAM bandwidth CSV logs.")
    parser.add_argument("ncu_dir", type=Path)
    parser.add_argument("--peak-gbps", type=float, default=0.0, help="Optional peak DRAM GB/s for computed utilization.")
    parser.add_argument("--detail", action="store_true", help="Print every profiled kernel row.")
    args = parser.parse_args()

    all_rows = []
    for path in sorted(args.ncu_dir.glob("*.csv")):
        all_rows.extend(parse_case(path))
    if not all_rows:
        raise SystemExit(f"no Nsight Compute bandwidth metric rows found under {args.ncu_dir}")

    aggregate = []
    for case in sorted({row["case"] for row in all_rows}):
        rows = [row for row in all_rows if row["case"] == case]
        aggregate.append(aggregate_case(case, rows, args.peak_gbps))

    print_aggregate(aggregate)
    if args.detail:
        print()
        print_detail(all_rows)


if __name__ == "__main__":
    main()
