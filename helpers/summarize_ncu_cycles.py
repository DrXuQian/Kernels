#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def parse_number(value):
    value = value.strip().replace(",", "")
    if not value or value.lower() in {"n/a", "nan"}:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_case(path):
    rows = []
    header = None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return rows

    for parsed in csv.reader(lines):
        if not parsed:
            continue
        if "Metric Name" in parsed and "Metric Value" in parsed:
            header = parsed
            continue
        if header is None or len(parsed) < len(header):
            continue

        record = {header[i]: parsed[i] for i in range(min(len(header), len(parsed)))}
        metric = record.get("Metric Name", "")
        if metric not in {"sm__cycles_elapsed.avg", "sm__cycles_elapsed.max", "gpu__time_duration.sum"}:
            continue

        kernel = record.get("Kernel Name", record.get("Name", ""))
        kernel_id = record.get("ID", "")
        value = parse_number(record.get("Metric Value", ""))
        if math.isnan(value):
            continue

        key = (kernel_id, kernel)
        existing = None
        for row in rows:
            if (row["kernel_id"], row["kernel_name"]) == key:
                existing = row
                break
        if existing is None:
            existing = {
                "case": path.stem,
                "kernel_id": kernel_id,
                "kernel_name": kernel,
                "cycles_avg": math.nan,
                "cycles_max": math.nan,
                "duration_ns": math.nan,
            }
            rows.append(existing)

        if metric == "sm__cycles_elapsed.avg":
            existing["cycles_avg"] = value
        elif metric == "sm__cycles_elapsed.max":
            existing["cycles_max"] = value
        elif metric == "gpu__time_duration.sum":
            existing["duration_ns"] = value

    return rows


def file_diagnostic(path):
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return f"read failed: {exc}"

    interesting = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if (
            "==error==" in lower
            or "==warning==" in lower
            or "err_" in lower
            or "error:" in lower
            or "failed" in lower
            or "no kernels" in lower
            or "permission" in lower
            or "launch-skip" in lower
        ):
            interesting.append(stripped)
        if len(interesting) >= 3:
            break

    if interesting:
        return " | ".join(interesting)
    if "Metric Name" not in text:
        return "no Nsight Compute metric header; inspect the case log for target program output"
    return "metric header found, but requested metrics were absent or non-numeric"


def print_missing_rows(files):
    if not files:
        return
    print("## Nsight Compute Files Without Metric Rows")
    print("| case | diagnostic |")
    print("|---|---|")
    for path in files:
        print(f"| `{path.stem}` | {file_diagnostic(path)} |")


def print_empty_dir_diagnostic(ncu_dir):
    print("## Nsight Compute Files Without Metric Rows")
    print("| case | diagnostic |")
    print("|---|---|")
    print(
        f"| `(none)` | no `.csv` files found under `{ncu_dir}`; selected cases likely failed before `ncu` "
        "launched, all selected cases were skipped, or `OUT_DIR` points at a different run |"
    )


def fmt_number(value):
    if value is None or math.isnan(value):
        return ""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.3f}"


def sanitize_kernel(name, max_len=96):
    name = name.replace("\n", " ")
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def print_table(rows, title):
    print(title)
    print("| case | kernel_id | cycles_avg | cycles_max | duration_ns | kernel |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| `{row['case']}` | {row['kernel_id']} | {fmt_number(row['cycles_avg'])} | "
            f"{fmt_number(row['cycles_max'])} | {fmt_number(row['duration_ns'])} | "
            f"`{sanitize_kernel(row['kernel_name'])}` |"
        )


def print_aggregate_table(rows):
    print("## Nsight Compute Case Aggregate")
    print("| case | kernels | sum_cycles_avg | sum_cycles_max | sum_duration_ns | slowest_kernel |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| `{row['case']}` | {row['kernels']} | {fmt_number(row['sum_cycles_avg'])} | "
            f"{fmt_number(row['sum_cycles_max'])} | {fmt_number(row['sum_duration_ns'])} | "
            f"`{sanitize_kernel(row['slowest_kernel'])}` |"
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
    parser = argparse.ArgumentParser(description="Summarize Nsight Compute cycle CSV logs.")
    parser.add_argument("ncu_dir", type=Path)
    parser.add_argument("--detail", action="store_true", help="Print every profiled kernel row.")
    args = parser.parse_args()

    all_rows = []
    files_without_rows = []
    csv_files = sorted(args.ncu_dir.glob("*.csv"))
    for path in csv_files:
        rows = parse_case(path)
        if rows:
            all_rows.extend(rows)
        else:
            files_without_rows.append(path)

    if not all_rows:
        if csv_files:
            print_missing_rows(files_without_rows)
        else:
            print_empty_dir_diagnostic(args.ncu_dir)
        raise SystemExit(f"no Nsight Compute metric rows found under {args.ncu_dir}")

    aggregate = []
    slowest = []
    for case in sorted({row["case"] for row in all_rows}):
        case_rows = [row for row in all_rows if row["case"] == case]
        case_rows.sort(
            key=lambda row: (
                -1 if math.isnan(row["cycles_avg"]) else row["cycles_avg"],
                -1 if math.isnan(row["cycles_max"]) else row["cycles_max"],
            ),
            reverse=True,
        )
        slowest.append(case_rows[0])
        aggregate.append(
            {
                "case": case,
                "kernels": len(case_rows),
                "sum_cycles_avg": nan_sum(row["cycles_avg"] for row in case_rows),
                "sum_cycles_max": nan_sum(row["cycles_max"] for row in case_rows),
                "sum_duration_ns": nan_sum(row["duration_ns"] for row in case_rows),
                "slowest_kernel": case_rows[0]["kernel_name"],
            }
        )

    print_aggregate_table(aggregate)
    print()
    print_table(slowest, "## Nsight Compute Slowest Kernel Per Case")
    if args.detail:
        print()
        print_table(all_rows, "## Nsight Compute Cycles Detail")
        if files_without_rows:
            print()
            print_missing_rows(files_without_rows)


if __name__ == "__main__":
    main()
