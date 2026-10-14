#!/usr/bin/env python3
"""Summarize ACU device durations; never infer hardware time from a GHz input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from model_latency_summary import (
    classify_module, classify_phase, compute_model_summary, console_summary,
    load_case_calls, parse_case_log_metadata, write_model_latency_summary,
)
from summarize_nsys_kernels import add_model_config_args, model_config_from_args, should_filter_kernel


DURATION_METRICS = ("ppu__time_duration.sum", "ppu_time_duration.sum")
UNITS_NS = {
    "ns": 1.0, "nsecond": 1.0, "nanosecond": 1.0, "nanoseconds": 1.0,
    "us": 1000.0, "usecond": 1000.0, "microsecond": 1000.0, "microseconds": 1000.0,
    "ms": 1e6, "msecond": 1e6, "millisecond": 1e6, "milliseconds": 1e6,
    "s": 1e9, "second": 1e9, "seconds": 1e9,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_name(path: Path) -> str:
    return path.name.removesuffix(".acu.csv") if path.name.endswith(".acu.csv") else path.stem


def metadata_path(path: Path) -> Path:
    return path.with_name(case_name(path) + ".metadata.json")


def unit_scale(unit: str) -> float:
    unit = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    if unit not in UNITS_NS:
        raise ValueError(f"unknown or missing ACU duration unit: {unit!r}")
    return UNITS_NS[unit]


def duration_ns(value: str, unit: str) -> float:
    result = float(value.strip().replace(",", "")) * unit_scale(unit)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"invalid ACU device duration: {value!r} {unit}")
    return result


def parse_case(path: Path, *, check_metadata: bool = True,
               include_kernel: str = "", exclude_kernels: tuple[str, ...] = ()) -> list[dict]:
    """Read long metric CSV or wide kernel CSV with an explicit units row.

    Launch identity includes process/device/context/stream and ID. Repeated
    copies of one metric are counted once; conflicting copies fail the case.
    A selected launch missing a duration invalidates the entire case.
    """
    metadata = {}
    sidecar = metadata_path(path)
    if check_metadata and sidecar.is_file():
        metadata = json.loads(sidecar.read_text())
        if metadata.get("status") != "COMPLETE":
            raise ValueError(f"capture is {metadata.get('status', 'incomplete')}: {sidecar}")
        if metadata.get("csv_sha256") != sha256(path):
            raise ValueError(f"CSV differs from captured report export: {path}")
        include_kernel = metadata.get("include_kernel", include_kernel)
        exclude_kernels = tuple(metadata.get("exclude_kernels", exclude_kernels))

    launches: dict[tuple, dict] = {}
    header = None
    column = None
    wide_unit = ""
    mode = ""
    found_header = False
    with path.open(newline="", errors="replace") as stream:
        for cells in csv.reader(stream):
            cells = [cell.strip() for cell in cells]
            if "Kernel Name" in cells and ("ID" in cells or "Kernel ID" in cells):
                found_header = True
                header = cells
                mode = "long" if "Metric Name" in cells and "Metric Value" in cells else "wide"
                column, wide_unit = None, ""
                for name in cells:
                    match = re.fullmatch(r"(ppu_+time_duration\.sum)(?:\s*[\[(]([^\])]+)[\])])?", name)
                    if match and match[1] in DURATION_METRICS:
                        column, wide_unit = name, match[2] or ""
                        break
                if mode == "wide" and column is None:
                    raise ValueError("ACU kernel table has no ppu__time_duration.sum column")
                continue
            if header is not None and len(cells) != len(header) and cells and cells[0].isdigit():
                raise ValueError("truncated ACU kernel/metric row")
            if header is None or len(cells) != len(header):
                continue
            record = dict(zip(header, cells))
            kernel = record.get("Kernel Name", "")
            if not kernel:
                if mode == "wide" and record.get(column, ""):
                    wide_unit = record[column]
                    unit_scale(wide_unit)
                continue
            if should_filter_kernel(kernel) or any(part in kernel for part in exclude_kernels):
                continue
            if include_kernel and include_kernel not in kernel:
                continue
            kernel_id = record.get("ID") or record.get("Kernel ID")
            if not kernel_id:
                raise ValueError(f"ACU launch has no ID: {kernel}")
            key = tuple(record.get(field, "") for field in ("Process ID", "Device", "Context", "Stream")) + (kernel_id, kernel)
            row = launches.setdefault(key, {
                "case": case_name(path), "kernel_id": kernel_id, "kernel_name": kernel,
                "duration_ns": None, "device": record.get("Device", ""),
            })
            if mode == "long":
                if record["Metric Name"] not in DURATION_METRICS:
                    continue
                value = duration_ns(record["Metric Value"], record.get("Metric Unit", ""))
            else:
                value = duration_ns(record[column], wide_unit)
            if row["duration_ns"] is not None and not math.isclose(row["duration_ns"], value, rel_tol=1e-12):
                raise ValueError(f"conflicting durations for launch {kernel_id}: {kernel}")
            row["duration_ns"] = value
    if not found_header:
        raise ValueError("no ACU raw CSV kernel table; inspect the capture/export log")
    if not launches:
        raise ValueError("no selected ACU kernel launches")
    if any(row["duration_ns"] is None for row in launches.values()):
        raise ValueError("at least one selected ACU launch has no device duration")
    expected = metadata.get("expected_kernels")
    if expected is not None and len(launches) != expected:
        raise ValueError(f"expected {expected} kernel launches, found {len(launches)}")
    return list(launches.values())


def collect_cases(directory: Path) -> tuple[list[dict], list[dict], list[dict]]:
    cases, kernels, missing = [], [], []
    paths = sorted(directory.glob("*.acu.csv"))
    # A failed capture may not have produced a CSV at all.
    labels = {case_name(path) for path in paths}
    for sidecar in sorted(directory.glob("*.metadata.json")):
        label = sidecar.name.removesuffix(".metadata.json")
        if label not in labels:
            missing.append({"case": label, "reason": "no exported ACU CSV"})
    for path in paths:
        try:
            log = directory.parent / (case_name(path) + ".log")
            if log.is_file():
                info = parse_case_log_metadata(log)
                if info.get("failed_at") or info.get("status") == "failed" or info.get("exit_status", "0") != "0":
                    raise ValueError("benchmark case failed; refusing its saved measurement")
            rows = parse_case(path)
            kernels.extend(rows)
            ns = math.fsum(row["duration_ns"] for row in rows)
            entry = {"case": case_name(path), "kernels": len(rows), "duration_ns": ns,
                     "latency_us": ns / 1000, "source": str(path)}
            sidecar = metadata_path(path)
            if sidecar.is_file():
                metadata = json.loads(sidecar.read_text())
                for kind in ("read", "write"):
                    if metadata.get(f"{kind}_bytes") is not None:
                        entry[f"{kind}_gb_s"] = metadata[f"{kind}_bytes"] / ns
                if "read_gb_s" in entry and "write_gb_s" in entry:
                    entry["traffic_gb_s"] = entry["read_gb_s"] + entry["write_gb_s"]
            cases.append(entry)
        except (OSError, ValueError) as exc:
            missing.append({"case": case_name(path), "reason": str(exc)})
    return cases, kernels, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acu_dir", type=Path)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--bench-out-dir", type=Path)
    parser.add_argument("--model-summary-dir", type=Path)
    parser.add_argument("--phase", choices=("auto", "all", "prefill", "decode"), default="auto")
    parser.add_argument("--no-model", action="store_true", help="Use for standalone SOL tests.")
    parser.add_argument("--case-calls", action="append", default=[])
    parser.add_argument("--calls-file", type=Path)
    add_model_config_args(parser)
    args = parser.parse_args()
    try:
        cases, kernels, missing = collect_cases(args.acu_dir)
        if not cases and not missing:
            raise ValueError(f"no *.acu.csv reports under {args.acu_dir}")
        print("# ACU device kernel latency\n")
        print("Times are ACU device durations under profiler replay/cache control, not host time or end-to-end model latency.\n")
        print("| case | kernels | sum_duration_ns | latency_us | read_GB/s | write_GB/s | read+write_GB/s |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for row in cases:
            bandwidth = [f"{row[key]:.3f}" if key in row else "" for key in ("read_gb_s", "write_gb_s", "traffic_gb_s")]
            print(f"| `{row['case']}` | {row['kernels']} | {row['duration_ns']:.6f} | {row['latency_us']:.6f} | " + " | ".join(bandwidth) + " |")
        if args.detail:
            print("\n| case | launch_id | duration_ns | kernel |\n|---|---|---:|---|")
            for row in kernels:
                name = row['kernel_name'].replace('|', '\\|')
                print(f"| `{row['case']}` | {row['kernel_id']} | {row['duration_ns']:.6f} | `{name}` |")
        if not args.no_model:
            bench_dir = args.bench_out_dir or args.acu_dir.parent
            config = model_config_from_args(args) or {}
            manifest = bench_dir / "expected_cases.txt"
            expected = [line.split("#", 1)[0].strip() for line in manifest.read_text().splitlines()] if manifest.is_file() else []
            phase = args.phase
            if phase == "auto":
                labels = {row['case'] for row in cases + missing} | set(expected)
                phases = {classify_phase(label) for label in labels if label and classify_module(label) != "Sampling"}
                phases &= {"prefill", "decode"}
                phase = next(iter(phases)) if len(phases) == 1 else "all"
            selected = [row for row in cases if phase == "all" or classify_phase(row['case']) == phase or classify_module(row['case']) == "Sampling"]
            missing = [row for row in missing if phase == "all" or classify_phase(row['case']) == phase or classify_module(row['case']) == "Sampling"]
            if phase in ("prefill", "decode"):
                config["sampling_decode_count" if phase == "prefill" else "sampling_prefill_count"] = 0
            calls = load_case_calls(args.case_calls, args.calls_file)
            computed = compute_model_summary(selected, bench_dir, config, calls)
            covered = {row['case'] for row in computed['covered_summary']['cases']}
            for label in expected:
                if label and label not in covered and not any(row['case'] == label for row in missing):
                    if phase == "all" or classify_phase(label) == phase or classify_module(label) == "Sampling":
                        missing.append({"case": label, "reason": "no valid ACU measurement (including dedupe source)"})
            print("\n" + console_summary(computed['model_summary']))
            if args.model_summary_dir and selected:
                write_model_latency_summary(
                    selected, args.model_summary_dir, title="ACU Model Kernel Latency Estimate",
                    source_name="acu", bench_out_dir=bench_dir, model_config=config,
                    case_calls=calls, missing_cases=missing,
                    chart_phases=("prefill", "decode") if phase == "all" else (phase,),
                )
        if missing:
            print("\n## Missing measurements (totals are partial)\n")
            for row in missing:
                print(f"- {row['case']}: {row['reason']}")
        return int(bool(missing) or not (cases if args.no_model else selected))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
