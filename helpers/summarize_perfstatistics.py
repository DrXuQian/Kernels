#!/usr/bin/env python3
"""Summarize per-benchmark perfstatistics reports.

Each benchmark case is expected to have its own directory containing
perfstatistics.log and, optionally, bench_metadata.txt written by bench_all.sh.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from model_latency_summary import parse_case_log_metadata, write_model_latency_summary


COMPUTE_CYCLES_RE = re.compile(r"\bcompute_cycles\s*=\s*([0-9][0-9,]*)")
MEMORY_READ_BYTES_RE = re.compile(r"\bmemory_read_bytes\s*=\s*([0-9][0-9,]*)")
MEMORY_WRITE_BYTES_RE = re.compile(r"\bmemory_write_bytes\s*=\s*([0-9][0-9,]*)")

MODEL_CONFIG_ARGS = (
    ("model_layers", "model_layers"),
    ("full_attn_layers", "full_attn_layers"),
    ("linear_attn_layers", "linear_attn_layers"),
    ("dense_ffn_layers", "dense_ffn_layers"),
    ("moe_ffn_layers", "moe_ffn_layers"),
    ("sampling_prefill_count", "sampling_prefill_count"),
    ("sampling_decode_count", "sampling_decode_count"),
)


def add_model_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-layers", dest="model_layers", type=int, help="Total transformer layers.")
    parser.add_argument("--full-attn-layers", dest="full_attn_layers", type=int, help="Full-attention layer count.")
    parser.add_argument("--linear-attn-layers", dest="linear_attn_layers", type=int, help="Linear-attention layer count.")
    parser.add_argument("--dense-ffn-layers", dest="dense_ffn_layers", type=int, help="Dense-FFN layer count.")
    parser.add_argument("--moe-ffn-layers", dest="moe_ffn_layers", type=int, help="MoE-FFN layer count.")
    parser.add_argument("--sampling-prefill-count", dest="sampling_prefill_count", type=int, help="Prefill sampling count.")
    parser.add_argument("--sampling-decode-count", dest="sampling_decode_count", type=int, help="Decode sampling count.")


def model_config_from_args(args: argparse.Namespace) -> dict[str, int] | None:
    config: dict[str, int] = {}
    for attr, key in MODEL_CONFIG_ARGS:
        value = getattr(args, attr)
        if value is not None:
            config[key] = value
    return config or None


def parse_metadata(path: Path) -> dict[str, str]:
    metadata_path = path / "bench_metadata.txt"
    metadata: dict[str, str] = {}
    if not metadata_path.is_file():
        return metadata

    for line in metadata_path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _parse_int_metric(pattern: re.Pattern[str], text: str) -> list[int]:
    return [int(m.group(1).replace(",", "")) for m in pattern.finditer(text)]


def parse_perf_metrics(log_path: Path) -> dict[str, list[int]]:
    text = log_path.read_text(errors="replace")
    return {
        "compute_cycles": _parse_int_metric(COMPUTE_CYCLES_RE, text),
        "memory_read_bytes": _parse_int_metric(MEMORY_READ_BYTES_RE, text),
        "memory_write_bytes": _parse_int_metric(MEMORY_WRITE_BYTES_RE, text),
    }


def shorten_executable(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if len(p.parts) <= 3:
        return path
    return str(Path(*p.parts[-3:]))


def discover_default_root() -> Path:
    bench_root = Path(".bench_logs")
    candidates = sorted(bench_root.glob("bench_*/perfstatistics"))
    if candidates:
        return candidates[-1]
    return Path(".")


def format_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    widths = {header: len(header) for header in headers}

    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def render(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[header]) for value, header in zip(values, headers))

    lines = [
        render(headers),
        render(["-" * widths[header] for header in headers]),
    ]
    for row in rows:
        lines.append(render([str(row.get(header, "")) for header in headers]))
    return "\n".join(lines)


def expand_deduped_summary_rows(
    rows: list[dict[str, object]],
    bench_out_dir: Path | None,
) -> list[dict[str, object]]:
    """Add skipped duplicate cases to the printed per-case table.

    `write_model_latency_summary` already expands deduped cases for model totals.
    This helper mirrors that behavior for the top-level perfstatistics table,
    while leaving the model-summary input unchanged to avoid double counting.
    """

    expanded = [dict(row) for row in rows]
    by_case = {str(row["case"]): row for row in expanded}
    if bench_out_dir is None or not bench_out_dir.is_dir():
        return expanded

    for log_path in sorted(bench_out_dir.glob("*.log")):
        metadata = parse_case_log_metadata(log_path)
        label = metadata.get("label")
        duplicate_of = metadata.get("dedupe_duplicate_of")
        if not label or not duplicate_of or label in by_case:
            continue
        source = by_case.get(duplicate_of)
        if source is None:
            continue
        copied = dict(source)
        copied["case"] = label
        copied["report_dir"] = f"deduped from {duplicate_of}"
        by_case[label] = copied
        expanded.append(copied)

    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="perfstatistics root(s) or individual report directories. Default: latest .bench_logs/bench_*/perfstatistics",
    )
    parser.add_argument("--ghz", type=float, default=1.5, help="Clock frequency for latency conversion. Default: 1.5")
    parser.add_argument("--peak-gbps", type=float, default=0.0, help="Peak memory bandwidth in GB/s for utilization calculation.")
    parser.add_argument("--tsv", action="store_true", help="Print TSV instead of a padded table.")
    parser.add_argument("--model-summary-dir", type=Path, help="Write model-level latency tables and SVG charts here.")
    parser.add_argument(
        "--bench-out-dir",
        type=Path,
        help="bench_all output directory used to expand deduped logical cases. Defaults to the perfstatistics parent.",
    )
    add_model_config_args(parser)
    args = parser.parse_args()

    roots = args.paths or [discover_default_root()]
    log_paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "perfstatistics.log":
            log_paths.append(root)
        elif (root / "perfstatistics.log").is_file():
            log_paths.append(root / "perfstatistics.log")
        elif root.is_dir():
            log_paths.extend(sorted(root.rglob("perfstatistics.log")))

    rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    latency_header = f"latency_us@{args.ghz:g}GHz"
    has_bandwidth = False
    bench_out_dir = args.bench_out_dir
    for log_path in sorted(set(log_paths)):
        report_dir = log_path.parent
        if bench_out_dir is None and report_dir.parent.name == "perfstatistics":
            bench_out_dir = report_dir.parent.parent
        metrics = parse_perf_metrics(log_path)
        cycles_list = metrics["compute_cycles"]
        if not cycles_list:
            continue

        metadata = parse_metadata(report_dir)
        label = metadata.get("label") or report_dir.name
        executable = shorten_executable(metadata.get("executable", ""))
        selected_cycles = cycles_list[-1]
        latency_us = selected_cycles / (args.ghz * 1000.0)

        read_bytes_list = metrics["memory_read_bytes"]
        write_bytes_list = metrics["memory_write_bytes"]
        read_bytes = read_bytes_list[-1] if read_bytes_list else 0
        write_bytes = write_bytes_list[-1] if write_bytes_list else 0
        total_bytes = read_bytes + write_bytes

        achieved_gbps = ""
        bw_util = ""
        if total_bytes > 0 and selected_cycles > 0:
            gbps = total_bytes * args.ghz / selected_cycles
            achieved_gbps = f"{gbps:.3f}"
            has_bandwidth = True
            if args.peak_gbps > 0:
                bw_util = f"{gbps / args.peak_gbps * 100.0:.2f}"

        row: dict[str, object] = {
            "case": label,
            "executable": executable or "-",
            "compute_cycles": selected_cycles,
            latency_header: f"{latency_us:.3f}",
            "read_bytes": read_bytes if total_bytes > 0 else "",
            "write_bytes": write_bytes if total_bytes > 0 else "",
            "total_bytes": total_bytes if total_bytes > 0 else "",
            "achieved_GBps": achieved_gbps,
            "report_dir": str(report_dir),
        }
        if args.peak_gbps > 0:
            row["bw_util%"] = bw_util
        rows.append(row)

        model_rows.append(
            {
                "case": label,
                "latency_us": latency_us,
                "cycles": selected_cycles,
                "source": str(report_dir),
            }
        )

    if not rows:
        print("No perfstatistics.log with compute_cycles was found.")
        return 1

    rows = expand_deduped_summary_rows(rows, bench_out_dir)
    rows.sort(key=lambda row: str(row["case"]))

    headers = ["case", "executable", "compute_cycles", latency_header]
    if has_bandwidth:
        headers.extend(["read_bytes", "write_bytes", "total_bytes", "achieved_GBps"])
        if args.peak_gbps > 0:
            headers.append("bw_util%")
    headers.append("report_dir")

    if has_bandwidth:
        total_read = sum(r["read_bytes"] for r in rows if isinstance(r["read_bytes"], int))
        total_write = sum(r["write_bytes"] for r in rows if isinstance(r["write_bytes"], int))
        total_all = total_read + total_write
        total_cycles = sum(r["compute_cycles"] for r in rows if isinstance(r.get("read_bytes"), int))
        summary_gbps = ""
        summary_util = ""
        if total_all > 0 and total_cycles > 0:
            gbps = total_all * args.ghz / total_cycles
            summary_gbps = f"{gbps:.3f}"
            if args.peak_gbps > 0:
                summary_util = f"{gbps / args.peak_gbps * 100.0:.2f}"
        total_row: dict[str, object] = {
            "case": "TOTAL",
            "executable": "",
            "compute_cycles": total_cycles,
            latency_header: f"{total_cycles / (args.ghz * 1000.0):.3f}",
            "read_bytes": total_read,
            "write_bytes": total_write,
            "total_bytes": total_all,
            "achieved_GBps": summary_gbps,
            "report_dir": "",
        }
        if args.peak_gbps > 0:
            total_row["bw_util%"] = summary_util
        rows.append(total_row)

    if args.tsv:
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(row.get(header, "")) for header in headers))
    else:
        print(format_table(rows, headers))

    if args.model_summary_dir:
        report_path, summary_text = write_model_latency_summary(
            model_rows,
            args.model_summary_dir,
            title="Perfstatistics Model Latency Summary",
            source_name="perfstatistics",
            bench_out_dir=bench_out_dir,
            model_config=model_config_from_args(args),
        )
        print()
        print(summary_text)
        print()
        print(f"Model latency summary: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
