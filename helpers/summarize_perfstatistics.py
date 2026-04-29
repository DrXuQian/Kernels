#!/usr/bin/env python3
"""Summarize per-benchmark perfstatistics reports.

Each benchmark case is expected to have its own directory containing
perfstatistics.log and, optionally, bench_metadata.txt written by bench_all.sh.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


COMPUTE_CYCLES_RE = re.compile(r"\bcompute_cycles\s*=\s*([0-9][0-9,]*)")


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


def parse_compute_cycles(log_path: Path) -> list[int]:
    text = log_path.read_text(errors="replace")
    cycles: list[int] = []
    for match in COMPUTE_CYCLES_RE.finditer(text):
        cycles.append(int(match.group(1).replace(",", "")))
    return cycles


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


def format_table(rows: list[dict[str, object]], latency_header: str) -> str:
    headers = ["case", "executable", "compute_cycles", latency_header, "report_dir"]
    widths = {header: len(header) for header in headers}

    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    def render(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[header]) for value, header in zip(values, headers))

    lines = [
        render(headers),
        render(["-" * widths[header] for header in headers]),
    ]
    for row in rows:
        lines.append(render([str(row[header]) for header in headers]))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="perfstatistics root(s) or individual report directories. Default: latest .bench_logs/bench_*/perfstatistics",
    )
    parser.add_argument("--ghz", type=float, default=1.5, help="Clock frequency for latency conversion. Default: 1.5")
    parser.add_argument("--tsv", action="store_true", help="Print TSV instead of a padded table.")
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
    latency_header = f"latency_us@{args.ghz:g}GHz"
    for log_path in sorted(set(log_paths)):
        report_dir = log_path.parent
        cycles = parse_compute_cycles(log_path)
        if not cycles:
            continue

        metadata = parse_metadata(report_dir)
        label = metadata.get("label") or report_dir.name
        executable = shorten_executable(metadata.get("executable", ""))
        selected_cycles = cycles[-1]
        latency_us = selected_cycles / (args.ghz * 1000.0)

        rows.append(
            {
                "case": label,
                "executable": executable or "-",
                "compute_cycles": selected_cycles,
                latency_header: f"{latency_us:.3f}",
                "report_dir": str(report_dir),
            }
        )

    if not rows:
        print("No perfstatistics.log with compute_cycles was found.")
        return 1

    rows.sort(key=lambda row: str(row["case"]))

    if args.tsv:
        headers = ["case", "executable", "compute_cycles", latency_header, "report_dir"]
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(row[header]) for header in headers))
    else:
        print(format_table(rows, latency_header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
