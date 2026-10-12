#!/usr/bin/env python3
"""Compare per-case latency from two benchmark result directories.

Accepts benchmark OUT_DIRs, perfstatistics directories, NCU CSV directories,
or directories containing nsys_latency_summary.md / ncu_cycles_summary.md.
Writes a grouped bar chart with matplotlib (use --no-plot for standard-library
text/CSV output only). Does not run profilers or benchmarks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import re
import sys
import textwrap
from urllib.parse import quote


CYCLES_RE = re.compile(r"\bcompute_cycles\s*=\s*([0-9][0-9,]*)")


@dataclass
class Measurement:
    latency_us: float | None = None
    cycles: float | None = None
    status: str = "no measurement"
    source: str = ""


def number(value: object) -> float | None:
    try:
        result = float(str(value).strip().replace(",", ""))
    except ValueError:
        return None
    return result if math.isfinite(result) and result >= 0 else None


def fields(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            result.setdefault(key.strip(), value.strip())
    return result


def insert(rows: dict[str, Measurement], label: str, value: Measurement) -> None:
    if label in rows:
        raise ValueError(f"duplicate case {label!r}: {rows[label].source} and {value.source}; select one run directory")
    rows[label] = value


def read_perfstatistics(paths: list[Path], ghz: float) -> dict[str, Measurement]:
    rows = {}
    for path in paths:
        metadata_path = path.parent / "bench_metadata.txt"
        metadata = fields(metadata_path.read_text()) if metadata_path.is_file() else {}
        label = metadata.get("label") or path.parent.name
        values = CYCLES_RE.findall(path.read_text(errors="replace"))
        if values:
            cycles = int(values[-1].replace(",", ""))
            value = Measurement(cycles / (ghz * 1000), cycles, "ok", str(path))
        else:
            value = Measurement(status="no compute_cycles", source=str(path))
        insert(rows, label, value)
    return rows


def read_ncu(paths: list[Path], ghz: float) -> dict[str, Measurement]:
    from summarize_ncu_cycles import parse_case

    rows = {}
    for path in paths:
        kernels = parse_case(path)
        latencies = []
        cycles = []
        for kernel in kernels:
            duration_ns = number(kernel["duration_ns"])
            kernel_cycles = number(kernel["cycles_avg"])
            cycles.append(kernel_cycles)
            latencies.append(duration_ns / 1000 if duration_ns is not None else
                             kernel_cycles / (ghz * 1000) if kernel_cycles is not None else None)
        complete = bool(latencies) and all(value is not None for value in latencies)
        value = Measurement(
            math.fsum(latencies) if complete else None,
            math.fsum(cycles) if cycles and all(value is not None for value in cycles) else None,
            "ok" if complete else "incomplete NCU metrics", str(path),
        )
        insert(rows, path.stem, value)
    return rows


def read_summary(path: Path, ghz: float) -> dict[str, Measurement]:
    """Read the complete case aggregate, not model reports' truncated top lists."""
    rows = {}
    header = None
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if "case" in cells:
            header = cells if "latency_us" in cells else None
            continue
        if header is None or len(cells) != len(header) or all(re.fullmatch(r"[-:]+", cell) for cell in cells):
            continue
        record = dict(zip(header, cells))
        cycles = number(record.get("sum_cycles_avg"))
        duration_ns = number(record.get("sum_duration_ns"))
        latency = (duration_ns / 1000 if duration_ns is not None else
                   cycles / (ghz * 1000) if cycles is not None else number(record["latency_us"]))
        insert(rows, record["case"], Measurement(
            latency, cycles,
            "ok" if latency is not None else "invalid latency", str(path),
        ))
    if not rows:
        raise ValueError(f"no complete case aggregate with latency_us found in {path}")
    return rows


def apply_case_logs(rows: dict[str, Measurement], bench_dir: Path) -> None:
    aliases = {}
    manifest = bench_dir / "expected_cases.txt"
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            label = line.split("#", 1)[0].strip()
            if label:
                rows.setdefault(label, Measurement(status="not run"))
    for path in sorted(bench_dir.glob("*.log")):
        text = path.read_text(errors="replace")
        header, _, output = text.partition("---- output ----")
        info = fields(header)
        label = info.get("label")
        if not label:
            continue
        footer = fields(output)
        if (info.get("status") == "failed" or info.get("failed_at") or footer.get("failed_at")
                or footer.get("exit_status", info.get("exit_status", "0")) != "0"):
            rows[label] = Measurement(status="failed", source=str(path))
            continue
        duplicate = info.get("dedupe_duplicate_of") or info.get("duplicate_of")
        if info.get("status") == "skipped_duplicate" and duplicate:
            aliases[label] = duplicate
        rows.setdefault(label, Measurement(source=str(path)))

    def resolve(label: str, visiting: set[str]) -> Measurement:
        if label in visiting:
            raise ValueError(f"dedupe cycle involving {label!r} in {bench_dir}")
        if label not in aliases:
            return rows.get(label, Measurement(status="missing dedupe source"))
        source = resolve(aliases[label], visiting | {label})
        value = replace(source, source=f"deduped from {aliases[label]}")
        rows[label] = value
        return value

    for label in aliases:
        resolve(label, set())


def load_directory(root: Path, requested: str, ghz: float) -> tuple[str, dict[str, Measurement]]:
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    perf_root = root / "perfstatistics" if (root / "perfstatistics").is_dir() else root
    perf_paths = sorted(perf_root.rglob("perfstatistics.log"))
    ncu_root = root / "ncu" if (root / "ncu").is_dir() else root
    ncu_paths = sorted(ncu_root.glob("*.csv"))
    bench_dir = root.parent if root.name in {"perfstatistics", "ncu", "nsys"} else root
    summaries = {name: bench_dir / filename for name, filename in (
        ("ncu", "ncu_cycles_summary.md"), ("nsys", "nsys_latency_summary.md"),
    )}
    available = []
    if perf_paths:
        available.append("perfstatistics")
    if ncu_paths or summaries["ncu"].is_file():
        available.append("ncu")
    if summaries["nsys"].is_file():
        available.append("nsys")
    if requested == "auto":
        if root.name in available:
            source = root.name
        elif len(available) > 1:
            raise ValueError(f"multiple report sources in {root}: {', '.join(available)}; choose --source or a specific report directory")
        elif not available:
            raise ValueError(f"no supported reports in {root}; expected perfstatistics.log, NCU CSV, or an exported NSYS/NCU case summary")
        else:
            source = available[0]
    else:
        source = requested
        if source not in available:
            raise ValueError(f"no {source} reports in {root}")
    if source == "perfstatistics":
        rows = read_perfstatistics(perf_paths, ghz)
    elif source == "ncu" and ncu_paths:
        rows = read_ncu(ncu_paths, ghz)
    else:
        rows = read_summary(summaries[source], ghz)
    apply_case_logs(rows, bench_dir)
    return source, rows


def phase(label: str) -> str:
    if "prefill" in label:
        return "prefill"
    if "decode" in label or label.startswith("sampling_"):
        return "decode"
    return "other"


def compare(label: str, baseline: Measurement, candidate: Measurement) -> dict[str, object]:
    left, right = baseline.latency_us, candidate.latency_us
    complete = left is not None and right is not None
    delta = right - left if complete else None
    status = ("faster" if delta < 0 else "slower" if delta > 0 else "unchanged") if complete else (
        f"baseline: {baseline.status}; candidate: {candidate.status}"
    )
    return {
        "case": label, "baseline_us": left, "candidate_us": right, "delta_us": delta,
        "delta_pct": delta / left * 100 if complete and left > 0 else None,
        "speedup": left / right if complete and right > 0 else None, "status": status,
        "baseline_cycles": baseline.cycles, "candidate_cycles": candidate.cycles,
        "baseline_source": baseline.source, "candidate_source": candidate.source,
    }


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, int)):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def positive_ghz(value: str) -> float:
    result = number(value)
    if result is None or result <= 0:
        raise argparse.ArgumentTypeError("frequency must be finite and greater than zero")
    return result


def write_bar_chart(rows: list[dict[str, object]], total: dict[str, object] | None,
                    path: Path, baseline_ghz: float, candidate_ghz: float) -> None:
    """Plot the same converted latencies as the table, preserving missing values."""
    if path.suffix.lower() not in {".svg", ".png", ".pdf"}:
        raise ValueError("--plot must have an .svg, .png, or .pdf extension")
    try:
        import matplotlib
    except ImportError as exc:
        raise ValueError("bar charts require matplotlib; install it with "
                         "'python3 -m pip install matplotlib', or use --no-plot") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [rows[start:start + 10] for start in range(0, len(rows), 10)]
    panels = [(group, "Case latency" if len(groups) == 1 else f"Case latency ({i + 1}/{len(groups)})")
              for i, group in enumerate(groups)]
    if total is not None:
        panels.append(([total], "Total of common cases (no layer multipliers)"))
    series = (("baseline_us", f"Baseline · {baseline_ghz:g} GHz", "#4E79A7", -0.2),
              ("candidate_us", f"Candidate · {candidate_ghz:g} GHz", "#F28E2B", 0.2))
    with plt.rc_context({"svg.fonttype": "none", "text.usetex": False}):
        fig, axes = plt.subplots(len(panels), 1, figsize=(max(9, min(len(rows), 10) * 1.6), 5 * len(panels)),
                                 squeeze=False, constrained_layout=True)
        try:
            for ax, (group, title) in zip(axes[:, 0], panels):
                positions = list(range(len(group)))
                for key, legend, color, offset in series:
                    values = [row[key] for row in group]
                    x = [position + offset for position in positions]
                    bars = ax.bar(x, [value if value is not None else float("nan") for value in values],
                                  width=0.36, label=legend, color=color)
                    ax.bar_label(bars, labels=[f"{value:.3f}" if value is not None else "" for value in values],
                                 padding=3, fontsize=8)
                    for position, value in zip(x, values):
                        if value is None:
                            ax.annotate("N/A", (position, 0), xytext=(0, 4), textcoords="offset points",
                                        ha="center", va="bottom", color=color, fontsize=8)
                labels = []
                for row in group:
                    words = str(row["case"]).replace("_", "_ ")
                    labels.append("\n".join(line.replace("_ ", "_") for line in textwrap.wrap(words, width=26)))
                ax.set_xticks(positions, labels, fontsize=8)
                ax.set_xlim(-0.6, len(group) - 0.4)
                maximum = max((row[key] for row in group for key, *_ in series if row[key] is not None), default=0)
                ax.set_ylim(0, maximum * 1.22 if maximum > 0 else 1)
                ax.set_ylabel("Latency (µs)")
                ax.set_title(title + " — lower is faster", pad=36)
                ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", alpha=0.25)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160, format=path.suffix[1:].lower())
        finally:
            plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--source", choices=("auto", "perfstatistics", "ncu", "nsys"), default="auto")
    parser.add_argument("--baseline-ghz", type=positive_ghz, default=1.5, help="cycle-to-us frequency for baseline (default: 1.5)")
    parser.add_argument("--candidate-ghz", type=positive_ghz, default=1.5, help="cycle-to-us frequency for candidate (default: 1.5)")
    parser.add_argument("--case", action="append", default=[], help="case-name substring filter; repeatable")
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--csv", type=Path, help="also write CSV, including cycles and report paths")
    parser.add_argument("-o", "--output", type=Path, help="also save the Markdown comparison")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--plot", type=Path, help="bar chart path (.svg/.png/.pdf; default: comparison.svg, or beside --output)")
    plots.add_argument("--no-plot", action="store_true", help="skip the chart; no matplotlib dependency")
    args = parser.parse_args()
    try:
        left_source, left = load_directory(args.baseline_dir, args.source, args.baseline_ghz)
        right_source, right = load_directory(args.candidate_dir, args.source, args.candidate_ghz)
        labels = sorted(label for label in left.keys() | right.keys()
                        if (not args.case or any(part in label for part in args.case))
                        and (args.phase == "all" or phase(label) == args.phase))
        if not labels:
            raise ValueError("no cases match the selected filters")
        missing = Measurement(status="missing case")
        rows = [compare(label, left.get(label, missing), right.get(label, missing)) for label in labels]
        common = [row for row in rows if row["delta_us"] is not None]
        rows.sort(key=lambda row: (row["delta_us"] is None, -abs(row["delta_us"] or 0), row["case"]))
        total = None
        if common:
            total = compare("TOTAL_COMMON", Measurement(math.fsum(row["baseline_us"] for row in common)),
                            Measurement(math.fsum(row["candidate_us"] for row in common)))
        plot_path = None if args.no_plot else args.plot or (args.output.with_suffix(".svg") if args.output else Path("comparison.svg"))
        output_paths = [path.resolve() for path in (plot_path, args.output, args.csv) if path is not None]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("chart, Markdown, and CSV output paths must be different")
        if plot_path is not None:
            write_bar_chart(rows, total, plot_path, args.baseline_ghz, args.candidate_ghz)
        if total is not None:
            rows.append(total)
        headers = ["case", "baseline_us", "candidate_us", "delta_us", "delta_pct", "speedup", "status"]
        lines = [
            "# Benchmark directory comparison", "",
            f"Baseline: `{args.baseline_dir.resolve()}` ({left_source}; cycle conversion {args.baseline_ghz:g} GHz).",
            f"Candidate: `{args.candidate_dir.resolve()}` ({right_source}; cycle conversion {args.candidate_ghz:g} GHz).", "",
            "delta = candidate - baseline; negative delta is faster. speedup = baseline / candidate; >1 is faster.",
            f"Comparable cases: {len(common)}/{len(labels)}. TOTAL_COMMON sums these logical cases once each; no layer multipliers.",
            "Missing/failed cases are excluded from totals; undefined ratios are shown as '-'.", "",
        ]
        if plot_path is not None:
            report_dir = args.output.resolve().parent if args.output else Path.cwd()
            plot_link = quote(Path(os.path.relpath(plot_path.resolve(), report_dir)).as_posix())
            prefix = "" if plot_path.suffix.lower() == ".pdf" else "!"
            lines.extend([f"Bar chart: `{plot_path.resolve()}`. Missing measurements are marked N/A.", "",
                          f"{prefix}[Case latency comparison]({plot_link})", ""])
        lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        lines.extend("| " + " | ".join(fmt(row[key]) for key in headers) + " |" for row in rows)
        report = "\n".join(lines) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(report, end="")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
