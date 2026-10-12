#!/usr/bin/env python3
"""Compare model latency from two benchmark result directories.

Accepts benchmark OUT_DIRs, perfstatistics directories, NCU CSV directories,
or directories containing nsys_latency_summary.md / ncu_cycles_summary.md.
Writes a horizontal grouped bar chart with matplotlib (use --no-plot for standard-library
text/CSV output only). Does not run profilers or benchmarks.
Model call counts and operator groups match summarize_perfstatistics.py.
Use --mode raw for an unweighted per-case comparison.
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
from urllib.parse import quote

from model_latency_summary import (
    DEFAULT_MODEL_CONFIG, aggregate_cases, classify_module, classify_operator,
    classify_phase, compute_model_summary, expand_to_model_estimate_cases,
    load_case_calls, operator_groups,
)
from summarize_perfstatistics import (
    CASE_REPLACEMENTS, add_model_config_args, describe_missing_case,
    expected_cases_from_script, model_config_from_args, parse_case_list, scan_case_logs,
)


CYCLES_RE = re.compile(r"\bcompute_cycles\s*=\s*([0-9][0-9,]*)")


@dataclass
class Measurement:
    latency_us: float | None = None
    cycles: float | None = None
    status: str = "no measurement"
    source: str = ""
    calls: int | None = 1
    base_latency_us: float | None = None


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


def load_directory(root: Path, requested: str, ghz: float, *, apply_logs: bool = True) -> tuple[str, dict[str, Measurement]]:
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
    if apply_logs:
        apply_case_logs(rows, bench_dir)
    return source, rows


def bench_directory(root: Path) -> Path:
    return root.parent if root.name in {"perfstatistics", "ncu", "nsys"} else root


def model_measurements(rows: dict[str, Measurement], root: Path, selected_phase: str,
                       config: dict[str, int], case_calls: dict,
                       expected: list[str] | None = None) -> dict[str, Measurement]:
    """Use the same input selection, dedupe, and model expansion as the summary."""
    bench_dir = bench_directory(root)
    logs = scan_case_logs(bench_dir)
    manifest = bench_dir / "expected_cases.txt"
    if expected is None:
        expected = parse_case_list(manifest.read_text()) if manifest.is_file() else list(logs)
    superseded = {old for old, new in CASE_REPLACEMENTS.items() if new in expected}
    inputs = [
        {"case": label, "latency_us": value.latency_us, "cycles": value.cycles, "source": value.source}
        for label, value in rows.items() if value.latency_us is not None and label not in superseded
        and (selected_phase == "all" or classify_phase(label) == selected_phase or classify_module(label) == "Sampling")
    ]
    effective_config = dict(config)
    if selected_phase in {"prefill", "decode"}:
        effective_config["sampling_decode_count" if selected_phase == "prefill" else "sampling_prefill_count"] = 0
    computed = compute_model_summary(inputs, bench_dir, effective_config, case_calls)
    result = {}
    for row in computed["model_summary"]["cases"]:
        if selected_phase != "all" and row["phase"] != selected_phase:
            continue
        result[str(row["case"])] = Measurement(
            row["latency_us"], row.get("cycles"), "ok", row.get("source", ""),
            row["multiplier"], row["base_latency_us"],
        )
    missing_inputs = [{"case": label, "latency_us": 0.0} for label in sorted(set(expected) | set(rows))
                      if label not in superseded]
    for row in expand_to_model_estimate_cases(missing_inputs, effective_config, case_calls):
        if selected_phase != "all" and row["phase"] != selected_phase:
            continue
        label, base = str(row["case"]), str(row["base_case"])
        result.setdefault(label, Measurement(status=describe_missing_case(logs.get(base)),
                                            source=rows.get(base, Measurement()).source, calls=row["multiplier"]))
    return result


def infer_phase(left: dict[str, Measurement], right: dict[str, Measurement]) -> str:
    phases = {classify_phase(label) for values in (left, right) for label, value in values.items()
              if value.latency_us is not None and classify_module(label) != "Sampling"}
    phases.discard("unknown")
    return next(iter(phases)) if len(phases) == 1 else "all"


def phase(label: str) -> str:
    return classify_phase(label)


def compare(label: str, baseline: Measurement, candidate: Measurement) -> dict[str, object]:
    left, right = baseline.latency_us, candidate.latency_us
    complete = left is not None and right is not None
    delta = right - left if complete else None
    status = ("faster" if delta < 0 else "slower" if delta > 0 else "unchanged") if complete else (
        f"baseline: {baseline.status}; candidate: {candidate.status}"
    )
    return {
        "case": label, "baseline_us": left, "candidate_us": right, "delta_us": delta,
        "baseline_ms": left / 1000 if left is not None else None,
        "candidate_ms": right / 1000 if right is not None else None,
        "delta_pct": delta / left * 100 if complete and left > 0 else None,
        "speedup": left / right if complete and right > 0 else None, "status": status,
        "baseline_cycles": baseline.cycles, "candidate_cycles": candidate.cycles,
        "baseline_source": baseline.source, "candidate_source": candidate.source,
        "baseline_calls": baseline.calls, "candidate_calls": candidate.calls,
        "baseline_base_us": baseline.base_latency_us, "candidate_base_us": candidate.base_latency_us,
    }


def measured_sum(values: dict[str, Measurement], labels: list[str]) -> Measurement:
    measured = [values[label].latency_us for label in labels if label in values and values[label].latency_us is not None]
    return Measurement(math.fsum(measured) if measured else None,
                       status=f"{len(measured)}/{len(labels)} cases measured", calls=None)


def compare_total(label: str, labels: list[str], left: dict[str, Measurement],
                  right: dict[str, Measurement]) -> dict[str, object]:
    result = compare(label, measured_sum(left, labels), measured_sum(right, labels))
    left_cases = {label for label in labels if label in left and left[label].latency_us is not None}
    right_cases = {label for label in labels if label in right and right[label].latency_us is not None}
    if left_cases != right_cases:
        result.update(delta_pct=None, speedup=None,
                      status=f"different coverage: baseline {len(left_cases)}/{len(labels)}, candidate {len(right_cases)}/{len(labels)}")
    elif len(left_cases) < len(labels):
        result["status"] = f"partial coverage: {len(left_cases)}/{len(labels)} cases measured"
    return result


def operator_comparisons(labels: list[str], left: dict[str, Measurement],
                         right: dict[str, Measurement], selected_phase: str) -> list[dict[str, object]]:
    """Use the model report's operator classification and aggregation."""
    result = []
    for selected in ("prefill", "decode", "unknown"):
        phase_labels = [label for label in labels if phase(label) == selected]
        summaries = []
        for values in (left, right):
            cases = [{"case": label, "latency_us": values[label].latency_us} for label in phase_labels
                     if label in values and values[label].latency_us is not None]
            summaries.append({str(row["operator"]): row["latency_us"]
                              for row in operator_groups(aggregate_cases(cases), selected)})
        for operator in sorted({classify_operator(label) for label in phase_labels}):
            members = [label for label in phase_labels if classify_operator(label) == operator]
            name = f"{selected} / {operator}" if selected_phase == "all" else operator
            coverage = compare_total(name, members, left, right)
            row = compare(name, Measurement(summaries[0].get(operator), calls=None),
                          Measurement(summaries[1].get(operator), calls=None))
            row["status"] = coverage["status"]
            for key in ("delta_pct", "speedup"):
                if coverage[key] is None:
                    row[key] = None
            result.append(row)
    return result


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def positive_ghz(value: str) -> float:
    result = number(value)
    if result is None or result <= 0:
        raise argparse.ArgumentTypeError("frequency must be finite and greater than zero")
    return result


def write_bar_chart(rows: list[dict[str, object]], totals: list[dict[str, object]],
                    path: Path, baseline_ghz: float, candidate_ghz: float,
                    title: str, total_title: str) -> None:
    """Plot converted latencies horizontally, with the largest pair at the top."""
    if path.suffix.lower() not in {".svg", ".png", ".pdf"}:
        raise ValueError("--plot must have an .svg, .png, or .pdf extension")
    try:
        import matplotlib
    except ImportError as exc:
        raise ValueError("bar charts require matplotlib; install it with "
                         "'python3 -m pip install matplotlib', or use --no-plot") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: (
        -max((row[key] for key in ("baseline_us", "candidate_us") if row[key] is not None), default=-1),
        str(row["case"]),
    ))
    panels = [(ordered, title)]
    if totals:
        panels.append((totals, total_title))
    series = (("baseline_us", f"Baseline · {baseline_ghz:g} GHz", "#4E79A7", -0.2),
              ("candidate_us", f"Candidate · {candidate_ghz:g} GHz", "#F28E2B", 0.2))
    heights = [max(3.4, 0.6 * len(group) + 1.4) for group, _ in panels]
    with plt.rc_context({"svg.fonttype": "none", "text.usetex": False}):
        fig, axes = plt.subplots(len(panels), 1, figsize=(12.5, sum(heights)),
                                 gridspec_kw={"height_ratios": heights},
                                 squeeze=False, constrained_layout=True)
        try:
            for ax, (group, title) in zip(axes[:, 0], panels):
                positions = list(range(len(group)))
                for key, legend, color, offset in series:
                    values = [row[key] for row in group]
                    y = [position + offset for position in positions]
                    bars = ax.barh(y, [value if value is not None else float("nan") for value in values],
                                   height=0.36, label=legend, color=color, edgecolor="white", linewidth=0.8)
                    ax.bar_label(bars, labels=[f"{value:.3f}" if value is not None else "" for value in values],
                                 padding=3, fontsize=8)
                    for position, value in zip(y, values):
                        if value is None:
                            ax.annotate("N/A", (0, position), xytext=(4, 0), textcoords="offset points",
                                        ha="left", va="center", color=color, fontsize=8)
                ax.set_yticks(positions, [str(row["case"]) for row in group], fontsize=8)
                ax.set_ylim(len(group) - 0.4, -0.6)
                maximum = max((row[key] for row in group for key, *_ in series if row[key] is not None), default=0)
                ax.set_xlim(0, maximum * 1.22 if maximum > 0 else 1)
                ax.set_xlabel("Latency (µs)")
                ax.set_title(title + " — lower is faster", pad=36, fontsize=14, fontweight="bold")
                ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
                ax.set_axisbelow(True)
                ax.grid(axis="x", alpha=0.25)
                ax.spines[["top", "right", "left"]].set_visible(False)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180, format=path.suffix[1:].lower())
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
    parser.add_argument("--phase", choices=("auto", "all", "prefill", "decode"), default="auto",
                        help="default: infer a single measured phase in model mode, otherwise show phases separately")
    parser.add_argument("--mode", choices=("model", "raw"), default="model", help="default: model call-weighted latency; raw keeps per-call latency")
    parser.add_argument("--group-by", choices=("operator", "case"), help="chart grouping (default: operator for model, case for raw)")
    parser.add_argument("--bench-script", type=Path, help="expected cases from SCRIPT --list, as in summarize_perfstatistics.py")
    parser.add_argument("--expected-cases", type=Path, help="shared expected-case list; otherwise each directory's manifest is used")
    parser.add_argument("--case-calls", action="append", default=[], metavar="LABEL[@PHASE]=N", help="model call-count override; repeatable")
    parser.add_argument("--calls-file", type=Path, help="model call-count overrides, as in summarize_perfstatistics.py")
    add_model_config_args(parser)
    parser.add_argument("--csv", type=Path, help="also write CSV, including cycles and report paths")
    parser.add_argument("-o", "--output", type=Path, help="also save the Markdown comparison")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--plot", type=Path, help="bar chart path (.svg/.png/.pdf; default: comparison.svg, or beside --output)")
    plots.add_argument("--no-plot", action="store_true", help="skip the chart; no matplotlib dependency")
    args = parser.parse_args()
    try:
        left_source, left = load_directory(args.baseline_dir, args.source, args.baseline_ghz, apply_logs=args.mode == "raw")
        right_source, right = load_directory(args.candidate_dir, args.source, args.candidate_ghz, apply_logs=args.mode == "raw")
        selected_phase = args.phase
        if selected_phase == "auto":
            selected_phase = infer_phase(left, right) if args.mode == "model" else "all"
        config = {**DEFAULT_MODEL_CONFIG, **(model_config_from_args(args) or {})}
        case_calls = load_case_calls(args.case_calls, args.calls_file)
        if args.mode == "model":
            expected = (parse_case_list(args.expected_cases.read_text()) if args.expected_cases else
                        expected_cases_from_script(args.bench_script) if args.bench_script else None)
            left = model_measurements(left, args.baseline_dir, selected_phase, config, case_calls, expected)
            right = model_measurements(right, args.candidate_dir, selected_phase, config, case_calls, expected)
        labels = sorted(label for label in left.keys() | right.keys()
                        if (not args.case or any(part in label for part in args.case))
                        and (selected_phase == "all" or phase(label) == selected_phase))
        if not labels:
            raise ValueError("no cases match the selected filters")
        missing = Measurement(status="missing case", calls=None)
        rows = [compare(label, left.get(label, missing), right.get(label, missing)) for label in labels]
        common = [row for row in rows if row["delta_us"] is not None]
        rows.sort(key=lambda row: (row["delta_us"] is None, -abs(row["delta_us"] or 0), row["case"]))
        totals, common_totals = [], []
        for selected in (("prefill", "decode", "unknown") if args.mode == "model" else ("all",)):
            phase_labels = [label for label in labels if selected == "all" or phase(label) == selected]
            if not phase_labels:
                continue
            suffix = f"_{selected}" if args.mode == "model" else ""
            totals.append(compare_total(f"TOTAL{suffix}" if suffix else "TOTAL_AVAILABLE", phase_labels, left, right))
            common_labels = [str(row["case"]) for row in common if row["case"] in phase_labels]
            if common_labels:
                common_totals.append(compare_total(f"TOTAL_COMMON{suffix}", common_labels, left, right))
        grouping = args.group_by or ("operator" if args.mode == "model" else "case")
        chart_rows = operator_comparisons(labels, left, right, selected_phase) if grouping == "operator" else rows
        plot_path = None if args.no_plot else args.plot or (args.output.with_suffix(".svg") if args.output else Path("comparison.svg"))
        output_paths = [path.resolve() for path in (plot_path, args.output, args.csv) if path is not None]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("chart, Markdown, and CSV output paths must be different")
        if plot_path is not None:
            title = f"{selected_phase.capitalize()} {grouping} latency" if selected_phase != "all" else f"{grouping.capitalize()} latency by phase"
            total_title = "Model latency by phase (each directory's measured cases)" if args.mode == "model" else "Per-call subtotal (each directory's measured cases)"
            write_bar_chart(chart_rows, totals, plot_path, args.baseline_ghz, args.candidate_ghz, title, total_title)
        headers = ["case", "baseline_us", "candidate_us", "delta_us", "delta_pct", "speedup", "status"]
        lines = [
            "# Benchmark directory comparison", "",
            f"Baseline: `{args.baseline_dir.resolve()}` ({left_source}; cycle conversion {args.baseline_ghz:g} GHz).",
            f"Candidate: `{args.candidate_dir.resolve()}` ({right_source}; cycle conversion {args.candidate_ghz:g} GHz).", "",
            f"Mode: **{args.mode}**; phase: **{selected_phase}**; chart grouping: **{grouping}**. Units: µs (totals also shown in ms).", "",
            "delta = candidate - baseline; negative delta is faster. speedup = baseline / candidate; >1 is faster.",
            f"Comparable cases: {len(common)}/{len(labels)}. TOTAL uses each directory's measured cases; TOTAL_COMMON uses only cases measured on both sides.",
            "Unmeasured cases are excluded. Percentage change and speedup are omitted when case coverage differs; undefined ratios are shown as '-'.", "",
        ]
        if args.mode == "model":
            lines.extend(["Model counts (same defaults and aggregation as summarize_perfstatistics.py): "
                          + ", ".join(f"{key}={value}" for key, value in config.items())
                          + f"; per-case overrides={len(case_calls)}.", ""])
        if args.case:
            lines.extend([f"Case filters: {', '.join(args.case)}. Totals cover only these selected cases.", ""])
        if plot_path is not None:
            report_dir = args.output.resolve().parent if args.output else Path.cwd()
            plot_link = quote(Path(os.path.relpath(plot_path.resolve(), report_dir)).as_posix())
            prefix = "" if plot_path.suffix.lower() == ".pdf" else "!"
            lines.extend([f"Bar chart: `{plot_path.resolve()}`. Missing measurements are marked N/A.", "",
                          f"{prefix}[Case latency comparison]({plot_link})", ""])
        def add_table(title: str, table_rows: list[dict[str, object]], columns: list[str]) -> None:
            lines.extend([title, "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"])
            lines.extend("| " + " | ".join(fmt(row[key]) for key in columns) + " |" for row in table_rows)
            lines.append("")

        add_table("## Totals", totals + common_totals, ["case", "baseline_us", "candidate_us", "baseline_ms", "candidate_ms", "speedup", "status"])
        if grouping == "operator":
            add_table("## Operator contributions", chart_rows, headers)
        add_table("## Case contributions", rows, headers[:3] + ["baseline_calls", "candidate_calls"] + headers[3:])
        report = "\n".join(lines) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows + totals + common_totals)
        print(report, end="")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
