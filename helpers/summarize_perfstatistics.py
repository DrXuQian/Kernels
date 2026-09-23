#!/usr/bin/env python3
"""Summarize per-benchmark perfstatistics reports.

Each benchmark case is expected to have its own directory containing
perfstatistics.log and, optionally, bench_metadata.txt written by bench_all.sh.

Each case measures one kernel call. The table also reports how many times that
kernel is called in one model forward of the case's phase (`calls`) and the
resulting model-level latency (`model_latency_us` = latency_us x calls). The
count follows the bench script case list: every case runs once per layer of
its module (--full-attn-layers, --linear-attn-layers, --dense-ffn-layers,
--moe-ffn-layers) and sampling cases use the sampling counts. Sampling runs
after prefill and after every decode step, so each sampling case is listed
twice, as `<case>__prefill` and `<case>__decode`. When bench_all.sh dedupes
identical commands, the logical cases that share one measured kernel are
listed as `deduped from <case>` rows, and the measured row also shows
`kernel_calls` / `kernel_model_latency_us`, the totals over every logical
case that kernel serves. The TOTAL_prefill / TOTAL_decode / TOTAL rows weight
every quantity by calls: cycles, latency, and bytes are summed as
value x calls, and achieved_GBps is the call-weighted bytes over the
call-weighted cycles of the cases that report bytes. Override single cases
with --case-calls LABEL[@prefill|@decode]=N or a --calls-file with one entry
per line; the model summary applies the same call counts.

Cases without a report are listed as `not run: ...` rows and excluded from
every total. The bench scripts write the selected labels to
`<OUT_DIR>/expected_cases.txt` before running; pass --expected-cases FILE or
--bench-script SCRIPT (its `--list` output) for older runs. Per-case logs in
the bench output directory tell whether a case failed, finished without a
perfstatistics report, or never started.

--scale SEQ[:USED] projects the measured latencies to SEQ prefill tokens and
a KV cache of USED tokens: in prefill every kernel except the attention core
scales linearly with SEQ and the attention core scales with SEQ x USED (SEQ
squared when USED = SEQ); in decode only the attention core changes, linearly
with USED. Sampling never scales. --measured-seq-len / --measured-used-len
give the lengths the benchmarks ran with (PREFILL_TOKENS / CTX_LEN).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from model_latency_summary import (
    DEFAULT_MODEL_CONFIG,
    case_call_count,
    classify_module,
    classify_phase,
    load_case_calls,
    strip_phase_suffix,
    write_model_latency_summary,
)


COMPUTE_CYCLES_RE = re.compile(r"\bcompute_cycles\s*=\s*([0-9][0-9,]*)")
MEMORY_READ_BYTES_RE = re.compile(r"\bmemory_read_bytes\s*=\s*([0-9][0-9,]*)")
MEMORY_WRITE_BYTES_RE = re.compile(r"\bmemory_write_bytes\s*=\s*([0-9][0-9,]*)")

CASE_LOG_HEADER_KEYS = ("label", "status", "duplicate_of", "dedupe_duplicate_of", "command", "executable")
CASE_LOG_FOOTER_KEYS = ("finished_at", "failed_at", "exit_status")
CASE_LOG_OUTPUT_MARKER = "---- output ----"

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


def parse_scale_arg(value: str) -> tuple[int, int]:
    seq_text, _, used_text = value.strip().partition(":")
    try:
        seq_len = int(seq_text)
        used_len = int(used_text) if used_text else seq_len
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected SEQ[:USED] with integers, got {value!r}") from None
    if seq_len <= 0 or used_len <= 0:
        raise argparse.ArgumentTypeError("SEQ and USED must be positive")
    return seq_len, used_len


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


def print_rows(rows: list[dict[str, object]], headers: list[str], tsv: bool) -> None:
    if tsv:
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(row.get(header, "")) for header in headers))
    else:
        print(format_table(rows, headers))


# --- case logs, expected cases, and missing-case detection -------------------


def parse_case_log_status(path: Path) -> dict[str, str]:
    """Read the bench script's per-case log header and completion footer."""

    info: dict[str, str] = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return info
    header, marker, output = text.partition(CASE_LOG_OUTPUT_MARKER)
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in CASE_LOG_HEADER_KEYS and key not in info:
            info[key] = value.strip()
    if marker:
        for line in output.splitlines():
            for key in CASE_LOG_FOOTER_KEYS:
                if line.startswith(key + ":"):
                    info[key] = line.split(":", 1)[1].strip()
    if "duplicate_of" in info and "dedupe_duplicate_of" not in info:
        info["dedupe_duplicate_of"] = info["duplicate_of"]
    return info


def scan_case_logs(bench_out_dir: Path | None) -> dict[str, dict[str, str]]:
    logs: dict[str, dict[str, str]] = {}
    if bench_out_dir is None or not bench_out_dir.is_dir():
        return logs
    for log_path in sorted(bench_out_dir.glob("*.log")):
        info = parse_case_log_status(log_path)
        label = info.get("label")
        if label:
            info["log"] = str(log_path)
            logs[label] = info
    return logs


def parse_case_list(text: str) -> list[str]:
    """Labels from expected_cases.txt or a bench script's --list output."""

    labels: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.endswith(":"):
            continue
        parts = line.split()
        if parts[0] in {"ok", "missing"} and len(parts) >= 2:
            label = parts[1]
        elif parts[0] in {"binary", "------"} or len(parts) != 1:
            continue
        else:
            label = parts[0]
        if label not in labels:
            labels.append(label)
    return labels


def expected_cases_from_script(script: Path) -> list[str]:
    result = subprocess.run([str(script.resolve()), "--list"], capture_output=True, text=True, check=False)
    labels = parse_case_list(result.stdout)
    if not labels:
        raise ValueError(f"{script} --list printed no case labels (exit status {result.returncode})")
    return labels


def describe_missing_case(info: dict[str, str] | None) -> str:
    if info is None:
        return "not run: no case log"
    if info.get("status") == "skipped_duplicate":
        return f"not run: deduped from {info.get('dedupe_duplicate_of', '?')}, which has no report"
    if "exit_status" in info or "failed_at" in info:
        return f"not run: benchmark failed (exit_status={info.get('exit_status', '?')})"
    if "finished_at" in info:
        return "not run: benchmark finished but no perfstatistics report"
    return "not run: benchmark did not finish (interrupted?)"


def expand_deduped_summary_rows(
    rows: list[dict[str, object]],
    case_logs: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    """Add skipped duplicate cases to the printed per-case table.

    `write_model_latency_summary` already expands deduped cases for model totals.
    This helper mirrors that behavior for the top-level perfstatistics table,
    while leaving the model-summary input unchanged to avoid double counting.
    """

    expanded = [dict(row) for row in rows]
    by_case = {str(row["case"]): row for row in expanded}
    for label, info in case_logs.items():
        duplicate_of = info.get("dedupe_duplicate_of")
        if not duplicate_of or label in by_case:
            continue
        source = by_case.get(duplicate_of)
        if source is None:
            continue
        copied = dict(source)
        copied["case"] = label
        copied["report_dir"] = f"deduped from {duplicate_of}"
        copied["_kernel_source"] = duplicate_of
        by_case[label] = copied
        expanded.append(copied)
    return expanded


def split_sampling_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """List every sampling case once per phase; sampling runs after prefill too."""

    result: list[dict[str, object]] = []
    renamed: dict[str, str] = {}
    for row in rows:
        label = str(row["case"])
        if row.get("_missing") or classify_module(label) != "Sampling":
            result.append(row)
            continue
        decode_row = dict(row)
        decode_row["case"] = f"{label}__decode"
        prefill_row = dict(row)
        prefill_row["case"] = f"{label}__prefill"
        prefill_row["_kernel_source"] = str(row.get("_kernel_source") or decode_row["case"])
        prefill_row["report_dir"] = f"sampling after prefill, measured as {decode_row['case']}"
        renamed[label] = str(decode_row["case"])
        result.extend([decode_row, prefill_row])
    for row in result:
        source = row.get("_kernel_source")
        if isinstance(source, str) and source in renamed:
            row["_kernel_source"] = renamed[source]
    return result


def aggregate_kernel_calls(rows: list[dict[str, object]]) -> None:
    """Sum calls of every logical row that shares one measured kernel.

    Deduped rows and prefill-sampling rows carry `_kernel_source`; the measured
    row gets `kernel_calls` and `kernel_model_latency_us` covering all of them.
    """

    measured = {str(row["case"]): row for row in rows if "_kernel_source" not in row and not row.get("_missing")}
    for row in measured.values():
        row["_kernel_calls"] = int(row["calls"])
        row["_kernel_model_latency_us"] = float(row["_model_latency_us"])
    for row in rows:
        if row.get("_missing"):
            continue
        source = measured.get(str(row.get("_kernel_source", "")))
        if source is None:
            continue
        source["_kernel_calls"] = int(source["_kernel_calls"]) + int(row["calls"])
        source["_kernel_model_latency_us"] = float(source["_kernel_model_latency_us"]) + float(row["_model_latency_us"])
    for row in rows:
        if "_kernel_calls" in row:
            row["kernel_calls"] = row["_kernel_calls"]
            row["kernel_model_latency_us"] = f"{float(row['_kernel_model_latency_us']):.3f}"
        else:
            row["kernel_calls"] = ""
            row["kernel_model_latency_us"] = ""


# --- seq len / KV len projection ---------------------------------------------


def is_attention_core(label: str) -> bool:
    return strip_phase_suffix(label).lower().endswith("_full_attn")


def scale_factor(label: str, phase: str, seq_ratio: float, used_ratio: float) -> float:
    """How a case's latency scales with prefill length and KV-cache length.

    Prefill: attention core ~ SEQ x USED, everything else ~ SEQ.
    Decode: attention core ~ USED, everything else is per token and fixed.
    Sampling depends on the vocabulary only.
    """

    if classify_module(label) == "Sampling":
        return 1.0
    attention = is_attention_core(label)
    if phase == "prefill":
        return seq_ratio * used_ratio if attention else seq_ratio
    if phase == "decode":
        return used_ratio if attention else 1.0
    return 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="perfstatistics root(s) or individual report directories. Default: latest .bench_logs/bench_*/perfstatistics",
    )
    parser.add_argument("--ghz", type=float, default=1.5, help="Clock frequency for latency conversion. Default: 1.5")
    parser.add_argument("--peak-gbps", type=float, default=0.0, help="Peak memory bandwidth in GB/s for utilization calculation.")
    parser.add_argument(
        "--case-calls",
        action="append",
        default=[],
        metavar="LABEL[@PHASE]=N",
        help="Override how many times a case's kernel is called in one model forward. "
        "PHASE is prefill or decode; without it the count applies to every phase. Repeatable.",
    )
    parser.add_argument(
        "--calls-file",
        type=Path,
        help="File with one LABEL[@PHASE]=N entry per line ('#' comments allowed). --case-calls entries win.",
    )
    parser.add_argument(
        "--expected-cases",
        type=Path,
        help="File listing the case labels that should have run, one per line. "
        "Default: <bench-out-dir>/expected_cases.txt written by the bench scripts.",
    )
    parser.add_argument(
        "--bench-script",
        type=Path,
        help="Bench script whose `--list` output gives the expected cases (for runs without expected_cases.txt).",
    )
    parser.add_argument("--status-only", action="store_true", help="Print only the case status report (which cases ran or are missing).")
    parser.add_argument("--print-missing", action="store_true", help="Print only the labels of expected cases without a report, one per line.")
    parser.add_argument(
        "--phase",
        choices=("prefill", "decode", "all"),
        default="all",
        help="Restrict the missing-case report and --print-missing to one phase. Sampling cases belong to both. Default: all",
    )
    parser.add_argument("--measured-seq-len", type=int, default=3823, help="Prefill tokens the benchmarks ran with (PREFILL_TOKENS). Default: 3823")
    parser.add_argument(
        "--measured-used-len",
        type=int,
        help="KV-cache length the attention benchmarks ran with (CTX_LEN). Default: same as --measured-seq-len",
    )
    parser.add_argument(
        "--scale",
        action="append",
        default=[],
        type=parse_scale_arg,
        metavar="SEQ[:USED]",
        help="Project latencies to SEQ prefill tokens and USED KV-cache tokens (USED defaults to SEQ). Repeatable.",
    )
    parser.add_argument("--tsv", action="store_true", help="Print TSV instead of a padded table.")
    parser.add_argument("--model-summary-dir", type=Path, help="Write model-level latency tables and SVG charts here.")
    parser.add_argument(
        "--bench-out-dir",
        type=Path,
        help="bench_all output directory holding the per-case logs and expected_cases.txt. Defaults to the perfstatistics parent.",
    )
    add_model_config_args(parser)
    args = parser.parse_args()

    try:
        case_calls = load_case_calls(args.case_calls, args.calls_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    model_config = model_config_from_args(args)
    effective_config = dict(DEFAULT_MODEL_CONFIG)
    if model_config:
        effective_config.update(model_config)
    measured_seq_len = args.measured_seq_len
    measured_used_len = args.measured_used_len or measured_seq_len
    if measured_seq_len <= 0 or measured_used_len <= 0:
        parser.error("--measured-seq-len and --measured-used-len must be positive")
    quiet = args.status_only or args.print_missing

    roots = args.paths or [discover_default_root()]
    log_paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "perfstatistics.log":
            log_paths.append(root)
        elif (root / "perfstatistics.log").is_file():
            log_paths.append(root / "perfstatistics.log")
        elif root.is_dir():
            log_paths.extend(sorted(root.rglob("perfstatistics.log")))

    bench_out_dir = args.bench_out_dir
    if bench_out_dir is None:
        for root in roots:
            candidate = root if root.is_dir() else root.parent
            if candidate.name == "perfstatistics":
                bench_out_dir = candidate.parent
                break
            if candidate.parent.name == "perfstatistics":
                bench_out_dir = candidate.parent.parent
                break

    rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    latency_header = f"latency_us@{args.ghz:g}GHz"
    has_bandwidth = False
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
            "_latency_us": latency_us,
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

    # Which cases should have run, and what their logs say happened.
    case_logs = scan_case_logs(bench_out_dir)
    expected_source = ""
    expected: list[str] = []
    try:
        if args.expected_cases is not None:
            expected = parse_case_list(args.expected_cases.read_text(errors="replace"))
            expected_source = str(args.expected_cases)
        elif args.bench_script is not None:
            expected = expected_cases_from_script(args.bench_script)
            expected_source = f"{args.bench_script} --list"
        elif bench_out_dir is not None and (bench_out_dir / "expected_cases.txt").is_file():
            expected = parse_case_list((bench_out_dir / "expected_cases.txt").read_text(errors="replace"))
            expected_source = str(bench_out_dir / "expected_cases.txt")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    rows = expand_deduped_summary_rows(rows, case_logs)
    present = {str(row["case"]) for row in rows}
    candidates = list(expected) + [label for label in case_logs if label not in expected]
    missing_rows: list[dict[str, object]] = []
    for label in candidates:
        if label in present:
            continue
        info = case_logs.get(label)
        phase = classify_phase(label)
        # Sampling runs after prefill and after decode, so it is missing from both.
        phases = ("prefill", "decode") if classify_module(label) == "Sampling" else (phase,)
        missing_rows.append(
            {
                "case": label,
                "executable": shorten_executable((info or {}).get("executable", "")) or "-",
                "compute_cycles": "",
                latency_header: "",
                "_latency_us": 0.0,
                "phase": phase,
                "calls": case_call_count(label, phase, effective_config, case_calls),
                "model_latency_us": "",
                "read_bytes": "",
                "write_bytes": "",
                "total_bytes": "",
                "achieved_GBps": "",
                "report_dir": describe_missing_case(info),
                "_missing": True,
                "_log": (info or {}).get("log", ""),
                "_phases": phases,
            }
        )
    missing_cases = [
        {
            "case": str(m["case"]),
            "calls": str(m["calls"]),
            "reason": str(m["report_dir"]),
            "log": str(m["_log"]),
            "phases": m["_phases"],
        }
        for m in missing_rows
    ]
    phase_filter = ("prefill", "decode") if args.phase == "all" else (args.phase,)
    missing_by_phase = {
        phase: [entry for entry in missing_cases if phase in entry["phases"]]  # type: ignore[operator]
        for phase in ("prefill", "decode", "unknown")
    }

    if args.print_missing:
        printed: list[str] = []
        for phase in phase_filter + (("unknown",) if args.phase == "all" else ()):
            for entry in missing_by_phase[phase]:
                if entry["case"] not in printed:
                    printed.append(str(entry["case"]))
        for label in printed:
            print(label)
        return 0

    measured_count = sum(1 for row in rows if "_kernel_source" not in row)
    deduped_count = len(rows) - measured_count
    if not quiet and not rows:
        print("No perfstatistics.log with compute_cycles was found.")

    if not quiet and rows:
        rows.extend(missing_rows)
        rows = split_sampling_rows(rows)
        rows.sort(key=lambda row: str(row["case"]))

        # Call counts depend on the (possibly deduped or phase-split) label, so
        # apply them after expansion.
        for row in rows:
            if row.get("_missing"):
                continue
            label = str(row["case"])
            phase = classify_phase(label)
            calls = case_call_count(label, phase, effective_config, case_calls)
            row["phase"] = phase
            row["calls"] = calls
            row["_model_latency_us"] = float(row["_latency_us"]) * calls
            row["model_latency_us"] = f"{row['_model_latency_us']:.3f}"
        aggregate_kernel_calls(rows)
        measured_rows = [row for row in rows if not row.get("_missing")]

        headers = [
            "case",
            "executable",
            "compute_cycles",
            latency_header,
            "phase",
            "calls",
            "model_latency_us",
            "kernel_calls",
            "kernel_model_latency_us",
        ]
        if has_bandwidth:
            headers.extend(["read_bytes", "write_bytes", "total_bytes", "achieved_GBps"])
            if args.peak_gbps > 0:
                headers.append("bw_util%")
        headers.append("report_dir")

        def weighted_total_row(label: str, phase: str, subset: list[dict[str, object]]) -> dict[str, object]:
            """Model-level totals for `subset`; every quantity is weighted by calls."""

            calls_sum = sum(int(r["calls"]) for r in subset)
            cycles_w = sum(int(r["compute_cycles"]) * int(r["calls"]) for r in subset)
            # Bandwidth uses only cases that report bytes, so numerator and
            # denominator cover the same launches.
            bw_rows = [r for r in subset if isinstance(r.get("read_bytes"), int)]
            read_w = sum(int(r["read_bytes"]) * int(r["calls"]) for r in bw_rows)
            write_w = sum(int(r["write_bytes"]) * int(r["calls"]) for r in bw_rows)
            bytes_w = read_w + write_w
            bw_cycles_w = sum(int(r["compute_cycles"]) * int(r["calls"]) for r in bw_rows)
            gbps_text = ""
            util_text = ""
            if bytes_w > 0 and bw_cycles_w > 0:
                gbps = bytes_w * args.ghz / bw_cycles_w
                gbps_text = f"{gbps:.3f}"
                if args.peak_gbps > 0:
                    util_text = f"{gbps / args.peak_gbps * 100.0:.2f}"
            coverage = f"{len(subset)} cases"
            if bw_rows and len(bw_rows) != len(subset):
                coverage += f", bandwidth from {len(bw_rows)}"
            omitted = sum(1 for r in missing_rows if not phase or phase in r["_phases"])  # type: ignore[operator]
            if omitted:
                coverage += f", {omitted} not run"
            total: dict[str, object] = {
                "case": label,
                "executable": "",
                "compute_cycles": cycles_w,
                latency_header: f"{cycles_w / (args.ghz * 1000.0):.3f}",
                "phase": phase,
                "calls": calls_sum,
                "model_latency_us": f"{sum(float(r['_model_latency_us']) for r in subset):.3f}",
                # Kernel totals span phases (sampling folds its prefill call into
                # the measured decode row), so only the overall TOTAL carries them.
                "kernel_calls": "" if phase else sum(int(r["_kernel_calls"]) for r in subset if "_kernel_calls" in r),
                "kernel_model_latency_us": ""
                if phase
                else f"{sum(float(r['_kernel_model_latency_us']) for r in subset if '_kernel_model_latency_us' in r):.3f}",
                "read_bytes": read_w if bw_rows else "",
                "write_bytes": write_w if bw_rows else "",
                "total_bytes": bytes_w if bw_rows else "",
                "achieved_GBps": gbps_text,
                "report_dir": coverage,
            }
            if args.peak_gbps > 0:
                total["bw_util%"] = util_text
            return total

        total_rows: list[dict[str, object]] = []
        for phase in ("prefill", "decode", "unknown"):
            subset = [r for r in measured_rows if r["phase"] == phase]
            if subset:
                total_rows.append(weighted_total_row(f"TOTAL_{phase}", phase, subset))
        total_rows.append(weighted_total_row("TOTAL", "", measured_rows))

        print_rows(rows + total_rows, headers, args.tsv)
        print()
        print(
            "model calls per forward: "
            + " ".join(
                f"{key}={effective_config[key]}"
                for key in (
                    "full_attn_layers",
                    "linear_attn_layers",
                    "dense_ffn_layers",
                    "moe_ffn_layers",
                    "sampling_prefill_count",
                    "sampling_decode_count",
                )
            )
            + (f" per_case_overrides={len(case_calls)}" if case_calls else "")
        )
        print("TOTAL rows weight cycles, latency, and bytes by calls; sampling is counted after prefill and after each decode step.")

        for target_seq, target_used in args.scale:
            seq_ratio = target_seq / measured_seq_len
            used_ratio = target_used / measured_used_len
            scaled_rows: list[dict[str, object]] = []
            for row in measured_rows:
                factor = scale_factor(str(row["case"]), str(row["phase"]), seq_ratio, used_ratio)
                scaled_latency = float(row["_latency_us"]) * factor
                scaled_rows.append(
                    {
                        "case": row["case"],
                        "phase": row["phase"],
                        "calls": row["calls"],
                        "latency_us": f"{float(row['_latency_us']):.3f}",
                        "scale": f"{factor:.3f}",
                        "scaled_latency_us": f"{scaled_latency:.3f}",
                        "model_latency_us": row["model_latency_us"],
                        "scaled_model_latency_us": f"{scaled_latency * int(row['calls']):.3f}",
                        "_scaled_model_latency_us": scaled_latency * int(row["calls"]),
                        "_model_latency_us": row["_model_latency_us"],
                    }
                )
            scaled_totals: list[dict[str, object]] = []
            for phase in ("prefill", "decode", "unknown", ""):
                subset = [r for r in scaled_rows if not phase or r["phase"] == phase]
                if not subset:
                    continue
                base_total = sum(float(r["_model_latency_us"]) for r in subset)
                scaled_total = sum(float(r["_scaled_model_latency_us"]) for r in subset)
                scaled_totals.append(
                    {
                        "case": f"TOTAL_{phase}" if phase else "TOTAL",
                        "phase": phase,
                        "calls": sum(int(r["calls"]) for r in subset),
                        "latency_us": f"{base_total:.3f}",
                        "scale": f"{scaled_total / base_total:.3f}" if base_total > 0 else "",
                        "scaled_latency_us": f"{scaled_total:.3f}",
                        "model_latency_us": f"{base_total:.3f}",
                        "scaled_model_latency_us": f"{scaled_total:.3f}",
                    }
                )
            print()
            print(
                f"=== projected latency @ seq_len={target_seq} used_len={target_used} "
                f"(measured seq_len={measured_seq_len} used_len={measured_used_len}; "
                f"prefill: x{seq_ratio:.3f}, prefill attention: x{seq_ratio * used_ratio:.3f}, "
                f"decode attention: x{used_ratio:.3f}) ==="
            )
            print_rows(
                scaled_rows + scaled_totals,
                ["case", "phase", "calls", "latency_us", "scale", "scaled_latency_us", "model_latency_us", "scaled_model_latency_us"],
                args.tsv,
            )

    if not args.tsv or quiet:
        print()
        expected_note = f"expected cases from {expected_source}" if expected_source else "no expected case list (only per-case logs were checked)"
        print(
            f"case status: measured={measured_count} deduped={deduped_count} not_run={len(missing_rows)} "
            f"(prefill={len(missing_by_phase['prefill'])} decode={len(missing_by_phase['decode'])}"
            + (f" unknown={len(missing_by_phase['unknown'])}" if missing_by_phase["unknown"] else "")
            + f"; {expected_note})"
        )
        for phase in phase_filter + (("unknown",) if args.phase == "all" and missing_by_phase["unknown"] else ()):
            entries = missing_by_phase[phase]
            print(f"  {phase}: {len(entries)} not run")
            for entry in entries:
                log_note = f"  [{entry['log']}]" if entry["log"] else ""
                both = "  (sampling: needed after prefill and after decode)" if len(entry["phases"]) > 1 else ""  # type: ignore[arg-type]
                print(f"    {entry['case']}: {entry['reason']}{log_note}{both}")
        if missing_rows:
            print("TOTAL rows and the model summary omit the cases above.")
    elif missing_rows:
        print(f"# not run: {len(missing_rows)} expected cases have no report (see --status-only)", file=sys.stderr)

    if quiet:
        return 0
    if not rows:
        return 1

    if args.model_summary_dir:
        report_path, summary_text = write_model_latency_summary(
            model_rows,
            args.model_summary_dir,
            title="Perfstatistics Model Latency Summary",
            source_name="perfstatistics",
            bench_out_dir=bench_out_dir,
            model_config=model_config,
            case_calls=case_calls,
            missing_cases=missing_cases,
        )
        print()
        print(summary_text)
        print()
        print(f"Model latency summary: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
