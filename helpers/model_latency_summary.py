#!/usr/bin/env python3
"""Model-level latency aggregation and SVG chart helpers."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Iterable


MODULES = ("Flash-Attn", "Linear-Attn", "MoE-FFN", "Sampling", "Other")
PHASES = ("prefill", "decode", "unknown")

MODULE_COLORS = {
    "Flash-Attn": "#4E79A7",
    "Linear-Attn": "#F28E2B",
    "MoE-FFN": "#59A14F",
    "Sampling": "#E15759",
    "Other": "#9C755F",
}

DEFAULT_MODEL_CONFIG = {
    "model_layers": 48,
    "full_attn_layers": 12,
    "linear_attn_layers": 36,
    "moe_ffn_layers": 48,
    "sampling_prefill_count": 1,
    "sampling_decode_count": 1,
}


def classify_phase(case: str) -> str:
    lower = case.lower()
    if "prefill" in lower:
        return "prefill"
    if "decode" in lower:
        return "decode"
    if lower.startswith("sampling_"):
        return "decode"
    return "unknown"


def classify_module(case: str) -> str:
    lower = case.lower()
    if lower.startswith("sampling_"):
        return "Sampling"
    if lower.startswith("flash_attn_") or "full_attn" in lower:
        return "Flash-Attn"
    if lower.startswith("linear_") or "linear_attn" in lower:
        return "Linear-Attn"
    if lower.startswith("moe_") or lower.startswith("moe_ffn_") or "consistent_expert" in lower:
        return "MoE-FFN"
    return "Other"


def finite(value: object) -> bool:
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def fmt_us(value: float) -> str:
    return f"{value:.3f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def sanitize_svg_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def parse_case_log_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return metadata
    for line in lines[:64]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def expand_deduped_cases(cases: list[dict[str, object]], bench_out_dir: Path | None) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Add logical cases skipped by bench_all.sh dedupe.

    bench_all.sh writes a top-level log for skipped duplicates with
    `dedupe_duplicate_of: <case>`. The measured case is still a valid proxy for
    the logical duplicate, so model totals should include both labels.
    """

    expanded = [dict(row) for row in cases if finite(row.get("latency_us"))]
    by_case = {str(row["case"]): row for row in expanded}
    duplicates: list[dict[str, str]] = []
    if bench_out_dir is None or not bench_out_dir.is_dir():
        return expanded, duplicates

    for log_path in sorted(bench_out_dir.glob("*.log")):
        metadata = parse_case_log_metadata(log_path)
        label = metadata.get("label")
        duplicate_of = metadata.get("dedupe_duplicate_of")
        if not label or not duplicate_of or label in by_case:
            continue
        source = by_case.get(duplicate_of)
        if source is None:
            duplicates.append({"case": label, "duplicate_of": duplicate_of, "status": "missing_source"})
            continue
        copied = dict(source)
        copied["case"] = label
        copied["deduped_from"] = duplicate_of
        copied["source"] = f"deduped from {duplicate_of}"
        by_case[label] = copied
        expanded.append(copied)
        duplicates.append({"case": label, "duplicate_of": duplicate_of, "status": "expanded"})

    expanded.sort(key=lambda row: str(row["case"]))
    return expanded, duplicates


def aggregate_cases(cases: Iterable[dict[str, object]]) -> dict[str, object]:
    phase_module: dict[tuple[str, str], dict[str, object]] = {}
    module_totals: dict[str, dict[str, object]] = {}
    phase_totals = {phase: 0.0 for phase in PHASES}
    total = 0.0

    annotated_cases: list[dict[str, object]] = []
    for row in cases:
        latency = float(row["latency_us"])
        case = str(row["case"])
        phase = str(row.get("phase") or classify_phase(case))
        module = str(row.get("module") or classify_module(case))
        annotated = dict(row)
        annotated["phase"] = phase
        annotated["module"] = module
        annotated["latency_us"] = latency
        annotated_cases.append(annotated)

        phase_totals[phase] = phase_totals.get(phase, 0.0) + latency
        total += latency

        key = (phase, module)
        phase_module.setdefault(key, {"phase": phase, "module": module, "latency_us": 0.0, "cases": 0})
        phase_module[key]["latency_us"] = float(phase_module[key]["latency_us"]) + latency
        phase_module[key]["cases"] = int(phase_module[key]["cases"]) + 1

        module_totals.setdefault(module, {"module": module, "latency_us": 0.0, "cases": 0})
        module_totals[module]["latency_us"] = float(module_totals[module]["latency_us"]) + latency
        module_totals[module]["cases"] = int(module_totals[module]["cases"]) + 1

    return {
        "cases": annotated_cases,
        "phase_module": sorted(
            phase_module.values(),
            key=lambda row: (PHASES.index(str(row["phase"])) if str(row["phase"]) in PHASES else 99,
                             MODULES.index(str(row["module"])) if str(row["module"]) in MODULES else 99),
        ),
        "module_totals": sorted(
            module_totals.values(),
            key=lambda row: (MODULES.index(str(row["module"])) if str(row["module"]) in MODULES else 99),
        ),
        "phase_totals": phase_totals,
        "total_us": total,
    }


def model_multiplier(module: str, config: dict[str, int]) -> int:
    if module == "Flash-Attn":
        return config["full_attn_layers"]
    if module == "Linear-Attn":
        return config["linear_attn_layers"]
    if module == "MoE-FFN":
        return config["moe_ffn_layers"]
    return 1


def expand_to_model_estimate_cases(cases: list[dict[str, object]], config: dict[str, int]) -> list[dict[str, object]]:
    model_cases: list[dict[str, object]] = []
    for row in cases:
        base_latency = float(row["latency_us"])
        base_case = str(row["case"])
        module = classify_module(base_case)
        phase = classify_phase(base_case)

        if module == "Sampling":
            for sampling_phase, count_key in (
                ("prefill", "sampling_prefill_count"),
                ("decode", "sampling_decode_count"),
            ):
                multiplier = config[count_key]
                if multiplier <= 0:
                    continue
                copied = dict(row)
                copied["case"] = f"{base_case}__{sampling_phase}"
                copied["base_case"] = base_case
                copied["phase"] = sampling_phase
                copied["module"] = module
                copied["base_latency_us"] = base_latency
                copied["multiplier"] = multiplier
                copied["latency_us"] = base_latency * multiplier
                model_cases.append(copied)
            continue

        multiplier = model_multiplier(module, config)
        if multiplier <= 0:
            continue
        copied = dict(row)
        copied["phase"] = phase
        copied["module"] = module
        copied["base_case"] = base_case
        copied["base_latency_us"] = base_latency
        copied["multiplier"] = multiplier
        copied["latency_us"] = base_latency * multiplier
        model_cases.append(copied)

    model_cases.sort(key=lambda row: str(row["case"]))
    return model_cases


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def console_summary(summary: dict[str, object]) -> str:
    total = float(summary["total_us"])
    phase_totals: dict[str, float] = summary["phase_totals"]  # type: ignore[assignment]
    phase_rows = []
    for phase in ("prefill", "decode"):
        value = phase_totals.get(phase, 0.0)
        phase_rows.append([phase, fmt_us(value)])
    if phase_totals.get("unknown", 0.0):
        value = phase_totals["unknown"]
        phase_rows.append(["unknown", fmt_us(value)])

    module_rows = []
    for row in summary["phase_module"]:  # type: ignore[index]
        latency = float(row["latency_us"])
        module_rows.append(
            [
                str(row["phase"]),
                str(row["module"]),
                str(row["cases"]),
                fmt_us(latency),
            ]
        )

    return "\n\n".join(
        [
            "## Model Latency Totals",
            markdown_table(
                ["scope", "latency_us"],
                [["model_estimate_total", fmt_us(total)]] + phase_rows,
            ),
            "## Module Latency By Phase",
            markdown_table(
                ["phase", "module", "logical_cases", "latency_us"],
                module_rows,
            ),
        ]
    )


def write_bar_svg(path: Path, title: str, rows: list[tuple[str, float, str]], x_label: str = "Latency (us)") -> None:
    rows = [(label, value, color) for label, value, color in rows if value > 0.0]
    width = 1120
    max_label_len = max((len(label) for label, _, _ in rows), default=24)
    left = min(560, max(220, max_label_len * 7 + 24))
    right = 150
    top = 70
    bar_h = 24
    gap = 15
    height = max(180, top + len(rows) * (bar_h + gap) + 55)
    plot_w = width - left - right
    max_value = max((value for _, value, _ in rows), default=1.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{sanitize_svg_text(title)}</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 15}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#555">{sanitize_svg_text(x_label)}</text>',
        f'<line x1="{left}" y1="{top - 10}" x2="{left}" y2="{height - 45}" stroke="#999" stroke-width="1"/>',
    ]
    for idx, (label, value, color) in enumerate(rows):
        y = top + idx * (bar_h + gap)
        bar_w = plot_w * value / max_value if max_value else 0.0
        parts.extend(
            [
                f'<text x="{left - 12}" y="{y + bar_h * 0.72:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#333">{sanitize_svg_text(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" fill="{color}" rx="2"/>',
                f'<text x="{left + bar_w + 8:.2f}" y="{y + bar_h * 0.72:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#333">{fmt_us(value)}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def arc_path(cx: float, cy: float, r: float, start: float, end: float) -> str:
    x1 = cx + r * math.cos(start)
    y1 = cy + r * math.sin(start)
    x2 = cx + r * math.cos(end)
    y2 = cy + r * math.sin(end)
    large = 1 if end - start > math.pi else 0
    return f"M {cx:.3f} {cy:.3f} L {x1:.3f} {y1:.3f} A {r:.3f} {r:.3f} 0 {large} 1 {x2:.3f} {y2:.3f} Z"


def write_pie_svg(path: Path, title: str, rows: list[tuple[str, float, str]]) -> None:
    rows = [(label, value, color) for label, value, color in rows if value > 0.0]
    total = sum(value for _, value, _ in rows)
    width = 760
    height = 360
    cx, cy, r = 180.0, 190.0, 118.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{sanitize_svg_text(title)}</text>',
    ]
    if total <= 0:
        parts.append("</svg>")
        path.write_text("\n".join(parts) + "\n")
        return

    if len(rows) == 1:
        label, value, color = rows[0]
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
    else:
        angle = -math.pi / 2
        for label, value, color in rows:
            next_angle = angle + 2.0 * math.pi * value / total
            parts.append(f'<path d="{arc_path(cx, cy, r, angle, next_angle)}" fill="{color}" stroke="white" stroke-width="2"/>')
            angle = next_angle

    legend_x = 360
    legend_y = 95
    for idx, (label, value, color) in enumerate(rows):
        y = legend_y + idx * 32
        pct = value / total * 100.0
        parts.extend(
            [
                f'<rect x="{legend_x}" y="{y - 13}" width="16" height="16" fill="{color}" rx="2"/>',
                f'<text x="{legend_x + 25}" y="{y}" font-family="Arial, sans-serif" font-size="13" fill="#333">{sanitize_svg_text(label)}: {fmt_us(value)} us ({fmt_pct(pct)})</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def module_rows(summary: dict[str, object], phase: str | None = None) -> list[tuple[str, float, str]]:
    if phase is None:
        rows = summary["module_totals"]  # type: ignore[index]
        return [(str(row["module"]), float(row["latency_us"]), MODULE_COLORS.get(str(row["module"]), "#999")) for row in rows]

    rows = [row for row in summary["phase_module"] if str(row["phase"]) == phase]  # type: ignore[index]
    return [(str(row["module"]), float(row["latency_us"]), MODULE_COLORS.get(str(row["module"]), "#999")) for row in rows]


def case_rows(summary: dict[str, object], phase: str, limit: int = 18) -> list[tuple[str, float, str]]:
    rows = [row for row in summary["cases"] if str(row["phase"]) == phase]  # type: ignore[index]
    rows.sort(key=lambda row: float(row["latency_us"]), reverse=True)
    selected = rows[:limit]
    return [
        (str(row["case"]), float(row["latency_us"]), MODULE_COLORS.get(str(row["module"]), "#999"))
        for row in selected
    ]


def write_markdown_report(
    path: Path,
    title: str,
    source_name: str,
    model_summary: dict[str, object],
    covered_summary: dict[str, object],
    model_config: dict[str, int],
    duplicates: list[dict[str, str]],
    chart_files: list[Path],
) -> None:
    total = float(model_summary["total_us"])
    covered_total = float(covered_summary["total_us"])

    lines = [
        f"# {title}",
        "",
        f"Source: `{source_name}`.",
        "",
        "This report contains two views: the raw covered-case subtotal from the selected benchmark cases, and a "
        "Qwen3.5-122B-A10B model estimate that applies layer-count multipliers.",
        "",
        "## Model Configuration",
        "",
        markdown_table(
            ["item", "count"],
            [
                ["model_layers", str(model_config["model_layers"])],
                ["full_attention_layers", str(model_config["full_attn_layers"])],
                ["linear_attention_layers", str(model_config["linear_attn_layers"])],
                ["moe_ffn_layers", str(model_config["moe_ffn_layers"])],
                ["prefill_sampling", str(model_config["sampling_prefill_count"])],
                ["decode_sampling", str(model_config["sampling_decode_count"])],
            ],
        ),
        "",
        console_summary(model_summary),
        "",
        "## Covered Case Subtotal",
        "",
        markdown_table(
            ["scope", "latency_us"],
            [
                ["covered_case_total_before_layer_multipliers", fmt_us(covered_total)],
                ["covered_case_prefill", fmt_us(covered_summary["phase_totals"].get("prefill", 0.0))],  # type: ignore[index]
                ["covered_case_decode", fmt_us(covered_summary["phase_totals"].get("decode", 0.0))],  # type: ignore[index]
            ],
        ),
        "",
        "## Charts",
        "",
    ]
    for chart in chart_files:
        lines.append(f"![{chart.stem}]({chart.name})")
        lines.append("")

    top_rows = []
    cases = list(model_summary["cases"])  # type: ignore[arg-type]
    cases.sort(key=lambda row: float(row["latency_us"]), reverse=True)
    for row in cases[:24]:
        latency = float(row["latency_us"])
        base_latency = float(row.get("base_latency_us", latency))
        multiplier = int(row.get("multiplier", 1))
        top_rows.append(
            [
                str(row["phase"]),
                str(row["module"]),
                f"`{row['case']}`",
                str(multiplier),
                fmt_us(base_latency),
                fmt_us(latency),
            ]
        )
    lines.extend(
        [
            "## Top Case Contributions",
            "",
            markdown_table(["phase", "module", "case", "multiplier", "base_latency_us", "model_latency_us"], top_rows),
            "",
        ]
    )

    expanded = [row for row in duplicates if row.get("status") == "expanded"]
    missing = [row for row in duplicates if row.get("status") != "expanded"]
    if expanded:
        lines.extend(["## Deduped Logical Cases Expanded", ""])
        lines.append(markdown_table(["case", "uses_latency_from"], [[f"`{d['case']}`", f"`{d['duplicate_of']}`"] for d in expanded]))
        lines.append("")
    if missing:
        lines.extend(["## Deduped Cases Missing Source", ""])
        lines.append(markdown_table(["case", "missing_source"], [[f"`{d['case']}`", f"`{d['duplicate_of']}`"] for d in missing]))
        lines.append("")

    if model_summary["phase_totals"].get("unknown", 0.0):  # type: ignore[index]
        lines.append("Unknown-phase cases are included in `model_estimate_total` but omitted from prefill/decode charts.")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n")


def write_model_latency_summary(
    cases: list[dict[str, object]],
    out_dir: Path,
    *,
    title: str,
    source_name: str,
    bench_out_dir: Path | None = None,
    model_config: dict[str, int] | None = None,
) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    expanded, duplicates = expand_deduped_cases(cases, bench_out_dir)
    effective_config = dict(DEFAULT_MODEL_CONFIG)
    if model_config:
        effective_config.update(model_config)
    covered_summary = aggregate_cases(expanded)
    model_cases = expand_to_model_estimate_cases(expanded, effective_config)
    model_summary = aggregate_cases(model_cases)

    chart_files = [
        out_dir / "model_latency_phase_bar.svg",
        out_dir / "model_latency_module_bar.svg",
        out_dir / "model_latency_prefill_modules_pie.svg",
        out_dir / "model_latency_decode_modules_pie.svg",
        out_dir / "model_latency_prefill_cases_bar.svg",
        out_dir / "model_latency_decode_cases_bar.svg",
    ]

    phase_totals: dict[str, float] = model_summary["phase_totals"]  # type: ignore[assignment]
    write_bar_svg(
        chart_files[0],
        "Model Latency By Phase",
        [
            ("prefill", phase_totals.get("prefill", 0.0), "#6B8FD6"),
            ("decode", phase_totals.get("decode", 0.0), "#D67C4E"),
        ],
    )
    write_bar_svg(chart_files[1], "Model Latency By Module", module_rows(model_summary))
    write_pie_svg(chart_files[2], "Prefill Module Share", module_rows(model_summary, "prefill"))
    write_pie_svg(chart_files[3], "Decode Module Share", module_rows(model_summary, "decode"))
    write_bar_svg(chart_files[4], "Prefill Top Case Latencies", case_rows(model_summary, "prefill"), "Latency (us)")
    write_bar_svg(chart_files[5], "Decode Top Case Latencies", case_rows(model_summary, "decode"), "Latency (us)")

    report_path = out_dir / "model_latency_summary.md"
    write_markdown_report(report_path, title, source_name, model_summary, covered_summary, effective_config, duplicates, chart_files)
    return report_path, console_summary(model_summary)
