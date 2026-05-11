#!/usr/bin/env python3
"""Generate Qwen3.5-122B H800-vs-PPU comparison charts from bench_122B.md."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_latency_summary import classify_module, classify_operator, classify_phase


CONFIG = {
    "Flash-Attn": 12,
    "Linear-Attn": 36,
    "MoE-FFN": 48,
    "Sampling": 1,
    "Other": 1,
}

DEVICE_COLORS = {"H800": "#4E79A7", "PPU": "#F28E2B"}
PIE_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#FF9DA7",
    "#499894",
    "#D37295",
    "#B6992D",
    "#A0CBE8",
    "#FFBE7D",
    "#8CD17D",
    "#FABFD2",
    "#86BCB6",
    "#79706E",
    "#D4A6C8",
]
OTHER_COLOR = "#B8B8B8"

ROW_RE = re.compile(
    r"^\| `(?P<case>[^`]+)` "
    r"\| (?P<h800_kernels>[^|]+) "
    r"\| (?P<h800_cycles_avg>[^|]+) "
    r"\| (?P<h800_cycles_max>[^|]+) "
    r"\| (?P<h800_duration_ns>[^|]+) "
    r"\| (?P<h800_latency_us>[^|]+) "
    r"\| (?P<ppu_cycles>[^|]+) "
    r"\| (?P<ppu_latency_us>[^|]+) \|$"
)


def parse_float(value: str) -> float | None:
    value = value.strip()
    if value == "-":
        return None
    return float(value)


def parse_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        match = ROW_RE.match(line.strip())
        if not match:
            continue
        case = match.group("case")
        rows.append(
            {
                "case": case,
                "h800_us": parse_float(match.group("h800_latency_us")),
                "ppu_us": parse_float(match.group("ppu_latency_us")),
            }
        )
    return rows


def phases_for_case(case: str) -> list[str]:
    if case.startswith("sampling_"):
        return ["prefill", "decode"]
    phase = classify_phase(case)
    return [phase] if phase in {"prefill", "decode"} else []


def build_model_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    model_rows: list[dict[str, object]] = []
    imputed: list[str] = []
    for row in rows:
        case = str(row["case"])
        module = classify_module(case)
        operator = classify_operator(case)
        multiplier = CONFIG[module]
        h800_us = row["h800_us"]
        ppu_us = row["ppu_us"]
        ppu_imputed = False
        if ppu_us is None and h800_us is not None and case in {"sampling_lm_head_gemm", "sampling_lm_head_vllm_linear"}:
            # The current PPU data is missing only lm_head. Use H800 as a
            # neutral placeholder so model totals remain comparable.
            ppu_us = h800_us
            ppu_imputed = True
            imputed.append(case)
        if h800_us is None or ppu_us is None:
            continue
        for phase in phases_for_case(case):
            model_rows.append(
                {
                    "case": case,
                    "phase": phase,
                    "module": module,
                    "operator": operator,
                    "multiplier": multiplier,
                    "h800_model_us": float(h800_us) * multiplier,
                    "ppu_model_us": float(ppu_us) * multiplier,
                    "ppu_imputed": ppu_imputed,
                }
            )
    return model_rows, imputed


def aggregate(rows: list[dict[str, object]], key: str) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"H800": 0.0, "PPU": 0.0})
    for row in rows:
        group = (str(row["phase"]), str(row[key]))
        out[group]["H800"] += float(row["h800_model_us"])
        out[group]["PPU"] += float(row["ppu_model_us"])
    return dict(out)


def phase_totals(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out = {"prefill": {"H800": 0.0, "PPU": 0.0}, "decode": {"H800": 0.0, "PPU": 0.0}}
    for row in rows:
        phase = str(row["phase"])
        out[phase]["H800"] += float(row["h800_model_us"])
        out[phase]["PPU"] += float(row["ppu_model_us"])
    return out


def case_totals(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    out: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["phase"]), str(row["case"]))
        dst = out.setdefault(
            key,
            {
                "phase": row["phase"],
                "case": row["case"],
                "module": row["module"],
                "operator": row["operator"],
                "H800": 0.0,
                "PPU": 0.0,
            },
        )
        dst["H800"] = float(dst["H800"]) + float(row["h800_model_us"])
        dst["PPU"] = float(dst["PPU"]) + float(row["ppu_model_us"])
    return out


def fmt_ms(us: float) -> str:
    return f"{us / 1000.0:.3f}"


def plot_totals(totals: dict[str, dict[str, float]], out: Path) -> None:
    phases = ["prefill", "decode"]
    x = range(len(phases))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    h800 = [totals[p]["H800"] / 1000.0 for p in phases]
    ppu = [totals[p]["PPU"] / 1000.0 for p in phases]
    ax.bar([v - width / 2 for v in x], h800, width, label="H800", color=DEVICE_COLORS["H800"])
    ax.bar([v + width / 2 for v in x], ppu, width, label="PPU", color=DEVICE_COLORS["PPU"])
    ax.set_xticks(list(x), ["Prefill", "Decode"])
    ax.set_ylabel("Model latency (ms)")
    ax.set_title("Qwen3.5-122B Model Latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(h800):
        ax.text(idx - width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(ppu):
        ax.text(idx + width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def top_with_other(items: list[tuple[str, float]], limit: int = 10) -> list[tuple[str, float]]:
    items = [(k, v) for k, v in items if v > 0]
    items.sort(key=lambda kv: kv[1], reverse=True)
    if len(items) <= limit:
        return items
    keep = items[: limit - 1]
    other = sum(v for _, v in items[limit - 1 :])
    return keep + [("Other", other)]


def pie_colors(labels: list[str]) -> list[str]:
    colors = []
    color_idx = 0
    for label in labels:
        if label == "Other":
            colors.append(OTHER_COLOR)
        else:
            colors.append(PIE_COLORS[color_idx % len(PIE_COLORS)])
            color_idx += 1
    return colors


def plot_operator_pies(operator_totals: dict[tuple[str, str], dict[str, float]], phase: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, device in zip(axes, ["H800", "PPU"]):
        items = [(op, vals[device]) for (ph, op), vals in operator_totals.items() if ph == phase]
        items = top_with_other(items, limit=11)
        names = [name for name, _ in items]
        labels = [name if value / sum(v for _, v in items) >= 0.025 else "" for name, value in items]
        values = [value / 1000.0 for _, value in items]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=pie_colors(names),
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.5 else "",
            startangle=90,
            counterclock=False,
            labeldistance=1.08,
            pctdistance=0.73,
            textprops={"fontsize": 8},
            wedgeprops={"edgecolor": "white", "linewidth": 0.9},
        )
        ax.set_title(f"{device} {phase.capitalize()} Operator Share")
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def short_case_label(case: str) -> str:
    label = case
    replacements = [
        ("w4a16_", ""),
        ("linear_attn_", "lin_"),
        ("flash_attn_", "flash_"),
        ("moe_", ""),
        ("_trtllm", ""),
        ("_vllm", ""),
        ("_linear", ""),
        ("_cuda_core", ""),
        ("_cutlass55", ""),
        ("_fpA_intB", ""),
        ("_cublas", ""),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    parts = label.split("_")
    lines = []
    line = ""
    for part in parts:
        candidate = part if not line else f"{line}_{part}"
        if len(candidate) > 24:
            lines.append(line)
            line = part
        else:
            line = candidate
    if line:
        lines.append(line)
    return "\n".join(lines[:3])


def plot_case_pies(case_rows: dict[tuple[str, str], dict[str, object]], phase: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0))
    for ax, device in zip(axes, ["H800", "PPU"]):
        items = [
            (str(row["case"]), float(row[device]))
            for (ph, _), row in case_rows.items()
            if ph == phase
        ]
        items = top_with_other(items, limit=12)
        total = sum(value for _, value in items)
        names = [name for name, _ in items]
        labels = [short_case_label(name) if value / total >= 0.03 else "" for name, value in items]
        values = [value / 1000.0 for _, value in items]
        ax.pie(
            values,
            labels=labels,
            colors=pie_colors(names),
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3.0 else "",
            startangle=90,
            counterclock=False,
            labeldistance=1.1,
            pctdistance=0.72,
            textprops={"fontsize": 7},
            wedgeprops={"edgecolor": "white", "linewidth": 0.9},
        )
        ax.set_title(f"{device} {phase.capitalize()} Kernel Share")
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_case_bars(case_rows: dict[tuple[str, str], dict[str, object]], phase: str, out: Path, limit: int = 24) -> None:
    rows = [row for (ph, _), row in case_rows.items() if ph == phase]
    rows.sort(key=lambda r: max(float(r["H800"]), float(r["PPU"])), reverse=True)
    rows = rows[:limit]
    rows.reverse()
    labels = [str(row["case"]) for row in rows]
    y = range(len(rows))
    width = 0.36
    h800 = [float(row["H800"]) / 1000.0 for row in rows]
    ppu = [float(row["PPU"]) / 1000.0 for row in rows]
    height = max(6.0, 0.38 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(13.5, height))
    ax.barh([v - width / 2 for v in y], h800, height=width, label="H800", color=DEVICE_COLORS["H800"])
    ax.barh([v + width / 2 for v in y], ppu, height=width, label="PPU", color=DEVICE_COLORS["PPU"])
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Model latency contribution (ms)")
    ax.set_title(f"Top {phase.capitalize()} Logical Kernel Contributions")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_operator_bars(
    operator_totals: dict[tuple[str, str], dict[str, float]],
    phase: str,
    out: Path,
    limit: int = 18,
) -> None:
    rows = [
        {"operator": operator, "H800": vals["H800"], "PPU": vals["PPU"]}
        for (ph, operator), vals in operator_totals.items()
        if ph == phase
    ]
    rows.sort(key=lambda r: max(float(r["H800"]), float(r["PPU"])), reverse=True)
    rows = rows[:limit]
    rows.reverse()
    labels = [str(row["operator"]) for row in rows]
    y = range(len(rows))
    width = 0.36
    h800 = [float(row["H800"]) / 1000.0 for row in rows]
    ppu = [float(row["PPU"]) / 1000.0 for row in rows]
    height = max(5.2, 0.42 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(12.5, height))
    ax.barh([v - width / 2 for v in y], h800, height=width, label="H800", color=DEVICE_COLORS["H800"])
    ax.barh([v + width / 2 for v in y], ppu, height=width, label="PPU", color=DEVICE_COLORS["PPU"])
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Model latency contribution (ms)")
    ax.set_title(f"Top {phase.capitalize()} Kernel-Type Contributions")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def markdown_summary(totals: dict[str, dict[str, float]], imputed: list[str]) -> str:
    rows = [
        "| Phase | H800 total ms | PPU total ms | PPU/H800 |",
        "|---|---:|---:|---:|",
    ]
    for phase in ["prefill", "decode"]:
        h = totals[phase]["H800"]
        p = totals[phase]["PPU"]
        rows.append(f"| {phase} | {fmt_ms(h)} | {fmt_ms(p)} | {p / h:.3f}x |")
    text = "\n".join(rows)
    if imputed:
        text += "\n\nPPU missing values imputed from H800: " + ", ".join(f"`{x}`" for x in imputed) + "."
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("bench_122B.md"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/bench_122B"))
    args = parser.parse_args()

    rows = parse_rows(args.input)
    model_rows, imputed = build_model_rows(rows)
    totals = phase_totals(model_rows)
    operator_totals = aggregate(model_rows, "operator")
    cases = case_totals(model_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_totals(totals, args.out_dir / "model_latency_totals.svg")
    plot_operator_pies(operator_totals, "prefill", args.out_dir / "prefill_operator_share.svg")
    plot_operator_pies(operator_totals, "decode", args.out_dir / "decode_operator_share.svg")
    plot_case_pies(cases, "prefill", args.out_dir / "prefill_kernel_share.svg")
    plot_case_pies(cases, "decode", args.out_dir / "decode_kernel_share.svg")
    plot_case_bars(cases, "prefill", args.out_dir / "prefill_kernel_contributions.svg")
    plot_case_bars(cases, "decode", args.out_dir / "decode_kernel_contributions.svg")
    plot_operator_bars(operator_totals, "prefill", args.out_dir / "prefill_kernel_type_contributions.svg")
    plot_operator_bars(operator_totals, "decode", args.out_dir / "decode_kernel_type_contributions.svg")

    print(markdown_summary(totals, imputed))


if __name__ == "__main__":
    main()
