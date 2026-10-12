#!/usr/bin/env python3
"""Regression checks against the real perfstatistics/model-summary CLI."""

import csv
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import compare_bench_dirs as compare
from model_latency_summary import DEFAULT_MODEL_CONFIG, operator_groups, compute_model_summary


HELPERS = Path(__file__).resolve().parent


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.left, self.right = self.root / "left", self.root / "right"
        self.fixture(self.left)
        self.fixture(self.right)

    def tearDown(self):
        self.temp.cleanup()

    def perf(self, root, label, cycles):
        path = root / "perfstatistics" / label
        path.mkdir(parents=True, exist_ok=True)
        (path / "perfstatistics.log").write_text(f"compute_cycles = 999\ncompute_cycles = {cycles}\n")

    def fixture(self, root):
        for label, cycles in (
            ("flash_attn_decode_full_attn", 100000),
            ("linear_decode_gdn", 100000),
            ("moe_gate_up_decode_vllm", 50000),
            ("moe_down_decode_vllm", 25000),
            ("dense_ffn_decode_down_cublas", 90000000),
            ("sampling_lm_head_gemm", 1000000),
        ):
            self.perf(root, label, cycles)
        (root / "alias.log").write_text("label: linear_decode_alias\nstatus: skipped_duplicate\nduplicate_of: linear_decode_gdn\n")
        (root / "expected_cases.txt").write_text("linear_decode_alias\nlinear_decode_missing\n")

    def cli(self, script, *arguments):
        result = subprocess.run([sys.executable, str(HELPERS / script), *map(str, arguments)],
                                cwd=self.root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def comparison(self, *arguments, clocks=(1.5, 1.0)):
        path = self.root / "comparison.csv"
        text = self.cli("compare_bench_dirs.py", self.left, self.right,
                        "--baseline-ghz", clocks[0], "--candidate-ghz", clocks[1],
                        "--csv", path, "--no-plot", *arguments)
        return text, {row["case"]: row for row in csv.DictReader(io.StringIO(path.read_text()))}

    def summary(self, root, ghz, selected_phase="decode", *arguments):
        text = self.cli("summarize_perfstatistics.py", root / "perfstatistics", "--bench-out-dir", root,
                        "--ghz", ghz, "--phase", selected_phase, "--peak-tflops", "250", "--tsv", *arguments)
        lines = text.splitlines()
        header = next(line.split("\t") for line in lines if line.startswith("case\t"))
        return {cells[0]: dict(zip(header, cells)) for line in lines
                if len(cells := line.split("\t")) == len(header) and cells[0] != "case"}

    def test_model_totals_and_every_case_match_summary_at_different_clocks(self):
        text, rows = self.comparison("--phase", "decode")
        for root, ghz, side in ((self.left, 1.5, "baseline"), (self.right, 1.0, "candidate")):
            summary = self.summary(root, ghz)
            self.assertAlmostEqual(float(rows["TOTAL_decode"][f"{side}_us"]),
                                   float(summary["TOTAL_decode"]["model_latency_us"]), places=3)
            for label, row in rows.items():
                if label.startswith("TOTAL") or not row[f"{side}_us"]:
                    continue
                self.assertAlmostEqual(float(row[f"{side}_us"]), float(summary[label]["model_latency_us"]), places=3)
                self.assertEqual(int(row[f"{side}_calls"]), int(summary[label]["calls"]))
        self.assertEqual(float(rows["TOTAL_decode"]["candidate_us"]), 13000)
        self.assertEqual(float(rows["TOTAL_decode"]["candidate_ms"]), 13)
        self.assertNotIn("dense_ffn_decode_down_cublas", rows)
        self.assertEqual(rows["linear_decode_missing"]["candidate_us"], "")
        self.assertIn("Mode: **model**", text)

    def test_own_totals_include_unmatched_cases_and_common_totals_do_not(self):
        self.perf(self.right, "moe_decode_candidate_only", 100000)
        _, rows = self.comparison("--phase", "decode")
        self.assertEqual(float(rows["TOTAL_decode"]["candidate_us"]), 17800)
        self.assertEqual(float(rows["TOTAL_COMMON_decode"]["candidate_us"]), 13000)
        self.assertEqual(rows["TOTAL_decode"]["speedup"], "")
        self.assertIn("different coverage", rows["TOTAL_decode"]["status"])
        self.assertNotEqual(rows["TOTAL_COMMON_decode"]["speedup"], "")
        self.assertEqual(float(self.summary(self.right, 1)["TOTAL_decode"]["model_latency_us"]), 17800)

    def test_sampling_phases_and_overrides_match_summary(self):
        for root in (self.left, self.right):
            self.perf(root, "linear_prefill_actlize_gdn", 200000)
        calls_file = self.root / "calls.txt"
        calls_file.write_text("linear_decode_gdn=2\nsampling_lm_head_gemm@decode=3\n")
        flags = ["--linear-attn-layers", "4", "--full-attn-layers", "2", "--moe-ffn-layers", "3",
                 "--calls-file", calls_file, "--case-calls", "linear_decode_gdn=5"]
        _, rows = self.comparison("--phase", "all", *flags)
        for selected in ("prefill", "decode"):
            summary = self.summary(self.right, 1, selected, *flags)
            self.assertAlmostEqual(float(rows[f"TOTAL_{selected}"]["candidate_us"]),
                                   float(summary[f"TOTAL_{selected}"]["model_latency_us"]), places=3)
        self.assertEqual(rows["sampling_lm_head_gemm__decode"]["candidate_calls"], "3")
        self.assertEqual(rows["sampling_lm_head_gemm__prefill"]["candidate_calls"], "1")
        self.assertEqual(rows["linear_decode_gdn"]["candidate_calls"], "5")

    def test_auto_phase_and_raw_mode(self):
        text, rows = self.comparison()
        self.assertIn("phase: **decode**", text)
        self.assertNotIn("TOTAL_prefill", rows)
        text, rows = self.comparison("--mode", "raw", "--phase", "decode")
        self.assertEqual(float(rows["linear_decode_gdn"]["candidate_us"]), 100)
        self.assertIn("TOTAL_COMMON", rows)
        self.assertIn("TOTAL_AVAILABLE", rows)

    def test_model_source_selection_matches_summary(self):
        for root in (self.left, self.right):
            self.perf(root, "linear_prefill_flashinfer_gdn", 99999999)
            self.perf(root, "linear_prefill_actlize_gdn", 200000)
            (root / "expected_cases.txt").write_text("linear_prefill_actlize_gdn\n")
            # The canonical model summary uses an existing measurement even if a later log failed.
            (root / "failed.log").write_text("label: linear_prefill_actlize_gdn\n---- output ----\nfailed_at: yesterday\n")
        _, rows = self.comparison("--phase", "prefill")
        summary = self.summary(self.right, 1, "prefill")
        self.assertNotIn("linear_prefill_flashinfer_gdn", rows)
        self.assertEqual(float(rows["TOTAL_prefill"]["candidate_us"]),
                         float(summary["TOTAL_prefill"]["model_latency_us"]))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "plotting dependency not installed")
    def test_operator_chart_matches_model_report_and_sorts_largest_first(self):
        report = self.root / "summary"
        self.cli("summarize_perfstatistics.py", self.right / "perfstatistics", "--bench-out-dir", self.right,
                 "--ghz", "1", "--phase", "decode", "--model-summary-dir", report)
        self.assertTrue((report / "model_latency_decode_operators_bar.png").is_file())
        path = self.root / "comparison.svg"
        self.cli("compare_bench_dirs.py", self.left, self.right, "--phase", "decode",
                 "--baseline-ghz", "1.5", "--candidate-ghz", "1", "--plot", path)
        _, raw = compare.load_directory(self.right, "auto", 1, apply_logs=False)
        model = compare.model_measurements(raw, self.right, "decode", DEFAULT_MODEL_CONFIG, {})
        operators = compare.operator_comparisons(list(model), model, model, "decode")
        canonical = compute_model_summary(
            [{"case": label, "latency_us": value.latency_us} for label, value in raw.items()],
            self.right, {"sampling_prefill_count": 0})["model_summary"]
        expected = {row["operator"]: row["latency_us"] for row in operator_groups(canonical, "decode")}
        for row in operators:
            if row["candidate_us"] is not None:
                self.assertEqual(row["candidate_us"], expected[row["case"]])
        texts = ET.parse(path).findall(".//{http://www.w3.org/2000/svg}text")
        positions = {text.text: float(text.attrib["y"]) for text in texts if "y" in text.attrib}
        ordered = sorted(expected, key=positions.get)
        self.assertEqual([expected[name] for name in ordered], sorted(expected.values(), reverse=True))
        self.assertIn("Baseline · 1.5 GHz", positions)
        self.assertIn("Candidate · 1 GHz", positions)
        self.assertIn("13000.000", positions)


if __name__ == "__main__":
    unittest.main()
