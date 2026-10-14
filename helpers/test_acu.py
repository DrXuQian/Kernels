#!/usr/bin/env python3
"""ACU parser, capture failure, model parity, and wrapper tests; no hardware needed."""

import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import compare_bench_dirs as compare
from model_latency_summary import DEFAULT_MODEL_CONFIG, compute_model_summary
from summarize_acu_kernels import collect_cases, parse_case, sha256


ROOT = Path(__file__).resolve().parents[1]


class AcuTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="acu-test-")
        self.root = Path(self.temp.name)
        self.acu_dir = self.root / "run" / "acu"
        self.acu_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, label, rows):
        path = self.acu_dir / f"{label}.acu.csv"
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        return path

    def wide(self, label="linear_decode_gdn", values=(10, 20), unit="us"):
        return self.write_csv(label, [
            ["ID", "Kernel Name", "ppu__time_duration.sum"], ["", "", unit],
            *[[str(i), "subject", str(value)] for i, value in enumerate(values)],
        ])

    def cli(self, script, *args, env=None, expected=0):
        result = subprocess.run([sys.executable, str(ROOT / "helpers" / script), *map(str, args)],
                                text=True, capture_output=True, env=env)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def fake_acu(self):
        path = self.root / "acu"
        path.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, subprocess, sys
a = sys.argv[1:]
if a == ['--version']:
    print('FAKE ACU for host tests only')
elif '--import' in a:
    if os.environ.get('FAKE_EXPORT_FAIL'):
        sys.exit(7)
    print(os.environ['FAKE_ACU_CSV'])
else:
    pathlib.Path(os.environ['FAKE_COMMAND']).write_text(json.dumps(a))
    prefix = a[a.index('-o') + 1]
    rc = subprocess.call(a[a.index('--check-exit-code') + 2:])
    if not os.environ.get('FAKE_MISSING_REPORT'):
        pathlib.Path(prefix + '.acurep').write_text('FAKE REPORT; no device executed')
    sys.exit(rc)
''')
        path.chmod(0o755)
        return path

    def fake_env(self):
        csv_text = self.wide(values=(10,)).read_text()
        # Use another folder for capture, so the fixture is never mistaken for an output.
        return dict(os.environ, ACU=str(self.fake_acu()), FAKE_ACU_CSV=csv_text,
                    FAKE_COMMAND=str(self.root / 'command.json'))

    def test_units_and_wide_header_units(self):
        for unit, value in (("ns", 1234), ("usecond", 1.234), ("µs", 1.234), ("ms", .001234), ("second", .000001234)):
            with self.subTest(unit=unit):
                self.assertAlmostEqual(parse_case(self.wide(values=(value,), unit=unit))[0]['duration_ns'], 1234)
        p = self.write_csv('inline', [['ID', 'Kernel Name', 'ppu__time_duration.sum [ns]'], ['0', 'target', '100']])
        self.assertEqual(parse_case(p)[0]['duration_ns'], 100)

    def test_long_csv_dedup_identity_and_initializer_filter(self):
        header = ['ID', 'Process ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value']
        data = [header,
                ['0', '9', 'init_tensor', 'ppu__time_duration.sum', 'ms', '99'],
                ['1', '9', 'main', 'ppu__time_duration.sum', 'ns', '1,000'],
                ['1', '9', 'main', 'ppu__time_duration.sum', 'us', '1'],
                ['1', '10', 'main', 'ppu__time_duration.sum', 'ns', '2,000'],
                ['2', '9', 'reduce', 'ppu__time_duration.sum', 'us', '3']]
        rows = parse_case(self.write_csv('long', data))
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(row['duration_ns'] for row in rows), 6000)
        data.append(['2', '9', 'reduce', 'ppu__time_duration.sum', 'us', '4'])
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            parse_case(self.write_csv('long', data))

    def test_incomplete_metrics_and_unknown_units_are_not_partial_timings(self):
        p = self.write_csv('incomplete', [
            ['ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'],
            ['0', 'producer', 'ppu__time_duration.sum', 'ns', '1000'],
            ['1', 'reducer', 'ce__cycles_elapsed.max', 'cycle', '1000'],
        ])
        with self.assertRaisesRegex(ValueError, 'no device duration'):
            parse_case(p)
        for unit, value in (('', 1), ('cycle', 1), ('us', 'nan'), ('us', -1), ('us', 0)):
            with self.subTest(unit=unit, value=value), self.assertRaises(ValueError):
                parse_case(self.wide(values=(value,), unit=unit))

    def test_capture_and_bandwidth_receipt(self):
        env = self.fake_env()
        prefix = self.root / 'capture' / 'acu' / 'copy_bw'
        self.cli('profile_acu.py', '--output-prefix', prefix, '--expect-kernels', 1,
                 '--read-bytes', 1000000, '--write-bytes', 1000000, '--', '/bin/true', env=env)
        receipt = json.loads(Path(str(prefix) + '.metadata.json').read_text())
        self.assertEqual(receipt['status'], 'COMPLETE')
        self.assertEqual(receipt['csv_sha256'], sha256(Path(str(prefix) + '.acu.csv')))
        args = json.loads(Path(env['FAKE_COMMAND']).read_text())
        self.assertIn('ppu__time_duration.sum', args)
        self.assertEqual(args[args.index('--cache-control') + 1], 'all')
        cases, _, missing = collect_cases(prefix.parent)
        self.assertFalse(missing)
        self.assertEqual(cases[0]['traffic_gb_s'], 200)
        self.cli('profile_acu.py', '--output-prefix', prefix, '--', '/bin/true', env=env, expected=1)
        self.assertEqual(json.loads(Path(str(prefix) + '.metadata.json').read_text()), receipt)
        Path(str(prefix) + '.acu.csv').write_text('modified')
        self.assertFalse(collect_cases(prefix.parent)[0])

    def test_failed_child_export_and_missing_report_cannot_be_measured(self):
        for name, command, extra in (
            ('child', '/bin/false', {}), ('export', '/bin/true', {'FAKE_EXPORT_FAIL': '1'}),
            ('no-report', '/bin/true', {'FAKE_MISSING_REPORT': '1'}),
        ):
            env = dict(self.fake_env(), **extra)
            prefix = self.root / name / 'acu' / 'subject'
            self.cli('profile_acu.py', '--output-prefix', prefix, '--', command, env=env, expected=1)
            receipt = json.loads(Path(str(prefix) + '.metadata.json').read_text())
            self.assertEqual(receipt['status'], 'FAILED')
            cases, _, missing = collect_cases(prefix.parent)
            self.assertFalse(cases)
            self.assertTrue(missing)

    def test_launch_count_guard(self):
        env = self.fake_env()
        env['FAKE_ACU_CSV'] = self.wide(values=(10, 20)).read_text()
        prefix = self.root / 'count' / 'acu' / 'copy_bw'
        self.cli('profile_acu.py', '--output-prefix', prefix, '--expect-kernels', 1,
                 '--', '/bin/true', env=env, expected=1)
        self.assertFalse(collect_cases(prefix.parent)[0])

    def test_model_parity_dedupe_missing_and_frequency_independence(self):
        self.wide(values=(10, 20))
        self.wide('sampling_top_p', (5,))
        bench = self.acu_dir.parent
        (bench / 'alias.log').write_text('label: linear_decode_alias\nstatus: skipped_duplicate\nduplicate_of: linear_decode_gdn\n')
        (bench / 'expected_cases.txt').write_text('linear_decode_gdn\nlinear_decode_alias\nsampling_top_p\n')
        cases, _, _ = collect_cases(self.acu_dir)
        config = {**DEFAULT_MODEL_CONFIG, 'sampling_prefill_count': 0}
        expected = compute_model_summary(cases, bench, config)['phase_totals']['decode']
        _, loaded = compare.load_directory(bench, 'auto', .1)
        _, different_clock = compare.load_directory(bench, 'auto', 9)
        self.assertEqual(loaded, different_clock)
        result = compare.model_measurements(loaded, bench, 'decode', config, {}, None)
        self.assertEqual(sum(row.latency_us or 0 for row in result.values()), expected)
        text = self.cli('summarize_acu_kernels.py', self.acu_dir).stdout
        self.assertIn(f'{expected:.3f}', text)
        perf = self.root / 'perfmodel' / 'perfstatistics' / 'linear_decode_gdn'
        perf.mkdir(parents=True)
        (perf / 'perfstatistics.log').write_text('compute_cycles = 30000\n')
        report = self.cli('compare_bench_dirs.py', perf.parents[1], bench, '--baseline-ghz', 1,
                          '--candidate-ghz', 9, '--mode', 'raw', '--no-plot').stdout
        self.assertIn('acu', report)
        self.assertIn('| linear_decode_gdn | 30.000 | 30.000 |', report)
        (bench / 'expected_cases.txt').write_text('linear_decode_gdn\nlinear_decode_missing\n')
        self.assertIn('totals are partial', self.cli('summarize_acu_kernels.py', self.acu_dir, expected=1).stdout)

    def test_failed_case_log_invalidates_an_older_capture(self):
        self.wide()
        (self.acu_dir.parent / 'linear_decode_gdn.log').write_text('label: linear_decode_gdn\nfailed_at: now\n')
        self.assertFalse(collect_cases(self.acu_dir)[0])

    def test_phase_selection_and_unmeasured_phase_coverage(self):
        self.wide('linear_decode_gdn')
        (self.acu_dir.parent / 'expected_cases.txt').write_text('linear_decode_gdn\nlinear_prefill_actlize_gdn\n')
        text = self.cli('summarize_acu_kernels.py', self.acu_dir, expected=1).stdout
        self.assertIn('linear_prefill_actlize_gdn: no valid', text)
        self.cli('summarize_acu_kernels.py', self.acu_dir, '--phase', 'decode')
        self.cli('summarize_acu_kernels.py', self.acu_dir, '--phase', 'prefill', expected=1)

    def test_suite_dry_run_and_conflicting_options(self):
        result = subprocess.run(['bash', str(ROOT / 'bench_ppu17_hardware.sh'), '--models', 'all',
                                 '--phase', 'all', '--dry-run'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('--acu-latency'), 6)
        self.assertEqual(result.stdout.count('--single-launch'), 2)
        self.assertNotIn('--case all', result.stdout)
        result = subprocess.run(['bash', str(ROOT / 'bench_Qwen3.5-122B-A10B-GPTQ.sh'),
                                 '--acu-latency', '--ncu', '--list'], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('choose only one profiler', result.stderr)

    def test_real_model_wrapper_capture_and_dedupe_with_fake_acu(self):
        env = self.fake_env()
        repo = self.root / 'repo'
        repo.mkdir()
        script = 'bench_Qwen3.5-122B-A10B-GPTQ.sh'
        shutil.copy2(ROOT / script, repo / script)
        shutil.copytree(ROOT / 'helpers', repo / 'helpers', ignore=shutil.ignore_patterns('__pycache__'))
        binary = repo / 'general' / 'bench_cuda_core_gemv'
        binary.parent.mkdir()
        binary.write_text('#!/bin/sh\nexit 0\n')
        binary.chmod(0o755)
        env.update(PYTHON=sys.executable, ACU_MODEL_SUMMARY='0', OUT_DIR=str(self.root / 'model'))
        result = subprocess.run(['bash', str(repo / script), '--acu-latency', '--case',
                                 'linear_attn_decode_in_proj_a_cuda_core,linear_attn_decode_in_proj_b_cuda_core'],
                                text=True, capture_output=True, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = Path(env['OUT_DIR'])
        self.assertEqual(len(list((out / 'acu').glob('*.acurep'))), 1)
        self.assertIn('skipped_duplicate', (out / 'linear_attn_decode_in_proj_b_cuda_core.log').read_text())
        self.assertTrue((out / 'acu_latency_summary.md').is_file())

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'requires matplotlib')
    def test_acu_chart_keeps_device_time_label(self):
        self.wide()
        plot = self.root / 'comparison.svg'
        self.cli('compare_bench_dirs.py', self.acu_dir, self.acu_dir, '--plot', plot,
                 '--baseline-label', 'PPU1.7 1.0 GHz', '--phase', 'decode')
        self.assertIn('PPU1.7 1.0 GHz', plot.read_text())
        self.assertIn('Candidate · ACU device time', plot.read_text())


if __name__ == '__main__':
    unittest.main()
