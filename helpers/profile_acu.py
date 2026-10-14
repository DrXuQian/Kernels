#!/usr/bin/env python3
"""Capture one benchmark invocation with ACU and validate its exported durations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from summarize_acu_kernels import parse_case, sha256


ROOT = Path(__file__).resolve().parents[1]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def capture(args) -> None:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("provide the benchmark command after --")
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(str(prefix) + suffix) for key, suffix in (
        ("metadata", ".metadata.json"), ("csv", ".acu.csv"),
        ("log", ".capture.log"), ("export_log", ".export.log"),
    )}
    candidates = (prefix, Path(str(prefix) + ".acurep"))
    for path in (*paths.values(), *candidates):
        if path.exists():
            raise ValueError(f"refusing to overwrite capture: {path}; choose a new OUT_DIR/prefix")
    acu = shutil.which(args.acu)
    if not acu:
        raise ValueError(f"ACU executable not found: {args.acu}")
    acu = str(Path(acu).resolve())
    cmd = [acu, "-f", "-o", str(prefix), "--metrics", args.metrics,
           "--replay-mode", "kernel", "--cache-control", args.cache_control,
           "--profile-from-start", "yes", "--check-exit-code", "yes", *command]
    metadata = dict(
        status="RUNNING", started_at=utc(), cwd=str(Path.cwd()), command=command,
        acu_command=cmd, acu=acu, acu_sha256=sha256(Path(acu)),
        duration_metric="ppu__time_duration.sum", cache_control=args.cache_control,
        replay_mode="kernel", profile_from_start="yes",
        include_kernel=args.include_kernel, exclude_kernels=args.exclude_kernel,
        expected_kernels=args.expect_kernels, read_bytes=args.read_bytes, write_bytes=args.write_bytes,
        timing_scope="ACU_DEVICE_KERNEL_DURATION_NOT_EVENT_TIME_OR_END_TO_END_MODEL_TIME",
        hardware={key: os.environ.get(key) for key in
                  ("PPU_DEVICE_LABEL", "PPU_SM_COUNT", "PPU_CLOCK_GHZ", "CUDA_VISIBLE_DEVICES")},
        backend_environment={key: os.environ.get(key) for key in
                             ("CUDA_ROOT", "CUDA_HOME", "PPU_SDK_ROOT", "ACTLIZE_LA_BACKEND",
                              "ACTLIZE_LA_SM90_CONFIGURATION", "GDN_QSA_SM90_EXTENSION", "LINEAR_GDN_MODE")},
    )

    def save():
        paths['metadata'].write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    try:
        version = subprocess.run([acu, "--version"], text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=30, check=False)
        metadata['acu_version'] = version.stdout.strip()
        if version.returncode:
            raise ValueError(f"ACU --version failed ({version.returncode}): {version.stdout.strip()}")
        for key, git_args in (("repo_commit", ["rev-parse", "HEAD"]), ("repo_status", ["status", "--short"])):
            result = subprocess.run(["git", "-C", str(ROOT), *git_args], text=True,
                                    capture_output=True, check=False)
            metadata[key] = result.stdout.strip() if result.returncode == 0 else "unknown"
        executable = shutil.which(command[0])
        sources = [Path(executable)] if executable else []
        sources += [Path(arg).resolve() for arg in command[1:] if arg.endswith((".py", ".so")) and Path(arg).is_file()]
        extension = os.environ.get("GDN_QSA_SM90_EXTENSION")
        if extension and Path(extension).is_file():
            sources.append(Path(extension).resolve())
        metadata['input_files_sha256'] = {str(path): sha256(path) for path in sources}
        save()
        print("[acu] " + shlex.join(cmd), flush=True)
        with paths['log'].open('w') as stream:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                rc = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if rc:
            raise ValueError(f"ACU/application exited {rc}; see {paths['log']}")
        reports = [path for path in candidates if path.is_file() and path.stat().st_size]
        if len(reports) != 1:
            raise ValueError(f"expected one nonempty ACU report, found {len(reports)}")
        report = reports[0]
        metadata.update(report=str(report), report_sha256=sha256(report))
        export = [acu, "--import", str(report), "--page", "raw", "--csv"]
        metadata['export_command'] = export
        with paths['csv'].open('w') as output, paths['export_log'].open('w') as errors:
            subprocess.run(export, stdout=output, stderr=errors, check=True)
        rows = parse_case(paths['csv'], check_metadata=False, include_kernel=args.include_kernel,
                          exclude_kernels=tuple(args.exclude_kernel))
        if args.expect_kernels is not None and len(rows) != args.expect_kernels:
            raise ValueError(f"expected {args.expect_kernels} target launches, found {len(rows)}")
        metadata.update(status="COMPLETE", csv_sha256=sha256(paths['csv']), kernels=len(rows),
                        devices=sorted({row['device'] for row in rows}), finished_at=utc())
        print(f"[acu] COMPLETE kernels={len(rows)} report={report}", flush=True)
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        metadata.update(status="FAILED", error=str(exc), finished_at=utc())
        raise
    finally:
        save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--acu", default=os.environ.get("ACU", "acu"))
    parser.add_argument("--metrics", default=os.environ.get("ACU_METRICS", "ppu__time_duration.sum"))
    parser.add_argument("--cache-control", choices=("all", "none"), default=os.environ.get("ACU_CACHE_CONTROL", "all"))
    parser.add_argument("--include-kernel", default="", help="Only summarize names containing this text.")
    parser.add_argument("--exclude-kernel", action="append", default=[], help="Additional initializer names to omit from the summary.")
    parser.add_argument("--expect-kernels", type=positive)
    parser.add_argument("--read-bytes", type=positive)
    parser.add_argument("--write-bytes", type=positive)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if "ppu__time_duration.sum" not in args.metrics.split(","):
            raise ValueError("--metrics must include ppu__time_duration.sum")
        capture(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[acu] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
