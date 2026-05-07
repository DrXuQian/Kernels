#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
KERNELS_ROOT = MOE_DIR.parents[2]

MARLIN_RE = re.compile(r"Kernel time: median=([0-9.]+) ms, avg=([0-9.]+) ms")
STANDALONE_RE = re.compile(
    r"dtype=(\w+) experts=(\d+) m_per_expert=(\d+) n=(\d+) k=(\d+) "
    r"group_size=(\d+) tile_enum=(\d+) stages=(\d+) avg_ms=([0-9.]+)"
)

CASES = [
    ("gate_up_prefill", 3823, 8, 8, 2048, 3072, "prefill"),
    ("down_prefill", 3823, 8, 8, 3072, 1024, "prefill"),
    ("gate_up_decode", 1, 8, 8, 2048, 3072, "decode"),
    ("down_decode", 1, 8, 8, 3072, 1024, "decode"),
]


def run_marlin(exe, case, warmup, iters):
    _, m, experts, topk, k, n, _ = case
    cmd = [
        str(exe),
        str(m),
        str(experts),
        str(topk),
        str(k),
        str(n),
        "--balanced",
        "--no-topk-weights",
        "--bench",
        str(warmup),
        str(iters),
    ]
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError("command failed:\n" + " ".join(cmd) + "\n" + cp.stdout + cp.stderr)
    match = MARLIN_RE.search(cp.stdout)
    if not match:
        raise RuntimeError("could not parse marlin output:\n" + cp.stdout)
    return {"median_ms": float(match.group(1)), "avg_ms": float(match.group(2))}


def run_standalone(exe, case, warmup, iters, tactic):
    _, m, _, topk, k, n, _ = case
    cmd = [
        str(exe),
        "--dtype=fp16",
        f"--experts={topk}",
        f"--m_per_expert={m}",
        f"--n={n}",
        f"--k={k}",
        "--group_size=128",
        f"--warmup={warmup}",
        f"--iters={iters}",
    ]
    if tactic:
        cmd.append(f"--tactic={tactic}")
    else:
        cmd.append("--sweep_configs")
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError("command failed:\n" + " ".join(cmd) + "\n" + cp.stdout + cp.stderr)
    final_lines = [line for line in cp.stdout.splitlines() if line.startswith("dtype=")]
    if not final_lines:
        raise RuntimeError("could not find standalone result:\n" + cp.stdout)
    match = STANDALONE_RE.search(final_lines[-1])
    if not match:
        raise RuntimeError("could not parse standalone result: " + final_lines[-1])
    return {"avg_ms": float(match.group(9)), "tile": int(match.group(7)), "stages": int(match.group(8))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", default=str(MOE_DIR / "build_sm90_clean/test_moe_w4a16_gemm"))
    parser.add_argument("--marlin", default=str(KERNELS_ROOT / "moe_ffn/w4a16/vllm/marlin/bench_marlin_moe"))
    parser.add_argument("--prefill_warmup", type=int, default=20)
    parser.add_argument("--prefill_iters", type=int, default=100)
    parser.add_argument("--decode_warmup", type=int, default=200)
    parser.add_argument("--decode_iters", type=int, default=2000)
    parser.add_argument("--tactic", default=None)
    args = parser.parse_args()

    standalone = Path(args.standalone)
    marlin = Path(args.marlin)
    rows = []
    for case in CASES:
        name, m, _, topk, k, n, mode = case
        warmup = args.prefill_warmup if mode == "prefill" else args.decode_warmup
        iters = args.prefill_iters if mode == "prefill" else args.decode_iters

        marlin_result = run_marlin(marlin, case, warmup, iters)
        standalone_result = run_standalone(standalone, case, warmup, iters, args.tactic)
        ratio = standalone_result["avg_ms"] / marlin_result["avg_ms"]
        rows.append((name, m, topk, k, n, marlin_result, standalone_result, ratio))
        print(
            f"{name:16s} marlin_avg={marlin_result['avg_ms']:.4f}ms "
            f"marlin_median={marlin_result['median_ms']:.4f}ms "
            f"standalone_avg={standalone_result['avg_ms']:.4f}ms "
            f"t{standalone_result['tile']}/s{standalone_result['stages']} "
            f"standalone/marlin={ratio:.3f}",
            flush=True,
        )

    print("\nCSV")
    print("case,M,topk,K,N,marlin_median_ms,marlin_avg_ms,standalone_avg_ms,standalone_tile,standalone_stages,standalone_over_marlin_avg")
    for name, m, topk, k, n, marlin_result, standalone_result, ratio in rows:
        print(
            f"{name},{m},{topk},{k},{n},{marlin_result['median_ms']:.4f},"
            f"{marlin_result['avg_ms']:.4f},{standalone_result['avg_ms']:.4f},"
            f"{standalone_result['tile']},{standalone_result['stages']},{ratio:.3f}"
        )


if __name__ == "__main__":
    main()
