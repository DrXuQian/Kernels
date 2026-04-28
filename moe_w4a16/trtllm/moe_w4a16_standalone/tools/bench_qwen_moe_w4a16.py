#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent

FINAL_RE = re.compile(
    r"dtype=(\w+) experts=(\d+) m_per_expert=(\d+) n=(\d+) k=(\d+) "
    r"group_size=(\d+) tile_enum=(\d+) stages=(\d+) avg_ms=([0-9.]+)"
)


CASES = [
    # Interprets the user-provided shapes as output(E,M,N), input(E,M,K).
    ("gate_up_prefill", 8, 3823, 3072, 2048, "prefill"),
    ("down_prefill", 8, 3823, 1024, 3072, "prefill"),
    ("gate_up_decode", 8, 1, 3072, 2048, "decode"),
    ("down_decode", 8, 1, 1024, 3072, "decode"),
]


def run_one(exe, dtype, experts, m_per_expert, n, k, group_size, warmup, iters):
    cmd = [
        str(exe),
        f"--dtype={dtype}",
        f"--experts={experts}",
        f"--m_per_expert={m_per_expert}",
        f"--n={n}",
        f"--k={k}",
        f"--group_size={group_size}",
        f"--warmup={warmup}",
        f"--iters={iters}",
        "--sweep_configs",
    ]
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + cp.stdout
            + "\nstderr:\n"
            + cp.stderr
        )

    final_lines = [line for line in cp.stdout.splitlines() if line.startswith("dtype=")]
    if not final_lines:
        raise RuntimeError("could not find final benchmark line in output:\n" + cp.stdout)
    match = FINAL_RE.search(final_lines[-1])
    if not match:
        raise RuntimeError("could not parse final benchmark line: " + final_lines[-1])

    return {
        "ms": float(match.group(9)),
        "tile": int(match.group(7)),
        "stages": int(match.group(8)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dir", default=str(MOE_DIR / "build_sm90_clean"))
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--prefill_warmup", type=int, default=20)
    parser.add_argument("--prefill_iters", type=int, default=100)
    parser.add_argument("--decode_warmup", type=int, default=200)
    parser.add_argument("--decode_iters", type=int, default=2000)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "both"], default="both")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    bins = {
        "standalone": build_dir / "test_moe_w4a16_gemm",
        "trtllm_src": build_dir / "test_moe_w4a16_gemm_trtllm",
    }
    dtypes = ["fp16", "bf16"] if args.dtype == "both" else [args.dtype]
    results = []

    for case_name, experts, m_per_expert, n, k, mode in CASES:
        warmup = args.prefill_warmup if mode == "prefill" else args.decode_warmup
        iters = args.prefill_iters if mode == "prefill" else args.decode_iters
        for dtype in dtypes:
            row = {"case": case_name, "dtype": dtype}
            for impl, exe in bins.items():
                row[impl] = run_one(exe, dtype, experts, m_per_expert, n, k, args.group_size, warmup, iters)
            row["ratio"] = row["standalone"]["ms"] / row["trtllm_src"]["ms"]
            results.append(row)
            print(
                f"{case_name:16s} {dtype:4s} "
                f"standalone={row['standalone']['ms']:.4f}ms"
                f"(t{row['standalone']['tile']}/s{row['standalone']['stages']}) "
                f"trtllm_src={row['trtllm_src']['ms']:.4f}ms"
                f"(t{row['trtllm_src']['tile']}/s{row['trtllm_src']['stages']}) "
                f"ratio={row['ratio']:.3f}",
                flush=True,
            )

    print("\nCSV")
    print("case,dtype,standalone_ms,standalone_tile,standalone_stages,trtllm_src_ms,trtllm_src_tile,trtllm_src_stages,ratio")
    for row in results:
        print(
            f"{row['case']},{row['dtype']},{row['standalone']['ms']:.4f},"
            f"{row['standalone']['tile']},{row['standalone']['stages']},"
            f"{row['trtllm_src']['ms']:.4f},{row['trtllm_src']['tile']},"
            f"{row['trtllm_src']['stages']},{row['ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
