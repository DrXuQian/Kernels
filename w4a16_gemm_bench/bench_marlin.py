"""
Benchmark: Marlin W4A16 GEMM vs BF16/FP16 cuBLAS
Shape: M=1024, N=1024, K=1024
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels', 'marlin'))

import torch
import numpy as np
import time
import marlin_cuda

M, N, K = 1024, 1024, 1024
WARMUP = 50
REPEAT = 200
GROUP_SIZE = 128
dev = torch.device('cuda:0')

def benchmark_fn(fn, warmup=WARMUP, repeat=REPEAT, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    median_ms = np.median(times_ms)
    flops = 2.0 * M * N * K
    tflops = flops / (median_ms / 1000.0) / 1e12

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Median : {median_ms:.4f} ms")
    print(f"  Min    : {np.min(times_ms):.4f} ms")
    print(f"  Max    : {np.max(times_ms):.4f} ms")
    print(f"  TFLOPS : {tflops:.4f}")
    print(f"{'='*60}")
    return median_ms, tflops


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Shape: M={M}, N={N}, K={K}, GroupSize={GROUP_SIZE}")
print(f"Warmup={WARMUP}, Repeat={REPEAT}")

# ─── Marlin W4A16 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Preparing Marlin W4A16 kernel inputs...")

# Input activation (M x K, FP16)
A = torch.randn(M, K, dtype=torch.half, device=dev)

# Quantized weight (packed INT4 → int32, K*N/8 elements)
# Marlin expects a flat tensor of size K*N//8 packed int32
B_packed = torch.randint(low=-2**31, high=2**31, size=(K * N // 8,), dtype=torch.int32, device=dev)

# Output (M x N, FP16)
C = torch.zeros(M, N, dtype=torch.half, device=dev)

# Scales: (K//group_size, N) in FP16
num_groups = K // GROUP_SIZE
s = torch.rand(num_groups, N, dtype=torch.half, device=dev) * 0.1 + 0.01

# Workspace for Marlin
workspace = torch.zeros(N // 128 * 16, dtype=torch.int32, device=dev)

# Test call
try:
    marlin_cuda.mul(A, B_packed, C, s, workspace, -1, -1, -1, 16)
    torch.cuda.synchronize()
    print(f"  Marlin test call OK, output shape: {C.shape}")
except Exception as e:
    print(f"  Marlin test call failed: {e}")
    sys.exit(1)

marlin_ms, marlin_tflops = benchmark_fn(
    lambda: marlin_cuda.mul(A, B_packed, C, s, workspace, -1, -1, -1, 16),
    label="Marlin W4A16 GEMM (original, FP16 activation)"
)

# ─── BF16 cuBLAS ─────────────────────────────────────────────────────────────
A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
B_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=dev)

bf16_ms, bf16_tflops = benchmark_fn(
    lambda: torch.mm(A_bf16, B_bf16),
    label="BF16 GEMM (cuBLAS via torch.mm)"
)

# ─── FP16 cuBLAS ─────────────────────────────────────────────────────────────
A_fp16 = torch.randn(M, K, dtype=torch.float16, device=dev)
B_fp16 = torch.randn(K, N, dtype=torch.float16, device=dev)

fp16_ms, fp16_tflops = benchmark_fn(
    lambda: torch.mm(A_fp16, B_fp16),
    label="FP16 GEMM (cuBLAS via torch.mm)"
)

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n\n" + "█" * 70)
print("  SUMMARY — GEMM Benchmark (1024 x 1024 x 1024) on", torch.cuda.get_device_name(0))
print("█" * 70)
print(f"{'Kernel':<45} {'Median(ms)':>10} {'TFLOPS':>8} {'vs BF16':>8}")
print("-" * 75)

results = [
    ("BF16 GEMM (cuBLAS)", bf16_ms, bf16_tflops),
    ("FP16 GEMM (cuBLAS)", fp16_ms, fp16_tflops),
    ("Marlin W4A16", marlin_ms, marlin_tflops),
]
for name, ms, tflops in results:
    speedup = bf16_ms / ms if ms > 0 else 0
    print(f"{name:<45} {ms:>10.4f} {tflops:>8.4f} {speedup:>7.2f}x")
print("-" * 75)
