#!/bin/bash
# Run all bench binaries and measure wall time (CUDA sync happens inside each)
# Qwen3.5-122B DeltaNet decode config: batch=1, heads=64, dim=128, experts=64, topk=8
set -e
cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════════════"
echo "  Kernel Timing — Qwen3.5-122B DeltaNet — H800 PCIe"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Each bench calls cudaDeviceSynchronize before printing "Done."
# So wall-clock time of the process is a reasonable proxy (includes init overhead).
# For precise kernel-only time, use ncu.

time_cmd() {
    local label="$1"; shift
    # Warmup: run once silently
    "$@" > /dev/null 2>&1 || true
    # Measure 10 iterations
    local total=0
    local N=10
    for i in $(seq 1 $N); do
        local t0=$(date +%s%N)
        "$@" > /dev/null 2>&1
        local t1=$(date +%s%N)
        local dt=$(( (t1 - t0) / 1000000 ))  # ms
        total=$((total + dt))
    done
    local avg=$((total / N))
    printf "  %-45s %6d ms (avg of %d runs, includes launch overhead)\n" "$label" "$avg" "$N"
}

echo "=== Linear Attention (Decode) ==="
time_cmd "conv1d_update (dim=12288, w=4)" \
    linear_attention/bench_conv1d_update 12288 4 1

time_cmd "gated_delta_net (heads=64, dim=128)" \
    linear_attention/bench_gated_delta_net

echo ""
echo "=== Linear Attention (Prefill, seq=3823) ==="
time_cmd "conv1d_fwd (seq=3823, dim=12288, w=4)" \
    linear_attention/bench_conv1d_fwd 3823 12288 4 1

time_cmd "kda_prefill (seq=3823, heads=64, dim=128)" \
    linear_attention/bench_kda_prefill 3823 64 128 1

echo ""
echo "=== MoE FFN (Decode, M=1, 64 experts, topk=8) ==="
time_cmd "topk_gating (experts=64, topk=8)" \
    moe_w4a16/vllm/auxiliary/bench_topk_gating 1 64 8

time_cmd "moe_align (experts=64, topk=8, block=16)" \
    moe_w4a16/vllm/auxiliary/bench_moe_align 1 64 8 16

time_cmd "Marlin MoE GEMM (K=2048, N=5632)" \
    moe_w4a16/vllm/marlin/bench_marlin_moe 1 64 8 2048 5632

time_cmd "silu_and_mul (topk=8, hidden=5632)" \
    moe_w4a16/vllm/auxiliary/bench_silu_and_mul 1 8 5632

time_cmd "moe_sum (topk=8, hidden=2048)" \
    moe_w4a16/vllm/auxiliary/bench_moe_sum 1 8 2048

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Note: Times include process launch + CUDA init overhead."
echo "  For kernel-only time, use: ncu --set full ./bench_xxx"
echo "════════════════════════════════════════════════════════════════"
