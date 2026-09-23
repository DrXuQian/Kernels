#!/usr/bin/env python3
"""FLOPs and traffic estimates for benchmark cases from their recorded commands.

The traffic model reuses summarize_nsys_effective_bandwidth.py (mandatory
bytes of each standalone benchmark) and adds the executables that helper does
not cover. FLOPs count multiply-adds as two operations for GEMM-like kernels,
MoE grouped GEMMs, attention, and linear attention; elementwise kernels count
as zero. Token-carrying arguments can be rescaled so a case can be projected
to a longer prefill or a longer KV cache.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

from summarize_nsys_effective_bandwidth import (
    estimate_bytes,
    int_opt,
    parse_args_from_command,
    pos_int,
    round_up,
    str_opt,
)


PYTHON_EXES = {"python", "python3"}
GEMM_EXES = {
    "bench_cublas_gemm",
    "bench_cuda_core_gemv",
    "bench_vllm_linear",
    "bench_cutlass_block_fp8_gemm",
    "test_machete_gemm",
    "test_fpA_intB_gemm",
    "cutlass_w4a16_bench",
    "cutlass55_gemm",
}
W4A16_GEMM_EXES = {"test_machete_gemm", "test_fpA_intB_gemm", "cutlass_w4a16_bench", "cutlass55_gemm"}
MOE_GROUPED_EXES = {"test_moe_w4a16_gemm", "bench_moe_fp8_blockscale_gemm"}
ELEMENTWISE_EXES = {
    "bench_rmsnorm",
    "bench_linear_ops",
    "bench_gated_activation",
    "bench_shared_expert_activation",
    "bench_expand_input_rows",
    "bench_finalize_moe_routing",
    "bench_custom_moe_routing",
    "bench_expert_map",
    "bench_topk_gating",
    "bench_moe_align",
    "bench_silu_and_mul",
    "bench_moe_sum",
    "bench_shared_expert",
    "bench_fused_rms_norm_gate",
    "bench_sampling",
}

# Where the prefill token count lives in each command: ("opt", name) or
# ("pos", index into the positional list) or ("attn",) for attention scripts.
# Executables missing here have no token argument (decode-only or fixed shape).
TOKEN_FIELDS: dict[str, Tuple[str, ...]] = {
    "bench_cublas_gemm": ("opt", "m"),
    "bench_cuda_core_gemv": ("opt", "m"),
    "bench_vllm_linear": ("opt", "m"),
    "bench_cutlass_block_fp8_gemm": ("opt", "m"),
    "test_machete_gemm": ("opt", "m"),
    "test_fpA_intB_gemm": ("opt", "m"),
    "cutlass_w4a16_bench": ("opt", "m"),
    "cutlass55_gemm": ("opt", "m"),
    "test_moe_w4a16_gemm": ("opt", "m_per_expert"),
    "bench_moe_fp8_blockscale_gemm": ("opt", "m_per_expert"),
    "bench_marlin_moe": ("pos", 0),
    "bench_rmsnorm": ("opt", "batch"),
    "bench_linear_ops": ("opt", "tokens"),
    "bench_shared_expert": ("opt", "tokens"),
    "bench_gated_activation": ("pos", 0),
    "bench_shared_expert_activation": ("pos", 0),
    "bench_expand_input_rows": ("pos", 0),
    "bench_finalize_moe_routing": ("pos", 0),
    "bench_custom_moe_routing": ("pos", 0),
    "bench_expert_map": ("pos", 0),
    "bench_topk_gating": ("pos", 0),
    "bench_moe_align": ("pos", 0),
    "bench_silu_and_mul": ("pos", 0),
    "bench_moe_sum": ("pos", 0),
    "bench_fused_rms_norm_gate": ("pos", 0),
    "bench_conv1d_fwd": ("pos", 0),
    "bench_gdn_prefill": ("pos", 0),
    "bench_conv1d_update": ("pos", 2),
    "bench_gated_delta_net": ("pos", 0),
    "bench_flash_attn.py": ("attn",),
    "bench_flash_infer.py": ("attn",),
    "bench_vllm_triton_gdn_prefill.py": ("pos", 1),
    "bench_vllm_triton_gdn_decode.py": ("pos", 1),
    "bench_flashinfer_cutlass_fp8_moe.py": ("opt", "tokens"),
}


def parse_case_command(command: str) -> tuple[str, dict]:
    """Executable name (script name for python cases) and parsed options."""

    exe, opts = parse_args_from_command(command)
    if exe in PYTHON_EXES:
        pos = opts.get("_positional", [])
        for index, value in enumerate(pos):
            if value.endswith(".py"):
                exe = value.rsplit("/", 1)[-1]
                # Drop the script path so positional indices start at the script's own args.
                opts = dict(opts)
                opts["_positional"] = pos[index + 1 :]
                opts["_script"] = value
                break
    return exe, opts


def attention_pairs(mode: str, seq_len: int, used_len: int) -> int:
    """(query token, key token) pairs, causal and bottom-right aligned."""

    if mode == "prefill":
        queries, context = seq_len, max(used_len, seq_len)
    else:
        queries, context = 1, max(used_len, 1)
    return queries * (context - queries) + queries * (queries + 1) // 2


def attention_shape(opts: dict) -> Optional[tuple[str, int, int, int, int, int]]:
    """mode, seq, heads, kv_heads, head_dim, ctx from a flash_attn/flash_infer command."""

    pos = opts.get("_positional", [])
    if len(pos) < 5:
        return None
    mode = pos[0]
    seq = pos_int(pos, 1)
    heads = pos_int(pos, 2)
    kv_heads = pos_int(pos, 3)
    head_dim = pos_int(pos, 4)
    ctx = int_opt(opts, "ctx", int_opt(opts, "context-len"))
    if mode not in {"prefill", "decode"} or None in (seq, heads, kv_heads, head_dim):
        return None
    if ctx is None:
        ctx = seq
    return mode, seq, heads, kv_heads, head_dim, ctx


def rescale_opts(exe: str, opts: dict, phase: str, seq_ratio: float, target_seq: int, target_used: int) -> dict:
    """Copy of `opts` with the token-carrying argument moved to the target lengths.

    Prefill cases scale their token count by `seq_ratio`; attention cases take
    the target lengths directly; decode cases keep their shape except the
    attention core, whose KV length becomes `target_used`.
    """

    field = TOKEN_FIELDS.get(exe)
    if field is None:
        return opts
    scaled = copy.deepcopy(opts)
    if field[0] == "attn":
        pos = list(scaled.get("_positional", []))
        if len(pos) >= 2:
            if pos[0] == "prefill" and phase == "prefill":
                pos[1] = str(target_seq)
                scaled["ctx"] = str(target_used)
            elif pos[0] == "decode" and phase == "decode":
                pos[1] = str(target_used)
        scaled["_positional"] = pos
        return scaled
    if phase != "prefill":
        return opts
    if field[0] == "opt":
        value = int_opt(scaled, str(field[1]))
        if value is not None:
            scaled[str(field[1])] = str(max(1, int(round(value * seq_ratio))))
    elif field[0] == "pos":
        pos = list(scaled.get("_positional", []))
        index = int(field[1])
        value = pos_int(pos, index)
        if value is not None:
            pos[index] = str(max(1, int(round(value * seq_ratio))))
            scaled["_positional"] = pos
    return scaled


def estimate_flops(exe: str, opts: dict) -> tuple[float, str]:
    """FLOPs of one benchmark call (multiply-add = 2). NaN when the case is not modeled."""

    pos = opts.get("_positional", [])
    if exe in GEMM_EXES:
        m, n, k = int_opt(opts, "m"), int_opt(opts, "n"), int_opt(opts, "k")
        if None in (m, n, k):
            return math.nan, ""
        return 2.0 * m * n * k, f"gemm({m},{n},{k})"
    if exe in MOE_GROUPED_EXES:
        experts = int_opt(opts, "experts", 8)
        m, n, k = int_opt(opts, "m_per_expert"), int_opt(opts, "n"), int_opt(opts, "k")
        if None in (experts, m, n, k):
            return math.nan, ""
        return 2.0 * experts * m * n * k, f"grouped_gemm(experts={experts},m={m},n={n},k={k})"
    if exe == "bench_marlin_moe":
        tokens, topk, k, n = pos_int(pos, 0), pos_int(pos, 2), pos_int(pos, 3), pos_int(pos, 4)
        if None in (tokens, topk, k, n):
            return math.nan, ""
        return 2.0 * tokens * topk * k * n, f"marlin_moe(tokens={tokens},topk={topk},k={k},n={n})"
    if exe == "bench_flashinfer_cutlass_fp8_moe.py":
        tokens, hidden, inter, topk = (
            int_opt(opts, "tokens"),
            int_opt(opts, "hidden"),
            int_opt(opts, "intermediate"),
            int_opt(opts, "topk", 8),
        )
        if None in (tokens, hidden, inter, topk):
            return math.nan, ""
        return 2.0 * tokens * topk * (hidden * 2 * inter + inter * hidden), f"fused_moe(tokens={tokens},topk={topk},hidden={hidden},inter={inter})"
    if exe == "bench_lm_head_gemv":
        n, k = int_opt(opts, "n"), int_opt(opts, "k")
        if None in (n, k):
            return math.nan, ""
        return 2.0 * n * k, f"gemv(n={n},k={k})"
    if exe in {"bench_flash_attn.py", "bench_flash_infer.py"}:
        shape = attention_shape(opts)
        if shape is None:
            return math.nan, ""
        mode, seq, heads, _kv_heads, head_dim, ctx = shape
        pairs = attention_pairs(mode, seq, ctx)
        return 4.0 * heads * head_dim * pairs, f"attention_{mode}(seq={seq},ctx={ctx},heads={heads},dim={head_dim},causal)"
    if exe in {"bench_gated_delta_net", "bench_gdn_prefill", "bench_vllm_triton_gdn_prefill.py", "bench_vllm_triton_gdn_decode.py"}:
        if exe == "bench_gated_delta_net":
            tokens, heads, head_dim, seqs = pos_int(pos, 0), pos_int(pos, 1), pos_int(pos, 2), pos_int(pos, 3, 1)
        else:
            tokens, heads, head_dim, seqs = pos_int(pos, 0), pos_int(pos, 2), pos_int(pos, 3), pos_int(pos, 4, 1)
        if None in (tokens, heads, head_dim, seqs):
            return math.nan, ""
        # Recurrent form per token and head: state update k^T v and readout q S, 2 d^2 MACs each.
        return 4.0 * tokens * seqs * heads * head_dim * head_dim, f"gated_delta_net(tokens={tokens},heads={heads},dim={head_dim},recurrent_lower_bound)"
    if exe == "bench_conv1d_fwd":
        tokens, dim, width = pos_int(pos, 0), pos_int(pos, 1), pos_int(pos, 2)
        if None in (tokens, dim, width):
            return math.nan, ""
        return 2.0 * tokens * dim * width, f"conv1d(tokens={tokens},dim={dim},width={width})"
    if exe == "bench_conv1d_update":
        dim, width, batch = pos_int(pos, 0), pos_int(pos, 1), pos_int(pos, 2, 1)
        if None in (dim, width, batch):
            return math.nan, ""
        return 2.0 * batch * dim * width, f"conv1d_update(batch={batch},dim={dim},width={width})"
    if exe in ELEMENTWISE_EXES:
        return 0.0, "elementwise (not counted)"
    return math.nan, f"flops not modeled for {exe}"


def _w4a16_gemm_bytes(m: int, n: int, k: int, group_size: int) -> float:
    groups = round_up(k, group_size) // group_size
    return m * k * 2 + n * k / 2 + groups * n * 2 + m * n * 2


def estimate_case_bytes(exe: str, opts: dict) -> tuple[float, str]:
    """Mandatory traffic of one benchmark call in bytes. NaN when not modeled."""

    pos = opts.get("_positional", [])
    if exe in W4A16_GEMM_EXES:
        m, n, k = int_opt(opts, "m"), int_opt(opts, "n"), int_opt(opts, "k")
        group = int_opt(opts, "group_size", int_opt(opts, "group-size", 128))
        if None in (m, n, k):
            return math.nan, ""
        return _w4a16_gemm_bytes(m, n, k, group), f"w4a16_gemm({m},{n},{k},group={group})"
    if exe == "test_moe_w4a16_gemm":
        experts = int_opt(opts, "experts", 8)
        m, n, k = int_opt(opts, "m_per_expert"), int_opt(opts, "n"), int_opt(opts, "k")
        group = int_opt(opts, "group_size", 128)
        if None in (experts, m, n, k):
            return math.nan, ""
        return experts * _w4a16_gemm_bytes(m, n, k, group), f"w4a16_grouped_gemm(experts={experts},m={m},n={n},k={k})"
    if exe == "bench_lm_head_gemv":
        n, k = int_opt(opts, "n"), int_opt(opts, "k")
        if None in (n, k):
            return math.nan, ""
        return k * 2 + n * k * 2 + n * 4, f"lm_head_gemv(n={n},k={k},fp32_out)"
    if exe in {"bench_flash_attn.py", "bench_flash_infer.py"}:
        shape = attention_shape(opts)
        if shape is None:
            return math.nan, ""
        mode, seq, heads, kv_heads, head_dim, ctx = shape
        queries = seq if mode == "prefill" else 1
        total = queries * heads * head_dim * 2 * 2 + 2 * ctx * kv_heads * head_dim * 2
        return total, f"attention_{mode}(seq={seq},ctx={ctx},lower_bound)"
    if exe in {"bench_gdn_prefill", "bench_vllm_triton_gdn_prefill.py", "bench_vllm_triton_gdn_decode.py"}:
        tokens, heads, head_dim, seqs = pos_int(pos, 0), pos_int(pos, 2), pos_int(pos, 3), pos_int(pos, 4, 1)
        if None in (tokens, heads, head_dim, seqs):
            return math.nan, ""
        mapped = dict(opts)
        mapped["_positional"] = [str(tokens), str(heads), str(head_dim), str(seqs)]
        return estimate_bytes("bench_gated_delta_net", mapped, "")
    if exe == "bench_conv1d_fwd":
        tokens, dim, width = pos_int(pos, 0), pos_int(pos, 1), pos_int(pos, 2)
        if None in (tokens, dim, width):
            return math.nan, ""
        return tokens * dim * 2 * 2 + dim * width * 2, f"conv1d(tokens={tokens},dim={dim},width={width})"
    if exe == "bench_flashinfer_cutlass_fp8_moe.py":
        tokens, hidden, inter, topk = int_opt(opts, "tokens"), int_opt(opts, "hidden"), int_opt(opts, "intermediate"), int_opt(opts, "topk", 8)
        experts = int_opt(opts, "experts", 8)
        if None in (tokens, hidden, inter, topk, experts):
            return math.nan, ""
        weights = experts * (hidden * 2 * inter + inter * hidden)
        acts = tokens * hidden * 2 + tokens * topk * (2 * inter + inter + hidden) * 2
        return weights + acts, f"fused_moe(tokens={tokens},experts={experts},topk={topk},lower_bound)"
    total, note = estimate_bytes(exe, opts, "")
    return total, note


def case_estimates(command: str) -> dict[str, object]:
    """FLOPs, bytes, and notes for a recorded benchmark command."""

    exe, opts = parse_case_command(command)
    flops, flops_note = estimate_flops(exe, opts)
    nbytes, bytes_note = estimate_case_bytes(exe, opts)
    return {"exe": exe, "opts": opts, "flops": flops, "flops_note": flops_note, "bytes": nbytes, "bytes_note": bytes_note}


def projected_estimates(exe: str, opts: dict, phase: str, seq_ratio: float, target_seq: int, target_used: int) -> tuple[float, float]:
    """FLOPs and bytes of the case at the target lengths."""

    scaled = rescale_opts(exe, opts, phase, seq_ratio, target_seq, target_used)
    flops, _ = estimate_flops(exe, scaled)
    nbytes, _ = estimate_case_bytes(exe, scaled)
    return flops, nbytes
