#!/usr/bin/env python3
"""Estimate effective bandwidth from nsys kernel duration and benchmark shapes.

This is a fallback for environments where NCU DRAM counters are blocked. It
uses the timed kernel duration from nsys and computes mandatory benchmark
traffic from the standalone benchmark command line. The result is effective
traffic bandwidth, not a hardware DRAM-counter measurement.
"""

import argparse
import math
import re
import shlex
from pathlib import Path


DTYPE_BYTES = {
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp16": 2,
    "bf16": 2,
    "float16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float": 4,
}


def parse_number(value):
    value = str(value).strip().replace(",", "")
    if not value or value in {"-", "—"}:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def fmt(value, digits=3):
    if value is None or math.isnan(value):
        return ""
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def dtype_nbytes(dtype):
    return DTYPE_BYTES.get(str(dtype).lower(), math.nan)


def parse_args_from_command(command):
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "", {}

    if not tokens:
        return "", {}
    if tokens[0] == "cd" and "&&" in tokens:
        tokens = tokens[tokens.index("&&") + 1 :]
    if not tokens:
        return "", {}

    exe = Path(tokens[0]).name
    opts = {}
    positional = []
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            positional.extend(tokens[i + 1 :])
            break
        if token.startswith("--"):
            if "=" in token:
                key, value = token[2:].split("=", 1)
                opts[key] = value
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                opts[token[2:]] = tokens[i + 1]
                i += 1
            else:
                opts[token[2:]] = "1"
        else:
            positional.append(token)
        i += 1
    opts["_positional"] = positional
    return exe, opts


def int_opt(opts, name, default=None):
    value = opts.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def str_opt(opts, name, default=None):
    return opts.get(name, default)


def pos_int(pos, index, default=None):
    try:
        return int(pos[index])
    except (IndexError, TypeError, ValueError):
        return default


def gemm_bytes(opts, allow_fp8=False):
    m = int_opt(opts, "m")
    n = int_opt(opts, "n")
    k = int_opt(opts, "k")
    dtype = str_opt(opts, "dtype", "fp16")
    out_dtype = str_opt(opts, "out-dtype", str_opt(opts, "output-dtype", "same"))
    if out_dtype == "same":
        out_dtype = "fp16" if dtype.startswith("fp8") else dtype
    in_b = dtype_nbytes(dtype)
    out_b = dtype_nbytes(out_dtype)
    if None in (m, n, k) or math.isnan(in_b) or math.isnan(out_b):
        return math.nan, ""
    if dtype.startswith("fp8") and not allow_fp8:
        return math.nan, "fp8 cuBLAS traffic is not modeled"
    total = m * k * in_b + k * n * in_b + m * n * out_b
    return total, f"gemm({m},{n},{k},{dtype}->{out_dtype})"


def block_fp8_bytes(opts):
    m = int_opt(opts, "m")
    n = int_opt(opts, "n")
    k = int_opt(opts, "k")
    out_dtype = str_opt(opts, "out-dtype", str_opt(opts, "output-dtype", "fp16"))
    out_b = dtype_nbytes(out_dtype)
    if None in (m, n, k) or math.isnan(out_b):
        return math.nan, ""
    kernel_m = round_up(m, 4)
    k_blocks = round_up(k, 128) // 128
    n_blocks = round_up(n, 128) // 128
    total = (
        kernel_m * k  # A fp8
        + n * k  # B fp8
        + m * n * out_b  # D
        + kernel_m * k_blocks * 4  # scale_a fp32
        + n_blocks * k_blocks * 4  # scale_b fp32
    )
    return total, f"block_fp8_gemm(m={m},kernel_m={kernel_m},n={n},k={k},out={out_dtype})"


def moe_fp8_bytes(opts):
    experts = int_opt(opts, "experts", 8)
    m_per_expert = int_opt(opts, "m_per_expert")
    n = int_opt(opts, "n")
    k = int_opt(opts, "k")
    if None in (experts, m_per_expert, n, k):
        return math.nan, ""
    total_m = experts * m_per_expert
    k_blocks = round_up(k, 128) // 128
    n_blocks = round_up(n, 128) // 128
    padded_rows = (total_m + experts * 31) // 32 * 32
    total = (
        total_m * k  # A fp8
        + experts * n * k  # B fp8
        + total_m * n * 2  # D bf16
        + padded_rows * k_blocks * 4  # scale_a fp32
        + experts * n_blocks * k_blocks * 4  # scale_b fp32
        + (experts + 1) * 8  # offsets
    )
    return total, f"moe_fp8(experts={experts},m_per={m_per_expert},n={n},k={k})"


def rmsnorm_bytes(opts, log_text):
    batch = int_opt(opts, "batch")
    embed = int_opt(opts, "embed")
    dtype = str_opt(opts, "dtype", "fp16")
    if batch is None or embed is None:
        match = re.search(r"bench rmsnorm: batch=(\d+) embed=(\d+) dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            batch = int(match.group(1))
            embed = int(match.group(2))
            dtype = match.group(3)
    b = dtype_nbytes(dtype)
    if None in (batch, embed) or math.isnan(b):
        return math.nan, ""
    total = batch * embed * b + embed * b + batch * embed * b
    return total, f"rmsnorm(batch={batch},embed={embed},dtype={dtype})"


def linear_ops_bytes(opts, log_text):
    op = str_opt(opts, "op", "")
    tokens = int_opt(opts, "tokens")
    hidden = int_opt(opts, "hidden")
    dtype = str_opt(opts, "dtype", "fp16")
    if tokens is None or hidden is None:
        match = re.search(r"linear ops bench: op=([A-Za-z0-9_]+) tokens=(\d+) hidden=(\d+).* dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            op = match.group(1)
            tokens = int(match.group(2))
            hidden = int(match.group(3))
            dtype = match.group(4)
    b = dtype_nbytes(dtype)
    if op != "residual_add" or None in (tokens, hidden) or math.isnan(b):
        return math.nan, ""
    total = 3 * tokens * hidden * b
    return total, f"residual_add(tokens={tokens},hidden={hidden},dtype={dtype})"


def gated_activation_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = topk = inter = None
    dtype = "fp16"
    if len(pos) >= 4:
        tokens, topk, inter, dtype = int(pos[0]), int(pos[1]), int(pos[2]), pos[3]
    else:
        match = re.search(r"gated_activation: tokens=(\d+) topk=(\d+) rows=\d+ inter=(\d+) dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            tokens = int(match.group(1))
            topk = int(match.group(2))
            inter = int(match.group(3))
            dtype = match.group(4)
    b = dtype_nbytes(dtype)
    if None in (tokens, topk, inter) or math.isnan(b):
        return math.nan, ""
    total = 3 * tokens * topk * inter * b
    return total, f"gated_activation(tokens={tokens},topk={topk},inter={inter},dtype={dtype})"


def fused_rms_norm_gate_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    n = pos_int(pos, 0)
    d = pos_int(pos, 1)
    dtype = str_opt(opts, "dtype", "bf16")
    if n is None or d is None:
        match = re.search(r"bench fused_rms_norm_gate: N=(\d+) D=(\d+) dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            n = int(match.group(1))
            d = int(match.group(2))
            dtype = match.group(3)
    b = dtype_nbytes(dtype)
    if None in (n, d) or math.isnan(b):
        return math.nan, ""
    total = 5 * n * d * b
    return total, f"fused_rms_norm_gate(N={n},D={d},dtype={dtype},kernel_load_model)"


def conv1d_update_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    dim = pos_int(pos, 0)
    width = pos_int(pos, 1)
    batch = pos_int(pos, 2)
    dtype = str_opt(opts, "dtype", "bf16")
    if None in (dim, width, batch):
        match = re.search(r"bench conv1d_update: dim=(\d+) width=(\d+) batch=(\d+) dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            dim = int(match.group(1))
            width = int(match.group(2))
            batch = int(match.group(3))
            dtype = match.group(4)
    b = dtype_nbytes(dtype)
    if None in (dim, width, batch) or math.isnan(b):
        return math.nan, ""
    state_len = max(width - 1, 0)
    total = (
        batch * dim * b  # x
        + dim * width * b  # depthwise weights
        + dim * b  # bias
        + 2 * batch * dim * state_len * b  # state read/write lower bound
        + batch * dim * b  # output
    )
    return total, f"conv1d_update(dim={dim},width={width},batch={batch},dtype={dtype},lower_bound)"


def gated_delta_net_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = pos_int(pos, 0)
    heads = pos_int(pos, 1)
    head_dim = pos_int(pos, 2)
    seqs = pos_int(pos, 3, 1)
    dtype = str_opt(opts, "dtype", "float")
    if None in (tokens, heads, head_dim):
        match = re.search(
            r"bench gated_delta_net: tokens=(\d+) heads=(\d+) dim=(\d+) seqs=(\d+)(?: dtype=([A-Za-z0-9_]+))?",
            log_text,
        )
        if match:
            tokens = int(match.group(1))
            heads = int(match.group(2))
            head_dim = int(match.group(3))
            seqs = int(match.group(4))
            if match.group(5):
                dtype = match.group(5)
    if None in (tokens, heads, head_dim, seqs):
        return math.nan, ""
    b = dtype_nbytes(dtype)
    if math.isnan(b):
        return math.nan, f"unsupported gated_delta_net dtype: {dtype}"
    qkv = head_dim * heads * tokens * seqs
    gb = heads * tokens * seqs
    state = seqs * heads * head_dim * head_dim
    total = (3 * qkv + qkv) * b + (2 * gb + 2 * state) * 4
    return total, f"gated_delta_net(tokens={tokens},heads={heads},dim={head_dim},seqs={seqs},dtype={dtype},fp32_state,lower_bound)"


def flash_attn_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    script_index = next((i for i, value in enumerate(pos) if value.endswith("bench_flash_attn.py")), None)
    mode = seq = heads = kv_heads = head_dim = None
    if script_index is not None and len(pos) >= script_index + 6:
        mode = pos[script_index + 1]
        seq = pos_int(pos, script_index + 2)
        heads = pos_int(pos, script_index + 3)
        kv_heads = pos_int(pos, script_index + 4)
        head_dim = pos_int(pos, script_index + 5)
    else:
        match = re.search(
            r"bench flash_attn (decode|prefill): heads=(\d+) kv_heads=(\d+) dim=(\d+) seq=(\d+)", log_text
        )
        if match:
            mode = match.group(1)
            heads = int(match.group(2))
            kv_heads = int(match.group(3))
            head_dim = int(match.group(4))
            seq = int(match.group(5))

    if None in (mode, seq, heads, kv_heads, head_dim):
        return math.nan, ""
    b = 2
    if mode == "decode":
        total = heads * head_dim * b + 2 * seq * kv_heads * head_dim * b + heads * head_dim * b
    elif mode == "prefill":
        total = seq * heads * head_dim * b + 2 * seq * kv_heads * head_dim * b + seq * heads * head_dim * b
    else:
        return math.nan, f"unsupported flash attention mode: {mode}"
    return total, f"flash_attn_{mode}_lower_bound(seq={seq},heads={heads},kv={kv_heads},dim={head_dim},fp16)"


def expand_input_rows_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = topk = hidden = None
    dtype = "fp16"
    if len(pos) >= 4:
        tokens, topk, hidden, dtype = pos_int(pos, 0), pos_int(pos, 1), pos_int(pos, 2), pos[3]
    else:
        match = re.search(r"expand_input_rows: tokens=(\d+) topk=(\d+) hidden=(\d+) dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            tokens, topk, hidden, dtype = int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
    b = dtype_nbytes(dtype)
    if None in (tokens, topk, hidden) or math.isnan(b):
        return math.nan, ""
    total = tokens * hidden * b + tokens * topk * hidden * b + tokens * topk * 4
    return total, f"expand_input_rows(tokens={tokens},topk={topk},hidden={hidden},dtype={dtype})"


def finalize_moe_routing_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = topk = hidden = None
    dtype = "fp16"
    if len(pos) >= 4:
        tokens, topk, hidden, dtype = pos_int(pos, 0), pos_int(pos, 1), pos_int(pos, 2), pos[3]
    else:
        match = re.search(
            r"finalize_moe_routing: tokens=(\d+) topk=(\d+) hidden=(\d+) dtype=([A-Za-z0-9_]+).* scales=(\d+)",
            log_text,
        )
        if match:
            tokens, topk, hidden, dtype = int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
    b = dtype_nbytes(dtype)
    if None in (tokens, topk, hidden) or math.isnan(b):
        return math.nan, ""
    total = (
        tokens * topk * hidden * b  # expanded rows
        + tokens * hidden * b  # final output
        + tokens * topk * 4  # scales
        + tokens * topk * 4  # token_selected_experts
        + tokens * topk * 4  # row mapping
    )
    return total, f"finalize_moe_routing(tokens={tokens},topk={topk},hidden={hidden},dtype={dtype},lower_bound)"


def custom_moe_routing_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = experts = topk = None
    dtype = "fp16"
    if len(pos) >= 4:
        tokens, experts, topk, dtype = pos_int(pos, 0), pos_int(pos, 1), pos_int(pos, 2), pos[3]
    else:
        match = re.search(r"custom_moe_routing: tokens=(\d+) experts=(\d+) topk=(\d+) dtype=([A-Za-z0-9_]+)", log_text)
        if match:
            tokens, experts, topk, dtype = int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
    b = dtype_nbytes(dtype)
    if None in (tokens, experts, topk) or math.isnan(b):
        return math.nan, ""
    total = tokens * experts * b + tokens * topk * 4 + tokens * topk * 4
    return total, f"custom_moe_routing(tokens={tokens},experts={experts},topk={topk},dtype={dtype},lower_bound)"


def expert_map_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = experts = topk = None
    mode = "auto"
    if len(pos) >= 4:
        tokens, experts, topk, mode = pos_int(pos, 0), pos_int(pos, 1), pos_int(pos, 2), pos[3]
    else:
        match = re.search(r"moe_expert_map: tokens=(\d+) experts=(\d+) topk=(\d+).* selected=([A-Za-z0-9_]+)", log_text)
        if match:
            tokens, experts, topk, mode = int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
    if None in (tokens, experts, topk):
        return math.nan, ""
    total = tokens * topk * 4 + 2 * tokens * topk * 4 + (experts + 1) * 8
    return total, f"expert_map(tokens={tokens},experts={experts},topk={topk},mode={mode},lower_bound)"


def vllm_topk_gating_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = pos_int(pos, 0)
    experts = pos_int(pos, 1)
    topk = pos_int(pos, 2)
    if None in (tokens, experts, topk):
        match = re.search(r"bench topk_gating: tokens=(\d+) experts=(\d+) topk=(\d+)", log_text)
        if match:
            tokens = int(match.group(1))
            experts = int(match.group(2))
            topk = int(match.group(3))
    if None in (tokens, experts, topk):
        return math.nan, ""
    total = tokens * experts * 4 + tokens * topk * 4 + tokens * topk * 4
    return total, f"vllm_topk_gating(tokens={tokens},experts={experts},topk={topk},fp32,lower_bound)"


def vllm_moe_align_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = pos_int(pos, 0)
    experts = pos_int(pos, 1)
    topk = pos_int(pos, 2)
    block = pos_int(pos, 3, 16)
    if None in (tokens, experts, topk):
        match = re.search(r"bench moe_align: tokens=(\d+) experts=(\d+) topk=(\d+) block=(\d+)", log_text)
        if match:
            tokens = int(match.group(1))
            experts = int(match.group(2))
            topk = int(match.group(3))
            block = int(match.group(4))
    if None in (tokens, experts, topk, block):
        return math.nan, ""
    expanded = tokens * topk
    padded = round_up(expanded, block)
    total = (
        expanded * 4  # selected expert ids
        + padded * 4  # sorted token ids
        + padded * 4  # sorted expert ids / block ids lower bound
        + (experts + 1) * 4  # expert offsets/counts
        + 4  # total token count
    )
    return total, f"vllm_moe_align(tokens={tokens},experts={experts},topk={topk},block={block},lower_bound)"


def vllm_marlin_moe_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = pos_int(pos, 0)
    experts = pos_int(pos, 1)
    topk = pos_int(pos, 2)
    k = pos_int(pos, 3)
    n = pos_int(pos, 4)
    group_size = int_opt(opts, "group-size", int_opt(opts, "group_size", 128))
    mul_topk = "no-topk-weights" not in opts
    if None in (tokens, experts, topk, k, n):
        match = re.search(
            r"Marlin MoE W4A16 bench: M=(\d+) experts=(\d+) top_k=(\d+) K=(\d+) N=(\d+) group=(\d+).* mul_topk=(\d+)",
            log_text,
        )
        if match:
            tokens = int(match.group(1))
            experts = int(match.group(2))
            topk = int(match.group(3))
            k = int(match.group(4))
            n = int(match.group(5))
            group_size = int(match.group(6))
            mul_topk = bool(int(match.group(7)))
    if None in (tokens, experts, topk, k, n, group_size):
        return math.nan, ""
    groups = round_up(k, group_size) // group_size
    active_rows = tokens * topk
    total = (
        tokens * k * 2  # activation, shared across selected experts
        + active_rows * k * n // 2  # selected int4 expert weights lower bound
        + active_rows * n * 2  # output rows
        + active_rows * n * groups * 2  # per-group scales lower bound
        + active_rows * 4  # routing metadata
    )
    if mul_topk:
        total += active_rows * 4
    return (
        total,
        f"vllm_marlin_moe(tokens={tokens},experts={experts},topk={topk},n={n},k={k},group={group_size},lower_bound)",
    )


def vllm_silu_and_mul_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = pos_int(pos, 0)
    topk = pos_int(pos, 1)
    hidden = pos_int(pos, 2)
    dtype = str_opt(opts, "dtype", "fp16")
    if None in (tokens, topk, hidden):
        match = re.search(r"bench silu_and_mul: tokens=(\d+) top_k=(\d+) rows=\d+ hidden=(\d+)", log_text)
        if match:
            tokens = int(match.group(1))
            topk = int(match.group(2))
            hidden = int(match.group(3))
    b = dtype_nbytes(dtype)
    if None in (tokens, topk, hidden) or math.isnan(b):
        return math.nan, ""
    total = 3 * tokens * topk * hidden * b
    return total, f"vllm_silu_and_mul(tokens={tokens},topk={topk},hidden={hidden},dtype={dtype})"


def vllm_moe_sum_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = pos_int(pos, 0)
    topk = pos_int(pos, 1)
    hidden = pos_int(pos, 2)
    dtype = str_opt(opts, "dtype", "fp16")
    if None in (tokens, topk, hidden):
        match = re.search(r"bench moe_sum: tokens=(\d+) topk=(\d+) hidden=(\d+)", log_text)
        if match:
            tokens = int(match.group(1))
            topk = int(match.group(2))
            hidden = int(match.group(3))
    b = dtype_nbytes(dtype)
    if None in (tokens, topk, hidden) or math.isnan(b):
        return math.nan, ""
    total = tokens * topk * hidden * b + tokens * hidden * b
    return total, f"vllm_moe_sum(tokens={tokens},topk={topk},hidden={hidden},dtype={dtype})"


def shared_expert_activation_bytes(opts, log_text):
    pos = opts.get("_positional", [])
    tokens = pos_int(pos, 0)
    inter = pos_int(pos, 1)
    dtype = pos[2] if len(pos) >= 3 else str_opt(opts, "dtype", "fp16")
    if None in (tokens, inter):
        match = re.search(
            r"bench trtllm shared_expert_activation: tokens=(\d+) rows=\d+ inter=(\d+) dtype=([A-Za-z0-9_]+)",
            log_text,
        )
        if match:
            tokens = int(match.group(1))
            inter = int(match.group(2))
            dtype = match.group(3)
    b = dtype_nbytes(dtype)
    if None in (tokens, inter) or math.isnan(b):
        return math.nan, ""
    total = 3 * tokens * inter * b
    return total, f"trtllm_shared_expert_activation(tokens={tokens},inter={inter},dtype={dtype})"


def shared_expert_bytes(opts, log_text):
    op = str_opt(opts, "op", "")
    tokens = int_opt(opts, "tokens", int_opt(opts, "batch"))
    hidden = int_opt(opts, "hidden", int_opt(opts, "embed"))
    out_dim = int_opt(opts, "out-dim", int_opt(opts, "experts", 1))
    dtype = str_opt(opts, "dtype", "fp16")
    if tokens is None or hidden is None:
        match = re.search(
            r"bench shared_expert: op=([A-Za-z0-9_]+) tokens=(\d+) hidden=(\d+)(?: out_dim=(\d+))? dtype=([A-Za-z0-9_]+)",
            log_text,
        )
        if match:
            op = match.group(1)
            tokens = int(match.group(2))
            hidden = int(match.group(3))
            if match.group(4):
                out_dim = int(match.group(4))
            dtype = match.group(5)
    b = dtype_nbytes(dtype)
    if None in (tokens, hidden) or math.isnan(b):
        return math.nan, ""
    if op in {"gate_gemv", "router_gate_gemm"}:
        total = tokens * hidden * b + hidden * out_dim * b + tokens * out_dim * b
    elif op == "sigmoid_mul_add":
        total = tokens * hidden * b * 4
    else:
        return math.nan, f"unsupported shared_expert op: {op}"
    return total, f"shared_expert_{op}(tokens={tokens},hidden={hidden},out={out_dim},dtype={dtype})"


def sampling_bytes(opts, log_text):
    op = str_opt(opts, "op", str_opt(opts, "kernel", ""))
    hidden = int_opt(opts, "hidden")
    vocab = int_opt(opts, "vocab")
    top_k = int_opt(opts, "top-k", int_opt(opts, "top_k", 50))
    if hidden is None or vocab is None:
        match = re.search(r"sampling bench: op=([A-Za-z0-9_]+) hidden=(\d+) vocab=(\d+) top_k=(\d+)", log_text)
        if match:
            op = match.group(1)
            hidden = int(match.group(2))
            vocab = int(match.group(3))
            top_k = int(match.group(4))
    if vocab is None:
        return math.nan, ""
    if op == "lm_head":
        if hidden is None:
            return math.nan, ""
        total = hidden * 2 + hidden * vocab * 2 + vocab * 4
    elif op in {"softmax", "topk_mask"}:
        total = 2 * vocab * 4
    elif op == "top_p":
        total = vocab * 4 + 8
    else:
        return math.nan, f"unsupported sampling op: {op}"
    return total, f"sampling_{op}(hidden={hidden},vocab={vocab},top_k={top_k},lower_bound)"


def estimate_bytes(exe, opts, log_text):
    if exe in {"bench_cublas_gemm", "bench_cuda_core_gemv", "bench_vllm_linear"}:
        return gemm_bytes(opts, allow_fp8=True)
    if exe in {"python3", "python"}:
        pos = opts.get("_positional", [])
        if any(value.endswith("bench_flash_attn.py") for value in pos):
            return flash_attn_bytes(opts, log_text)
    if exe == "bench_cutlass_block_fp8_gemm":
        return block_fp8_bytes(opts)
    if exe == "bench_moe_fp8_blockscale_gemm":
        return moe_fp8_bytes(opts)
    if exe == "bench_rmsnorm":
        return rmsnorm_bytes(opts, log_text)
    if exe == "bench_linear_ops":
        return linear_ops_bytes(opts, log_text)
    if exe == "bench_gated_activation":
        return gated_activation_bytes(opts, log_text)
    if exe == "bench_fused_rms_norm_gate":
        return fused_rms_norm_gate_bytes(opts, log_text)
    if exe == "bench_conv1d_update":
        return conv1d_update_bytes(opts, log_text)
    if exe == "bench_gated_delta_net":
        return gated_delta_net_bytes(opts, log_text)
    if exe == "bench_expand_input_rows":
        return expand_input_rows_bytes(opts, log_text)
    if exe == "bench_finalize_moe_routing":
        return finalize_moe_routing_bytes(opts, log_text)
    if exe == "bench_custom_moe_routing":
        return custom_moe_routing_bytes(opts, log_text)
    if exe == "bench_expert_map":
        return expert_map_bytes(opts, log_text)
    if exe == "bench_topk_gating":
        return vllm_topk_gating_bytes(opts, log_text)
    if exe == "bench_moe_align":
        return vllm_moe_align_bytes(opts, log_text)
    if exe == "bench_marlin_moe":
        return vllm_marlin_moe_bytes(opts, log_text)
    if exe == "bench_silu_and_mul":
        return vllm_silu_and_mul_bytes(opts, log_text)
    if exe == "bench_moe_sum":
        return vllm_moe_sum_bytes(opts, log_text)
    if exe == "bench_shared_expert_activation":
        return shared_expert_activation_bytes(opts, log_text)
    if exe == "bench_shared_expert":
        return shared_expert_bytes(opts, log_text)
    if exe == "bench_sampling":
        return sampling_bytes(opts, log_text)
    return math.nan, f"unsupported executable: {exe}"


def parse_nsys_aggregate(summary_path):
    rows = {}
    if not summary_path.exists():
        return rows
    in_case_aggregate = False
    for line in summary_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_case_aggregate = stripped == "## Nsight Systems Case Aggregate"
            continue
        if not in_case_aggregate:
            continue
        if not stripped.startswith("| `"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 5:
            continue
        case = parts[0].strip("`")
        duration_ns = parse_number(parts[3])
        latency_us = parse_number(parts[4])
        if not math.isnan(duration_ns):
            rows[case] = {"duration_ns": duration_ns, "latency_us": latency_us}
    return rows


def read_command_from_log(log_path):
    if not log_path.exists():
        return "", ""
    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        if line.startswith("command:"):
            return line[len("command:") :].strip(), text
    return "", text


def main():
    parser = argparse.ArgumentParser(description="Estimate effective bandwidth from nsys benchmark output.")
    parser.add_argument("out_dir", type=Path, help="Benchmark OUT_DIR containing nsys_latency_summary.md and case logs.")
    parser.add_argument("--peak-gbps", type=float, default=3350.0)
    args = parser.parse_args()

    aggregate = parse_nsys_aggregate(args.out_dir / "nsys_latency_summary.md")
    if not aggregate:
        raise SystemExit(f"no nsys aggregate rows found under {args.out_dir}")

    supported = []
    unsupported = []
    for case, timing in sorted(aggregate.items()):
        command, log_text = read_command_from_log(args.out_dir / f"{case}.log")
        exe, opts = parse_args_from_command(command)
        est_bytes, estimator = estimate_bytes(exe, opts, log_text)
        duration_ns = timing["duration_ns"]
        if math.isnan(est_bytes) or duration_ns <= 0:
            unsupported.append((case, estimator or "missing command/shape metadata"))
            continue
        gbps = est_bytes / duration_ns
        pct = 100.0 * gbps / args.peak_gbps if args.peak_gbps > 0 else math.nan
        supported.append((case, duration_ns, est_bytes, gbps, pct, estimator))

    print("## Nsight Systems Effective Bandwidth Estimate")
    print()
    print(
        "Effective bandwidth is computed from theoretical benchmark traffic divided by nsys CUDA kernel duration. "
        "It is a fallback when NCU DRAM counters are unavailable."
    )
    print()
    print("| case | duration_ns | estimated_bytes | effective_GBps | peak_pct | estimator |")
    print("|---|---:|---:|---:|---:|---|")
    for case, duration_ns, est_bytes, gbps, pct, estimator in supported:
        print(f"| `{case}` | {fmt(duration_ns)} | {fmt(est_bytes)} | {fmt(gbps)} | {fmt(pct)} | `{estimator}` |")

    if unsupported:
        print()
        print("## Unsupported Cases")
        print("| case | reason |")
        print("|---|---|")
        for case, reason in unsupported:
            print(f"| `{case}` | {reason} |")


if __name__ == "__main__":
    main()
