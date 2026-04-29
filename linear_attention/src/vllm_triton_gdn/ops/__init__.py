# SPDX-License-Identifier: Apache-2.0

"""Minimal vLLM FLA/GDN Triton op surface used by standalone benches."""

from .chunk import chunk_gated_delta_rule
from .fused_gdn_prefill_post_conv import fused_post_conv_prep
from .fused_recurrent import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from .fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

__all__ = [
    "chunk_gated_delta_rule",
    "fused_post_conv_prep",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_sigmoid_gating_delta_rule_update",
]
