#!/usr/bin/env bash
# MiniMax-M2.7 TP=1 standalone benchmark wrapper.
#
# Source: MiniMaxAI/MiniMax-M2.7 config.
# MiniMax-M2.7 uses FP8 E4M3 block-wise quantized weights. Until the
# standalone vLLM/TRT-LLM block-wise FP8 kernels are extracted, quantized GEMM
# labels are explicit fp8 cuBLAS baselines, not W4A16 kernels.
# Shape policy: per-rank TP shape. TP=1 keeps full dimensions.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP=1

export MODEL_NAME="${MODEL_NAME:-MiniMax-M2.7-TP1}"
export HIDDEN_DIM=3072
export FULL_ATTN_Q_HEADS=$((48 / TP))
export FULL_ATTN_KV_HEADS=$((8 / TP))
export FULL_ATTN_HEAD_DIM=128
export W4A16_FULL_ATTN_Q_PROJ_GATE_N=$((6144 / TP))
export W4A16_FULL_ATTN_Q_PROJ_GATE_K=3072
export W4A16_FULL_ATTN_K_PROJ_N=$((1024 / TP))
export W4A16_FULL_ATTN_K_PROJ_K=3072
export W4A16_FULL_ATTN_V_PROJ_N=$((1024 / TP))
export W4A16_FULL_ATTN_V_PROJ_K=3072
export W4A16_FULL_ATTN_O_PROJ_N=3072
export W4A16_FULL_ATTN_O_PROJ_K=$((6144 / TP))

export MOE_EXPERTS=8
export MOE_ROUTER_EXPERTS=256
export MOE_TOPK=8
export MOE_INTERMEDIATE=$((1536 / TP))
export MOE_GATE_N=$((3072 / TP))
export MOE_GATE_K=3072
export MOE_DOWN_N=3072
export MOE_DOWN_K=$((1536 / TP))
export MOE_SHARED_HIDDEN=3072

export SAMPLING_VOCAB=$((200064 / TP))

export MODEL_LAYERS=62
export MODEL_FULL_ATTN_LAYERS=62
export MODEL_LINEAR_ATTN_LAYERS=0
export MODEL_DENSE_FFN_LAYERS=0
export MODEL_MOE_FFN_LAYERS=62

export ENABLE_LINEAR_ATTN=0
export ENABLE_FULL_ATTN=1
export ENABLE_DENSE_FFN=0
export ENABLE_MOE_FFN=1
export ENABLE_SHARED_EXPERT=0
export ENABLE_SAMPLING=1
export QUANTIZED_GEMM_DTYPE=fp8
export MOE_GEMM_BACKEND=cublas
export DECODE_MOE_BACKEND=trtllm

exec "$ROOT_DIR/run_all_qwen3.5-122B-A10B_GPTQ.sh" "$@"
