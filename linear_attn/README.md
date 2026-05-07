# Linear Attention (DeltaNet) Kernels

Qwen3.5 DeltaNet layer 的 standalone CUDA/Triton kernel bench。

## Kernel 列表

| Bench | Kernel | 来源 | GPU | 阶段 |
|-------|--------|------|-----|------|
| `bench_conv1d_fwd` | `causal_conv1d_fwd` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | SM80+ | Prefill |
| `bench_conv1d_update` | `causal_conv1d_update` | Dao-AILab/causal-conv1d | SM80+ | Decode |
| `bench_linear_ops` | `in_proj_a/b` cuBLAS GEMM, residual add | standalone cuBLAS/CUDA | SM80+ | Prefill + Decode |
| `bench_gated_delta_net` | `gated_delta_net_cuda` | [llama.cpp](https://github.com/ggml-org/llama.cpp) | SM80+ | Prefill + Decode |
| `bench_kda_prefill` | `launch_kda_fwd_prefill_kernel` | [cuLA](https://github.com/inclusionAI/cuLA) | SM90 | Prefill (chunked) |
| `src/bench_vllm_triton_gdn_prefill.py` | `fused_post_conv_prep` + `chunk_gated_delta_rule` | vLLM FLA/GDN Triton | NVIDIA CUDA/Triton | Prefill |
| `src/bench_vllm_triton_gdn_decode.py` | `fused_recurrent_gated_delta_rule_packed_decode` | vLLM FLA/GDN Triton | NVIDIA CUDA/Triton | Decode |

## vLLM Triton GDN 提取

`src/vllm_triton_gdn/ops/` 是从 vLLM 的 Gated Delta Net / FLA Triton 路径抽出来的最小 standalone copy，去掉了 vLLM runtime 依赖，只依赖 PyTorch 和 Triton。

来源：

- vLLM commit: `e48cb85185d792f5b4a595c2af3cbc37ac742aac`
- upstream path: `vllm/model_executor/layers/fla/ops/`
- Qwen3-Next/Qwen3.5 dispatch path: `vllm/model_executor/layers/mamba/gdn_linear_attn.py`

对应关系：

- Prefill: vLLM 在 causal conv1d 之后先跑 `fused_post_conv_prep`，把 `mixed_qkv + a/b` 变成 contiguous 的 `q/k/v/g/beta`，然后跑 `chunk_gated_delta_rule`。这不是单个 kernel，而是一组 Triton kernel，包括 `l2norm`、`cumsum`、`chunk_scaled_dot_kkt`、`solve_tril`、`recompute_w_u`、`chunk_delta_h`、`chunk_o`。
- Decode: vLLM 的 non-spec decode fast path 跑 `fused_recurrent_gated_delta_rule_packed_decode`，输入是 causal conv1d update 之后的 packed `mixed_qkv`，这是单个 Triton kernel，内部完成 Q/K L2Norm、`g/beta`、state update 和 output。

## Triton 编译/执行方式

Triton 不是 `nvcc source.cu -> ELF executable` 这种 ahead-of-time 编译方式。这里的可执行入口是 Python 脚本；`@triton.jit` kernel 会在第一次 launch 时按当前 shape、dtype、constexpr 参数 JIT 编译，并把 cubin/metadata 缓存在 `TRITON_CACHE_DIR`。

建议步骤：

以下命令从 repo 根目录执行：

```bash
# 1. 语法检查，不触发 GPU JIT
python3 -m py_compile \
  linear_attn/src/bench_vllm_triton_gdn_prefill.py \
  linear_attn/src/bench_vllm_triton_gdn_decode.py \
  linear_attn/src/vllm_triton_gdn/ops/*.py

# 2. 设置 Triton cache。后续 nsys/ncu/bench 使用同一个目录可复用编译产物。
export TRITON_CACHE_DIR=$PWD/.triton_cache/vllm_gdn

# 3. 首次运行会 JIT 编译。要预编译目标 shape，就运行目标 shape 一次。
python3 linear_attn/src/bench_vllm_triton_gdn_decode.py
python3 linear_attn/src/bench_vllm_triton_gdn_prefill.py

# 4. 脚本本身也可以作为 executable 运行。
chmod +x linear_attn/src/bench_vllm_triton_gdn_decode.py
./linear_attn/src/bench_vllm_triton_gdn_decode.py
```

注意：Triton cache 是按 kernel specialization 区分的。小 shape 的预热只能检查可运行性；如果要避免 Qwen3.5 shape 第一次 profile 时包含 JIT 编译，必须用同一个 `TRITON_CACHE_DIR` 先跑一次 Qwen3.5 shape。

## 编译

```bash
make                        # 全部
make bench_conv1d_fwd       # 单个
make bench_linear_ops
```

也可以从 repo 根目录使用统一编译脚本选择目标，并让脚本检查当前环境：

```bash
./compile.sh env
./compile.sh build linear_attn
./compile.sh build flashinfer-gdn
```

## 运行

```bash
# Conv1d prefill
./bench_conv1d_fwd [seq_len] [dim] [width] [batch]         # default: 3823 12288 4 1

# Conv1d decode
./bench_conv1d_update [dim] [width] [batch]                 # default: 12288 4 1

# GDN 递推 (decode: n_tokens=1, prefill: n_tokens>1)
./bench_gated_delta_net [n_tokens] [heads] [head_dim] [n_seqs]  # default: 1 64 128 1

# FP16/BF16 dense adjuncts
./bench_linear_ops --op=in_proj_a --tokens=3823 --hidden=3072 --out-dim=64 --dtype fp16 --bench 0 1
./bench_linear_ops --op=in_proj_b --tokens=3823 --hidden=3072 --out-dim=64 --dtype fp16 --bench 0 1
./bench_linear_ops --op=residual_add --tokens=3823 --hidden=3072 --dtype fp16 --bench 0 1

# cuLA chunked prefill (Hopper only)
./bench_kda_prefill [seq_len] [num_heads] [head_dim] [num_seqs] # default: 3823 64 128 1

# vLLM Triton GDN prefill after causal_conv1d
python3 src/bench_vllm_triton_gdn_prefill.py [seq_len] [q_heads] [v_heads] [head_dim] [num_seqs]

# vLLM Triton GDN packed decode after causal_conv1d_update
python3 src/bench_vllm_triton_gdn_decode.py [batch] [q_heads] [v_heads] [head_dim]
```

## ncu Profile

```bash
ncu --set full --kernel-name "causal_conv1d_fwd"    -o conv_fwd    ./bench_conv1d_fwd 3823 12288 4 1
ncu --set full --kernel-name "causal_conv1d_update"  -o conv_update ./bench_conv1d_update 12288 4 1
ncu --set full --kernel-name "gated_delta_net"       -o gdn         ./bench_gated_delta_net 1 64 128 1
ncu --set full                                       -o kda         ./bench_kda_prefill 3823 64 128 1

# vLLM Triton decode is a single kernel.
nsys profile --force-overwrite=true -o vllm_gdn_decode \
  python3 src/bench_vllm_triton_gdn_decode.py

# vLLM Triton prefill is a sequence of chunk kernels.
nsys profile --force-overwrite=true -o vllm_gdn_prefill \
  python3 src/bench_vllm_triton_gdn_prefill.py
```
