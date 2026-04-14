#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Kernel 1: Prefill causal conv1d
void causal_conv1d_fn(
    const __nv_bfloat16* x,     // (batch, dim, seq_len)
    const __nv_bfloat16* weight, // (dim, 1, kernel_size)
    const __nv_bfloat16* bias,   // (dim,) or nullptr
    __nv_bfloat16* y,            // (batch, dim, seq_len) output
    __nv_bfloat16* state_out,    // (batch, dim, kernel_size-1) final conv state
    int batch, int dim, int seq_len, int kernel_size,
    cudaStream_t stream = 0);

// Kernel 2: Decode conv1d (single-step update)
void causal_conv1d_update(
    const __nv_bfloat16* x,      // (batch, dim, 1)
    __nv_bfloat16* state,        // (batch, dim, kernel_size-1) in/out
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_bfloat16* y,            // (batch, dim, 1) output
    int batch, int dim, int kernel_size,
    cudaStream_t stream = 0);

// Kernel 3: Prefill gated delta rule (recurrent implementation for correctness)
void chunk_gated_delta_rule(
    const __nv_bfloat16* Q,     // (batch, seq_len, heads, head_dim)
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float* g,              // (batch, seq_len, heads) — log-space decay gate
    const float* beta,           // (batch, seq_len, heads) — sigmoid write strength
    __nv_bfloat16* output,       // (batch, seq_len, heads, head_dim)
    float* state_out,            // (batch, heads, head_dim, head_dim) — fp32
    int batch, int seq_len, int heads, int head_dim, int chunk_size,
    bool use_l2norm,
    cudaStream_t stream = 0);

// Kernel 4: Decode gated delta rule (single-step recurrent)
void fused_recurrent_gated_delta_rule(
    const __nv_bfloat16* Q,      // (batch, 1, heads, head_dim)
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float* g,               // (batch, 1, heads)
    const float* beta,            // (batch, 1, heads)
    float* state,                 // (batch, heads, head_dim, head_dim) in/out — fp32
    __nv_bfloat16* output,        // (batch, 1, heads, head_dim)
    int batch, int heads, int head_dim,
    bool use_l2norm,
    cudaStream_t stream = 0);

// Kernel 5: Fused RMS norm + sigmoid gate
void fused_rms_norm_gate(
    const __nv_bfloat16* x,      // (N, D)
    const __nv_bfloat16* z,      // (N, D) gate input
    const __nv_bfloat16* weight,  // (D,)
    __nv_bfloat16* output,        // (N, D)
    int N, int D, float eps,
    cudaStream_t stream = 0);
