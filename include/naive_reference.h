#pragma once
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_bf16.h>

static inline float bf16_to_float(const __nv_bfloat16& v) {
    return __bfloat162float(v);
}

// CPU reference: single-step fused recurrent gated delta rule
// Matches the Triton kernel semantics from flash-linear-attention
static inline void naive_fused_recurrent_step(
    const __nv_bfloat16* Q,   // (heads, head_dim)
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float* g,            // (heads,)
    const float* beta,         // (heads,)
    float* state,              // (heads, head_dim, head_dim) in/out — state[h][k][v]
    float* output_fp32,        // (heads, head_dim) — fp32 for comparison
    int heads, int head_dim,
    bool use_l2norm)
{
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < heads; h++) {
        std::vector<float> q(head_dim), k(head_dim), v(head_dim);
        for (int d = 0; d < head_dim; d++) {
            q[d] = bf16_to_float(Q[h * head_dim + d]);
            k[d] = bf16_to_float(K[h * head_dim + d]);
            v[d] = bf16_to_float(V[h * head_dim + d]);
        }

        // L2 normalize q, k if needed
        if (use_l2norm) {
            float qn = 0, kn = 0;
            for (int d = 0; d < head_dim; d++) { qn += q[d]*q[d]; kn += k[d]*k[d]; }
            qn = sqrtf(qn + 1e-6f);
            kn = sqrtf(kn + 1e-6f);
            for (int d = 0; d < head_dim; d++) { q[d] /= qn; k[d] /= kn; }
        }

        // Scale q
        for (int d = 0; d < head_dim; d++) q[d] *= scale;

        float* S = state + (long long)h * head_dim * head_dim; // S[k_dim][v_dim]
        float decay = expf(g[h]);
        float b = beta[h];

        // 1. Decay state
        for (int i = 0; i < head_dim * head_dim; i++)
            S[i] *= decay;

        // 2. Compute h^T @ k: result[vd] = sum_{kd} S[kd][vd] * k[kd]
        std::vector<float> Sk(head_dim, 0.0f);
        for (int vd = 0; vd < head_dim; vd++)
            for (int kd = 0; kd < head_dim; kd++)
                Sk[vd] += S[kd * head_dim + vd] * k[kd];

        // 3. Delta: v_new = beta * (v - S^T @ k)
        std::vector<float> v_new(head_dim);
        for (int vd = 0; vd < head_dim; vd++)
            v_new[vd] = b * (v[vd] - Sk[vd]);

        // 4. Rank-1 update: S += k * v_new^T
        for (int kd = 0; kd < head_dim; kd++)
            for (int vd = 0; vd < head_dim; vd++)
                S[kd * head_dim + vd] += k[kd] * v_new[vd];

        // 5. Output: o[vd] = sum_{kd} S[kd][vd] * q[kd]
        for (int vd = 0; vd < head_dim; vd++) {
            float val = 0;
            for (int kd = 0; kd < head_dim; kd++)
                val += S[kd * head_dim + vd] * q[kd];
            output_fp32[h * head_dim + vd] = val;
        }
    }
}

// CPU reference: RMS norm + sigmoid gate
static inline void naive_rms_norm_gate(
    const __nv_bfloat16* x,      // (D,) single row
    const __nv_bfloat16* z,      // (D,) gate
    const __nv_bfloat16* weight,  // (D,)
    float* output_fp32,           // (D,)
    int D, float eps)
{
    // RMS norm
    float sum_sq = 0;
    for (int i = 0; i < D; i++) {
        float v = bf16_to_float(x[i]);
        sum_sq += v * v;
    }
    float rms = sqrtf(sum_sq / D + eps);
    float inv_rms = 1.0f / rms;

    for (int i = 0; i < D; i++) {
        float xv = bf16_to_float(x[i]) * inv_rms;
        float wv = bf16_to_float(weight[i]);
        float zv = bf16_to_float(z[i]);
        float gate = 1.0f / (1.0f + expf(-zv)); // sigmoid
        output_fp32[i] = xv * wv * gate;
    }
}

// CPU reference: conv1d update (single step) for a single channel
// state: (kernel_size-1,) values, updated in-place
static inline void naive_causal_conv1d_update_channel(
    float new_x,
    float* state,             // (kernel_size-1,) in/out
    const __nv_bfloat16* weight, // (kernel_size,)
    float bias,
    float* output_fp32,       // scalar
    int kernel_size)
{
    // Build window: [state[0], state[1], ..., state[ks-2], new_x]
    float buf[8] = {0};
    for (int i = 0; i < kernel_size - 1; i++)
        buf[i] = state[i];
    buf[kernel_size - 1] = new_x;

    float out = bias;
    for (int k = 0; k < kernel_size; k++)
        out += bf16_to_float(weight[k]) * buf[k];
    *output_fp32 = out;

    // Update state: shift left
    for (int i = 0; i < kernel_size - 2; i++)
        state[i] = buf[i + 1];
    state[kernel_size - 2] = new_x;
}

// CPU reference: full recurrent gated delta rule over a sequence (for K3 validation)
// Q, K, V: (seq_len, heads, head_dim) in bf16
// g, beta: (seq_len, heads) in float
// state: (heads, head_dim, head_dim) in/out fp32
// output_fp32: (seq_len, heads, head_dim)
static inline void naive_chunk_gated_delta_rule_full(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float* g,
    const float* beta,
    float* state,
    float* output_fp32,
    int seq_len, int heads, int head_dim,
    bool use_l2norm)
{
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < heads; h++) {
            long long bhd = (long long)t * heads * head_dim + (long long)h * head_dim;
            long long bh  = (long long)t * heads + h;

            std::vector<float> q(head_dim), k(head_dim), v(head_dim);
            for (int d = 0; d < head_dim; d++) {
                q[d] = bf16_to_float(Q[bhd + d]);
                k[d] = bf16_to_float(K[bhd + d]);
                v[d] = bf16_to_float(V[bhd + d]);
            }

            if (use_l2norm) {
                float qn = 0, kn = 0;
                for (int d = 0; d < head_dim; d++) { qn += q[d]*q[d]; kn += k[d]*k[d]; }
                qn = sqrtf(qn + 1e-6f);
                kn = sqrtf(kn + 1e-6f);
                for (int d = 0; d < head_dim; d++) { q[d] /= qn; k[d] /= kn; }
            }
            for (int d = 0; d < head_dim; d++) q[d] *= scale;

            float* S = state + (long long)h * head_dim * head_dim;
            float decay = expf(g[bh]);
            float b = beta[bh];

            for (int i = 0; i < head_dim * head_dim; i++) S[i] *= decay;

            std::vector<float> Sk(head_dim, 0.0f);
            for (int vd = 0; vd < head_dim; vd++)
                for (int kd = 0; kd < head_dim; kd++)
                    Sk[vd] += S[kd * head_dim + vd] * k[kd];

            std::vector<float> v_new(head_dim);
            for (int vd = 0; vd < head_dim; vd++)
                v_new[vd] = b * (v[vd] - Sk[vd]);

            for (int kd = 0; kd < head_dim; kd++)
                for (int vd = 0; vd < head_dim; vd++)
                    S[kd * head_dim + vd] += k[kd] * v_new[vd];

            for (int vd = 0; vd < head_dim; vd++) {
                float val = 0;
                for (int kd = 0; kd < head_dim; kd++)
                    val += S[kd * head_dim + vd] * q[kd];
                output_fp32[bhd + vd] = val;
            }
        }
    }
}

// CPU reference: causal conv1d for a single channel
static inline void naive_causal_conv1d_channel(
    const __nv_bfloat16* x,      // (seq_len,) single channel
    const __nv_bfloat16* weight,  // (kernel_size,)
    float bias,
    float* output_fp32,           // (seq_len,)
    int seq_len, int kernel_size)
{
    for (int t = 0; t < seq_len; t++) {
        float out = bias;
        for (int w = 0; w < kernel_size; w++) {
            int idx = t - (kernel_size - 1) + w;
            float x_val = (idx >= 0) ? bf16_to_float(x[idx]) : 0.0f;
            out += bf16_to_float(weight[w]) * x_val;
        }
        output_fp32[t] = out;
    }
}
