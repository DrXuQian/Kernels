#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "deltanet.h"
#include "naive_reference.h"
#include "bench_utils.h"

// ============================================================================
// Model config (Qwen3.5-122B-A10B DeltaNet layer)
// ============================================================================
constexpr int BATCH = 1;
constexpr int SEQ_LEN = 3823;
constexpr int NUM_K_HEADS = 16;
constexpr int NUM_V_HEADS = 64;
constexpr int HEAD_DIM = 128;
constexpr int KEY_DIM = NUM_K_HEADS * HEAD_DIM;      // 2048
constexpr int VALUE_DIM = NUM_V_HEADS * HEAD_DIM;     // 8192
constexpr int CONV_DIM = KEY_DIM * 2 + VALUE_DIM;     // 12288
constexpr int CONV_KERNEL = 4;
constexpr int CHUNK_SIZE = 64;
constexpr float EPS = 1e-6f;
constexpr int DECODE_STEPS = 10;

// ============================================================================
// Error checking and timing
// ============================================================================
#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    float end(cudaStream_t s = 0) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Extract channels [ch_start, ch_start + dim) from (batch, total_dim, seq_len)
// and transpose to (batch, seq_len, heads, head_dim) layout.
__global__ void extract_transpose_bshd_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int batch, int total_dim, int seq_len,
    int ch_start, int heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % heads;
    int t = (idx / (head_dim * heads)) % seq_len;
    int b = idx / (head_dim * heads * seq_len);

    int ch = ch_start + h * head_dim + d;
    output[idx] = input[(long long)b * total_dim * seq_len + (long long)ch * seq_len + t];
}

// Repeat-interleave heads: (batch, seq, src_heads, dim) → (batch, seq, dst_heads, dim)
__global__ void repeat_interleave_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int batch, int seq_len, int src_heads, int dst_heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * dst_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h_dst = (idx / head_dim) % dst_heads;
    int t = (idx / (head_dim * dst_heads)) % seq_len;
    int b = idx / (head_dim * dst_heads * seq_len);

    int repeat = dst_heads / src_heads;
    int h_src = h_dst / repeat;

    long long src_idx = (long long)b * seq_len * src_heads * head_dim
                      + (long long)t * src_heads * head_dim
                      + (long long)h_src * head_dim + d;
    output[idx] = input[src_idx];
}

// ============================================================================
// Helper launch functions (init uses host-side from bench_utils.h)
// ============================================================================
void init_bf16(__nv_bfloat16* ptr, long long n, float scale, unsigned seed) {
    host_rand_bf16(ptr, n, scale, seed);
}

void init_logsig(float* ptr, long long n, unsigned seed) {
    host_rand_logsig(ptr, n, seed);
}

void init_sig(float* ptr, long long n, unsigned seed) {
    host_rand_sig(ptr, n, seed);
}

void extract_transpose(const __nv_bfloat16* in, __nv_bfloat16* out,
                       int batch, int total_dim, int seq_len,
                       int ch_start, int heads, int head_dim) {
    int total = batch * seq_len * heads * head_dim;
    extract_transpose_bshd_kernel<<<(total+255)/256, 256>>>(
        in, out, batch, total_dim, seq_len, ch_start, heads, head_dim);
}

void repeat_interleave(const __nv_bfloat16* in, __nv_bfloat16* out,
                       int batch, int seq_len, int src_heads, int dst_heads, int head_dim) {
    int total = batch * seq_len * dst_heads * head_dim;
    repeat_interleave_kernel<<<(total+255)/256, 256>>>(
        in, out, batch, seq_len, src_heads, dst_heads, head_dim);
}

// ============================================================================
// Numerical comparison
// ============================================================================
void compare_outputs(const char* name, const float* ref, const __nv_bfloat16* gpu_data,
                     int n, float atol = 1e-3f, float rtol = 1e-2f)
{
    std::vector<__nv_bfloat16> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), gpu_data, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    int pass = 0, fail = 0;
    float max_abs_err = 0, max_rel_err = 0;
    for (int i = 0; i < n; i++) {
        float gpu_val = __bfloat162float(host[i]);
        float ref_val = ref[i];
        float abs_err = fabsf(gpu_val - ref_val);
        float rel_err = (fabsf(ref_val) > 1e-8f) ? abs_err / fabsf(ref_val) : abs_err;
        max_abs_err = fmaxf(max_abs_err, abs_err);
        max_rel_err = fmaxf(max_rel_err, rel_err);
        if (abs_err <= atol + rtol * fabsf(ref_val)) pass++; else fail++;
    }
    printf("  [%s] %d/%d passed | max_abs_err=%.6f max_rel_err=%.6f %s\n",
           name, pass, n, max_abs_err, max_rel_err, (fail == 0) ? "PASS" : "FAIL");
}

// ============================================================================
// Main test
// ============================================================================
int main() {
    printf("=== DeltaNet CUDA Kernels Test ===\n");
    printf("BATCH=%d  SEQ_LEN=%d  NUM_K_HEADS=%d  NUM_V_HEADS=%d  HEAD_DIM=%d\n",
           BATCH, SEQ_LEN, NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM);
    printf("KEY_DIM=%d  VALUE_DIM=%d  CONV_DIM=%d  CONV_KERNEL=%d\n\n",
           KEY_DIM, VALUE_DIM, CONV_DIM, CONV_KERNEL);

    GpuTimer timer;

    // ===================== Allocate global buffers =====================

    // Conv weights & bias
    __nv_bfloat16 *d_conv_weight, *d_conv_bias;
    CUDA_CHECK(cudaMalloc(&d_conv_weight, (long long)CONV_DIM * CONV_KERNEL * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_conv_bias, CONV_DIM * sizeof(__nv_bfloat16)));
    init_bf16(d_conv_weight, (long long)CONV_DIM * CONV_KERNEL, 0.1f, 42);
    init_bf16(d_conv_bias, CONV_DIM, 0.01f, 43);

    // RMS norm weight
    __nv_bfloat16 *d_norm_weight;
    CUDA_CHECK(cudaMalloc(&d_norm_weight, VALUE_DIM * sizeof(__nv_bfloat16)));
    init_bf16(d_norm_weight, VALUE_DIM, 1.0f, 99);

    // Conv state (persistent across prefill → decode)
    __nv_bfloat16 *d_conv_state;
    CUDA_CHECK(cudaMalloc(&d_conv_state,
        (long long)BATCH * CONV_DIM * (CONV_KERNEL - 1) * sizeof(__nv_bfloat16)));

    // Recurrent state (persistent, fp32)
    float *d_recurrent_state;
    long long state_size = (long long)BATCH * NUM_V_HEADS * HEAD_DIM * HEAD_DIM;
    CUDA_CHECK(cudaMalloc(&d_recurrent_state, state_size * sizeof(float)));

    // =================================================================
    // ===================== PREFILL PHASE =============================
    // =================================================================
    printf("--- Prefill Phase (seq_len=%d) ---\n", SEQ_LEN);

    // Allocate prefill buffers
    long long conv_io = (long long)BATCH * CONV_DIM * SEQ_LEN;
    __nv_bfloat16 *d_x_conv, *d_y_conv;
    CUDA_CHECK(cudaMalloc(&d_x_conv, conv_io * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_y_conv, conv_io * sizeof(__nv_bfloat16)));
    init_bf16(d_x_conv, conv_io, 0.5f, 44);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Kernel 1: Causal conv1d prefill ---
    timer.begin();
    causal_conv1d_fn(d_x_conv, d_conv_weight, d_conv_bias, d_y_conv, d_conv_state,
                     BATCH, CONV_DIM, SEQ_LEN, CONV_KERNEL);
    float ms1 = timer.end();
    CUDA_CHECK(cudaGetLastError());
    printf("  Kernel 1 (causal_conv1d_fn):  %.3f ms\n", ms1);

    // Extract Q_raw, K_raw, V from conv output and transpose to BSHD
    long long qk_raw = (long long)BATCH * SEQ_LEN * NUM_K_HEADS * HEAD_DIM;
    long long v_total = (long long)BATCH * SEQ_LEN * NUM_V_HEADS * HEAD_DIM;
    __nv_bfloat16 *d_Q_raw, *d_K_raw, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q_raw, qk_raw * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K_raw, qk_raw * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, v_total * sizeof(__nv_bfloat16)));

    extract_transpose(d_y_conv, d_Q_raw, BATCH, CONV_DIM, SEQ_LEN, 0,         NUM_K_HEADS, HEAD_DIM);
    extract_transpose(d_y_conv, d_K_raw, BATCH, CONV_DIM, SEQ_LEN, KEY_DIM,   NUM_K_HEADS, HEAD_DIM);
    extract_transpose(d_y_conv, d_V,     BATCH, CONV_DIM, SEQ_LEN, 2*KEY_DIM, NUM_V_HEADS, HEAD_DIM);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Expand Q, K from NUM_K_HEADS=16 to NUM_V_HEADS=64 (repeat_interleave ×4)
    long long qk_expanded = (long long)BATCH * SEQ_LEN * NUM_V_HEADS * HEAD_DIM;
    __nv_bfloat16 *d_Q, *d_K;
    CUDA_CHECK(cudaMalloc(&d_Q, qk_expanded * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, qk_expanded * sizeof(__nv_bfloat16)));
    repeat_interleave(d_Q_raw, d_Q, BATCH, SEQ_LEN, NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM);
    repeat_interleave(d_K_raw, d_K, BATCH, SEQ_LEN, NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free raw Q/K and conv I/O
    cudaFree(d_Q_raw); cudaFree(d_K_raw);
    cudaFree(d_x_conv); cudaFree(d_y_conv);

    // Compute g (log-sigmoid decay) and beta (sigmoid write strength)
    long long gb_size = (long long)BATCH * SEQ_LEN * NUM_V_HEADS;
    float *d_g, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_g, gb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, gb_size * sizeof(float)));
    init_logsig(d_g, gb_size, 55);
    init_sig(d_beta, gb_size, 56);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Kernel 3: Chunk gated delta rule (prefill) ---
    __nv_bfloat16 *d_attn_out;
    CUDA_CHECK(cudaMalloc(&d_attn_out, v_total * sizeof(__nv_bfloat16)));

    timer.begin();
    chunk_gated_delta_rule(d_Q, d_K, d_V, d_g, d_beta,
                           d_attn_out, d_recurrent_state,
                           BATCH, SEQ_LEN, NUM_V_HEADS, HEAD_DIM, CHUNK_SIZE,
                           /*use_l2norm=*/true);
    float ms3 = timer.end();
    CUDA_CHECK(cudaGetLastError());
    printf("  Kernel 3 (chunk_gated_delta_rule): %.3f ms\n", ms3);

    // Free prefill Q, K, V, g, beta
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_g); cudaFree(d_beta);

    // --- Kernel 5: Fused RMS norm + gate (on prefill output) ---
    // Reshape attn_out to (BATCH * SEQ_LEN, NUM_V_HEADS * HEAD_DIM) = (3823, 8192)
    int norm_N = BATCH * SEQ_LEN;
    int norm_D = NUM_V_HEADS * HEAD_DIM;  // 8192

    __nv_bfloat16 *d_gate_input, *d_norm_out;
    CUDA_CHECK(cudaMalloc(&d_gate_input, (long long)norm_N * norm_D * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_norm_out, (long long)norm_N * norm_D * sizeof(__nv_bfloat16)));
    init_bf16(d_gate_input, (long long)norm_N * norm_D, 0.5f, 70);
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    fused_rms_norm_gate(d_attn_out, d_gate_input, d_norm_weight, d_norm_out,
                        norm_N, norm_D, EPS);
    float ms5 = timer.end();
    CUDA_CHECK(cudaGetLastError());
    printf("  Kernel 5 (fused_rms_norm_gate): %.3f ms\n", ms5);

    cudaFree(d_attn_out); cudaFree(d_gate_input); cudaFree(d_norm_out);
    printf("  Prefill complete. Conv state and recurrent state saved.\n\n");

    // =================================================================
    // ===================== DECODE PHASE ==============================
    // =================================================================
    printf("--- Decode Phase (%d steps) ---\n", DECODE_STEPS);

    // Allocate decode buffers (small: single timestep)
    __nv_bfloat16 *d_x_dec, *d_y_dec;
    CUDA_CHECK(cudaMalloc(&d_x_dec, (long long)BATCH * CONV_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_y_dec, (long long)BATCH * CONV_DIM * sizeof(__nv_bfloat16)));

    // Decode BSHD buffers (seq_len=1)
    long long dec_qk_raw_n = (long long)BATCH * 1 * NUM_K_HEADS * HEAD_DIM;
    long long dec_v_n = (long long)BATCH * 1 * NUM_V_HEADS * HEAD_DIM;
    long long dec_qk_n = (long long)BATCH * 1 * NUM_V_HEADS * HEAD_DIM;
    __nv_bfloat16 *d_dec_Q_raw, *d_dec_K_raw, *d_dec_V;
    __nv_bfloat16 *d_dec_Q, *d_dec_K;
    __nv_bfloat16 *d_dec_out;
    float *d_dec_g, *d_dec_beta;

    CUDA_CHECK(cudaMalloc(&d_dec_Q_raw, dec_qk_raw_n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dec_K_raw, dec_qk_raw_n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dec_V, dec_v_n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dec_Q, dec_qk_n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dec_K, dec_qk_n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dec_out, dec_v_n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dec_g, (long long)BATCH * NUM_V_HEADS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_beta, (long long)BATCH * NUM_V_HEADS * sizeof(float)));

    // Norm buffers for decode
    __nv_bfloat16 *d_dec_gate_in, *d_dec_norm_out;
    CUDA_CHECK(cudaMalloc(&d_dec_gate_in, dec_v_n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dec_norm_out, dec_v_n * sizeof(__nv_bfloat16)));

    float total_decode_ms = 0;
    for (int step = 0; step < DECODE_STEPS; step++) {
        unsigned seed_base = 1000 + step * 10;

        // Random input for this decode step
        init_bf16(d_x_dec, (long long)BATCH * CONV_DIM, 0.5f, seed_base);
        init_logsig(d_dec_g, (long long)BATCH * NUM_V_HEADS, seed_base + 1);
        init_sig(d_dec_beta, (long long)BATCH * NUM_V_HEADS, seed_base + 2);
        init_bf16(d_dec_gate_in, dec_v_n, 0.5f, seed_base + 3);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.begin();

        // Kernel 2: Conv update
        causal_conv1d_update(d_x_dec, d_conv_state, d_conv_weight, d_conv_bias,
                             d_y_dec, BATCH, CONV_DIM, CONV_KERNEL);

        // Extract and transpose (seq_len=1)
        extract_transpose(d_y_dec, d_dec_Q_raw, BATCH, CONV_DIM, 1, 0,         NUM_K_HEADS, HEAD_DIM);
        extract_transpose(d_y_dec, d_dec_K_raw, BATCH, CONV_DIM, 1, KEY_DIM,   NUM_K_HEADS, HEAD_DIM);
        extract_transpose(d_y_dec, d_dec_V,     BATCH, CONV_DIM, 1, 2*KEY_DIM, NUM_V_HEADS, HEAD_DIM);

        // Expand Q, K
        repeat_interleave(d_dec_Q_raw, d_dec_Q, BATCH, 1, NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM);
        repeat_interleave(d_dec_K_raw, d_dec_K, BATCH, 1, NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM);

        // Kernel 4: Fused recurrent delta rule
        fused_recurrent_gated_delta_rule(
            d_dec_Q, d_dec_K, d_dec_V,
            d_dec_g, d_dec_beta,
            d_recurrent_state, d_dec_out,
            BATCH, NUM_V_HEADS, HEAD_DIM,
            /*use_l2norm=*/true);

        // Kernel 5: Norm + gate
        fused_rms_norm_gate(d_dec_out, d_dec_gate_in, d_norm_weight, d_dec_norm_out,
                            BATCH, norm_D, EPS);

        float step_ms = timer.end();
        CUDA_CHECK(cudaGetLastError());
        total_decode_ms += step_ms;

        if (step == 0 || step == DECODE_STEPS - 1)
            printf("  Step %d: %.3f ms\n", step, step_ms);
    }
    printf("  Avg decode step: %.3f ms\n\n", total_decode_ms / DECODE_STEPS);

    // =================================================================
    // ===================== CPU REFERENCE VALIDATION ==================
    // =================================================================
    printf("--- Numerical Validation ---\n");

    // Validate Kernel 4 (fused_recurrent) against CPU reference
    // We'll run one more decode step and compare
    {
        unsigned seed_v = 9999;
        init_bf16(d_x_dec, (long long)BATCH * CONV_DIM, 0.5f, seed_v);
        init_logsig(d_dec_g, (long long)BATCH * NUM_V_HEADS, seed_v + 1);
        init_sig(d_dec_beta, (long long)BATCH * NUM_V_HEADS, seed_v + 2);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Run conv update to get Q, K, V
        causal_conv1d_update(d_x_dec, d_conv_state, d_conv_weight, d_conv_bias,
                             d_y_dec, BATCH, CONV_DIM, CONV_KERNEL);
        extract_transpose(d_y_dec, d_dec_Q_raw, BATCH, CONV_DIM, 1, 0,         NUM_K_HEADS, HEAD_DIM);
        extract_transpose(d_y_dec, d_dec_K_raw, BATCH, CONV_DIM, 1, KEY_DIM,   NUM_K_HEADS, HEAD_DIM);
        extract_transpose(d_y_dec, d_dec_V,     BATCH, CONV_DIM, 1, 2*KEY_DIM, NUM_V_HEADS, HEAD_DIM);
        repeat_interleave(d_dec_Q_raw, d_dec_Q, BATCH, 1, NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM);
        repeat_interleave(d_dec_K_raw, d_dec_K, BATCH, 1, NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy inputs to host for CPU reference
        int qkv_n = NUM_V_HEADS * HEAD_DIM;
        std::vector<__nv_bfloat16> h_Q(qkv_n), h_K(qkv_n), h_V(qkv_n);
        std::vector<float> h_g(NUM_V_HEADS), h_beta(NUM_V_HEADS);
        CUDA_CHECK(cudaMemcpy(h_Q.data(), d_dec_Q, qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_K.data(), d_dec_K, qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_V.data(), d_dec_V, qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_g.data(), d_dec_g, NUM_V_HEADS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_beta.data(), d_dec_beta, NUM_V_HEADS * sizeof(float), cudaMemcpyDeviceToHost));

        // Copy state to host (for CPU) and save a backup
        std::vector<float> h_state(state_size);
        CUDA_CHECK(cudaMemcpy(h_state.data(), d_recurrent_state, state_size * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> h_state_cpu(h_state); // copy for CPU reference

        // Run GPU kernel
        fused_recurrent_gated_delta_rule(
            d_dec_Q, d_dec_K, d_dec_V,
            d_dec_g, d_dec_beta,
            d_recurrent_state, d_dec_out,
            BATCH, NUM_V_HEADS, HEAD_DIM,
            /*use_l2norm=*/true);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Run CPU reference
        std::vector<float> cpu_output(qkv_n, 0.0f);
        naive_fused_recurrent_step(
            h_Q.data(), h_K.data(), h_V.data(),
            h_g.data(), h_beta.data(),
            h_state_cpu.data(), cpu_output.data(),
            NUM_V_HEADS, HEAD_DIM, /*use_l2norm=*/true);

        // Compare
        compare_outputs("Kernel 4 (fused_recurrent)", cpu_output.data(), d_dec_out, qkv_n);

        // Also validate state
        std::vector<float> h_state_gpu(state_size);
        CUDA_CHECK(cudaMemcpy(h_state_gpu.data(), d_recurrent_state, state_size * sizeof(float), cudaMemcpyDeviceToHost));

        int state_pass = 0, state_fail = 0;
        float state_max_err = 0;
        for (long long i = 0; i < state_size; i++) {
            float err = fabsf(h_state_gpu[i] - h_state_cpu[i]);
            state_max_err = fmaxf(state_max_err, err);
            if (err <= 1e-3f + 1e-2f * fabsf(h_state_cpu[i])) state_pass++; else state_fail++;
        }
        printf("  [State] %d/%lld passed | max_err=%.6f %s\n",
               state_pass, state_size, state_max_err, (state_fail == 0) ? "PASS" : "FAIL");
    }

    // Validate Kernel 5 (fused_rms_norm_gate) against CPU reference
    {
        // Use the decode output as test input
        __nv_bfloat16 *d_test_x, *d_test_z, *d_test_out;
        int D = norm_D;
        CUDA_CHECK(cudaMalloc(&d_test_x, D * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_test_z, D * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_test_out, D * sizeof(__nv_bfloat16)));
        init_bf16(d_test_x, D, 1.0f, 777);
        init_bf16(d_test_z, D, 1.0f, 778);
        CUDA_CHECK(cudaDeviceSynchronize());

        fused_rms_norm_gate(d_test_x, d_test_z, d_norm_weight, d_test_out, 1, D, EPS);
        CUDA_CHECK(cudaDeviceSynchronize());

        // CPU reference
        std::vector<__nv_bfloat16> h_x(D), h_z(D), h_w(D);
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_test_x, D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_z.data(), d_test_z, D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_w.data(), d_norm_weight, D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        std::vector<float> cpu_norm_out(D);
        naive_rms_norm_gate(h_x.data(), h_z.data(), h_w.data(), cpu_norm_out.data(), D, EPS);

        compare_outputs("Kernel 5 (fused_rms_norm_gate)", cpu_norm_out.data(), d_test_out, D);

        cudaFree(d_test_x); cudaFree(d_test_z); cudaFree(d_test_out);
    }

    // Validate Kernel 1 (causal_conv1d) against CPU reference for a few channels
    {
        printf("  Validating Kernel 1 (causal_conv1d) on 4 random channels...\n");
        int test_dim = 4;
        int test_seq = 256;
        __nv_bfloat16 *d_cx, *d_cy, *d_cw, *d_cb;
        CUDA_CHECK(cudaMalloc(&d_cx, (long long)test_dim * test_seq * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_cy, (long long)test_dim * test_seq * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_cw, (long long)test_dim * CONV_KERNEL * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_cb, test_dim * sizeof(__nv_bfloat16)));
        init_bf16(d_cx, (long long)test_dim * test_seq, 0.5f, 333);
        init_bf16(d_cw, (long long)test_dim * CONV_KERNEL, 0.3f, 334);
        init_bf16(d_cb, test_dim, 0.01f, 335);
        CUDA_CHECK(cudaDeviceSynchronize());

        causal_conv1d_fn(d_cx, d_cw, d_cb, d_cy, nullptr, 1, test_dim, test_seq, CONV_KERNEL);
        CUDA_CHECK(cudaDeviceSynchronize());

        // CPU reference for each channel
        std::vector<__nv_bfloat16> h_cx(test_dim * test_seq), h_cw(test_dim * CONV_KERNEL), h_cb(test_dim);
        CUDA_CHECK(cudaMemcpy(h_cx.data(), d_cx, test_dim * test_seq * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cw.data(), d_cw, test_dim * CONV_KERNEL * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cb.data(), d_cb, test_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        std::vector<float> cpu_conv_ref(test_dim * test_seq);
        for (int d = 0; d < test_dim; d++) {
            float bias = __bfloat162float(h_cb[d]);
            naive_causal_conv1d_channel(
                h_cx.data() + d * test_seq,
                h_cw.data() + d * CONV_KERNEL,
                bias,
                cpu_conv_ref.data() + d * test_seq,
                test_seq, CONV_KERNEL);
        }

        // Compare all channels together
        std::vector<__nv_bfloat16> h_cy(test_dim * test_seq);
        CUDA_CHECK(cudaMemcpy(h_cy.data(), d_cy, test_dim * test_seq * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        int pass = 0, fail = 0;
        float max_err = 0;
        for (int i = 0; i < test_dim * test_seq; i++) {
            float gpu_val = __bfloat162float(h_cy[i]);
            float ref_val = cpu_conv_ref[i];
            float err = fabsf(gpu_val - ref_val);
            max_err = fmaxf(max_err, err);
            if (err <= 1e-3f + 1e-2f * fabsf(ref_val)) pass++; else fail++;
        }
        printf("  [Kernel 1 (conv1d)] %d/%d passed | max_err=%.6f %s\n",
               pass, test_dim * test_seq, max_err, (fail == 0) ? "PASS" : "FAIL");

        cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_cw); cudaFree(d_cb);
    }

    // Validate Kernel 2 (causal_conv1d_update) against CPU reference
    {
        printf("  Validating Kernel 2 (causal_conv1d_update) on 4 channels x 5 steps...\n");
        int test_dim = 4;
        __nv_bfloat16 *d2_x, *d2_y, *d2_w, *d2_b, *d2_st;
        CUDA_CHECK(cudaMalloc(&d2_w, (long long)test_dim * CONV_KERNEL * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d2_b, test_dim * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d2_st, (long long)test_dim * (CONV_KERNEL - 1) * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d2_x, test_dim * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d2_y, test_dim * sizeof(__nv_bfloat16)));

        init_bf16(d2_w, (long long)test_dim * CONV_KERNEL, 0.3f, 400);
        init_bf16(d2_b, test_dim, 0.01f, 401);
        // Initialize state to known values (simulate end of a prefill)
        init_bf16(d2_st, (long long)test_dim * (CONV_KERNEL - 1), 0.2f, 402);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy to host for CPU reference
        std::vector<__nv_bfloat16> h2_w(test_dim * CONV_KERNEL), h2_b(test_dim);
        std::vector<__nv_bfloat16> h2_st_gpu(test_dim * (CONV_KERNEL - 1));
        CUDA_CHECK(cudaMemcpy(h2_w.data(), d2_w, test_dim * CONV_KERNEL * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h2_b.data(), d2_b, test_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h2_st_gpu.data(), d2_st, test_dim * (CONV_KERNEL - 1) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        // CPU state (fp32 copy)
        std::vector<float> cpu_state(test_dim * (CONV_KERNEL - 1));
        for (int i = 0; i < test_dim * (CONV_KERNEL - 1); i++)
            cpu_state[i] = __bfloat162float(h2_st_gpu[i]);

        int total_pass = 0, total_fail = 0;
        float total_max_err = 0;

        for (int step = 0; step < 5; step++) {
            init_bf16(d2_x, test_dim, 0.5f, 500 + step);
            CUDA_CHECK(cudaDeviceSynchronize());

            // GPU: update
            causal_conv1d_update(d2_x, d2_st, d2_w, d2_b, d2_y, 1, test_dim, CONV_KERNEL);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy input to host
            std::vector<__nv_bfloat16> h2_x(test_dim);
            CUDA_CHECK(cudaMemcpy(h2_x.data(), d2_x, test_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

            // CPU: update each channel
            std::vector<float> cpu_out(test_dim);
            for (int d = 0; d < test_dim; d++) {
                float bias_val = __bfloat162float(h2_b[d]);
                float new_x = __bfloat162float(h2_x[d]);
                naive_causal_conv1d_update_channel(
                    new_x,
                    cpu_state.data() + d * (CONV_KERNEL - 1),
                    h2_w.data() + d * CONV_KERNEL,
                    bias_val,
                    &cpu_out[d],
                    CONV_KERNEL);
            }

            // Compare
            std::vector<__nv_bfloat16> h2_y(test_dim);
            CUDA_CHECK(cudaMemcpy(h2_y.data(), d2_y, test_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            for (int d = 0; d < test_dim; d++) {
                float gpu_val = __bfloat162float(h2_y[d]);
                float ref_val = cpu_out[d];
                float err = fabsf(gpu_val - ref_val);
                total_max_err = fmaxf(total_max_err, err);
                if (err <= 1e-3f + 1e-2f * fabsf(ref_val)) total_pass++; else total_fail++;
            }

            // Sync CPU state with GPU state (read back GPU state for next step)
            CUDA_CHECK(cudaMemcpy(h2_st_gpu.data(), d2_st,
                test_dim * (CONV_KERNEL - 1) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            for (int i = 0; i < test_dim * (CONV_KERNEL - 1); i++)
                cpu_state[i] = __bfloat162float(h2_st_gpu[i]);
        }
        printf("  [Kernel 2 (conv1d_update)] %d/%d passed | max_err=%.6f %s\n",
               total_pass, total_pass + total_fail, total_max_err,
               (total_fail == 0) ? "PASS" : "FAIL");

        cudaFree(d2_x); cudaFree(d2_y); cudaFree(d2_w); cudaFree(d2_b); cudaFree(d2_st);
    }

    // Validate Kernel 3 (chunk_gated_delta_rule) against CPU reference
    // Use small dimensions to make CPU reference tractable
    {
        printf("  Validating Kernel 3 (chunk_gated_delta_rule) on small problem...\n");
        int t_heads = 2;
        int t_hdim = 32;  // small head_dim for CPU feasibility
        int t_seq = 64;   // one chunk
        int t_batch = 1;

        long long t_qkv_n = (long long)t_batch * t_seq * t_heads * t_hdim;
        long long t_gb_n   = (long long)t_batch * t_seq * t_heads;
        long long t_state_n = (long long)t_batch * t_heads * t_hdim * t_hdim;

        __nv_bfloat16 *d3_Q, *d3_K, *d3_V, *d3_out;
        float *d3_g, *d3_beta, *d3_state;
        CUDA_CHECK(cudaMalloc(&d3_Q, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d3_K, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d3_V, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d3_out, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d3_g, t_gb_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d3_beta, t_gb_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d3_state, t_state_n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d3_state, 0, t_state_n * sizeof(float)));

        init_bf16(d3_Q, t_qkv_n, 0.5f, 600);
        init_bf16(d3_K, t_qkv_n, 0.5f, 601);
        init_bf16(d3_V, t_qkv_n, 0.5f, 602);
        init_logsig(d3_g, t_gb_n, 603);
        init_sig(d3_beta, t_gb_n, 604);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Note: kernel 3 uses HEAD_DIM_MAX=128 in the kernel, but for validation
        // we need head_dim to equal blockDim.x. Let me use 128-dim and 2 heads.
        // Actually the kernel uses head_dim as blockDim, so we must pass head_dim
        // as the actual value. Let's use 128 with 2 heads and short seq.
        cudaFree(d3_Q); cudaFree(d3_K); cudaFree(d3_V); cudaFree(d3_out);
        cudaFree(d3_g); cudaFree(d3_beta); cudaFree(d3_state);

        // Redo with head_dim=128 (required by kernel)
        t_hdim = 128;
        t_heads = 2;
        t_seq = 32;  // short seq for CPU speed
        t_qkv_n = (long long)t_batch * t_seq * t_heads * t_hdim;
        t_gb_n   = (long long)t_batch * t_seq * t_heads;
        t_state_n = (long long)t_batch * t_heads * t_hdim * t_hdim;

        CUDA_CHECK(cudaMalloc(&d3_Q, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d3_K, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d3_V, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d3_out, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d3_g, t_gb_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d3_beta, t_gb_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d3_state, t_state_n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d3_state, 0, t_state_n * sizeof(float)));

        init_bf16(d3_Q, t_qkv_n, 0.3f, 600);
        init_bf16(d3_K, t_qkv_n, 0.3f, 601);
        init_bf16(d3_V, t_qkv_n, 0.3f, 602);
        init_logsig(d3_g, t_gb_n, 603);
        init_sig(d3_beta, t_gb_n, 604);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Run GPU
        chunk_gated_delta_rule(d3_Q, d3_K, d3_V, d3_g, d3_beta,
                               d3_out, d3_state,
                               t_batch, t_seq, t_heads, t_hdim, CHUNK_SIZE,
                               /*use_l2norm=*/true);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy to host for CPU reference
        std::vector<__nv_bfloat16> h3_Q(t_qkv_n), h3_K(t_qkv_n), h3_V(t_qkv_n);
        std::vector<float> h3_g(t_gb_n), h3_beta(t_gb_n);
        CUDA_CHECK(cudaMemcpy(h3_Q.data(), d3_Q, t_qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h3_K.data(), d3_K, t_qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h3_V.data(), d3_V, t_qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h3_g.data(), d3_g, t_gb_n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h3_beta.data(), d3_beta, t_gb_n * sizeof(float), cudaMemcpyDeviceToHost));

        // CPU reference
        std::vector<float> cpu3_state(t_state_n, 0.0f);
        std::vector<float> cpu3_out(t_qkv_n, 0.0f);
        naive_chunk_gated_delta_rule_full(
            h3_Q.data(), h3_K.data(), h3_V.data(),
            h3_g.data(), h3_beta.data(),
            cpu3_state.data(), cpu3_out.data(),
            t_seq, t_heads, t_hdim, /*use_l2norm=*/true);

        // Compare output
        compare_outputs("Kernel 3 (chunk_gated_delta) output", cpu3_out.data(), d3_out, (int)t_qkv_n);

        // Compare state
        std::vector<float> h3_state_gpu(t_state_n);
        CUDA_CHECK(cudaMemcpy(h3_state_gpu.data(), d3_state, t_state_n * sizeof(float), cudaMemcpyDeviceToHost));
        int s3_pass = 0, s3_fail = 0;
        float s3_max_err = 0;
        for (long long i = 0; i < t_state_n; i++) {
            float err = fabsf(h3_state_gpu[i] - cpu3_state[i]);
            s3_max_err = fmaxf(s3_max_err, err);
            if (err <= 1e-2f + 1e-1f * fabsf(cpu3_state[i])) s3_pass++; else s3_fail++;
        }
        printf("  [Kernel 3 state] %d/%lld passed | max_err=%.6f %s\n",
               s3_pass, t_state_n, s3_max_err, (s3_fail == 0) ? "PASS" : "FAIL");

        cudaFree(d3_Q); cudaFree(d3_K); cudaFree(d3_V); cudaFree(d3_out);
        cudaFree(d3_g); cudaFree(d3_beta); cudaFree(d3_state);
    }

    // Validate Kernel 2+4 continuity: prefill then decode, compare vs pure recurrent
    {
        printf("  Validating prefill→decode state continuity (K1→K2, K3→K4)...\n");
        int t_heads = 2;
        int t_hdim = 128;
        int t_seq = 16;  // short prefill
        int t_batch = 1;
        int t_dim = 4;   // small conv dim for testing

        // === Conv continuity test (K1 → K2) ===
        long long cio = (long long)t_dim * t_seq;
        __nv_bfloat16 *dc_x, *dc_y, *dc_w, *dc_b, *dc_st;
        CUDA_CHECK(cudaMalloc(&dc_x, cio * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dc_y, cio * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dc_w, (long long)t_dim * CONV_KERNEL * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dc_b, t_dim * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dc_st, (long long)t_dim * (CONV_KERNEL - 1) * sizeof(__nv_bfloat16)));
        init_bf16(dc_x, cio, 0.5f, 700);
        init_bf16(dc_w, (long long)t_dim * CONV_KERNEL, 0.3f, 701);
        init_bf16(dc_b, t_dim, 0.01f, 702);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Prefill
        causal_conv1d_fn(dc_x, dc_w, dc_b, dc_y, dc_st, 1, t_dim, t_seq, CONV_KERNEL);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Now decode one step
        __nv_bfloat16 *dc_xd, *dc_yd;
        CUDA_CHECK(cudaMalloc(&dc_xd, t_dim * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dc_yd, t_dim * sizeof(__nv_bfloat16)));
        init_bf16(dc_xd, t_dim, 0.5f, 710);
        CUDA_CHECK(cudaDeviceSynchronize());

        causal_conv1d_update(dc_xd, dc_st, dc_w, dc_b, dc_yd, 1, t_dim, CONV_KERNEL);
        CUDA_CHECK(cudaDeviceSynchronize());

        // CPU reference: run conv over (t_seq+1) timesteps, check last output matches
        std::vector<__nv_bfloat16> hc_x(cio), hc_xd(t_dim), hc_w(t_dim * CONV_KERNEL), hc_b(t_dim);
        CUDA_CHECK(cudaMemcpy(hc_x.data(), dc_x, cio * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hc_xd.data(), dc_xd, t_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hc_w.data(), dc_w, t_dim * CONV_KERNEL * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hc_b.data(), dc_b, t_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        // Build full sequence: [prefill_x..., decode_x]
        int full_seq = t_seq + 1;
        std::vector<__nv_bfloat16> full_x(t_dim * full_seq);
        for (int d = 0; d < t_dim; d++) {
            for (int t = 0; t < t_seq; t++)
                full_x[d * full_seq + t] = hc_x[d * t_seq + t];
            full_x[d * full_seq + t_seq] = hc_xd[d];
        }

        std::vector<float> cpu_full_out(t_dim * full_seq);
        for (int d = 0; d < t_dim; d++) {
            float bias_val = __bfloat162float(hc_b[d]);
            naive_causal_conv1d_channel(
                full_x.data() + d * full_seq,
                hc_w.data() + d * CONV_KERNEL,
                bias_val,
                cpu_full_out.data() + d * full_seq,
                full_seq, CONV_KERNEL);
        }

        // Compare decode output (last timestep of full seq) vs GPU decode output
        std::vector<__nv_bfloat16> hc_yd(t_dim);
        CUDA_CHECK(cudaMemcpy(hc_yd.data(), dc_yd, t_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        int cp = 0, cf = 0; float cme = 0;
        for (int d = 0; d < t_dim; d++) {
            float gpu_val = __bfloat162float(hc_yd[d]);
            float ref_val = cpu_full_out[d * full_seq + t_seq];
            float err = fabsf(gpu_val - ref_val);
            cme = fmaxf(cme, err);
            if (err <= 1e-3f + 1e-2f * fabsf(ref_val)) cp++; else cf++;
        }
        printf("  [K1→K2 continuity] %d/%d passed | max_err=%.6f %s\n",
               cp, t_dim, cme, (cf == 0) ? "PASS" : "FAIL");

        cudaFree(dc_x); cudaFree(dc_y); cudaFree(dc_w); cudaFree(dc_b); cudaFree(dc_st);
        cudaFree(dc_xd); cudaFree(dc_yd);

        // === Delta rule continuity test (K3 → K4) ===
        long long t_qkv_n = (long long)t_batch * t_seq * t_heads * t_hdim;
        long long t_gb_n   = (long long)t_batch * t_seq * t_heads;
        long long t_state_n = (long long)t_batch * t_heads * t_hdim * t_hdim;

        __nv_bfloat16 *dr_Q, *dr_K, *dr_V, *dr_out;
        float *dr_g, *dr_beta, *dr_state;
        CUDA_CHECK(cudaMalloc(&dr_Q, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dr_K, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dr_V, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dr_out, t_qkv_n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dr_g, t_gb_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dr_beta, t_gb_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dr_state, t_state_n * sizeof(float)));
        CUDA_CHECK(cudaMemset(dr_state, 0, t_state_n * sizeof(float)));
        init_bf16(dr_Q, t_qkv_n, 0.3f, 800);
        init_bf16(dr_K, t_qkv_n, 0.3f, 801);
        init_bf16(dr_V, t_qkv_n, 0.3f, 802);
        init_logsig(dr_g, t_gb_n, 803);
        init_sig(dr_beta, t_gb_n, 804);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Prefill (K3)
        chunk_gated_delta_rule(dr_Q, dr_K, dr_V, dr_g, dr_beta,
                               dr_out, dr_state,
                               t_batch, t_seq, t_heads, t_hdim, CHUNK_SIZE,
                               /*use_l2norm=*/true);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Now one decode step (K4)
        long long dec_qkv = (long long)t_heads * t_hdim;
        __nv_bfloat16 *dr_dQ, *dr_dK, *dr_dV, *dr_dout;
        float *dr_dg, *dr_dbeta;
        CUDA_CHECK(cudaMalloc(&dr_dQ, dec_qkv * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dr_dK, dec_qkv * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dr_dV, dec_qkv * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dr_dout, dec_qkv * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dr_dg, t_heads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dr_dbeta, t_heads * sizeof(float)));
        init_bf16(dr_dQ, dec_qkv, 0.3f, 810);
        init_bf16(dr_dK, dec_qkv, 0.3f, 811);
        init_bf16(dr_dV, dec_qkv, 0.3f, 812);
        init_logsig(dr_dg, t_heads, 813);
        init_sig(dr_dbeta, t_heads, 814);
        CUDA_CHECK(cudaDeviceSynchronize());

        fused_recurrent_gated_delta_rule(
            dr_dQ, dr_dK, dr_dV, dr_dg, dr_dbeta,
            dr_state, dr_dout,
            t_batch, t_heads, t_hdim,
            /*use_l2norm=*/true);
        CUDA_CHECK(cudaDeviceSynchronize());

        // CPU: run the full (t_seq+1) steps as pure recurrent
        // Build concatenated inputs
        int full_s = t_seq + 1;
        long long full_qkv = (long long)full_s * t_heads * t_hdim;
        long long full_gb  = (long long)full_s * t_heads;
        std::vector<__nv_bfloat16> hr_Q(full_qkv), hr_K(full_qkv), hr_V(full_qkv);
        std::vector<float> hr_g(full_gb), hr_beta(full_gb);

        // Copy prefill data
        std::vector<__nv_bfloat16> tmp_Q(t_qkv_n), tmp_K(t_qkv_n), tmp_V(t_qkv_n);
        std::vector<float> tmp_g(t_gb_n), tmp_beta(t_gb_n);
        CUDA_CHECK(cudaMemcpy(tmp_Q.data(), dr_Q, t_qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_K.data(), dr_K, t_qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_V.data(), dr_V, t_qkv_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_g.data(), dr_g, t_gb_n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_beta.data(), dr_beta, t_gb_n * sizeof(float), cudaMemcpyDeviceToHost));

        // Copy decode data
        std::vector<__nv_bfloat16> tmp_dQ(dec_qkv), tmp_dK(dec_qkv), tmp_dV(dec_qkv);
        std::vector<float> tmp_dg(t_heads), tmp_dbeta(t_heads);
        CUDA_CHECK(cudaMemcpy(tmp_dQ.data(), dr_dQ, dec_qkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_dK.data(), dr_dK, dec_qkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_dV.data(), dr_dV, dec_qkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_dg.data(), dr_dg, t_heads * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_dbeta.data(), dr_dbeta, t_heads * sizeof(float), cudaMemcpyDeviceToHost));

        // Concatenate
        for (int t = 0; t < t_seq; t++) {
            for (int i = 0; i < t_heads * t_hdim; i++) {
                hr_Q[t * t_heads * t_hdim + i] = tmp_Q[t * t_heads * t_hdim + i];
                hr_K[t * t_heads * t_hdim + i] = tmp_K[t * t_heads * t_hdim + i];
                hr_V[t * t_heads * t_hdim + i] = tmp_V[t * t_heads * t_hdim + i];
            }
            for (int i = 0; i < t_heads; i++) {
                hr_g[t * t_heads + i] = tmp_g[t * t_heads + i];
                hr_beta[t * t_heads + i] = tmp_beta[t * t_heads + i];
            }
        }
        for (int i = 0; i < t_heads * t_hdim; i++) {
            hr_Q[t_seq * t_heads * t_hdim + i] = tmp_dQ[i];
            hr_K[t_seq * t_heads * t_hdim + i] = tmp_dK[i];
            hr_V[t_seq * t_heads * t_hdim + i] = tmp_dV[i];
        }
        for (int i = 0; i < t_heads; i++) {
            hr_g[t_seq * t_heads + i] = tmp_dg[i];
            hr_beta[t_seq * t_heads + i] = tmp_dbeta[i];
        }

        // Run CPU full recurrent over all (t_seq+1) steps
        std::vector<float> cpu_dr_state(t_state_n, 0.0f);
        std::vector<float> cpu_dr_out(full_qkv, 0.0f);
        naive_chunk_gated_delta_rule_full(
            hr_Q.data(), hr_K.data(), hr_V.data(),
            hr_g.data(), hr_beta.data(),
            cpu_dr_state.data(), cpu_dr_out.data(),
            full_s, t_heads, t_hdim, /*use_l2norm=*/true);

        // Compare last timestep output
        std::vector<__nv_bfloat16> hr_dout(dec_qkv);
        CUDA_CHECK(cudaMemcpy(hr_dout.data(), dr_dout, dec_qkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        int rp = 0, rf = 0; float rme = 0;
        for (int i = 0; i < (int)dec_qkv; i++) {
            float gpu_val = __bfloat162float(hr_dout[i]);
            float ref_val = cpu_dr_out[t_seq * t_heads * t_hdim + i];
            float err = fabsf(gpu_val - ref_val);
            rme = fmaxf(rme, err);
            if (err <= 5e-3f + 5e-2f * fabsf(ref_val)) rp++; else rf++;
        }
        printf("  [K3→K4 continuity] %d/%d passed | max_err=%.6f %s\n",
               rp, (int)dec_qkv, rme, (rf == 0) ? "PASS" : "FAIL");

        // Compare final state
        std::vector<float> hr_state_gpu(t_state_n);
        CUDA_CHECK(cudaMemcpy(hr_state_gpu.data(), dr_state, t_state_n * sizeof(float), cudaMemcpyDeviceToHost));
        int sp = 0, sf = 0; float sme = 0;
        for (long long i = 0; i < t_state_n; i++) {
            float err = fabsf(hr_state_gpu[i] - cpu_dr_state[i]);
            sme = fmaxf(sme, err);
            if (err <= 5e-3f + 5e-2f * fabsf(cpu_dr_state[i])) sp++; else sf++;
        }
        printf("  [K3→K4 state] %d/%lld passed | max_err=%.6f %s\n",
               sp, t_state_n, sme, (sf == 0) ? "PASS" : "FAIL");

        cudaFree(dr_Q); cudaFree(dr_K); cudaFree(dr_V); cudaFree(dr_out);
        cudaFree(dr_g); cudaFree(dr_beta); cudaFree(dr_state);
        cudaFree(dr_dQ); cudaFree(dr_dK); cudaFree(dr_dV); cudaFree(dr_dout);
        cudaFree(dr_dg); cudaFree(dr_dbeta);
    }

    // =================================================================
    // Cleanup
    // =================================================================
    cudaFree(d_conv_weight); cudaFree(d_conv_bias); cudaFree(d_norm_weight);
    cudaFree(d_conv_state); cudaFree(d_recurrent_state);
    cudaFree(d_x_dec); cudaFree(d_y_dec);
    cudaFree(d_dec_Q_raw); cudaFree(d_dec_K_raw); cudaFree(d_dec_V);
    cudaFree(d_dec_Q); cudaFree(d_dec_K); cudaFree(d_dec_out);
    cudaFree(d_dec_g); cudaFree(d_dec_beta);
    cudaFree(d_dec_gate_in); cudaFree(d_dec_norm_out);

    printf("\n=== All tests complete ===\n");
    return 0;
}
