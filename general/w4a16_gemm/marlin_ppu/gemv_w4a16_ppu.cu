// W4A16 GEMV (M=1 decode) reading Marlin's INT4 B format in place.
//
// WHY THIS EXISTS. Marlin is a GEMM: its tile is 16x128 and decode uses 1 of those 16 rows. Worse, its (tile, k)
// decomposition obeys blocks * iters ~= n_tiles * k_tiles, so grid size and pipeline depth trade against each other and
// decode tops out around 37% of HBM no matter how the knobs are set (measured; see marlin_classic_ppu.cuh). A GEMV has
// no such conservation law: each block owns a slice of N columns and walks all of K, so `iters` is long (pipeline is
// fine) and `blocks` is large (warps are plentiful), independently.
//
// B LAYOUT (decoded from Marlin's b_gl_rd + the packq in test_marlin_classic_num.cu). Crucially it is INDEPENDENT of
// thread_n_blocks -- folding slice_col*b_sh_stride + warp_n*32 into (nblock/4)*32 makes the config drop out -- so
// prefill and decode share ONE packed weight tensor.
//
//   int4 index   idx = (N/2) * ktile  +  (nblock/4) * 32  +  lane          ktile = k/16, nblock = n/16
//   the j-th uint32 of that int4 (j = nblock % 4) holds B[n][k] at
//       lane  = 4*(n % 8) + (k % 8)/2
//       shift = 4*(k%16 >= 8) + 8*(n%16 >= 8) + 16*(k % 2)
//       value = ((q >> shift) & 0xf) - 8
//
// So ONE int4 covers 4 consecutive nblocks = 64 n-columns x 4 k-values. A block must consume all 64 columns or it
// wastes 12 of every 16 bytes -- that pins grid.x = N/64.
//
// Portable: no ppu.* asm, so this builds and runs on stock CUDA too (that is how correctness was checked off-box).
//
// build:  nvcc -O3 -std=c++17 -o gemv_w4a16 gemv_w4a16_ppu.cu     (on the box: no -arch)
// run:    ./gemv_w4a16                # correctness + the decode shape sweep
//         ./gemv_w4a16 N K [iters]

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while(0)

static const int N_PER_BLOCK = 64;   // one int4 spans exactly this many columns
static const int GEMV_THREADS = 128; // 4 warps; each warp takes a different ktile

// Each thread consumes one int4 per step: 4 nblocks (j) x {lo,hi} n-halves = 8 columns, x 4 k-values.
// acc[j][hi] accumulates column n = (nb_group*4 + j)*16 + lane/4 + 8*hi.
template <int SPLIT_K>
__global__ void __launch_bounds__(GEMV_THREADS)
gemv_w4a16(const int4* __restrict__ B, const half* __restrict__ A, const half* __restrict__ s,
           float* __restrict__ partial, int N, int K) {
    const int nb_group = blockIdx.x;                 // 4 nblocks = 64 columns
    const int slice    = blockIdx.y;
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;

    const int k_tiles = K / 16;
    const int kt_per_slice = k_tiles / SPLIT_K;
    const int kt_begin = slice * kt_per_slice;
    const int kt_end   = kt_begin + kt_per_slice;

    const int b_stride = N / 2;                      // int4 per ktile
    const int lane_k = (lane % 4) * 2;               // this lane's k phase within the 16-wide ktile

    float acc[4][2] = {{0.f}};

    // 4 warps stride the ktile range; each thread reads one int4 per ktile it owns.
    for (int kt = kt_begin + wid; kt < kt_end; kt += GEMV_THREADS / 32) {
        const int4 q4 = B[(long long) b_stride * kt + nb_group * 32 + lane];
        const int kb = kt * 16 + lane_k;
        // the 4 k values this lane's int4 carries
        const float a0 = __half2float(A[kb]);
        const float a1 = __half2float(A[kb + 1]);
        const float a2 = __half2float(A[kb + 8]);
        const float a3 = __half2float(A[kb + 9]);

        const unsigned q[4] = { (unsigned) q4.x, (unsigned) q4.y, (unsigned) q4.z, (unsigned) q4.w };
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const unsigned qq = q[j];
            // shift = 4*hi_k + 8*hi_n + 16*parity
            acc[j][0] += (int((qq >>  0) & 0xf) - 8) * a0    // hi_n=0 hi_k=0 p=0 -> k=kb
                       + (int((qq >>  4) & 0xf) - 8) * a2    // hi_n=0 hi_k=1 p=0 -> k=kb+8
                       + (int((qq >> 16) & 0xf) - 8) * a1    // hi_n=0 hi_k=0 p=1 -> k=kb+1
                       + (int((qq >> 20) & 0xf) - 8) * a3;   // hi_n=0 hi_k=1 p=1 -> k=kb+9
            acc[j][1] += (int((qq >>  8) & 0xf) - 8) * a0    // hi_n=1 ...
                       + (int((qq >> 12) & 0xf) - 8) * a2
                       + (int((qq >> 24) & 0xf) - 8) * a1
                       + (int((qq >> 28) & 0xf) - 8) * a3;
        }
    }

    // A column n is held by the 4 lanes sharing lane/4 (they differ only in k phase). Fold them.
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            acc[j][h] += __shfl_xor_sync(0xffffffff, acc[j][h], 1);
            acc[j][h] += __shfl_xor_sync(0xffffffff, acc[j][h], 2);
        }
    // now lane%4 == 0 holds the warp's partial for column (nb_group*4+j)*16 + lane/4 + 8*h

    __shared__ float sm[GEMV_THREADS / 32][N_PER_BLOCK];
    if (lane % 4 == 0) {
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < 2; h++) sm[wid][j * 16 + lane / 4 + 8 * h] = acc[j][h];
    }
    __syncthreads();

    // 64 columns, 128 threads -> first 64 threads each reduce one column across the 4 warps.
    if (threadIdx.x < N_PER_BLOCK) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < GEMV_THREADS / 32; w++) v += sm[w][threadIdx.x];
        const int n = nb_group * N_PER_BLOCK + threadIdx.x;
        partial[(long long) slice * N + n] = v;
    }
}

// out[n] = half(scale[n] * sum_slice partial[slice][n])
__global__ void gemv_reduce(const float* __restrict__ partial, const half* __restrict__ s,
                            half* __restrict__ C, int N, int split_k) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float v = 0.f;
    for (int i = 0; i < split_k; i++) v += partial[(long long) i * N + n];
    C[n] = __float2half(v * __half2float(s[n]));
}

// ---------------------------------------------------------------------------------------------------------------

// SPLIT_K trades block count against per-thread loop length, exactly the tension that caps Marlin -- except here the
// two are NOT tied to the tile decomposition, so we can sit anywhere on the curve. grid = (N/64) x SPLIT_K, and each
// warp walks (K/16/SPLIT_K)/4 int4s. N=4096 gives only 64 blocks per slice, so it needs a larger SPLIT_K than N=14336.
static int get_split_k() { const char* e = getenv("GEMV_SPLIT_K"); int v = e ? atoi(e) : 8; return v > 0 ? v : 8; }
static int SPLIT_K = 8;

static void launch(const int4* B, const half* A, const half* s, float* partial, half* C, int N, int K) {
    dim3 grid(N / N_PER_BLOCK, SPLIT_K);
    switch (SPLIT_K) {
        case  1: gemv_w4a16< 1><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case  2: gemv_w4a16< 2><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case  4: gemv_w4a16< 4><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case  8: gemv_w4a16< 8><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case 16: gemv_w4a16<16><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        case 32: gemv_w4a16<32><<<grid, GEMV_THREADS>>>(B, A, s, partial, N, K); break;
        default: printf("  GEMV_SPLIT_K=%d unsupported (1/2/4/8/16/32)\n", SPLIT_K); exit(1);
    }
    if (SPLIT_K > 1) gemv_reduce<<<(N + 255) / 256, 256>>>(partial, s, C, N, SPLIT_K);
    else             gemv_reduce<<<(N + 255) / 256, 256>>>(partial, s, C, N, 1);
}

// Marlin's packing, verbatim from test_marlin_classic_num.cu's packq, generalized to any N.
static void pack_B(const std::vector<int>& Bdeq, std::vector<int>& hB, int N, int K) {
    auto Bu = [&](int n, int k) { return (Bdeq[(size_t) n * K + k] + 8) & 0xf; };
    for (int ktile = 0; ktile < K / 16; ktile++)
        for (int nblock = 0; nblock < N / 16; nblock++)
            for (int lane = 0; lane < 32; lane++) {
                int n = nblock * 16 + lane / 4, kb = ktile * 16 + (lane % 4) * 2, q = 0;
                q |= Bu(n, kb)         << 0;  q |= Bu(n, kb + 1)     << 16;
                q |= Bu(n, kb + 8)     << 4;  q |= Bu(n, kb + 9)     << 20;
                q |= Bu(n + 8, kb)     << 8;  q |= Bu(n + 8, kb + 1) << 24;
                q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
                size_t idx = (size_t) (N / 2) * ktile + (nblock / 4) * 32 + lane;   // int4
                hB[idx * 4 + (nblock % 4)] = q;
            }
}

static double bench_one(int N, int K, int iters, bool check) {
    SPLIT_K = get_split_k();
    // each slice needs a whole number of ktiles, and the 4 warps must divide them evenly
    if (N % 64 || K % (16 * SPLIT_K) || (K / 16 / SPLIT_K) % (GEMV_THREADS / 32)) {
        printf("  N=%d K=%d SPLIT_K=%d unsupported (N%%64, K%%%d, ktiles/slice %% 4)\n", N, K, SPLIT_K, 16 * SPLIT_K); return -1; }

    std::vector<half> hA(K), hS(N, __float2half(1.0f)), hC(N);
    std::vector<int> Bdeq((size_t) N * K), hB((size_t) (K / 16) * (N / 2) * 4);
    srand(1234);
    for (auto& x : hA) x = __float2half(0.05f * (rand() % 40 - 20));
    for (auto& x : Bdeq) x = (rand() & 0xf) - 8;
    pack_B(Bdeq, hB, N, K);

    int4 *dB; half *dA, *dS, *dC; float* dP;
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * 4));
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
    CUDA_CHECK(cudaMalloc(&dS, hS.size() * 2));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * 2));
    CUDA_CHECK(cudaMalloc(&dP, (size_t) 32 * N * 4));   // max SPLIT_K
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice));

    launch(dB, dA, dS, dP, dC, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (check) {
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));
        double max_abs = 0, ref_max = 0;
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double) __half2float(hA[k]) * Bdeq[(size_t) n * K + k];
            max_abs = fmax(max_abs, fabs(__half2float(hC[n]) - acc));
            ref_max = fmax(ref_max, fabs(acc));
        }
        double rel = max_abs / (ref_max + 1e-9);
        printf("  correctness N=%d K=%d: max_abs=%.4e rel=%.2e (|ref|max=%.2f) -> %s\n",
               N, K, max_abs, rel, ref_max, rel < 3e-2 ? "MATCH" : "MISMATCH");
        if (rel >= 3e-2) { printf("  MISMATCH -- aborting bench\n"); return -1; }
    }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 10; i++) launch(dB, dA, dS, dP, dC, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) launch(dB, dA, dS, dP, dC, N, K);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;

    const char* e = getenv("MARLIN_HBM_GBS");
    double peak = e ? atof(e) : 2700.0;
    double bytes = (double) N * K / 2 + (double) K * 2 + (double) N * 2 + (double) N * 2;  // B + A + C + scales
    double gbs = bytes / (ms * 1e6);
    printf("  M=1     N=%-5d K=%-5d : %8.2f us  %7.1f TFLOP/s  %7.0f GB/s  %5.1f%% HBM\n",
           N, K, ms * 1e3, 2.0 * N * K / (ms * 1e9), gbs, 100.0 * gbs / peak);

    cudaFree(dB); cudaFree(dA); cudaFree(dS); cudaFree(dC); cudaFree(dP);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1e3;
}

int main(int argc, char** argv) {
    if (argc >= 3) {
        int N = atoi(argv[1]), K = atoi(argv[2]);
        int iters = (argc >= 4) ? atoi(argv[3]) : 200;
        printf("W4A16 GEMV (Marlin B format), SPLIT_K=%d:\n", get_split_k());
        return bench_one(N, K, iters, true) < 0 ? 1 : 0;
    }
    printf("W4A16 GEMV (Marlin B format), SPLIT_K=%d, 200 iters:\n", get_split_k());
    int shapes[][2] = { {4096, 4096}, {14336, 4096}, {4096, 14336} };
    for (auto& s : shapes) if (bench_one(s[0], s[1], 200, true) < 0) return 1;
    return 0;
}
