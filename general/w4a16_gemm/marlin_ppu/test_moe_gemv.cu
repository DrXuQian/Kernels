// Q4_K MoE decode GEMV vs a CPU reference, plus %HBM (decode is bandwidth bound; read that, not TFLOP/s).
#include "marlin_moe_gemv_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstdlib>

#define CK(x) do { cudaError_t e_=(x); if(e_){printf("cuda %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);} } while(0)

template <int THREADS, int SPLIT_K, int U = 1>
static int run(int tokens, int topk, int n_experts, int N, int K, bool bench) {
  using namespace marlin_moe_gemv_ppu;
  const int gs = 32, G = K / gs, n_rows = tokens * topk;
  std::mt19937 rng(5 + N + K + tokens);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> qd(-8, 7);

  std::vector<int> rexp(n_rows), rtok(n_rows);
  for (int t = 0; t < tokens; t++)
    for (int k = 0; k < topk; k++) { rexp[t * topk + k] = (int) (rng() % n_experts); rtok[t * topk + k] = t; }
  std::vector<half> hA((size_t) tokens * K);
  for (auto& x : hA) x = __float2half(0.1f * nd(rng));
  std::vector<int> Bq((size_t) n_experts * N * K);
  for (auto& x : Bq) x = qd(rng);
  std::vector<half> sP((size_t) n_experts * G * N);
  for (auto& x : sP) x = __float2half(0.4f + 0.6f * ((rng() % 100) / 100.0f));

  std::vector<double> ref((size_t) n_rows * N);
  if (!bench)
    for (int r = 0; r < n_rows; r++) {
      const int e = rexp[r], t = rtok[r];
      for (int n = 0; n < N; n++) {
        double acc = 0;
        for (int k = 0; k < K; k++)
          acc += (double) __half2float(hA[(size_t) t * K + k]) * Bq[((size_t) e * N + n) * K + k]
               * __half2float(sP[(size_t) e * G * N + (size_t)(k / gs) * N + n]);
        ref[(size_t) r * N + n] = acc;
      }
    }

  std::vector<int> hB((size_t) n_experts * (K / 16) * (N / 2) * 4);
  for (int e = 0; e < n_experts; e++) {
    const int* Be = Bq.data() + (size_t) e * N * K;
    auto Bu = [&](int n, int k) { return (Be[(size_t) n * K + k] + 8) & 0xf; };
    int* out = hB.data() + (size_t) e * (K / 16) * (N / 2) * 4;
    for (int kt = 0; kt < K / 16; kt++)
      for (int nb = 0; nb < N / 16; nb++)
        for (int l = 0; l < 32; l++) {
          int n = nb * 16 + l / 4, kb = kt * 16 + (l % 4) * 2, q = 0;
          q |= Bu(n, kb) << 0;          q |= Bu(n, kb + 1) << 16;
          q |= Bu(n, kb + 8) << 4;      q |= Bu(n, kb + 9) << 20;
          q |= Bu(n + 8, kb) << 8;      q |= Bu(n + 8, kb + 1) << 24;
          q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
          size_t idx = (size_t) (N / 2) * kt + (nb / 4) * 32 + l;
          out[idx * 4 + (nb % 4)] = q;
        }
  }

  int4* dB; half *dA, *dS, *dC; float* dP; int *dRe, *dRt;
  CK(cudaMalloc(&dB, hB.size() * 4));  CK(cudaMalloc(&dA, hA.size() * 2));
  CK(cudaMalloc(&dS, sP.size() * 2));  CK(cudaMalloc(&dC, (size_t) n_rows * N * 2));
  CK(cudaMalloc(&dP, (size_t) n_rows * SPLIT_K * N * 4));
  // A slice must hold whole scale groups, or g = g0/kt_per_group lands mid-group and reads the wrong
  // scale row -- silently, exactly as it did in the dense GEMV's auto_split_k.
  // A slice must hold at least ONE whole scale group. Checking only divisibility let kt_per_slice=0
  // through -- 0 % 2 == 0 -- so FC2 at sk=64 launched a kernel that computes nothing and the bench happily
  // timed it at 16.08 us. A guard that passes the degenerate case is not a guard.
  const int kt_per_slice = K / 16 / SPLIT_K, ktpg = gs / 16;
  if (kt_per_slice < ktpg || kt_per_slice % ktpg) {
      printf("  T=%d sk=%-2d: %d ktiles/slice vs a group of %d -- skipped (need >= and a multiple)\n",
             THREADS, SPLIT_K, kt_per_slice, ktpg); return 0; }
  CK(cudaMalloc(&dRe, n_rows * 4));    CK(cudaMalloc(&dRt, n_rows * 4));
  CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dS, sP.data(), sP.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dRe, rexp.data(), n_rows * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dRt, rtok.data(), n_rows * 4, cudaMemcpyHostToDevice));

  dim3 grid(N / MOEV_NPB, SPLIT_K, n_rows);
  auto go = [&] {
    moe_gemv_q4k<THREADS, SPLIT_K, U><<<grid, THREADS>>>(dB, dA, dS, dRe, dRt, dP, dC, N, K, gs, n_rows);
    if (SPLIT_K > 1) moe_gemv_reduce<<<(int) (((long long) n_rows * N + 255) / 256), 256>>>(dP, dC, N, SPLIT_K, n_rows);
  };
  go(); CK(cudaDeviceSynchronize());
  // MOEV_NCU=1: one launch of each kernel, no warmup, no timing loop -- a clean capture. Same convention
  // as GEMV_NCU / MOE_NCU. The printed us is meaningless then.
  if (getenv("MOEV_NCU")) {
    printf("  MOEV_NCU: single launch T=%d sk=%d blocks=%d -- timing skipped\n",
           THREADS, SPLIT_K, (N / MOEV_NPB) * SPLIT_K * n_rows);
    cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dC);cudaFree(dP);cudaFree(dRe);cudaFree(dRt);
    return 0;
  }

  if (!bench) {
    std::vector<half> got((size_t) n_rows * N);
    CK(cudaMemcpy(got.data(), dC, got.size() * 2, cudaMemcpyDeviceToHost));
    double ma = 0, rm = 0;
    for (size_t i = 0; i < got.size(); i++) { ma = fmax(ma, fabs(__half2float(got[i]) - ref[i])); rm = fmax(rm, fabs(ref[i])); }
    const double rel = ma / (rm + 1e-9);
    printf("  T=%-3d sk=%-2d tok=%-3d topk=%d E=%-4d N=%-5d K=%-5d rows=%-4d rel %.2e -> %s\n",
           THREADS, SPLIT_K, tokens, topk, n_experts, N, K, n_rows, rel, rel < 3e-2 ? "MATCH" : "MISMATCH");
    cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dC);cudaFree(dP);cudaFree(dRe);cudaFree(dRt);
    return rel < 3e-2 ? 0 : 1;
  }
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 5; i++) go(); CK(cudaDeviceSynchronize());
  cudaEventRecord(a); for (int i = 0; i < 50; i++) go(); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 50;
  // The weights DOMINATE and only the routed experts are touched. Distinct experts, not rows.
  std::vector<char> seen(n_experts, 0); int distinct = 0;
  for (int r = 0; r < n_rows; r++) if (!seen[rexp[r]]) { seen[rexp[r]] = 1; distinct++; }
  const double wb = (double) distinct * N * K / 2.0;
  const int blocks = (N / MOEV_NPB) * SPLIT_K * n_rows;
  printf("  T=%-3d sk=%-2d U=%d blocks=%-6d %-11s | %7.2f us | %6.0f GB/s | %5.1f%% HBM (floor %.1f us)\n",
         THREADS, SPLIT_K, blocks, SPLIT_K > 1 ? "2 launches" : "1 launch", ms * 1e3,
         wb / (ms * 1e6), 100.0 * wb / (ms * 1e6) / 2766.0, wb / 2766.0 / 1e3);
  (void) distinct;
  cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dC);cudaFree(dP);cudaFree(dRe);cudaFree(dRt);
  return 0;
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 1024, K = argc > 2 ? atoi(argv[2]) : 2048;
  printf("Q4_K MoE DECODE GEMV (bandwidth bound -- read %%HBM, not TFLOP/s)\n");
  // argv 3/4 pick ONE config so a profiler capture holds one kernel instead of the whole sweep.
  if (argc > 4) {
    const int T = atoi(argv[3]), sk = atoi(argv[4]);
    if (T == 32) { if (sk == 8) run<32,8>(1,8,256,N,K,true); else if (sk == 32) run<32,32>(1,8,256,N,K,true); else run<32,16>(1,8,256,N,K,true); }
    else         { if (sk == 8) run<64,8>(1,8,256,N,K,true); else if (sk == 32) run<64,32>(1,8,256,N,K,true); else run<64,16>(1,8,256,N,K,true); }
    return 0;
  }
  int bad = 0;
  printf("--- correctness (small E so the CPU reference is affordable) ---\n");
  bad |= run<64, 1>(1, 8, 8,  256, 512,  false);
  bad |= run<64, 4>(1, 8, 8,  256, 512,  false);
  bad |= run<32, 8>(2, 8, 16, 256, 512,  false);
  bad |= run<64, 8>(4, 4, 16, 512, 1024, false);
  bad |= run<32, 8, 2>(2, 8, 16, 256, 512, false);    // unrolled path, and a tail (n_kt not a multiple of U)
  bad |= run<64, 4, 4>(1, 8, 8,  256, 512, false);
  printf("%s\n", bad ? "SOME CASES FAILED" : "all decode cases MATCH");
  if (bad) return 1;
  printf("--- perf, batch 1 x top-8 over 256 experts (the real decode shape) ---\n");
  // SPLIT_K=1 issues ONE kernel; everything above it also launches a reduce. This problem is small -- 8.4
  // MB of weights, a 3.0 us floor at 2766 GB/s -- so a second launch is a large fixed fraction, and the
  // dense GEMV measured exactly that (~2.1 us launch = 30%% of a 7 us kernel, which is why it has
  // GEMV_FUSED_REDUCE). Sweeping sk=1/2/4 alongside 8/16 separates launch overhead from bandwidth before
  // anything gets optimized: if sk=1 is already near the others, the reduce launch is the cost, not the loop.
  run<32, 1> (1, 8, 256, N, K, true);
  run<64, 1> (1, 8, 256, N, K, true);
  run<32, 2> (1, 8, 256, N, K, true);
  run<32, 4> (1, 8, 256, N, K, true);
  run<32, 8> (1, 8, 256, N, K, true);
  run<64, 8> (1, 8, 256, N, K, true);
  run<64, 16>(1, 8, 256, N, K, true);
  run<32, 16>(1, 8, 256, N, K, true);
  // sk=1 came back 6x SLOWER (59.6 vs 9.8 us) at 128 blocks = 1.8 per CU, so the second launch was never
  // the cost -- this is parallelism-starved, and adding blocks has paid monotonically all the way to 2048
  // with no sign of saturating at 30.8% HBM against a 3.0 us floor. Push further: kt_per_slice must stay a
  // whole number of gs=32 groups (2 ktiles), so K=2048's 128 ktiles allow sk up to 64.
  run<32, 32>(1, 8, 256, N, K, true);
  run<32, 64>(1, 8, 256, N, K, true);
  run<64, 32>(1, 8, 256, N, K, true);
  // U = loads in flight per warp. sk=16 leaves 8 ktiles per warp, room for U up to 8; sk=32 leaves 4.
  // Swept together because they pull opposite ways -- sk buys blocks and spends ktiles-per-warp.
  printf("  -- unroll (loads in flight per warp; latency, not bandwidth, is the stall) --\n");
  run<32, 16, 2>(1, 8, 256, N, K, true);
  run<32, 16, 4>(1, 8, 256, N, K, true);
  run<32, 8,  4>(1, 8, 256, N, K, true);
  run<32, 32, 2>(1, 8, 256, N, K, true);
  run<64, 16, 2>(1, 8, 256, N, K, true);
  return 0;
}
