// Q4_K MoE decode GEMV vs a CPU reference, plus %HBM (decode is bandwidth bound; read that, not TFLOP/s).
#include "marlin_moe_gemv_ppu.cuh"
#include <cstdio>
#include <cstdlib>
#include <string>
#ifdef MOEV_FUSED_REDUCE
static const bool MOEV_FUSED = true;
#else
static const bool MOEV_FUSED = false;
#endif
#include <vector>
#include <random>
#include <cstdlib>
#include <string>


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
  // A slice no longer has to hold a whole scale group. That constraint came from the OLD group-major loop,
  // where g = g0/kt_per_group with g0 = kt_begin read the wrong group row when a slice started mid-group.
  // The restructured loop computes g = kt/kt_per_group PER KTILE, so it is right wherever the slice starts,
  // and the arithmetic agrees: the scale is applied to a group's partial, and (p_A + p_B)*s == p_A*s +
  // p_B*s, so splitting a group across slices and summing the partials is identical. Only whole ktiles and
  // a non-empty slice are actually required.
  const int kt_per_slice = K / 16 / SPLIT_K;
  if (kt_per_slice < 1 || (K / 16) % SPLIT_K) {
      printf("  T=%d sk=%-3d: %d ktiles/slice -- skipped (need >= 1 and k_tiles %% sk == 0)\n",
             THREADS, SPLIT_K, kt_per_slice); return 0; }
  CK(cudaMalloc(&dRe, n_rows * 4));    CK(cudaMalloc(&dRt, n_rows * 4));
  int* dCnt; CK(cudaMalloc(&dCnt, (size_t) n_rows * (N / MOEV_NPB) * 4));
  CK(cudaMemset(dCnt, 0, (size_t) n_rows * (N / MOEV_NPB) * 4));   // the winning CTA rearms it each launch
  CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dS, sP.data(), sP.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dRe, rexp.data(), n_rows * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dRt, rtok.data(), n_rows * 4, cudaMemcpyHostToDevice));

#ifdef MOEV_PERSIST
  // One wave: #CU * blocks-per-CU. Shared is tiny here, so the blocks/CU bound is warps: 64 / (T/32).
  int cus = 0; CK(cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, 0));
  // The wave must come from the ACTUAL occupancy, not the theoretical warp maximum. At T=128 the kernel
  // uses 66 registers, so 66*128 = 8448 per block and 131072/8448 = 15.5 caps it at 14 blocks/CU -- a wave
  // of 72*14 = 1008, not the 72*16 = 1152 I assumed. Grid 1024 therefore already exceeded one wave and the
  // 16-block remainder nearly doubled the runtime.
  int bpc = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpc, moe_gemv_q4k<THREADS, SPLIT_K, U>, THREADS, 0));
  const int wave = cus * (bpc > 0 ? bpc : 1);
  const int tot = n_rows * (N / MOEV_NPB) * SPLIT_K;
  // And the objective is to minimise ceil(tot/grid) -- how many tiles the SLOWEST block gets -- subject to
  // grid <= wave. Not "largest divisor": at tot=2048, wave=1008, an even 1024 costs 2 waves (4 tile-times)
  // and an even 512 uses half the machine (also 4), while grid = wave gives most blocks 2 and some 3 (3).
  // Slightly uneven inside one wave beats perfectly even across two.
  int g1 = tot < wave ? tot : wave;
  if (const char* e = getenv("MOEV_GRID")) g1 = atoi(e);
  dim3 grid(g1, 1, 1);
#else
  dim3 grid(N / MOEV_NPB, SPLIT_K, n_rows);
#endif
  auto go = [&] {
    // dynamic shared: the block's A slice (K/SPLIT_K halves) then the reduce scratch
    const size_t shm = (size_t) (K / SPLIT_K) * sizeof(half) + (size_t) (THREADS / 32) * MOEV_NPB * sizeof(float);
    moe_gemv_q4k<THREADS, SPLIT_K, U><<<grid, THREADS, shm>>>(dB, dA, dS, dRe, dRt, dP, dC, dCnt, N, K, gs, n_rows);
#ifndef MOEV_FUSED_REDUCE
    if (SPLIT_K > 1) moe_gemv_reduce<<<(int) (((long long) n_rows * N + 255) / 256), 256>>>(dP, dC, N, SPLIT_K, n_rows);
#endif
  };
  go(); CK(cudaDeviceSynchronize());
  // MOEV_NCU=1: one launch of each kernel, no warmup, no timing loop -- a clean capture. Same convention
  // as GEMV_NCU / MOE_NCU. The printed us is meaningless then.
  if (getenv("MOEV_NCU")) {
    printf("  MOEV_NCU: single launch T=%d sk=%d blocks=%d -- timing skipped\n",
           THREADS, SPLIT_K, (N / MOEV_NPB) * SPLIT_K * n_rows);
    cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dC);cudaFree(dP);cudaFree(dRe);cudaFree(dRt);cudaFree(dCnt);
    return 0;
  }

  if (!bench) {
    // Run TWICE and check the second: the fused reduce rearms `counter` from inside the winning CTA, so a
    // leaked count corrupts every launch after the first -- which a single-shot check cannot see.
    go(); CK(cudaDeviceSynchronize());
    std::vector<half> got((size_t) n_rows * N);
    CK(cudaMemcpy(got.data(), dC, got.size() * 2, cudaMemcpyDeviceToHost));
    double ma = 0, rm = 0;
    for (size_t i = 0; i < got.size(); i++) { ma = fmax(ma, fabs(__half2float(got[i]) - ref[i])); rm = fmax(rm, fabs(ref[i])); }
    const double rel = ma / (rm + 1e-9);
    printf("  T=%-3d sk=%-2d tok=%-3d topk=%d E=%-4d N=%-5d K=%-5d rows=%-4d rel %.2e -> %s\n",
           THREADS, SPLIT_K, tokens, topk, n_experts, N, K, n_rows, rel, rel < 3e-2 ? "MATCH" : "MISMATCH");
    cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dC);cudaFree(dP);cudaFree(dRe);cudaFree(dRt);cudaFree(dCnt);
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
  const int blocks = grid.x * grid.y * grid.z;
#ifdef MOEV_PERSIST
  {   // uneven tiles per block IS the tail; print it rather than leaving it to be inferred
    const int tot2 = n_rows * (N / MOEV_NPB) * SPLIT_K;
    printf("  persist: %d tiles / %d blocks = %d..%d per block, wave=%d (%.2f waves)\n",
           tot2, blocks, tot2 / blocks, (tot2 + blocks - 1) / blocks, wave, (double) blocks / wave);
  }
#endif
  printf("  T=%-3d sk=%-2d U=%d blocks=%-6d %-11s | %7.2f us | %6.0f GB/s | %5.1f%% HBM (floor %.1f us)\n",
         THREADS, SPLIT_K, U, blocks, (SPLIT_K > 1 && !MOEV_FUSED) ? "2 launches" : "1 launch", ms * 1e3,
         wb / (ms * 1e6), 100.0 * wb / (ms * 1e6) / 2766.0, wb / 2766.0 / 1e3);
  (void) distinct;
  cudaFree(dB);cudaFree(dA);cudaFree(dS);cudaFree(dC);cudaFree(dP);cudaFree(dRe);cudaFree(dRt);cudaFree(dCnt);
  return 0;
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 1024, K = argc > 2 ? atoi(argv[2]) : 2048;
  printf("Q4_K MoE DECODE GEMV (bandwidth bound -- read %%HBM, not TFLOP/s)\n");
  // argv 3/4 pick ONE config so a profiler capture holds one kernel instead of the whole sweep.
  // argv: N K [THREADS SPLIT_K [U]]. U was missing and defaulted to 1, so a capture of "the best config"
  // would have profiled T=32/sk=16/U=1 (9.74 us) instead of U=2 (9.11) -- the wrong kernel, silently.
  if (argc > 4) {
    const int T = atoi(argv[3]), sk = atoi(argv[4]), u = argc > 5 ? atoi(argv[5]) : 1;
    // Verify and profile the SAME template instantiation. The previous version checked run<128,4,2> while
    // profiling run<128,16,2> -- different SPLIT_K is a different compiled kernel, so it verified something
    // else and said nothing about the thing being measured. Only the problem SIZE is shrunk (8 experts,
    // 256x512) so the CPU reference is affordable; T, SPLIT_K and U are exactly the profiled ones.
    const char* ncu = getenv("MOEV_NCU");
    const std::string saved = ncu ? ncu : "";
    // MOEV_NCU is a PROFILING mode: exactly one launch, or the capture holds several kernels and the
    // single-launch hook is pointless. Adding the verification to this same invocation broke that -- it
    // put the check's main+reduce (and the post-bench re-check) into the capture alongside the kernel
    // under study. So verification runs only WITHOUT MOEV_NCU, and its absence is announced rather than
    // silent, since profiling an unverified build is the failure this was added to prevent.
#define MOEV_ONE(TT, SK, UU) do {                                                        \
      if (!saved.empty()) {                                                               \
        printf("--- MOEV_NCU set: verification SKIPPED so the capture holds one kernel.\n"     \
               "    Verify first with the same args and no MOEV_NCU.\n");                     \
      } else {                                                                            \
        printf("--- verifying <T=%d, sk=%d, U=%d> (the instantiation being profiled) ---\n", TT, SK, UU); \
        if (run<TT, SK, UU>(1, 8, 8, 256, 512, false)) { printf("  WRONG -- not profiling it\n"); return 1; } \
      }                                                                                   \
      return run<TT, SK, UU>(1, 8, 256, N, K, true);                                      \
    } while (0)

    if (T == 32) {
      if (sk == 4)       { if (u == 8) MOEV_ONE(32,4,8);  else if (u == 2) MOEV_ONE(32,4,2);  else MOEV_ONE(32,4,1); }
      else if (sk == 8)  { if (u == 8) MOEV_ONE(32,8,8);  else if (u == 4) MOEV_ONE(32,8,4);  else if (u == 2) MOEV_ONE(32,8,2); else MOEV_ONE(32,8,1); }
      else if (sk == 32) { if (u == 4) MOEV_ONE(32,32,4); else if (u == 2) MOEV_ONE(32,32,2); else MOEV_ONE(32,32,1); }
      else               { if (u == 8) MOEV_ONE(32,16,8); else if (u == 4) MOEV_ONE(32,16,4); else if (u == 2) MOEV_ONE(32,16,2); else MOEV_ONE(32,16,1); }
    } else if (T == 64) {
      if (sk == 8)       { if (u == 2) MOEV_ONE(64,8,2);  else MOEV_ONE(64,8,1); }
      else if (sk == 32) { if (u == 2) MOEV_ONE(64,32,2); else MOEV_ONE(64,32,1); }
      else               { if (u == 2) MOEV_ONE(64,16,2); else MOEV_ONE(64,16,1); }
    } else if (T == 128) {
      if (sk == 32)      { if (u == 2) MOEV_ONE(128,32,2); else MOEV_ONE(128,32,1); }
      else if (sk == 8)  { if (u == 4) MOEV_ONE(128,8,4);  else MOEV_ONE(128,8,2); }
      else               { if (u == 4) MOEV_ONE(128,16,4); else if (u == 2) MOEV_ONE(128,16,2); else MOEV_ONE(128,16,1); }
    } else {
      printf("  T=%d not in the single-config dispatcher (32/64/128)\n", T); return 1;
    }
#undef MOEV_ONE
  }

  int bad = 0;
  printf("--- correctness (small E so the CPU reference is affordable) ---\n");
  bad |= run<64, 1>(1, 8, 8,  256, 512,  false);
  bad |= run<64, 4>(1, 8, 8,  256, 512,  false);
  bad |= run<32, 8>(2, 8, 16, 256, 512,  false);
  bad |= run<64, 8>(4, 4, 16, 512, 1024, false);
  bad |= run<32, 8, 2>(2, 8, 16, 256, 512, false);    // unrolled path, and a tail (n_kt not a multiple of U)
  bad |= run<64, 4, 4>(1, 8, 8,  256, 512, false);
  bad |= run<32, 4, 8>(2, 8, 16, 256, 512, false);   // U spans several scale groups -- the new path
  bad |= run<32, 2, 8>(1, 8, 8,  256, 512, false);
  bad |= run<128, 4, 2>(2, 8, 16, 256, 512, false);   // 4 warps/block: the cross-warp reduce path
  bad |= run<128, 8, 4>(1, 8, 8,  256, 512, false);
  bad |= run<128, 16, 2>(1, 8, 8,  256, 512, false);   // the instantiation the acu captures profile
  bad |= run<32,  16, 2>(2, 8, 16, 256, 512, false);
  // slices that hold ONE ktile, i.e. a scale group split across two slices -- newly legal, so it needs a
  // reference rather than the argument above
  bad |= run<32, 32, 1>(1, 8, 8,  256, 512, false);
  bad |= run<64, 32, 1>(2, 8, 16, 256, 512, false);
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
  // U is no longer capped by the scale group (2 ktiles), so these are reachable for the first time.
  // sk=16 leaves 8 ktiles per warp; U=8 issues every one of them before consuming any.
  run<32, 16, 8>(1, 8, 256, N, K, true);
  run<32, 8,  8>(1, 8, 256, N, K, true);
  run<32, 4,  8>(1, 8, 256, N, K, true);
  // With the reduce fused there is ONE launch, so SPLIT_K is free to buy blocks again -- and blocks is the
  // only lever left: the grid was under one wave (4096 warps vs 4608 slots) and occupancy averaged 63%.
  printf("  -- fused reduce: SPLIT_K can buy blocks again --\n");
  run<32, 32, 2>(1, 8, 256, N, K, true);
  run<32, 64, 2>(1, 8, 256, N, K, true);
  run<32, 32, 4>(1, 8, 256, N, K, true);
  run<32, 64, 1>(1, 8, 256, N, K, true);
  // RE-SWEEP THE THREAD COUNT. "T=32 beats T=64" was measured with the group-boundary bug in place, which
  // punished larger T specifically: T=64 gives step=2, so each warp got ONE ktile per scale group, main_n
  // went to 0 and the hoisted path never ran. That penalty is gone now.
  //
  // And 1 warp per block is the wrong shape on its own terms: hardware caps blocks per CU (32 is typical),
  // so 1 warp/block caps resident warps at 32/CU = 50% however the theoretical figure is computed, and the
  // cross-warp reduce through shared plus __syncthreads() is dead weight for a single warp. T=64 also
  // doubles resident warps to 8192 against 4608 slots -- from under one wave to nearly two, which is
  // exactly the occupancy problem.
  printf("  -- thread count re-swept (the earlier T comparison predates the hoist fix) --\n");
  run<64,  16, 2>(1, 8, 256, N, K, true);
  run<64,  32, 2>(1, 8, 256, N, K, true);
  run<128, 16, 2>(1, 8, 256, N, K, true);
  run<128, 32, 2>(1, 8, 256, N, K, true);
  run<128, 16, 4>(1, 8, 256, N, K, true);
  // The sk cap was my own guard, not the hardware's: with the per-ktile group index a slice may hold ONE
  // ktile, so K=2048's 128 ktiles allow sk up to 128 -> 16384 blocks. Grid was the last lever left.
  printf("  -- SPLIT_K past the old cap (the guard was obsolete after the loop restructure) --\n");
  run<32,  128, 1>(1, 8, 256, N, K, true);
  run<64,  128, 1>(1, 8, 256, N, K, true);
  run<128, 128, 1>(1, 8, 256, N, K, true);
  run<128, 64,  2>(1, 8, 256, N, K, true);
  return 0;
}
