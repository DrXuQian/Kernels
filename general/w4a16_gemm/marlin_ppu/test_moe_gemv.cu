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

  // A 50-iteration loop over ONE B buffer measures CACHE, not HBM: only 8 experts are routed to, so the
  // working set is 8*N*K/2 -- 8.4 MB at N=1024 K=2048 -- which sits in LLC after the first iteration.
  // acu on a "cold" single launch already showed it: 5.32 MB from device memory against 8.39 MB of unique
  // weight, because the setup memcpy of all 256 experts (256 MB) leaves its tail resident and three of the
  // drawn experts (237, 153, 132) live there. Every %HBM number in this bench has been optimistic.
  // MOEV_WORKSET_MB rotates through identical copies so consecutive iterations cannot reuse lines, the
  // same device gemv_w4a16_ppu.cu has for exactly this reason and which I did not carry over.
  const size_t b_bytes = hB.size() * 4;
  int rot = 1;
  if (const char* e = getenv("MOEV_WORKSET_MB")) {
    const size_t want = (size_t) atoi(e) << 20;
    rot = (int) ((want + b_bytes - 1) / b_bytes); if (rot < 1) rot = 1;
  }
  int4* dB; half *dA, *dS, *dC; float* dP; int *dRe, *dRt;
  CK(cudaMalloc(&dB, b_bytes * (size_t) rot));  CK(cudaMalloc(&dA, hA.size() * 2));
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
  CK(cudaMemcpy(dB, hB.data(), b_bytes, cudaMemcpyHostToDevice));
  for (int r2 = 1; r2 < rot; r2++)     // identical copies; correctness unaffected, cache residency is not
    CK(cudaMemcpy((char*) dB + (size_t) r2 * b_bytes, dB, b_bytes, cudaMemcpyDeviceToDevice));
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
#ifdef MOEV_COUNT
  unsigned long long* dbg; CK(cudaMalloc(&dbg, 2 * sizeof(unsigned long long)));
  CK(cudaMemset(dbg, 0, 2 * sizeof(unsigned long long)));
#endif
  int it_ = 0;
  auto Brot = [&](int i) { return (const int4*) ((char*) dB + (size_t) (i % rot) * b_bytes); };
  auto go = [&] {
#ifdef MOEV_ATOMIC
    CK(cudaMemset(dP, 0, (size_t) n_rows * N * 4));
#endif
    // dynamic shared: the block's A slice (K/SPLIT_K halves) then the reduce scratch
    const size_t shm = (size_t) (K / SPLIT_K) * sizeof(half) + (size_t) (THREADS / 32) * MOEV_NPB * sizeof(float);
    moe_gemv_q4k<THREADS, SPLIT_K, U><<<grid, THREADS, shm>>>(Brot(it_++), dA, dS, dRe, dRt, dP, dC, dCnt, N, K, gs, n_rows
#ifdef MOEV_COUNT
      , dbg
#endif
      );
#if defined(MOEV_ATOMIC)
    // one f32 plane, already summed by the atomics -- just narrow it to half
    moe_gemv_reduce<<<(int) (((long long) n_rows * N + 255) / 256), 256>>>(dP, dC, N, 1, n_rows);
#elif !defined(MOEV_FUSED_REDUCE)
    if (SPLIT_K > 1) moe_gemv_reduce<<<(int) (((long long) n_rows * N + 255) / 256), 256>>>(dP, dC, N, SPLIT_K, n_rows);
#endif
  };
#ifdef MOEV_ATOMIC
  CK(cudaMemset(dP, 0, (size_t) n_rows * N * 4));    // atomics accumulate into it; must start at zero
#endif
  go(); CK(cudaDeviceSynchronize());
#ifdef MOEV_COUNT
  {   // what the kernel ACTUALLY touched, against what the shape says it should
    unsigned long long h[2]; CK(cudaMemcpy(h, dbg, sizeof(h), cudaMemcpyDeviceToHost));
    const long long exp_tiles = (long long) n_rows * (N / MOEV_NPB) * SPLIT_K;
    const long long exp_ld = exp_tiles * (K / 16 / SPLIT_K) * 32;
    printf("  COUNT: tiles %llu / %lld expected | B int4 loads %llu / %lld (%.2f MB / %.2f MB)\n",
           h[0], exp_tiles, h[1], exp_ld, h[1] * 16.0 / 1e6, exp_ld * 16.0 / 1e6);
    CK(cudaMemset(dbg, 0, sizeof(h)));
  }
#endif
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
  if (rot == 1) printf("       (workset %.1f MB fits LLC -- this is CACHE bandwidth; set MOEV_WORKSET_MB to force DRAM)\n",
                       wb / 1e6);
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
        /* and again with a grid that FORCES the tile loop to wrap. The default verification config has  */ \
        /* 512 tiles and a 512-block grid, so it never wraps -- which is exactly how a `return` inside   */ \
        /* the tile loop survived and dropped 44% of the profiled config's work.                        */ \
        setenv("MOEV_GRID", "64", 1);                                                     \
        const int vw_ = run<TT, SK, UU>(1, 8, 8, 256, 512, false);                        \
        unsetenv("MOEV_GRID");                                                            \
        if (vw_) { printf("  WRONG with a wrapping grid -- not profiling it\n"); return 1; } \
      }                                                                                   \
      return run<TT, SK, UU>(1, 8, 256, N, K, true);                                      \
    } while (0)

    // Every unrecognised value must REFUSE, not fall through. sk=64 and sk=128 landed in the sk=16 branch
    // and ran 2048 blocks while printing "sk=16" -- so the whole point of the atomic change (that high
    // SPLIT_K becomes affordable) was never actually tested. I added this guard for T and for U and left
    // sk without one, which is the third time a silent config substitution has cost a round here.
    if (T == 32) {
      if (sk == 4)        { if (u == 8) MOEV_ONE(32,4,8);   else if (u == 2) MOEV_ONE(32,4,2);   else MOEV_ONE(32,4,1); }
      else if (sk == 8)   { if (u == 8) MOEV_ONE(32,8,8);   else if (u == 4) MOEV_ONE(32,8,4);   else if (u == 2) MOEV_ONE(32,8,2);   else MOEV_ONE(32,8,1); }
      else if (sk == 16)  { if (u == 8) MOEV_ONE(32,16,8);  else if (u == 4) MOEV_ONE(32,16,4);  else if (u == 2) MOEV_ONE(32,16,2);  else MOEV_ONE(32,16,1); }
      else if (sk == 32)  { if (u == 4) MOEV_ONE(32,32,4);  else if (u == 2) MOEV_ONE(32,32,2);  else MOEV_ONE(32,32,1); }
      else if (sk == 64)  { if (u == 2) MOEV_ONE(32,64,2);  else MOEV_ONE(32,64,1); }
      else if (sk == 128) { MOEV_ONE(32,128,1); }
      else { printf("  sk=%d not dispatched (4/8/16/32/64/128)\n", sk); return 1; }
    } else if (T == 64) {
      if (sk == 8)       { if (u == 2) MOEV_ONE(64,8,2);  else MOEV_ONE(64,8,1); }
      else if (sk == 32) { if (u == 2) MOEV_ONE(64,32,2); else MOEV_ONE(64,32,1); }
      else if (sk == 16)  { if (u == 2) MOEV_ONE(64,16,2); else MOEV_ONE(64,16,1); }
      else if (sk == 64)  { MOEV_ONE(64,64,1); }
      else { printf("  sk=%d not dispatched for T=64\n", sk); return 1; }
    } else if (T == 128) {
      if (sk == 32)      { if (u == 2) MOEV_ONE(128,32,2); else MOEV_ONE(128,32,1); }
      else if (sk == 8)  { if (u == 4) MOEV_ONE(128,8,4);  else MOEV_ONE(128,8,2); }
      else if (sk == 16)  { if (u == 4) MOEV_ONE(128,16,4); else if (u == 2) MOEV_ONE(128,16,2); else MOEV_ONE(128,16,1); }
      else { printf("  sk=%d not dispatched for T=128\n", sk); return 1; }
    } else {
      printf("  T=%d not in the single-config dispatcher (32/64/128)\n", T); return 1;
    }
#undef MOEV_ONE
  }

  int bad = 0;
  // ONE configuration. The sweep had grown to four binaries, six compile flags and twenty-odd
  // combinations, which made the debugging surface larger than the problem. T=32/sk=16/U=2 on the static
  // grid with the separate reduce was best or tied best in every run, and it is the only path the
  // persistent `return` bug never touched. Everything else is still reachable by argv or a -D flag; it
  // just no longer runs by default.
  printf("--- correctness ---\n");
  bad |= run<32, 16, 2>(1, 8, 8,  256, 512,  false);   // the shipped config
  bad |= run<32, 16, 2>(2, 8, 16, 256, 512,  false);   // several tokens, more experts
  bad |= run<32, 16, 2>(4, 4, 16, 512, 1024, false);   // different N/K/topk
  bad |= run<32, 4,  8>(1, 8, 8,  256, 512,  false);   // U spanning scale groups, and a tail
  printf("%s\n", bad ? "SOME CASES FAILED" : "all decode cases MATCH");
  if (bad) return 1;

  printf("\n--- T=32 sk=16 U=2, batch 1 x top-8 over 256 experts ---\n");
  printf("  warm (weights sit in LLC -- this is CACHE bandwidth):\n");
  run<32, 16, 2>(1, 8, 256, N, K, true);
  if (!getenv("MOEV_WORKSET_MB")) {
    printf("  cold: re-run with MOEV_WORKSET_MB=512 to force the weights out of LLC.\n");
    printf("        Every %%HBM figure above the workset line is cache, not HBM.\n");
  }
  return 0;
}
