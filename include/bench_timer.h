// Common CUDA event timer for all bench binaries.
// Usage: call bench_timer_init() after kernel args are ready,
//        wrap kernel call in BENCH_TIMER_RUN(kernel_call),
//        call bench_timer_report().
//
// Activated by --bench [warmup] [iters] on command line.
// Default (no --bench): runs once, no timing.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

struct BenchTimer {
    int warmup = 0;
    int iters = 0;
    bool active = false;

    // Parse --bench from argv. Returns number of args consumed (0 if not found).
    // Call BEFORE parsing positional args, or scan from end.
    int parse(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--bench") == 0) {
                warmup = (i + 1 < argc) ? atoi(argv[i + 1]) : 10;
                iters  = (i + 2 < argc) ? atoi(argv[i + 2]) : 50;
                active = true;
                return 3;  // consumed: --bench W I
            }
        }
        return 0;
    }

    // Remove --bench and its args from argv, return new argc
    static int strip_bench_args(int argc, char** argv) {
        int out = 0;
        for (int i = 0; i < argc; i++) {
            if (strcmp(argv[i], "--bench") == 0) {
                i += 2;  // skip W and I
                continue;
            }
            argv[out++] = argv[i];
        }
        return out;
    }

    // Run kernel_fn warmup+iters times with CUDA event timing.
    // kernel_fn should be a lambda that launches the kernel (no sync needed inside).
    template <typename F>
    void run(F kernel_fn) {
        if (!active) {
            kernel_fn();
            cudaDeviceSynchronize();
            return;
        }

        // Warmup
        for (int i = 0; i < warmup; i++) {
            kernel_fn();
        }
        cudaDeviceSynchronize();

        // Timed iterations
        std::vector<float> times(iters);
        for (int i = 0; i < iters; i++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            kernel_fn();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&times[i], start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        std::sort(times.begin(), times.end());
        float median = times[iters / 2];
        float min_t = times[0];
        float max_t = times[iters - 1];
        float sum = 0;
        for (auto t : times) sum += t;
        float avg = sum / iters;

        printf("  Kernel time: median=%.4f ms, avg=%.4f ms, min=%.4f ms, max=%.4f ms  (warmup=%d, iters=%d)\n",
               median, avg, min_t, max_t, warmup, iters);
    }
};
