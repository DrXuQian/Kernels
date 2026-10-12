// PPU 1.0 only. Build with the PPU nvcc default target, not -arch=sm_XX.
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDA_ARCH__) && (!defined(__HGGC_ARCH__) || __HGGC_ARCH__ != 100)
#error "This benchmark requires PPU 1.0; use the PPU SDK nvcc without -arch=sm_XX"
#endif

static void check(cudaError_t rc, const char * what) {
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
    }
}
#define CUDA_CHECK(expr) check((expr), #expr)

__host__ __device__ static uint32_t tag(uint64_t word) {
    uint32_t x = uint32_t(word) ^ (uint32_t(word >> 32) * 0x9e3779b9u) ^ 0xa511e9b3u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

__global__ void initialize(uint32_t * data, uint64_t words) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < words; i += uint64_t(gridDim.x) * blockDim.x) {
        data[i] = tag(i);
    }
}

// Read a separate allocation before each sample. No full-size write stream.
__global__ void evict_cache(const uint4 * data, uint64_t vectors, uint32_t * sink) {
    uint32_t value = 0;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vectors; i += uint64_t(gridDim.x) * blockDim.x) {
        const uint4 v = data[i];
        value ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    sink[uint64_t(blockIdx.x) * blockDim.x + threadIdx.x] = value;
}

template<int TileBytes>
__device__ __forceinline__ void aiu_load(uint32_t * shared, const uint32_t * global) {
    static_assert(TileBytes % 128 == 0, "whole 128-byte rows required");
    const unsigned address = unsigned(__cvta_generic_to_shared(shared));
    // b8 descriptor: row width/pitch is 128 bytes; height counts rows.
    // Linear shared layout: no swizzle/transpose or matrix consumer is needed.
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b8 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(address), "l"(global),
           "r"(1), "r"(TileBytes / 128), "r"(128), "r"(0), "r"(0), "r"(0),
           "r"(1), "r"(TileBytes / 128), "r"(128), "r"(1) : "memory");
}

template<int Pending>
__device__ __forceinline__ void wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending) : "memory");
}

struct Sentinel {
    uint32_t value;
    uint32_t tiles;
};

template<int TileBytes>
__device__ uint32_t check_tile(const uint32_t * shared, uint64_t tile) {
    uint32_t bad = 0;
    for (unsigned i = threadIdx.x; i < TileBytes / 4; i += blockDim.x) {
        bad += shared[i] != tag(tile * (TileBytes / 4) + i);
    }
    return bad;
}

// Each CTA owns tile b, b+grid, b+2*grid, ... . Slots are reused only after
// their previous AIU group completes. Check=true consumes every word, untimed.
template<int TileBytes, int Stages, bool Check>
__global__ void aiu_read(const uint32_t * input, uint64_t tiles,
                        Sentinel * sink, uint32_t * errors) {
    static_assert(Stages == 1 || Stages == 2 || Stages == 4 || Stages == 8, "stage count");
    extern __shared__ __align__(128) uint32_t storage[];
    uint32_t bad = 0;
    uint32_t count = 0;
    for (uint64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x, ++count) {
        uint32_t * slot = storage + (count & (Stages - 1)) * (TileBytes / 4);
        if (count >= Stages) {
            wait_group<Stages - 1>();
            if constexpr (Check) {
                __syncthreads();
                bad += check_tile<TileBytes>(slot, tile - uint64_t(Stages) * gridDim.x);
                __syncthreads();
            }
        }
        if (threadIdx.x == 0) {
            aiu_load<TileBytes>(slot, input + tile * (TileBytes / 4));
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    wait_group<0>();
    if constexpr (Check) {
        __syncthreads();
        const uint32_t first = count > Stages ? count - Stages : 0;
        for (uint32_t i = first; i < count; ++i) {
            bad += check_tile<TileBytes>(storage + (i & (Stages - 1)) * (TileBytes / 4),
                                        blockIdx.x + uint64_t(i) * gridDim.x);
        }
        errors[uint64_t(blockIdx.x) * blockDim.x + threadIdx.x] = bad;
    }
    if (threadIdx.x == 0 && count) {
        // Only eight output bytes per CTA; all bulk copies are volatile asm.
        // The final read also checks that the timed launch reached its last tile.
        volatile uint32_t * last = storage + ((count - 1) & (Stages - 1)) * (TileBytes / 4);
        sink[blockIdx.x] = {last[blockIdx.x % (TileBytes / 4)], count};
    }
}

struct Options {
    uint64_t mib = 2048;
    uint64_t l2_mib = 0;
    int samples = 9;
    int warmups = 2;
    int device = 0;
    int tile_kib = 0;
    int stages = 0;
    int blocks_per_sm = 0;
    double peak_gbps = 0;
};

static uint64_t integer(const char * text) {
    if (!*text || *text == '-') throw std::runtime_error("expected a nonnegative integer");
    char * end = nullptr;
    errno = 0;
    const auto n = std::strtoull(text, &end, 10);
    if (errno || *end || n > 1048576) throw std::runtime_error("invalid or excessive integer argument");
    return n;
}

static bool member(int value, std::initializer_list<int> allowed) {
    return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
}

static Options arguments(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::puts("PPU 1.0 AIU HBM-to-shared read benchmark\n"
                      "  --mib 2048          Source size, at least 16 x L2/LLC\n"
                      "  --l2-mib N          Verified cache capacity if SDK cannot report it\n"
                      "  --samples 9 --warmups 2 --device 0\n"
                      "  --tile-kib 0        0=all; or 4,8,16\n"
                      "  --stages 0          0=all; or 1,2,4,8\n"
                      "  --blocks-per-sm 0   0=all; or 1,2,4,8\n"
                      "  --peak-gbps N       Optional decimal GB/s reference, e.g. 2700\n"
                      "  --self-test         Host-only tile/pipeline coverage check");
            std::exit(0);
        }
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        if (key == "--peak-gbps") {
            char * end = nullptr;
            o.peak_gbps = std::strtod(argv[i], &end);
            if (!*argv[i] || *end || !std::isfinite(o.peak_gbps) || o.peak_gbps <= 0)
                throw std::runtime_error("invalid --peak-gbps");
            continue;
        }
        const auto n = integer(argv[i]);
        if (key == "--mib") o.mib = n;
        else if (key == "--l2-mib") o.l2_mib = n;
        else if (key == "--samples") o.samples = int(n);
        else if (key == "--warmups") o.warmups = int(n);
        else if (key == "--device") o.device = int(n);
        else if (key == "--tile-kib") o.tile_kib = int(n);
        else if (key == "--stages") o.stages = int(n);
        else if (key == "--blocks-per-sm") o.blocks_per_sm = int(n);
        else throw std::runtime_error("unknown argument: " + key);
    }
    if (!o.mib || o.samples < 1 || o.samples > 10000 || o.warmups < 1 || o.warmups > 1000 ||
        !member(o.tile_kib, {0,4,8,16}) || !member(o.stages, {0,1,2,4,8}) ||
        !member(o.blocks_per_sm, {0,1,2,4,8})) throw std::runtime_error("argument outside supported range");
    return o;
}

static void self_test() {
    size_t cases = 0;
    for (uint64_t tiles : {1,2,3,7,8,9,31,32,33,127,511,1025}) {
        for (unsigned grid : {1,2,3,8,17,72}) {
            for (unsigned stages : {1,2,4,8}) {
                std::vector<unsigned> seen(tiles, 0);
                for (unsigned b = 0; b < grid; ++b) {
                    std::vector<uint64_t> slots(stages, UINT64_MAX);
                    unsigned count = 0;
                    for (uint64_t tile = b; tile < tiles; tile += grid, ++count) {
                        if (count >= stages) {
                            const auto old = tile - uint64_t(stages) * grid;
                            if (slots[count & (stages-1)] != old) throw std::runtime_error("slot reused early");
                            ++seen[old];
                        }
                        slots[count & (stages-1)] = tile;
                    }
                    for (unsigned i = count > stages ? count-stages : 0; i < count; ++i) {
                        const auto tile = b + uint64_t(i) * grid;
                        if (slots[i & (stages-1)] != tile) throw std::runtime_error("drain differs");
                        ++seen[tile];
                    }
                }
                for (auto n : seen) if (n != 1) throw std::runtime_error("coverage is not exactly once");
                ++cases;
            }
        }
    }
    std::printf("AIU_READ_SELF_TEST PASS cases=%zu device_test=NOT_RUN\n", cases);
}

struct Memory {
    void * data = nullptr;
    explicit Memory(size_t bytes) { CUDA_CHECK(cudaMalloc(&data, bytes)); }
    ~Memory() { if (data) cudaFree(data); }
};

struct Timing {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    Timing() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~Timing() {
        if (stream) cudaStreamSynchronize(stream);
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (stream) cudaStreamDestroy(stream);
    }
};

struct Bench {
    Options options;
    uint64_t bytes, eviction_bytes;
    int sms;
    Memory input, eviction, evict_sink, output, errors;
    Timing timing;
    double best_gbps = 0;
    int best_tile = 0, best_stages = 0, best_blocks = 0;
    int measured = 0, skipped = 0;

    Bench(Options o, uint64_t n, uint64_t eviction_size, int compute_units) :
        options(o), bytes(n), eviction_bytes(eviction_size), sms(compute_units),
        input(n), eviction(eviction_size), evict_sink(size_t(sms)*16*256*4),
        output(size_t(sms)*8*sizeof(Sentinel)), errors(size_t(sms)*8*32*4) {
        initialize<<<sms*16,256,0,timing.stream>>>((uint32_t *) input.data, bytes/4);
        CUDA_CHECK(cudaGetLastError());
        initialize<<<sms*16,256,0,timing.stream>>>((uint32_t *) eviction.data, eviction_bytes/4);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(timing.stream));
    }

    void evict() {
        evict_cache<<<sms*16,256,0,timing.stream>>>((const uint4 *) eviction.data, eviction_bytes/16,
                                                  (uint32_t *) evict_sink.data);
        CUDA_CHECK(cudaGetLastError());
    }

    template<int Tile, int Stages>
    void run(int blocks_per_sm, size_t shared_limit) {
        if ((options.tile_kib && options.tile_kib*1024 != Tile) ||
            (options.stages && options.stages != Stages) ||
            (options.blocks_per_sm && options.blocks_per_sm != blocks_per_sm)) return;
        constexpr size_t shared = size_t(Tile)*Stages;
        const int blocks = sms*blocks_per_sm;
        if (shared > shared_limit || bytes/Tile < uint64_t(blocks)) {
            ++skipped;
            std::printf("AIU_READ_SKIP tile=%d stages=%d blocks=%d reason=RESOURCE_LIMIT\n", Tile, Stages, blocks);
            return;
        }
        CUDA_CHECK(cudaFuncSetAttribute(aiu_read<Tile,Stages,false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
        CUDA_CHECK(cudaFuncSetAttribute(aiu_read<Tile,Stages,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
        int resident = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, aiu_read<Tile,Stages,false>, 32, shared));
        if (!resident) throw std::runtime_error("zero occupancy");
        std::printf("AIU_READ_PROGRESS tile=%d stages=%d blocks=%d phase=verify\n", Tile, Stages, blocks);
        aiu_read<Tile,Stages,true><<<blocks,32,shared,timing.stream>>>(
            (const uint32_t *) input.data, bytes/Tile, (Sentinel *) output.data, (uint32_t *) errors.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(timing.stream));
        std::vector<uint32_t> host_errors(size_t(blocks)*32);
        CUDA_CHECK(cudaMemcpy(host_errors.data(), errors.data, host_errors.size()*4, cudaMemcpyDeviceToHost));
        uint64_t bad = 0;
        for (auto count : host_errors) bad += count;
        if (bad) throw std::runtime_error("AIU word validation failed: " + std::to_string(bad));

        std::vector<double> samples;
        for (int i = -options.warmups; i < options.samples; ++i) {
            evict();
            CUDA_CHECK(cudaEventRecord(timing.start, timing.stream));
            aiu_read<Tile,Stages,false><<<blocks,32,shared,timing.stream>>>(
                (const uint32_t *) input.data, bytes/Tile, (Sentinel *) output.data, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(timing.stop, timing.stream));
            CUDA_CHECK(cudaEventSynchronize(timing.stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, timing.start, timing.stop));
            if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("invalid event interval");
            if (i >= 0) samples.push_back(double(ms)*1000);
        }
        std::vector<Sentinel> sentinels(blocks);
        CUDA_CHECK(cudaMemcpy(sentinels.data(), output.data, sentinels.size()*sizeof(Sentinel), cudaMemcpyDeviceToHost));
        for (int b = 0; b < blocks; ++b) {
            const uint64_t count = (bytes/Tile - 1 - b)/blocks + 1;
            const uint64_t last = b + (count-1)*blocks;
            if (sentinels[b].tiles != count || sentinels[b].value != tag(last*(Tile/4) + b%(Tile/4)))
                throw std::runtime_error("timed kernel sentinel differs");
        }
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double median = (sorted[(sorted.size()-1)/2] + sorted[sorted.size()/2])/2;
        const double gbps = bytes/median/1000;
        std::printf("AIU_READ_RESULT tile_bytes=%d stages=%d blocks=%d threads=32 shared_bytes=%zu "
                    "resident_ctas_per_sm=%d read_bytes=%llu median_us=%.3f min_us=%.3f max_us=%.3f "
                    "read_GBs=%.3f correctness=PASS sink_bytes=%zu",
                    Tile, Stages, blocks, shared, resident, (unsigned long long) bytes,
                    median, sorted.front(), sorted.back(), gbps, sentinels.size()*sizeof(Sentinel));
        if (options.peak_gbps) std::printf(" peak_pct=%.2f", 100*gbps/options.peak_gbps);
        std::printf(" samples_us=[");
        for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.3f", i ? "," : "", samples[i]);
        std::puts("]");
        ++measured;
        if (gbps > best_gbps) {
            best_gbps = gbps; best_tile = Tile; best_stages = Stages; best_blocks = blocks;
        }
    }
};

int main(int argc, char ** argv) try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) { self_test(); return 0; }
    const auto options = arguments(argc, argv);
    CUDA_CHECK(cudaSetDevice(options.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
    if (prop.multiProcessorCount <= 0) throw std::runtime_error("invalid compute-unit count");
    constexpr uint64_t mib = 1024*1024;
    const uint64_t reported_l2 = prop.l2CacheSize > 0 ? uint64_t(prop.l2CacheSize) : 0;
    const uint64_t cache_bytes = options.l2_mib ? options.l2_mib*mib : reported_l2;
    if (!cache_bytes) throw std::runtime_error("SDK reports no cache size; supply verified --l2-mib (ZW810: 64 MiB if confirmed on your device)");
    if (cache_bytes < reported_l2) throw std::runtime_error("--l2-mib is smaller than reported cache");
    const uint64_t bytes = options.mib*mib;
    if (bytes < 16*cache_bytes) throw std::runtime_error("--mib must be at least 16 times the cache size");
    const uint64_t eviction_bytes = 4*cache_bytes;
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (bytes + eviction_bytes + 16*mib > free_bytes ||
        bytes > std::numeric_limits<size_t>::max()) throw std::runtime_error("not enough free device memory; reduce --mib while retaining >=16x cache");
    int shared_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, options.device));
    const size_t shared_limit = std::max(size_t(std::max(shared_optin, 0)), prop.sharedMemPerBlock);
    char pci[32]{};
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), options.device));
    int runtime = 0, driver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::printf("AIU_READ_DEVICE name=%s pci=%s ordinal=%d compute_units=%d runtime=%d driver=%d "
                "cache_bytes=%llu cache_source=%s reported_l2_bytes=%llu input_bytes=%llu eviction_bytes=%llu "
                "shared_limit=%zu warmups=%d samples=%d\n",
                prop.name, pci, options.device, prop.multiProcessorCount, runtime, driver,
                (unsigned long long) cache_bytes, options.l2_mib ? "OPERATOR" : "SDK",
                (unsigned long long) reported_l2, (unsigned long long) bytes,
                (unsigned long long) eviction_bytes, shared_limit, options.warmups, options.samples);
    Bench bench(options, bytes, eviction_bytes, prop.multiProcessorCount);
    for (int grid : {1,2,4,8}) {
#define RUN(T) bench.run<T,1>(grid, shared_limit); bench.run<T,2>(grid, shared_limit); \
               bench.run<T,4>(grid, shared_limit); bench.run<T,8>(grid, shared_limit)
        RUN(4096); RUN(8192); RUN(16384);
#undef RUN
    }
    if (!bench.measured) throw std::runtime_error("no admissible configuration measured");
    std::printf("AIU_READ_BEST tile_bytes=%d stages=%d blocks=%d read_GBs=%.3f measured=%d skipped=%d "
                "scope=REQUESTED_BYTES_PER_KERNEL_EVENT dram_counters=NOT_COLLECTED\n",
                bench.best_tile, bench.best_stages, bench.best_blocks, bench.best_gbps, bench.measured, bench.skipped);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "AIU_READ_FAIL %s\n", error.what());
    return 1;
}
