/*
 * Standalone test for fpA_intB GEMM (FP16 x INT4, GPTQ).
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_gemm_sm80_wrappers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
struct Args
{
    int m = 1;
    int n = 4096;
    int k = 4096;
    int group_size = 128;
    int warmup = 10;
    int iters = 100;
    bool warmup_set = false;
    bool iters_set = false;
    bool verify = false;
    bool list_configs = false;
    bool force_config = false;
    bool force_cuda = false;
    std::string tactic_file;
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig forced_config{};
};

void print_usage(char const* name)
{
    std::printf(
        "Usage: %s [--m=N] [--n=N] [--k=N] [--group_size=N] [--warmup=N] [--iters=N] [--verify] [--list_configs] "
        "[--config=cuda|sm80:tile_mxtile_nxtile_k:stages:split_k|sm90:tile_mxtile_nxtile_k:cluster_mxcluster_nxcluster_k] "
        "[--tactic=<file>]\n",
        name);
}

bool parse_int(char const* arg, char const* key, int& out)
{
    size_t const len = std::strlen(key);
    if (std::strncmp(arg, key, len) != 0)
    {
        return false;
    }
    out = std::strtol(arg + len, nullptr, 10);
    return true;
}

bool tile_from_shape(int tile_m, int tile_n, int tile_k,
    tensorrt_llm::cutlass_extensions::CutlassTileConfig& out)
{
    using Config = tensorrt_llm::cutlass_extensions::CutlassTileConfig;
    if (tile_k != 64)
    {
        return false;
    }
    if (tile_m == 16 && tile_n == 128)
    {
        out = Config::CtaShape16x128x64_WarpShape16x32x64;
        return true;
    }
    if (tile_m == 16 && tile_n == 256)
    {
        out = Config::CtaShape16x256x64_WarpShape16x64x64;
        return true;
    }
    if (tile_m == 32 && tile_n == 128)
    {
        out = Config::CtaShape32x128x64_WarpShape32x32x64;
        return true;
    }
    if (tile_m == 64 && tile_n == 128)
    {
        out = Config::CtaShape64x128x64_WarpShape64x32x64;
        return true;
    }
    if (tile_m == 128 && tile_n == 128)
    {
        out = Config::CtaShape128x128x64_WarpShape128x32x64;
        return true;
    }
    return false;
}

bool shape_from_tile80(tensorrt_llm::cutlass_extensions::CutlassTileConfig tile, int& tile_m, int& tile_n, int& tile_k)
{
    using Config = tensorrt_llm::cutlass_extensions::CutlassTileConfig;
    tile_k = 64;
    switch (tile)
    {
    case Config::CtaShape16x128x64_WarpShape16x32x64:
        tile_m = 16;
        tile_n = 128;
        return true;
    case Config::CtaShape16x256x64_WarpShape16x64x64:
        tile_m = 16;
        tile_n = 256;
        return true;
    case Config::CtaShape32x128x64_WarpShape32x32x64:
        tile_m = 32;
        tile_n = 128;
        return true;
    case Config::CtaShape64x128x64_WarpShape64x32x64:
        tile_m = 64;
        tile_n = 128;
        return true;
    case Config::CtaShape128x128x64_WarpShape128x32x64:
        tile_m = 128;
        tile_n = 128;
        return true;
    default:
        return false;
    }
}

bool tile90_from_shape(int tile_m, int tile_n, int tile_k,
    tensorrt_llm::cutlass_extensions::CutlassTileConfigSM90& out)
{
    using Config = tensorrt_llm::cutlass_extensions::CutlassTileConfigSM90;
    if (tile_k != 128)
    {
        return false;
    }
    if (tile_m == 64 && tile_n == 16)
    {
        out = Config::CtaShape64x16x128B;
        return true;
    }
    if (tile_m == 64 && tile_n == 32)
    {
        out = Config::CtaShape64x32x128B;
        return true;
    }
    if (tile_m == 64 && tile_n == 64)
    {
        out = Config::CtaShape64x64x128B;
        return true;
    }
    if (tile_m == 64 && tile_n == 128)
    {
        out = Config::CtaShape64x128x128B;
        return true;
    }
    if (tile_m == 64 && tile_n == 256)
    {
        out = Config::CtaShape64x256x128B;
        return true;
    }
    if (tile_m == 128 && tile_n == 16)
    {
        out = Config::CtaShape128x16x128B;
        return true;
    }
    if (tile_m == 128 && tile_n == 32)
    {
        out = Config::CtaShape128x32x128B;
        return true;
    }
    if (tile_m == 128 && tile_n == 64)
    {
        out = Config::CtaShape128x64x128B;
        return true;
    }
    if (tile_m == 128 && tile_n == 128)
    {
        out = Config::CtaShape128x128x128B;
        return true;
    }
    if (tile_m == 128 && tile_n == 256)
    {
        out = Config::CtaShape128x256x128B;
        return true;
    }
    return false;
}

bool decode_shape_tuple(int encoded, int& m, int& n, int& k)
{
    if (encoded <= 0 || encoded >= 1000000000)
    {
        return false;
    }
    m = encoded / 1000000;
    n = (encoded % 1000000) / 1000;
    k = encoded % 1000;
    return m > 0 && n > 0 && k > 0;
}

std::string shape3_to_string(int m, int n, int k)
{
    std::ostringstream out;
    out << m << "x" << n << "x" << k;
    return out.str();
}

bool cluster_from_shape(int cluster_m, int cluster_n, int cluster_k,
    tensorrt_llm::cutlass_extensions::ClusterShape& out)
{
    using Cluster = tensorrt_llm::cutlass_extensions::ClusterShape;
    if (cluster_k != 1)
    {
        return false;
    }
    if (cluster_m == 1 && cluster_n == 1)
    {
        out = Cluster::ClusterShape_1x1x1;
        return true;
    }
    if (cluster_m == 2 && cluster_n == 1)
    {
        out = Cluster::ClusterShape_2x1x1;
        return true;
    }
    if (cluster_m == 1 && cluster_n == 2)
    {
        out = Cluster::ClusterShape_1x2x1;
        return true;
    }
    if (cluster_m == 2 && cluster_n == 2)
    {
        out = Cluster::ClusterShape_2x2x1;
        return true;
    }
    return false;
}

bool parse_positive_int(std::string const& spec, int& out)
{
    char* end = nullptr;
    long const value = std::strtol(spec.c_str(), &end, 10);
    if (end == spec.c_str() || *end != '\0' || value <= 0 || value > std::numeric_limits<int>::max())
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool parse_shape3(std::string spec, int& x, int& y, int& z)
{
    for (char& ch : spec)
    {
        if (ch == 'x' || ch == 'X' || ch == ',')
        {
            ch = ' ';
        }
    }
    std::istringstream is(spec);
    if (!(is >> x >> y >> z))
    {
        return false;
    }
    int extra = 0;
    if (is >> extra)
    {
        return false;
    }
    return x > 0 && y > 0 && z > 0;
}

std::vector<std::string> split_string(std::string const& str, char delim)
{
    std::vector<std::string> parts;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        parts.push_back(item);
    }
    return parts;
}

bool parse_sm80_config(std::vector<std::string> const& parts, Args& args)
{
    if (parts.size() != 4)
    {
        return false;
    }

    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;
    if (!parse_shape3(parts[1], tile_m, tile_n, tile_k))
    {
        return false;
    }

    int stages = 0;
    int split_k = 0;
    if (!parse_positive_int(parts[2], stages) || !parse_positive_int(parts[3], split_k))
    {
        return false;
    }

    tensorrt_llm::cutlass_extensions::CutlassTileConfig tile_config;
    if (!tile_from_shape(tile_m, tile_n, tile_k, tile_config))
    {
        return false;
    }

    args.forced_config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig{};
    args.forced_config.tile_config_sm80 = tile_config;
    args.forced_config.stages = stages;
    args.forced_config.split_k_factor = split_k;
    args.forced_config.split_k_style = split_k > 1
        ? tensorrt_llm::cutlass_extensions::SplitKStyle::SPLIT_K_SERIAL
        : tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K;
    args.forced_config.enableCudaKernel = false;
    args.forced_config.sm_version = 80;
    args.force_config = true;
    return true;
}

bool parse_sm90_config(std::vector<std::string> const& parts, Args& args)
{
    if (parts.size() != 3)
    {
        return false;
    }

    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;
    if (!parse_shape3(parts[1], tile_m, tile_n, tile_k))
    {
        return false;
    }

    int cluster_m = 0;
    int cluster_n = 0;
    int cluster_k = 0;
    if (!parse_shape3(parts[2], cluster_m, cluster_n, cluster_k))
    {
        return false;
    }

    tensorrt_llm::cutlass_extensions::CutlassTileConfigSM90 tile_config;
    if (!tile90_from_shape(tile_m, tile_n, tile_k, tile_config))
    {
        return false;
    }

    tensorrt_llm::cutlass_extensions::ClusterShape cluster_shape;
    if (!cluster_from_shape(cluster_m, cluster_n, cluster_k, cluster_shape))
    {
        return false;
    }

    args.forced_config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig(tile_config,
        tensorrt_llm::cutlass_extensions::MainloopScheduleType::AUTO,
        tensorrt_llm::cutlass_extensions::EpilogueScheduleType::AUTO, cluster_shape);
    args.force_config = true;
    return true;
}

bool parse_config_spec(char const* spec, Args& args)
{
    if (spec == nullptr || *spec == '\0')
    {
        return false;
    }
    if (std::strcmp(spec, "cuda") == 0 || std::strcmp(spec, "cuda_kernel") == 0)
    {
        args.force_cuda = true;
        args.force_config = true;
        return true;
    }

    std::string config_spec(spec);
    auto parts = split_string(config_spec, ':');
    if (!parts.empty())
    {
        if (parts[0] == "sm80")
        {
            return parse_sm80_config(parts, args);
        }
        if (parts[0] == "sm90" || parts[0] == "tma")
        {
            return parse_sm90_config(parts, args);
        }
    }

    // Backward-compatible SM80 spelling: tile_m,tile_n,tile_k,stages,split_k.
    std::string normalized(spec);
    for (char& ch : normalized)
    {
        if (ch == 'x' || ch == 'X')
        {
            ch = ',';
        }
    }

    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;
    int stages = 0;
    int split_k = 1;
    int parsed = std::sscanf(normalized.c_str(), "%d,%d,%d,%d,%d", &tile_m, &tile_n, &tile_k, &stages, &split_k);
    if (parsed < 4)
    {
        return false;
    }
    if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0 || stages <= 0 || split_k <= 0)
    {
        return false;
    }

    tensorrt_llm::cutlass_extensions::CutlassTileConfig tile_config;
    if (!tile_from_shape(tile_m, tile_n, tile_k, tile_config))
    {
        return false;
    }

    args.forced_config.tile_config_sm80 = tile_config;
    args.forced_config.stages = stages;
    args.forced_config.split_k_factor = split_k;
    args.forced_config.split_k_style = split_k > 1
        ? tensorrt_llm::cutlass_extensions::SplitKStyle::SPLIT_K_SERIAL
        : tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K;
    args.forced_config.enableCudaKernel = false;
    args.forced_config.sm_version = 80;
    args.force_config = true;
    return true;
}

// Tactic file: portable serialization of CutlassGemmConfig by semantic fields.
// Format per line:
//   m,n,k,gs|cuda=1                                            (CUDA core kernel)
//   m,n,k,gs|cuda=0,tma=0,tile80=5,stages=4,splitk=1,sk_style=0  (SM80 CUTLASS)
//   m,n,k,gs|cuda=0,tma=1,sm=90,tile90=64016128,ml=1,el=2,cl=1001001  (SM90 TMA)
//
// All enum values are stored as integers. The tile/cluster enums encode MxNxK
// via shape_tuple_to_enum (m*1e6 + n*1e3 + k), so they are self-describing
// and portable across builds.

std::string serialize_config(tensorrt_llm::cutlass_extensions::CutlassGemmConfig const& c)
{
    std::ostringstream s;
    if (c.enableCudaKernel)
    {
        s << "cuda=1";
    }
    else if (c.is_tma_warp_specialized)
    {
        s << "cuda=0,tma=1,sm=" << c.sm_version
          << ",tile90=" << static_cast<int>(c.tile_config_sm90)
          << ",ml=" << static_cast<int>(c.mainloop_schedule)
          << ",el=" << static_cast<int>(c.epilogue_schedule)
          << ",cl=" << static_cast<int>(c.cluster_shape);
    }
    else
    {
        s << "cuda=0,tma=0"
          << ",tile80=" << static_cast<int>(c.tile_config_sm80)
          << ",stages=" << c.stages
          << ",splitk=" << c.split_k_factor
          << ",sk_style=" << static_cast<int>(c.split_k_style);
    }
    return s.str();
}

std::string format_config_for_cli(tensorrt_llm::cutlass_extensions::CutlassGemmConfig const& c)
{
    if (c.enableCudaKernel)
    {
        return "cuda";
    }

    if (c.is_tma_warp_specialized)
    {
        if (c.mainloop_schedule != tensorrt_llm::cutlass_extensions::MainloopScheduleType::AUTO
            || c.epilogue_schedule != tensorrt_llm::cutlass_extensions::EpilogueScheduleType::AUTO)
        {
            return serialize_config(c);
        }

        int tile_m = 0;
        int tile_n = 0;
        int tile_k = 0;
        int cluster_m = 0;
        int cluster_n = 0;
        int cluster_k = 0;
        bool const ok = decode_shape_tuple(static_cast<int>(c.tile_config_sm90), tile_m, tile_n, tile_k)
            && decode_shape_tuple(static_cast<int>(c.cluster_shape), cluster_m, cluster_n, cluster_k);
        if (ok)
        {
            return "sm90:" + shape3_to_string(tile_m, tile_n, tile_k) + ":"
                + shape3_to_string(cluster_m, cluster_n, cluster_k);
        }
        return serialize_config(c);
    }

    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;
    if (shape_from_tile80(c.tile_config_sm80, tile_m, tile_n, tile_k))
    {
        return "sm80:" + shape3_to_string(tile_m, tile_n, tile_k) + ":" + std::to_string(c.stages) + ":"
            + std::to_string(c.split_k_factor);
    }
    return serialize_config(c);
}

bool deserialize_config(std::string const& str, tensorrt_llm::cutlass_extensions::CutlassGemmConfig& out)
{
    using namespace tensorrt_llm::cutlass_extensions;
    out = CutlassGemmConfig{};

    // Parse key=value pairs
    auto get_int = [&](char const* key) -> int {
        std::string needle = std::string(key) + "=";
        auto pos = str.find(needle);
        if (pos == std::string::npos) return -999;
        return std::atoi(str.c_str() + pos + needle.size());
    };

    int cuda = get_int("cuda");
    if (cuda == 1)
    {
        out.enableCudaKernel = true;
        return true;
    }

    int tma = get_int("tma");
    if (tma == 1)
    {
        out.is_tma_warp_specialized = true;
        out.sm_version = get_int("sm");
        out.tile_config_sm90 = static_cast<CutlassTileConfigSM90>(get_int("tile90"));
        out.mainloop_schedule = static_cast<MainloopScheduleType>(get_int("ml"));
        out.epilogue_schedule = static_cast<EpilogueScheduleType>(get_int("el"));
        out.cluster_shape = static_cast<ClusterShape>(get_int("cl"));
        return true;
    }

    // SM80
    out.tile_config_sm80 = static_cast<CutlassTileConfig>(get_int("tile80"));
    out.stages = get_int("stages");
    out.split_k_factor = get_int("splitk");
    out.split_k_style = static_cast<SplitKStyle>(get_int("sk_style"));
    out.sm_version = 80;
    return true;
}

bool load_tactic(std::string const& path, int m, int n, int k, int gs,
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig& config)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;

    char key_prefix[128];
    std::snprintf(key_prefix, sizeof(key_prefix), "%d,%d,%d,%d|", m, n, k, gs);
    std::string prefix(key_prefix);

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        if (line.compare(0, prefix.size(), prefix) == 0)
        {
            return deserialize_config(line.substr(prefix.size()), config);
        }
    }
    return false;
}

void save_tactic(std::string const& path, int m, int n, int k, int gs,
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig const& config)
{
    std::ofstream f(path, std::ios::app);
    if (!f.is_open())
    {
        std::fprintf(stderr, "Warning: cannot write tactic file %s\n", path.c_str());
        return;
    }
    f << m << "," << n << "," << k << "," << gs << "|" << serialize_config(config) << "\n";
}

void parse_args(int argc, char** argv, Args& args)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (parse_int(argv[i], "--m=", args.m))
        {
            continue;
        }
        if (parse_int(argv[i], "--n=", args.n))
        {
            continue;
        }
        if (parse_int(argv[i], "--k=", args.k))
        {
            continue;
        }
        if (parse_int(argv[i], "--group_size=", args.group_size))
        {
            continue;
        }
        if (parse_int(argv[i], "--warmup=", args.warmup))
        {
            args.warmup_set = true;
            continue;
        }
        if (parse_int(argv[i], "--iters=", args.iters))
        {
            args.iters_set = true;
            continue;
        }
        if (std::strcmp(argv[i], "--verify") == 0)
        {
            args.verify = true;
            continue;
        }
        if (std::strcmp(argv[i], "--list_configs") == 0)
        {
            args.list_configs = true;
            continue;
        }
        if (std::strncmp(argv[i], "--config=", 9) == 0)
        {
            if (!parse_config_spec(argv[i] + 9, args))
            {
                std::fprintf(stderr,
                    "Invalid --config format. Use --config=cuda, --config=sm80:128x128x64:3:1, "
                    "or --config=sm90:128x256x128:2x1x1.\n");
                print_usage(argv[0]);
                std::exit(1);
            }
            continue;
        }
        if (std::strncmp(argv[i], "--tactic=", 9) == 0)
        {
            args.tactic_file = argv[i] + 9;
            continue;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(1);
    }
}

void print_all_configs()
{
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
    Config const* configs = nullptr;
    size_t const count = tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_get_all_configs(&configs);
    std::printf("Supported configs (%zu):\n", count);
    for (size_t i = 0; i < count; ++i)
    {
        std::printf("  %zu: %s\n", i, format_config_for_cli(configs[i]).c_str());
    }
}

bool get_cuda_config(tensorrt_llm::cutlass_extensions::CutlassGemmConfig& out)
{
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
    Config const* configs = nullptr;
    size_t const count = tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_get_all_configs(&configs);
    for (size_t i = 0; i < count; ++i)
    {
        if (configs[i].enableCudaKernel)
        {
            out = configs[i];
            return true;
        }
    }
    return false;
}

void quantize_groupwise_int4(std::vector<half>& scales, std::vector<half>& zeros, std::vector<int8_t>& packed_weights,
    std::vector<int8_t>& signed_weights, std::vector<half> const& weights, int k, int n, int group_size)
{
    int const groups = k / group_size;
    scales.assign(static_cast<size_t>(groups) * n, __float2half_rn(0.0f));
    zeros.assign(static_cast<size_t>(groups) * n, __float2half_rn(0.0f));
    signed_weights.assign(static_cast<size_t>(k) * n, 0);

    for (int g = 0; g < groups; ++g)
    {
        int const k_start = g * group_size;
        int const k_end = k_start + group_size;
        for (int col = 0; col < n; ++col)
        {
            float max_abs = 0.0f;
            for (int row = k_start; row < k_end; ++row)
            {
                float const val = __half2float(weights[static_cast<size_t>(row) * n + col]);
                max_abs = std::max(max_abs, std::fabs(val));
            }
            float const scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
            scales[static_cast<size_t>(g) * n + col] = __float2half_rn(scale);

            int const qzero = 7; // GPTQ zero-point encoding (zero point = qzero + 1)
            zeros[static_cast<size_t>(g) * n + col] = __float2half_rn((7 - qzero) * scale);

            for (int row = k_start; row < k_end; ++row)
            {
                float const val = __half2float(weights[static_cast<size_t>(row) * n + col]);
                float const scaled = scale > 0.0f ? std::round(val / scale) : 0.0f;
                int q = static_cast<int>(scaled);
                q = std::max(-8, std::min(7, q));
                signed_weights[static_cast<size_t>(row) * n + col] = static_cast<int8_t>(q);
            }
        }
    }

    int const bytes_per_row = n / 2;
    packed_weights.assign(static_cast<size_t>(k) * bytes_per_row, 0);
    for (int row = 0; row < k; ++row)
    {
        for (int byte_idx = 0; byte_idx < bytes_per_row; ++byte_idx)
        {
            int const col0 = 2 * byte_idx;
            int const col1 = col0 + 1;
            int8_t q0 = signed_weights[static_cast<size_t>(row) * n + col0];
            int8_t q1 = signed_weights[static_cast<size_t>(row) * n + col1];
            uint8_t u0 = static_cast<uint8_t>(q0) & 0x0F;
            uint8_t u1 = static_cast<uint8_t>(q1) & 0x0F;
            packed_weights[static_cast<size_t>(row) * bytes_per_row + byte_idx] = static_cast<int8_t>(u0 | (u1 << 4));
        }
    }
}

struct DiffStats
{
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t max_idx = 0;
};

DiffStats compare_half_to_float(std::vector<half> const& gpu, std::vector<float> const& ref)
{
    DiffStats stats;
    size_t const n = std::min(gpu.size(), ref.size());
    for (size_t i = 0; i < n; ++i)
    {
        float const gpu_val = __half2float(gpu[i]);
        float const ref_val = ref[i];
        float const abs_err = std::fabs(gpu_val - ref_val);
        float const rel_err = abs_err / (std::fabs(ref_val) + 1e-6f);
        if (abs_err > stats.max_abs)
        {
            stats.max_abs = abs_err;
            stats.max_idx = i;
        }
        if (rel_err > stats.max_rel)
        {
            stats.max_rel = rel_err;
        }
    }
    return stats;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    parse_args(argc, argv, args);

    if (args.list_configs)
    {
        print_all_configs();
        return 0;
    }

    if (args.m <= 0 || args.n <= 0 || args.k <= 0)
    {
        std::fprintf(stderr, "m, n, k must be positive.\n");
        return 1;
    }
    if (args.warmup < 0 || args.iters <= 0)
    {
        std::fprintf(stderr, "warmup must be >= 0 and iters must be > 0.\n");
        return 1;
    }
    if (args.group_size != 128 && args.group_size != 64)
    {
        std::fprintf(stderr, "group_size must be 64 or 128.\n");
        return 1;
    }
    if ((args.k % args.group_size) != 0)
    {
        std::fprintf(stderr, "k must be multiple of group_size.\n");
        return 1;
    }
    if ((args.k % 64) != 0)
    {
        std::fprintf(stderr, "k must be multiple of 64 for SM80 fpA_intB.\n");
        return 1;
    }
    if ((args.n % 2) != 0)
    {
        std::fprintf(stderr, "n must be even for int4 packing.\n");
        return 1;
    }

    std::printf("m=%d n=%d k=%d group_size=%d\n", args.m, args.n, args.k, args.group_size);
    std::fflush(stdout);

    auto check_cuda = [](cudaError_t status, char const* label)
    {
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "CUDA error at %s: %s (%d)\n", label, cudaGetErrorString(status),
                static_cast<int>(status));
            std::abort();
        }
    };

    std::vector<half> h_a(static_cast<size_t>(args.m) * args.k);
    std::vector<half> h_w(static_cast<size_t>(args.k) * args.n);

    for (size_t i = 0; i < h_a.size(); ++i)
    {
        h_a[i] = __float2half_rn(static_cast<float>(i % 101) * 0.01f);
    }
    for (size_t i = 0; i < h_w.size(); ++i)
    {
        h_w[i] = __float2half_rn(static_cast<float>(i % 79) * 0.02f);
    }

    std::vector<half> h_scales;
    std::vector<half> h_zeros;
    std::vector<int8_t> h_packed;
    std::vector<int8_t> h_signed;
    quantize_groupwise_int4(h_scales, h_zeros, h_packed, h_signed, h_w, args.k, args.n, args.group_size);

    std::vector<int8_t> h_preprocessed(h_packed.size(), 0);
    std::vector<size_t> shape = {static_cast<size_t>(args.k), static_cast<size_t>(args.n)};
    tensorrt_llm::kernels::cutlass_kernels::preprocess_weights_for_mixed_gemm(
        h_preprocessed.data(), h_packed.data(), shape, tensorrt_llm::kernels::cutlass_kernels::QuantType::W4_A16, false);

    half* d_a = nullptr;
    int8_t* d_b = nullptr;
    half* d_scales = nullptr;
    half* d_zeros = nullptr;
    half* d_c = nullptr;

    size_t const a_bytes = h_a.size() * sizeof(half);
    size_t const b_bytes = h_preprocessed.size();
    size_t const scale_bytes = h_scales.size() * sizeof(half);
    size_t const c_bytes = static_cast<size_t>(args.m) * args.n * sizeof(half);

    check_cuda(cudaMalloc(&d_a, a_bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, b_bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_scales, scale_bytes), "cudaMalloc d_scales");
    check_cuda(cudaMalloc(&d_zeros, scale_bytes), "cudaMalloc d_zeros");
    check_cuda(cudaMalloc(&d_c, c_bytes), "cudaMalloc d_c");

    check_cuda(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    check_cuda(cudaMemcpy(d_b, h_preprocessed.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_b");
    check_cuda(cudaMemcpy(d_scales, h_scales.data(), scale_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_scales");
    check_cuda(cudaMemcpy(d_zeros, h_zeros.data(), scale_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_zeros");
    std::printf("device buffers ready\n");
    std::fflush(stdout);

    size_t workspace_bytes = tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_get_workspace_size(
        args.m, args.n, args.k);
    std::printf("workspace bytes: %zu\n", workspace_bytes);
    std::fflush(stdout);
    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
    {
        check_cuda(cudaMalloc(&d_workspace, workspace_bytes), "cudaMalloc d_workspace");
    }

    cudaStream_t stream{};
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    std::printf("stream created\n");
    std::fflush(stdout);

    tensorrt_llm::cutlass_extensions::CutlassGemmConfig config{};
    bool tactic_hit = false;
    if (args.force_config)
    {
        if (args.force_cuda)
        {
            if (!get_cuda_config(config))
            {
                std::fprintf(stderr, "CUDA kernel config not found in candidate list.\\n");
                return 1;
            }
        }
        else
        {
            config = args.forced_config;
        }
        if (!tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_is_supported_config(config))
        {
            std::fprintf(stderr, "Forced config not supported. Use --list_configs.\n");
            return 1;
        }
    }
    else if (!args.tactic_file.empty())
    {
        // Try loading from tactic file first
        if (load_tactic(args.tactic_file, args.m, args.n, args.k, args.group_size, config))
        {
            tactic_hit = true;
            std::printf("tactic cache HIT from %s\n", args.tactic_file.c_str());
        }
        if (!tactic_hit)
        {
            // Miss: profile and save
            std::printf("tactic cache MISS: profiling...\n");
            bool ok = tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_select_config_fp16_int4_gptq(
                config, args.m, args.n, args.k, args.group_size, 0, true);
            if (!ok)
            {
                std::fprintf(stderr, "No valid config found for the given shape.\n");
                return 1;
            }
            save_tactic(args.tactic_file, args.m, args.n, args.k, args.group_size, config);
            std::printf("tactic saved to %s\n", args.tactic_file.c_str());
        }
    }
    else
    {
        bool ok = tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_select_config_fp16_int4_gptq(
            config, args.m, args.n, args.k, args.group_size, 0, true);
        if (!ok)
        {
            std::fprintf(stderr, "No valid config found for the given shape.\n");
            return 1;
        }
    }

    std::printf("selected config: %s\n", format_config_for_cli(config).c_str());

    for (int i = 0; i < args.warmup; ++i)
    {
        tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_gemm_fp16_int4_gptq_with_config(
            d_a, d_b, d_scales, d_zeros, d_c, args.m, args.n, args.k, args.group_size, stream, d_workspace,
            workspace_bytes, config);
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (int i = 0; i < args.iters; ++i)
    {
        tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_gemm_fp16_int4_gptq_with_config(
            d_a, d_b, d_scales, d_zeros, d_c, args.m, args.n, args.k, args.group_size, stream, d_workspace,
            workspace_bytes, config);
    }
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    std::vector<half> h_out(static_cast<size_t>(args.m) * args.n);
    check_cuda(cudaMemcpy(h_out.data(), d_c, c_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_c");

    if (args.verify)
    {
        bool const small_case = args.m <= 4 && args.n <= 256 && args.k <= 256;
        if (!small_case)
        {
            std::printf("verify requested, but sizes are too large; skipping CPU reference.\n");
        }
        else
        {
            std::vector<float> ref(h_out.size(), 0.0f);
            for (int m = 0; m < args.m; ++m)
            {
                for (int n = 0; n < args.n; ++n)
                {
                    float acc = 0.0f;
                    for (int k = 0; k < args.k; ++k)
                    {
                        int const group = k / args.group_size;
                        half const scale_h = h_scales[static_cast<size_t>(group) * args.n + n];
                        half const zero_h = h_zeros[static_cast<size_t>(group) * args.n + n];
                        float const scale = __half2float(scale_h);
                        float const zero = __half2float(zero_h);

                        int q = static_cast<int>(h_signed[static_cast<size_t>(k) * args.n + n]);
                        float const w = static_cast<float>(q) * scale + zero;
                        float const a = __half2float(h_a[static_cast<size_t>(m) * args.k + k]);
                        acc += a * w;
                    }
                    ref[static_cast<size_t>(m) * args.n + n] = acc;
                }
            }

            DiffStats stats = compare_half_to_float(h_out, ref);
            float const abs_tol = 5e-2f;
            float const rel_tol = 5e-2f;
            bool const pass = (stats.max_abs <= abs_tol) || (stats.max_rel <= rel_tol);
            std::printf("verify: max_abs=%.6f max_rel=%.6f %s\n", stats.max_abs, stats.max_rel,
                pass ? "PASS" : "FAIL");
        }
    }
    else
    {
        float checksum = 0.0f;
        size_t const sample = std::min<size_t>(h_out.size(), 1024);
        for (size_t i = 0; i < sample; ++i)
        {
            checksum += __half2float(h_out[i]);
        }
        std::printf("checksum=%.6f\n", checksum);
    }

    std::printf("Avg kernel time: %.3f us (%d iters, %d warmup)\n",
        (ms * 1000.0f) / static_cast<float>(args.iters), args.iters, args.warmup);

    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_scales), "cudaFree d_scales");
    check_cuda(cudaFree(d_zeros), "cudaFree d_zeros");
    check_cuda(cudaFree(d_c), "cudaFree d_c");
    if (d_workspace)
    {
        check_cuda(cudaFree(d_workspace), "cudaFree d_workspace");
    }
    tensorrt_llm::common::check_cuda_error(cudaStreamDestroy(stream));

    return 0;
}
