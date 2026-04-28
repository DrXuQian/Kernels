/*
 * Minimal NvInferRuntime stub for standalone fpA_intB build.
 */
#pragma once

namespace nvinfer1
{
enum class DataType
{
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kBF16 = 5,
    kFP8 = 6,
    kFP4 = 7
};
}
