/*
 * Minimal string utilities for standalone fpA_intB build.
 */
#pragma once

#include "tensorrt_llm/common/config.h"
#include <cstdarg>
#include <cstdio>
#include <string>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
inline std::string fmtstr(std::string const& s)
{
    return s;
}

inline std::string fmtstr(std::string&& s)
{
    return s;
}

inline std::string fmtstr(char const* format, ...)
{
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    int const needed = std::vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);
    if (needed <= 0)
    {
        va_end(args);
        return std::string();
    }
    std::string result;
    result.resize(static_cast<size_t>(needed));
    std::vsnprintf(result.data(), result.size() + 1, format, args);
    va_end(args);
    return result;
}
} // namespace common

TRTLLM_NAMESPACE_END
