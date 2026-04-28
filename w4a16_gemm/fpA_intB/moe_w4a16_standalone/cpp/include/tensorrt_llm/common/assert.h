/*
 * Minimal assert utilities for the standalone fpA_intB build.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include <cstdio>
#include <cstdlib>

#define TLLM_CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(val))                                                                                                    \
        {                                                                                                              \
            std::fprintf(stderr, "TLLM_CHECK failed: %s (%s:%d)\n", #val, __FILE__, __LINE__);                         \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(val))                                                                                                    \
        {                                                                                                              \
            auto _msg = tensorrt_llm::common::fmtstr(info, ##__VA_ARGS__);                                              \
            std::fprintf(stderr, "TLLM_CHECK failed: %s (%s:%d)\n", _msg.c_str(), __FILE__, __LINE__);                 \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)
