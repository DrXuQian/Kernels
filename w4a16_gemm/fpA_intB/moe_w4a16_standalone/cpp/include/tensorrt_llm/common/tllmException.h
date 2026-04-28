/*
 * Minimal exception utilities for standalone fpA_intB build.
 */
#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"
#include <stdexcept>

#define TLLM_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw std::runtime_error(tensorrt_llm::common::fmtstr(__VA_ARGS__));                                           \
    } while (0)

TRTLLM_NAMESPACE_BEGIN
namespace common
{
using TllmException = std::runtime_error;
} // namespace common
TRTLLM_NAMESPACE_END
