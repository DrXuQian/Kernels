/*
 * Minimal logger stubs for standalone fpA_intB build.
 */
#pragma once

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN
namespace common
{
struct Logger
{
    enum Level
    {
        TRACE = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40
    };

    static Logger* getLogger()
    {
        static Logger logger;
        return &logger;
    }

    bool isEnabled(Level) const
    {
        return false;
    }

    template <typename... Args>
    void log(Level, char const*, Args const&...)
    {
    }
};
} // namespace common
TRTLLM_NAMESPACE_END

#define TLLM_LOG(level, ...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define TLLM_LOG_TRACE(...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define TLLM_LOG_DEBUG(...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define TLLM_LOG_INFO(...)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define TLLM_LOG_WARNING(...)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define TLLM_LOG_ERROR(...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define TLLM_LOG_EXCEPTION(ex, ...)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
