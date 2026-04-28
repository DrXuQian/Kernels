#pragma once

#include <stdexcept>
#include <sstream>

namespace machete_standalone {

template <typename... Args>
[[noreturn]] inline void throw_check_failure(char const* expr, char const* file, int line, Args const&... args)
{
    std::ostringstream os;
    os << file << ":" << line << ": check failed: " << expr;
    ((os << " " << args), ...);
    throw std::runtime_error(os.str());
}

} // namespace machete_standalone

#define MACHETE_CHECK(expr, ...)                                                                                      \
    do                                                                                                                \
    {                                                                                                                 \
        if (!(expr))                                                                                                  \
        {                                                                                                             \
            ::machete_standalone::throw_check_failure(#expr, __FILE__, __LINE__, ##__VA_ARGS__);                     \
        }                                                                                                             \
    } while (0)

#define TORCH_CHECK(expr, ...) MACHETE_CHECK(expr, ##__VA_ARGS__)
