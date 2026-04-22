#pragma once
// Standalone compatibility layer: replaces torch/ATen/c10 dependencies

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>

// Replace TORCH_CHECK with assert + message
#define TORCH_CHECK(cond, ...)                                     \
  do {                                                             \
    if (!(cond)) {                                                 \
      fprintf(stderr, "TORCH_CHECK failed: %s at %s:%d\n",        \
              #cond, __FILE__, __LINE__);                          \
      abort();                                                     \
    }                                                              \
  } while (0)
