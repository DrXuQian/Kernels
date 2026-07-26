#pragma once
#include <cstdint>
#include <cstring>
struct __ppu_bfloat16 { uint16_t __x; };
struct __ppu_bfloat16_raw { uint16_t x; __ppu_bfloat16_raw() : x(0) {} __ppu_bfloat16_raw(__ppu_bfloat16 v){ std::memcpy(&x,&v.__x,2); } };
using bfloat16 = __ppu_bfloat16;
