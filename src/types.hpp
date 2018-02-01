#pragma once

#include <cstdint>


using int_t = std::int32_t;
using uint_t = std::uint32_t;
using real_t = float;


struct uvec2 {
    uint_t x;
    uint_t y;
};

struct ivec2 {
    int_t x;
    int_t y;
};

struct rvec2 {
    real_t x;
    real_t y;
};
