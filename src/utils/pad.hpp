#pragma once


namespace utils {

//! Pads the given `value` upwards to be divisible by the specified `divisor`.
template <typename T>
inline auto pad_up(T value, T divisor) -> T {
    return (value / divisor) * divisor + (value % divisor == 0 ? 0 : divisor);
}

}   /* namespace utils */
