#pragma once

#include "sdl/sdl.hpp"

#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>


namespace sdl {
namespace error {

class ErrorCategory : public std::error_category {
    inline virtual auto name()                 const noexcept -> char const* override;
    inline virtual auto message(int condition) const noexcept -> std::string override;
};

auto ErrorCategory::name() const noexcept -> char const* {
    return "SDL2";
}

auto ErrorCategory::message(int condition) const noexcept -> std::string {
    std::stringstream msg;
    msg << "Error code 0x" << std::hex << condition;

    return msg.str();
}


inline auto category() noexcept -> std::error_category const& {
    static ErrorCategory category;
    return category;
}

}   /* namespace error */


class Exception : public std::system_error {
public:
    explicit Exception(int errc)
        : std::system_error{errc, error::category()} {}

    Exception(int errc, std::string const& what)
        : std::system_error{errc, error::category(), what} {}
};


void except(int errc) {
    if (errc < 0) {
        char const* msg = SDL_GetError();
        throw Exception{-errc, msg};
    }
}

template <typename T>
auto except_null(T* ptr) -> T* {
    if (ptr == nullptr) {
        char const* msg = SDL_GetError();
        throw Exception{1, msg};
    } else {
        return ptr;
    }
}

}   /* namespace sdl */
