//! Standard compliant exceptions and error-checking facilities for OpenGL, GLFW and SLD2.
//!


#pragma once

#include "opengl/opengl.hpp"

#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>


namespace glew {
namespace error {

class ErrorCategory : public std::error_category {
    inline virtual auto name()                 const noexcept -> char const* override;
    inline virtual auto message(int condition) const noexcept -> std::string override;
};

auto ErrorCategory::name() const noexcept -> char const* {
    return "GLEW";
}

auto ErrorCategory::message(int condition) const noexcept -> std::string {
    auto err = static_cast<GLenum>(condition);

    std::stringstream msg;
    msg << glewGetErrorString(err);
    msg << " (0x" << std::hex << std::setw(4) << std::setfill('0') << err << ")";

    return msg.str();
}


inline auto category() noexcept -> std::error_category const& {
    static ErrorCategory category;
    return category;
}

}   /* namespace error */


class Exception : public std::system_error {
public:
    explicit Exception(GLenum errc)
        : std::system_error{static_cast<int>(errc), error::category()} {}

    Exception(GLenum errc, std::string const& what)
        : std::system_error{static_cast<int>(errc), error::category(), what} {}
};


void except(GLenum errc) {
    if (errc != GLEW_OK) {
        throw Exception{errc};
    }
}

}   /* namespace glew */


namespace opengl {
namespace error {


auto get_string(GLenum err) -> std::string {
    switch (err) {
    case GL_NO_ERROR:                       return "GL_NO_ERROR";
    case GL_INVALID_ENUM:                   return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:                  return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:              return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION:  return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY:                  return "GL_OUT_OF_MEMORY";
    case GL_STACK_UNDERFLOW:                return "GL_STACK_UNDERFLOW";
    case GL_STACK_OVERFLOW:                 return "GL_STACK_OVERFLOW";
    case GL_CONTEXT_LOST:                   return "GL_CONTEXT_LOST";
    default:                                return "Unknown error";
    }
}


class ErrorCategory : public std::error_category {
    inline virtual auto name()                 const noexcept -> char const* override;
    inline virtual auto message(int condition) const noexcept -> std::string override;
};

auto ErrorCategory::name() const noexcept -> char const* {
    return "OpenGL";
}

auto ErrorCategory::message(int condition) const noexcept -> std::string {
    auto err = static_cast<GLenum>(condition);

    std::stringstream msg;
    msg << get_string(err) << " (0x" << std::setw(4) << std::setfill('0') << std::hex << err << ")";

    return msg.str();
}


inline auto category() noexcept -> std::error_category const& {
    static ErrorCategory category;
    return category;
}

}   /* namespace error */


class Exception : public std::system_error {
public:
    explicit Exception(GLenum errc)
        : std::system_error{static_cast<int>(errc), error::category()} {}

    Exception(GLenum errc, std::string const& what)
        : std::system_error{static_cast<int>(errc), error::category(), what} {}
};


void except(GLenum errc) {
    if (errc != GL_NO_ERROR) {
        throw Exception{errc};
    }
}

void check_error() {
    except(glGetError());
}

}   /* namespace opengl */
