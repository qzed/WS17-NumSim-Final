#pragma once

#include "opengl/opengl.hpp"
#include "opengl/errors.hpp"

#include <utility>


namespace opengl {

class VertexArray {
public:
    inline static auto create() -> VertexArray;

    inline explicit VertexArray(GLuint handle);
    inline VertexArray(VertexArray const& other) = delete;
    inline VertexArray(VertexArray&& other);
    inline ~VertexArray();

    inline auto operator= (VertexArray const& other) -> VertexArray& = delete;
    inline auto operator= (VertexArray&& other) -> VertexArray&;

    inline auto handle() const -> GLuint;

    inline void bind() const;
    inline void unbind() const;

private:
    GLuint m_handle;
};


auto VertexArray::create() -> VertexArray {
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);

    if (!vao) {
        opengl::check_error();
        throw Exception{static_cast<GLenum>(-1), "Failed to create OpenGL vertex array"};
    }

    return VertexArray{vao};
}

VertexArray::VertexArray(GLuint handle)
    : m_handle{handle} {}

VertexArray::VertexArray(VertexArray&& other)
    : m_handle{std::exchange(other.m_handle, 0)} {}

VertexArray::~VertexArray() {
    if (m_handle) {
        glDeleteVertexArrays(1, &m_handle);
    }
}

auto VertexArray::operator= (VertexArray&& other) -> VertexArray& {
    m_handle = std::exchange(other.m_handle, 0);
    return *this;
}

auto VertexArray::handle() const -> GLuint {
    return m_handle;
}

void VertexArray::bind() const {
    glBindVertexArray(m_handle);
}

void VertexArray::unbind() const {
    glBindVertexArray(0);
}

}   /* namespace opengl */
