#pragma once

#include "opengl/opengl.hpp"
#include "opengl/errors.hpp"

#include <utility>


namespace opengl {

struct Extent2d {
    GLsizei x;
    GLsizei y;
};


class Texture {
public:
    inline static auto create(GLenum target) -> Texture;

    inline Texture();
    inline Texture(GLenum type, GLuint handle);
    inline Texture(Texture const& other) = delete;
    inline Texture(Texture&& other);
    inline ~Texture();

    inline auto operator= (Texture const& other) -> Texture& = delete;
    inline auto operator= (Texture&& other) -> Texture&;

    inline auto handle() const -> GLuint;
    inline auto target() const -> GLenum;

    inline void image_2d(GLint level, GLint internal_format, Extent2d size, GLenum format,
                         GLenum type, const void* pixels) const;
    
    inline void bind() const;
    inline void bind(GLuint unit) const;
    inline void unbind() const;

private:
    GLenum m_target;
    GLuint m_handle;
};


auto Texture::create(GLenum target) -> Texture {
    GLuint handle = 0;

    glGenTextures(1, &handle);
    opengl::check_error();

    return {target, handle};
}

Texture::Texture()
    : m_target{0}, m_handle{0} {}

Texture::Texture(GLenum type, GLuint handle)
    : m_target{type}, m_handle{handle} {}

Texture::Texture(Texture&& other)
    : m_target{other.m_target}, m_handle{std::exchange(other.m_target, 0)} {}

Texture::~Texture() {
    if (m_handle) {
        glDeleteTextures(1, &m_handle);
    }
}

auto Texture::operator= (Texture&& other) -> Texture& {
    m_target = other.m_target;
    m_handle = std::exchange(other.m_handle, 0);
    return *this;
}

auto Texture::handle() const -> GLuint {
    return m_handle;
}

auto Texture::target() const -> GLenum {
    return m_target;
}

void Texture::image_2d(GLint level, GLint internal_format, Extent2d size, GLenum format,
                       GLenum type, const void* pixels) const
{
    glTexImage2D(m_target, level, internal_format, size.x, size.y, 0, format, type, pixels);
    opengl::check_error();
}
    
void Texture::bind() const {
    glBindTexture(m_target, m_handle);
}
    
void Texture::bind(GLuint unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(m_target, m_handle);
}

void Texture::unbind() const {
    glBindTexture(m_target, 0);
}

}   /* namespace opengl */
