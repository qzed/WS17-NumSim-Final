#pragma once

#include "opengl/opengl.hpp"
#include "opengl/errors.hpp"

#include <utility>


namespace opengl {

class Sampler {
public:
    inline static auto create() -> Sampler;

    inline Sampler(GLuint handle);
    inline Sampler(Sampler const& other) = delete;
    inline Sampler(Sampler&& other);
    inline ~Sampler();

    inline auto operator= (Sampler const& other) -> Sampler& = delete;
    inline auto operator= (Sampler&& other) -> Sampler&;

    inline auto handle() const -> GLuint;
    
    inline void bind(GLuint unit) const;
    inline void unbind(GLuint unit) const;

    inline auto set(GLenum pname, GLint param) const;

private:
    GLuint m_handle;
};


auto Sampler::create() -> Sampler {
    GLuint handle = 0;

    glGenSamplers(1, &handle);
    opengl::check_error();

    return {handle};
}

Sampler::Sampler(GLuint handle)
    : m_handle{handle} {}

Sampler::Sampler(Sampler&& other)
    : m_handle{std::exchange(other.m_handle, 0)} {}

Sampler::~Sampler() {
    if (m_handle) {
        glDeleteSamplers(1, &m_handle);
    }
}

auto Sampler::operator= (Sampler&& other) -> Sampler& {
    m_handle = std::exchange(other.m_handle, 0);
    return *this;
}

auto Sampler::handle() const -> GLuint {
    return m_handle;
}

void Sampler::bind(GLuint unit) const {
    glBindSampler(unit, m_handle);
}
void Sampler::unbind(GLuint unit) const {
    glBindSampler(unit, 0);
}

auto Sampler::set(GLenum pname, GLint param) const {
    glSamplerParameteri(m_handle, pname, param);
    opengl::check_error();
}

}   /* namespace opengl */
