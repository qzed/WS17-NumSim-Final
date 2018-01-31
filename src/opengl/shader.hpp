#pragma once

#include "opengl/opengl.hpp"
#include "opengl/errors.hpp"

#include "utils/resource.hpp"

#include <utility>
#include <vector>


namespace opengl {

class CompileError : public Exception {
public:
    inline CompileError(GLenum errc, std::string log);
    inline CompileError(GLenum errc, std::string const& msg, std::string log);

    inline auto log() const -> std::string const&;

private:
    std::string m_log;
};


class LinkError : public Exception {
public:
    inline LinkError(GLenum errc, std::string log);
    inline LinkError(GLenum errc, std::string const& msg, std::string log);

    inline auto log() const -> std::string const&;

private:
    std::string m_log;
};


class Shader {
public:
    inline static auto create(GLenum type) -> Shader;

    inline Shader(GLenum type, GLuint handle);
    inline Shader(Shader const& other) = delete;
    inline Shader(Shader&& other);
    inline ~Shader();

    inline auto operator= (Shader const& other) -> Shader& = delete;
    inline auto operator= (Shader&& other) -> Shader&;

    inline auto handle() const -> GLuint;
    inline auto type() const -> GLenum;

    inline auto set_source(std::string const& source) const;
    inline auto set_source(utils::Resource const& source) const;

    inline void compile(std::string const& name = "") const;
    inline auto get_compile_status() const -> bool;
    inline auto get_info_log() const -> std::string;

private:
    GLenum m_type;
    GLuint m_handle;
};


class Program {
public:
    inline static auto create() -> Program;

    inline explicit Program(GLuint handle);
    inline Program(Program const& other) = delete;
    inline Program(Program&& other);
    inline ~Program();

    inline auto operator= (Program const& other) -> Program& = delete;
    inline auto operator= (Program&& other) -> Program&;

    inline auto handle() const -> GLuint;

    inline void bind() const;
    inline void unbind() const;

    inline void attach(Shader const& shader) const;
    inline void detach(Shader const& shader) const;

    inline void link() const;
    inline auto get_link_status() const -> bool;
    inline auto get_info_log() const -> std::string;

    inline auto get_uniform_location(GLchar const* name) -> GLint;
    inline void set_uniform(GLint loc, GLint val);

private:
    GLuint m_handle;
};



CompileError::CompileError(GLenum errc, std::string log)
    : Exception{errc}, m_log{std::move(log)} {}

CompileError::CompileError(GLenum errc, std::string const& msg, std::string log)
    : Exception{errc, msg}, m_log{std::move(log)} {}

auto CompileError::log() const -> std::string const& {
    return m_log;
}


LinkError::LinkError(GLenum errc, std::string log)
    : Exception{errc}, m_log{std::move(log)} {}

LinkError::LinkError(GLenum errc, std::string const& msg, std::string log)
    : Exception{errc, msg}, m_log{std::move(log)} {}

auto LinkError::log() const -> std::string const& {
    return m_log;
}


auto Shader::create(GLenum type) -> Shader {
    GLuint shader = glCreateShader(type);

    if (!shader) {
        opengl::check_error();
        throw Exception{static_cast<GLenum>(-1), "Failed to create OpenGL shader"};
    }

    return {type, shader};
}

Shader::Shader(GLenum type, GLuint handle)
    : m_type{type}, m_handle{handle} {}

Shader::Shader(Shader&& other)
    : m_type{other.m_type}, m_handle{std::exchange(other.m_handle, 0)} {}

Shader::~Shader() {
    if (m_handle) {
        glDeleteShader(m_handle);
    }
}

auto Shader::operator= (Shader&& other) -> Shader& {
    m_type = other.m_type;
    m_handle = std::exchange(other.m_handle, 0);
    return *this;
}

auto Shader::handle() const -> GLuint {
    return m_handle;
}

auto Shader::type() const -> GLenum {
    return m_type;
}

auto Shader::set_source(std::string const& source) const {
    GLchar const* data = source.data();
    GLint length = source.length();

    glShaderSource(m_handle, 1, &data, &length);
}

auto Shader::set_source(utils::Resource const& source) const {
    GLchar const* data = reinterpret_cast<GLchar const*>(source.data());
    GLint length = source.size();

    glShaderSource(m_handle, 1, &data, &length);
}

void Shader::compile(std::string const& name) const {
    glCompileShader(m_handle);

    if (!get_compile_status()) {
        std::string msg = name.empty()
            ? "Failed to compile shader"
            : "Failed to compile shader `" + name + "`";

        GLenum errc = glGetError();
        throw CompileError{errc, msg, get_info_log()};
    }

    opengl::check_error();
}

auto Shader::get_compile_status() const -> bool {
    GLint status = GL_FALSE;
    glGetShaderiv(m_handle, GL_COMPILE_STATUS, &status);

    return (status == GL_TRUE);
}

auto Shader::get_info_log() const -> std::string {
    GLint len = 0;
    glGetShaderiv(m_handle, GL_INFO_LOG_LENGTH, &len);

    GLint actual = 0;
    std::vector<GLchar> buffer(len);
    glGetShaderInfoLog(m_handle, len, &actual, buffer.data());
    buffer.resize(actual);

    return std::string{buffer.begin(), buffer.end()};
}


auto Program::create() -> Program {
    GLuint handle = glCreateProgram();

    if (!handle) {
        opengl::check_error();
        throw Exception{static_cast<GLenum>(-1), "Failed to create OpenGL program"};
    }

    return Program{handle};
}

Program::Program(GLuint handle)
    : m_handle{handle} {}

Program::Program(Program&& other)
    : m_handle{std::exchange(other.m_handle, 0)} {}

Program::~Program() {
    if (m_handle) {
        glDeleteProgram(m_handle);
    }
}

auto Program::operator= (Program&& other) -> Program& {
    m_handle = std::exchange(other.m_handle, 0);
    return *this;
}

auto Program::handle() const -> GLuint {
    return m_handle;
}

void Program::bind() const {
    glUseProgram(m_handle);
}

void Program::unbind() const {
    glUseProgram(0);
}

void Program::attach(Shader const& shader) const {
    glAttachShader(m_handle, shader.handle());
}

void Program::detach(Shader const& shader) const {
    glDetachShader(m_handle, shader.handle());
}

void Program::link() const {
    glLinkProgram(m_handle);

    if (!get_link_status()) {
        GLenum errc = glGetError();
        throw LinkError(errc, "Failed to link program", get_info_log());
    }

    opengl::check_error();
}

auto Program::get_link_status() const -> bool {
    GLint status = GL_FALSE;
    glGetProgramiv(m_handle, GL_LINK_STATUS, &status);

    return (status == GL_TRUE);
}

auto Program::get_info_log() const -> std::string {
    GLint len = 0;
    glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &len);

    GLint actual = 0;
    std::vector<GLchar> buffer(len);
    glGetProgramInfoLog(m_handle, len, &actual, buffer.data());
    buffer.resize(actual);

    return std::string{buffer.begin(), buffer.end()};
}

auto Program::get_uniform_location(GLchar const* name) -> GLint {
    GLint loc = glGetUniformLocation(m_handle, name);

    if (loc == -1) {
        opengl::check_error();
        throw Exception{GL_INVALID_VALUE};
    }

    return loc;
}

void Program::set_uniform(GLint loc, GLint val) {
    glUniform1i(loc, val);
}

}    /* namespace opengl */
