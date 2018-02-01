#pragma once

#include "types.hpp"

#include "opengl/vertex_array.hpp"
#include "opengl/shader.hpp"
#include "opengl/sampler.hpp"
#include "opengl/texture.hpp"

#include "vis/shader/resources.hpp"


namespace vis {

enum class SamplerType {
    Nearest = 0,
    Linear = 1,
};


class Visualizer {
public:
    inline Visualizer();
    inline Visualizer(Visualizer const&) = delete;
    inline Visualizer(Visualizer&& other) = default;

    inline auto operator= (Visualizer const&) -> Visualizer& = delete;
    inline auto operator= (Visualizer&& other) -> Visualizer& = default;

    inline void initialize(ivec2 screen, ivec2 data_size);
    inline void resize(ivec2 screen);

    inline void draw();

    inline void set_sampler(SamplerType sampler);
    inline auto get_sampler() const -> SamplerType;

    inline auto cl_target_texture() const -> opengl::Texture const&;

private:
    ivec2 m_screen_size;
    ivec2 m_data_size;

    opengl::VertexArray m_vao;
    opengl::Program m_shader;
    opengl::Texture m_texture;
    opengl::Sampler m_sampler_nearest;
    opengl::Sampler m_sampler_linear;

    SamplerType m_sampler_type;
};


Visualizer::Visualizer()
    : m_sampler_type{SamplerType::Nearest} {}

void Visualizer::initialize(ivec2 screen, ivec2 data_size) {
    m_screen_size = screen;
    m_data_size = data_size;

    // create emplty vertex-array
    auto vao = opengl::VertexArray::create();

    // create debug shader
    auto shader_vert = opengl::Shader::create(GL_VERTEX_SHADER);
    shader_vert.set_source(vis::shader::resources::fullscreen_vs);
    shader_vert.compile("fullscreen.vs");

    auto shader_frag = opengl::Shader::create(GL_FRAGMENT_SHADER);
    shader_frag.set_sources({
        &vis::shader::resources::debug_fs,
        &vis::shader::resources::cubehelix_glsl,
    });
    shader_frag.compile("debug.fs");

    auto shader = opengl::Program::create();
    shader.attach(shader_vert);
    shader.attach(shader_frag);
    shader.link();
    shader.detach(shader_frag);
    shader.detach(shader_vert);

    // create opencl target texture
    auto texture = opengl::Texture::create(GL_TEXTURE_2D);
    texture.bind();
    texture.image_2d(0, GL_R32F, {data_size.x, data_size.y}, GL_RG, GL_FLOAT, nullptr);
    texture.unbind();

    // create samplers
    auto sampler_nearest = opengl::Sampler::create();
    sampler_nearest.set(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    sampler_nearest.set(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    sampler_nearest.set(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    sampler_nearest.set(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    auto sampler_linear = opengl::Sampler::create();
    sampler_linear.set(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    sampler_linear.set(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    sampler_linear.set(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    sampler_linear.set(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // set uniform
    shader.bind();
    shader.set_uniform(shader.get_uniform_location("u_tex_data"), 0);
    shader.unbind();

    // update
    m_vao = std::move(vao);
    m_shader = std::move(shader);
    m_texture = std::move(texture);
    m_sampler_nearest = std::move(sampler_nearest);
    m_sampler_linear = std::move(sampler_linear);
}

void Visualizer::resize(ivec2 screen) {
    m_screen_size = screen;
}

void Visualizer::draw() {
    auto const& active_sampler = [&]() -> opengl::Sampler const& {
        if (m_sampler_type == SamplerType::Nearest) {
            return m_sampler_nearest;
        } else {
            return m_sampler_linear;
        }
    }();

    m_shader.bind();
    m_vao.bind();
    m_texture.bind(0);
    active_sampler.bind(0);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    m_texture.unbind();
    m_vao.unbind();
    m_shader.unbind();
}

void Visualizer::set_sampler(SamplerType sampler) {
    m_sampler_type = sampler;
}

auto Visualizer::get_sampler() const -> SamplerType {
    return m_sampler_type;
}

auto Visualizer::cl_target_texture() const -> opengl::Texture const& {
    return m_texture;
}

}   /* namespace vis */
