#pragma once

#include "types.hpp"
#include "utils/cached.hpp"

#include "opengl/vertex_array.hpp"
#include "opengl/shader.hpp"
#include "opengl/sampler.hpp"
#include "opengl/texture.hpp"

#include "vis/shader/resources.hpp"


namespace vis {

enum class SamplerType {
    Nearest,
    Linear,
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

    inline auto get_cl_target_texture() const -> opengl::Texture const&;

    inline void set_data_range(real_t min, real_t max);
    inline auto get_data_range_min();
    inline auto get_data_range_max();

private:
    ivec2 m_screen_size;
    ivec2 m_data_size;

    opengl::VertexArray m_vao;
    opengl::Program m_shader;
    opengl::Texture m_texture;
    opengl::Sampler m_sampler_nearest;
    opengl::Sampler m_sampler_linear;

    GLuint m_shader_loc_tex_data;
    GLuint m_shader_loc_norm_min;
    GLuint m_shader_loc_norm_max;

    SamplerType m_sampler_type;

    utils::Cached<real_t> m_shader_u_norm_min;
    utils::Cached<real_t> m_shader_u_norm_max;
};


Visualizer::Visualizer()
    : m_sampler_type{SamplerType::Nearest} {}

void Visualizer::initialize(ivec2 screen, ivec2 data_size) {
    m_screen_size = screen;
    m_data_size = data_size;

    // create empty vertex-array
    auto vao = opengl::VertexArray::create();

    // create debug shader
    auto shader_vert = opengl::Shader::create(GL_VERTEX_SHADER);
    shader_vert.set_source(vis::shader::resources::fullscreen_vs);
    shader_vert.compile("fullscreen.vs");

    auto shader_frag = opengl::Shader::create(GL_FRAGMENT_SHADER);
    shader_frag.set_sources({
        &vis::shader::resources::map_fs,
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

    // get uniform locations
    GLint loc_tex_data = shader.get_uniform_location("u_tex_data");
    GLint loc_norm_min = shader.get_uniform_location("u_norm_min");
    GLint loc_norm_max = shader.get_uniform_location("u_norm_max");

    // set texture unit in shader
    shader.bind();
    shader.set_uniform(loc_tex_data, 0);
    shader.unbind();

    // update
    m_vao = std::move(vao);
    m_shader = std::move(shader);
    m_texture = std::move(texture);
    m_sampler_nearest = std::move(sampler_nearest);
    m_sampler_linear = std::move(sampler_linear);

    m_shader_loc_tex_data = loc_tex_data;
    m_shader_loc_norm_min = loc_norm_min;
    m_shader_loc_norm_max = loc_norm_max;

    m_shader_u_norm_min = 0.0f;
    m_shader_u_norm_max = 1.0f;
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
    m_shader_u_norm_min.when_dirty([&](real_t val) {
        m_shader.set_uniform(m_shader_loc_norm_min, static_cast<GLfloat>(val));
    });
    m_shader_u_norm_max.when_dirty([&](real_t val) {
        m_shader.set_uniform(m_shader_loc_norm_max, static_cast<GLfloat>(val));
    });

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

auto Visualizer::get_cl_target_texture() const -> opengl::Texture const& {
    return m_texture;
}

void Visualizer::set_data_range(real_t min, real_t max) {
    m_shader_u_norm_min = min;
    m_shader_u_norm_max = max;
}

auto Visualizer::get_data_range_min() {
    return m_shader_u_norm_min.get();
}

auto Visualizer::get_data_range_max() {
    return m_shader_u_norm_max.get();
}

}   /* namespace vis */
