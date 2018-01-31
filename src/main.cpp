#include "opengl/opengl.hpp"
#include "opengl/init.hpp"
#include "opengl/shader.hpp"
#include "opengl/vertex_array.hpp"

#include "sdl/opengl/window.hpp"
#include "sdl/opengl/utils.hpp"

#include "vis/shader/resources.hpp"

#include <iostream>
#include <iomanip>


int main(int argc, char** argv) try {
    auto window = sdl::opengl::Window::builder("Numerical Simulations Course 2017/18", 800, 600)
        .set(SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        .set(SDL_GL_CONTEXT_MINOR_VERSION, 3)
        .set(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
        .set(SDL_GL_DOUBLEBUFFER, 1)
        .set(SDL_WINDOW_RESIZABLE)
        .build();

    opengl::init();
    sdl::opengl::set_swap_interval(1);

    auto shader_vert = opengl::Shader::create(GL_VERTEX_SHADER);
    shader_vert.set_source(vis::shader::resources::fullscreen_vs);
    shader_vert.compile("fullscreen.vs");

    auto shader_frag = opengl::Shader::create(GL_FRAGMENT_SHADER);
    shader_frag.set_source(vis::shader::resources::debug_fs);
    shader_frag.compile("debug.fs");

    auto shader_prog = opengl::Program::create();
    shader_prog.attach(shader_vert);
    shader_prog.attach(shader_frag);
    shader_prog.link();
    shader_prog.detach(shader_frag);
    shader_prog.detach(shader_vert);

    auto vertex_array = opengl::VertexArray::create();

    glClearColor(0.0, 0.0, 0.0, 1.0);

    bool running = true;
    while (running) {
        SDL_Event e;

        // handle input
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {   // received on SIGINT or when all windows have been closed
                running = false;
                std::cout << "Terminating\n";
            }

            else if (e.type == SDL_WINDOWEVENT && e.window.windowID == window.id()) {
                if (e.window.event == SDL_WINDOWEVENT_CLOSE) {
                    window.hide();      // hide on close
                } else if (e.window.event == SDL_WINDOWEVENT_RESIZED) {
                    glViewport(0, 0, e.window.data1, e.window.data2);
                }
            }

            else if (e.type == SDL_KEYDOWN && e.key.windowID == window.id()) {
                std::cout << "Key pressed: "
                    << "0x" << std::hex << std::setfill('0') << std::setw(2) << e.key.keysym.sym
                    << "\n";
            }
        }

        // render
        glClear(GL_COLOR_BUFFER_BIT);

        shader_prog.bind();
        vertex_array.bind();

        glDrawArrays(GL_TRIANGLES, 0, 3);

        vertex_array.unbind();
        shader_prog.unbind();

        window.swap_buffers();
        opengl::check_error();
    }

} catch (opengl::CompileError const& err) {
    std::cerr << "Error: " << err.what() << "\n";
    std::cerr << "-- LOG -------------------------------------------------------------------------\n";
    std::cerr << err.log();
    std::cerr << "--------------------------------------------------------------------------------\n";
    throw err;

} catch (opengl::LinkError const& err) {
    std::cerr << "Error: " << err.what() << "\n";
    std::cerr << "-- LOG -------------------------------------------------------------------------\n";
    std::cerr << err.log();
    std::cerr << "--------------------------------------------------------------------------------\n";
    throw err;
}
