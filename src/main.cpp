#include "sdl/opengl/window.hpp"
#include "vis/shader/resources.hpp"

#include <iostream>
#include <iomanip>


int main(int argc, char** argv) {
    auto window = sdl::opengl::Window::builder("Numerical Simulations Course 2017/18", 800, 600)
        .set(SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        .set(SDL_GL_CONTEXT_MINOR_VERSION, 3)
        .set(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
        .set(SDL_GL_DOUBLEBUFFER, 1)
        .set(SDL_WINDOW_RESIZABLE)
        .build();

    // TODO
    glewExperimental = true;
    glew::except(glewInit());

    SDL_GL_SetSwapInterval(1);

    auto shader_prog = glCreateProgram();
    opengl::check_error();

    auto shader_vert = glCreateShader(GL_VERTEX_SHADER);
    {
        auto source = reinterpret_cast<GLchar const*>(vis::shader::resources::fullscreen_vs.data());
        auto source_len = static_cast<GLint>(vis::shader::resources::fullscreen_vs.size());

        glShaderSource(shader_vert, 1, &source, &source_len);
        glCompileShader(shader_vert);

        GLint shader_compile_status = GL_FALSE;
        glGetShaderiv(shader_vert, GL_COMPILE_STATUS, &shader_compile_status);
        if (shader_compile_status != GL_TRUE) {
            std::cerr << "Failed to compile vertex shader:\n";

            int len = 0;
            glGetShaderiv(shader_vert, GL_INFO_LOG_LENGTH, &len);

            std::vector<char> buffer(len);
            glGetShaderInfoLog(shader_vert, len, nullptr, buffer.data());

            std::cerr << buffer.data() << "\n";
            std::exit(1);
        }
    }
    opengl::check_error();

    auto shader_frag = glCreateShader(GL_FRAGMENT_SHADER);
    {
        auto source = reinterpret_cast<GLchar const*>(vis::shader::resources::debug_fs.data());
        auto source_len = static_cast<GLint>(vis::shader::resources::debug_fs.size());

        glShaderSource(shader_frag, 1, &source, &source_len);
        glCompileShader(shader_frag);

        GLint shader_compile_status = GL_FALSE;
        glGetShaderiv(shader_frag, GL_COMPILE_STATUS, &shader_compile_status);
        if (shader_compile_status != GL_TRUE) {
            std::cerr << "Failed to compile vertex shader:\n";

            int len = 0;
            glGetShaderiv(shader_frag, GL_INFO_LOG_LENGTH, &len);

            std::vector<char> buffer(len);
            glGetShaderInfoLog(shader_frag, len, nullptr, buffer.data());

            std::cerr << buffer.data() << "\n";
            std::exit(1);
        }
    }
    opengl::check_error();

    glAttachShader(shader_prog, shader_vert);
    glAttachShader(shader_prog, shader_frag);
    glLinkProgram(shader_prog);

    {
        GLint program_link_status = GL_FALSE;
        glGetProgramiv(shader_prog, GL_LINK_STATUS, &program_link_status);
        if (program_link_status != GL_TRUE) {
            std::cerr << "Failed to link shader program:\n";

            GLint len = 0;
            glGetProgramiv(shader_prog, GL_INFO_LOG_LENGTH, &len);

            std::vector<char> buffer(len, 0);
            glGetProgramInfoLog(shader_prog, len, nullptr, buffer.data());

            std::cerr << buffer.data() << "\n";
            std::exit(1);
        }
    }
    opengl::check_error();

    glDetachShader(shader_prog, shader_vert);
    glDetachShader(shader_prog, shader_frag);
    opengl::check_error();

    glDeleteShader(shader_vert);
    glDeleteShader(shader_frag);
    opengl::check_error();

    GLuint vao = 0;

    glGenVertexArrays(1, &vao);
    opengl::check_error();

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
        opengl::check_error();

        glUseProgram(shader_prog);
        opengl::check_error();

        glBindVertexArray(vao);
        opengl::check_error();

        glDrawArrays(GL_TRIANGLES, 0, 3);
        opengl::check_error();

        glBindVertexArray(0);
        opengl::check_error();

        glUseProgram(0);
        opengl::check_error();

        window.swap_buffers();
        opengl::check_error();
    }
}
