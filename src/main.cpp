#include "opengl/opengl.hpp"
#include "opengl/init.hpp"
#include "opengl/shader.hpp"
#include "opengl/vertex_array.hpp"
#include "opencl/opengl_interop.hpp"

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


    // get OpenCL platform
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform;
    for (auto const& p : platforms) {
        auto extensions = p.getInfo<CL_PLATFORM_EXTENSIONS>();
        if (extensions.find(opencl::opengl::EXT_CL_GL_SHARING) != std::string::npos) {
            platform = p;
        }
    }

    std::cout << "Using platform:\n";
    std::cout << "  Name:       " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "  Vendor:     " << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
    std::cout << "  Version:    " << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
    std::cout << "  Profile:    " << platform.getInfo<CL_PLATFORM_PROFILE>() << "\n";
    std::cout << "  Extensions: " << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << "\n";
    std::cout << "\n";

    // get OpenCL device
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    cl::Device device;
    for (auto const& d : devices) {
        auto extensions = d.getInfo<CL_DEVICE_EXTENSIONS>();
        if (extensions.find(opencl::opengl::EXT_CL_GL_SHARING) != std::string::npos) {
            device = d;
        }
    }

    std::cout << "Using device:\n";
    std::cout << "  Name:       " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "  Vendor:     " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
    std::cout << "  Version:    " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
    std::cout << "  Profile:    " << device.getInfo<CL_DEVICE_PROFILE>() << "\n";
    std::cout << "  Extensions: " << device.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";
    std::cout << "\n";

    // create OpenCL context
    std::vector<cl_context_properties> properties;
    for (auto const& prop : opencl::opengl::get_context_share_properties(window)) {
        properties.push_back(prop.type);
        properties.push_back(prop.value);
    }
    properties.push_back(CL_CONTEXT_PLATFORM);
    properties.push_back((cl_context_properties) platform());
    properties.push_back(0);


    // set-up visualization
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
