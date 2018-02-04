#include "types.hpp"

#include "opengl/opengl.hpp"
#include "opengl/init.hpp"
#include "opengl/shader.hpp"
#include "opengl/texture.hpp"
#include "opengl/sampler.hpp"
#include "opengl/vertex_array.hpp"

#include "opencl/opencl.hpp"
#include "opencl/opengl_interop.hpp"

#include "sdl/opengl/window.hpp"
#include "sdl/opengl/utils.hpp"

#include "vis/visualizer.hpp"

#include "core/kernel/resources.hpp"
#include "core/parameters.hpp"
#include "core/geometry.hpp"

#include <iostream>
#include <iomanip>


const std::string WINDOW_TITLE = "Numerical Simulations Course 2017/18";
const ivec2 INITIAL_SCREEN_SIZE = {800, 800};
const ivec2 SIMULATION_SIZE = {32, 32};


int main(int argc, char** argv) try {
    auto params = core::Parameters{};
    auto geom = core::Geometry::lid_driven_cavity(SIMULATION_SIZE);

    auto window = sdl::opengl::Window::builder(WINDOW_TITLE, INITIAL_SCREEN_SIZE)
        .set(SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        .set(SDL_GL_CONTEXT_MINOR_VERSION, 3)
        .set(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
        .set(SDL_GL_DOUBLEBUFFER, 1)
        .set(SDL_WINDOW_RESIZABLE)
        .build();

    opengl::init();
    sdl::opengl::set_swap_interval(1);

    auto visualizer = vis::Visualizer{};
    visualizer.initialize(INITIAL_SCREEN_SIZE, SIMULATION_SIZE);

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

    if (!platform()) {
        std::cout << "Error: No OpenCl platform with support for extension ";
        std::cout << "`" << opencl::opengl::EXT_CL_GL_SHARING << "`";
        std::cout << " found.";
        return 1;
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

    if (!device()) {
        std::cout << "Error: No OpenCL device with support for extension ";
        std::cout << "`" << opencl::opengl::EXT_CL_GL_SHARING << "`";
        std::cout << " found.";
        return 1;
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

    auto cl_context = cl::Context(device, properties.data());

    // progam: zero
    cl::Program::Sources cl_zero_sources;
    cl_zero_sources.push_back(core::kernel::resources::zero_cl.to_string());

    cl::Program cl_zero_program{cl_context, cl_zero_sources};
    cl_zero_program.build({device});

    // progam: visualize
    cl::Program::Sources cl_visualize_sources;
    cl_visualize_sources.push_back(core::kernel::resources::visualize_cl.to_string());

    cl::Program cl_visualize_program{cl_context, cl_visualize_sources};
    cl_visualize_program.build({device});

    // progam: boundaries
    cl::Program::Sources cl_boundaries_sources;
    cl_boundaries_sources.push_back(core::kernel::resources::boundaries_cl.to_string());

    cl::Program cl_boundaries_program{cl_context, cl_boundaries_sources};
    cl_boundaries_program.build({device});

    // progam: momentum (preliminary velocities)
    cl::Program::Sources cl_momentum_sources;
    cl_momentum_sources.push_back(core::kernel::resources::momentum_cl.to_string());

    cl::Program cl_momentum_program{cl_context, cl_momentum_sources};
    cl_momentum_program.build({device});

    // progam: rhs (right-hand-side of pressure equation)
    cl::Program::Sources cl_rhs_sources;
    cl_rhs_sources.push_back(core::kernel::resources::rhs_cl.to_string());

    cl::Program cl_rhs_program{cl_context, cl_rhs_sources};
    cl_rhs_program.build({device});


    cl::CommandQueue cl_queue{cl_context, device};

    // set boundary buffer
    auto buf_boundary = cl::Buffer{cl_context, CL_MEM_READ_ONLY, geom.data().size() * sizeof(std::uint8_t)};
    cl::copy(cl_queue, geom.data().begin(), geom.data().end(), buf_boundary);

    // create component buffers
    auto buf_u_size = (SIMULATION_SIZE.x + 1) * SIMULATION_SIZE.y * sizeof(cl_float);
    auto buf_u = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_u_size};
    auto buf_f = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_u_size};

    auto buf_v_size = SIMULATION_SIZE.x * (SIMULATION_SIZE.y + 1) * sizeof(cl_float);
    auto buf_v = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_v_size};
    auto buf_g = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_v_size};

    auto buf_p_size = SIMULATION_SIZE.x * SIMULATION_SIZE.y * sizeof(cl_float);
    auto buf_p = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_p_size};

    auto buf_rhs_size = (SIMULATION_SIZE.x - 2) * (SIMULATION_SIZE.y - 2) * sizeof(cl_float);
    auto buf_rhs = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_rhs_size};

    {   // initialize u
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_u);

        auto range = cl::NDRange((SIMULATION_SIZE.x + 1) * SIMULATION_SIZE.y);
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize v
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_v);

        auto range = cl::NDRange(SIMULATION_SIZE.x * (SIMULATION_SIZE.y + 1));
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize f
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_f);

        auto range = cl::NDRange((SIMULATION_SIZE.x + 1) * SIMULATION_SIZE.y);
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize g
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_g);

        auto range = cl::NDRange(SIMULATION_SIZE.x * (SIMULATION_SIZE.y + 1));
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize p
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_p);

        auto range = cl::NDRange(SIMULATION_SIZE.x * SIMULATION_SIZE.y);
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize rhs
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_rhs);

        auto range = cl::NDRange((SIMULATION_SIZE.x - 2) * (SIMULATION_SIZE.y - 2));
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }


    glClearColor(0.0, 0.0, 0.0, 1.0);

    // create OpenCL reference to OpenGL texture
    auto const& texture = visualizer.get_cl_target_texture();
    auto cl_image = cl::ImageGL{cl_context, CL_MEM_WRITE_ONLY, texture.target(), 0, texture.handle()};
    auto cl_req = std::vector<cl::Memory>{cl_image};

    bool running = true;
    while (running) {
        SDL_Event e;

        // handle input
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {   // received on SIGINT or when all windows have been closed
                running = false;
            }

            else if (e.type == SDL_WINDOWEVENT && e.window.windowID == window.id()) {
                if (e.window.event == SDL_WINDOWEVENT_CLOSE) {
                    window.hide();      // hide on close
                } else if (e.window.event == SDL_WINDOWEVENT_RESIZED) {
                    glViewport(0, 0, e.window.data1, e.window.data2);
                }
            }

            else if (e.type == SDL_KEYDOWN && e.key.windowID == window.id()) {
                if (e.key.keysym.sym == SDLK_l) {
                    visualizer.set_sampler(vis::SamplerType::Linear);
                } else if (e.key.keysym.sym == SDLK_n) {
                    visualizer.set_sampler(vis::SamplerType::Nearest);
                } else if (e.key.keysym.sym == SDLK_q) {
                    running = false;
                }
            }
        }


        {   // set u boundary
            cl::Kernel kernel_boundary_u{cl_boundaries_program, "set_boundary_u"};
            kernel_boundary_u.setArg(0, buf_u);
            kernel_boundary_u.setArg(1, buf_boundary);
            kernel_boundary_u.setArg(2, static_cast<cl_float>(geom.boundary_velocity().x));

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel_boundary_u, cl::NullRange, range, cl::NullRange);
        }

        {   // set v boundary
            cl::Kernel kernel_boundary_v{cl_boundaries_program, "set_boundary_v"};
            kernel_boundary_v.setArg(0, buf_v);
            kernel_boundary_v.setArg(1, buf_boundary);
            kernel_boundary_v.setArg(2, static_cast<cl_float>(geom.boundary_velocity().y));

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel_boundary_v, cl::NullRange, range, cl::NullRange);
        }

        {   // set pressure boundary
            cl::Kernel kernel_boundary_p{cl_boundaries_program, "set_boundary_p"};
            kernel_boundary_p.setArg(0, buf_p);
            kernel_boundary_p.setArg(1, buf_boundary);
            kernel_boundary_p.setArg(2, static_cast<cl_float>(geom.boundary_pressure()));

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel_boundary_p, cl::NullRange, range, cl::NullRange);
        }

        {   // calculate preliminary velocities: f
            cl_float2 h = {{ static_cast<cl_float>(geom.mesh().x), static_cast<cl_float>(geom.mesh().y) }};

            cl::Kernel kernel_momentum_f{cl_momentum_program, "momentum_eq_f"};
            kernel_momentum_f.setArg(0, buf_u);
            kernel_momentum_f.setArg(1, buf_v);
            kernel_momentum_f.setArg(2, buf_f);
            kernel_momentum_f.setArg(3, buf_boundary);
            kernel_momentum_f.setArg(4, static_cast<cl_float>(params.alpha));
            kernel_momentum_f.setArg(5, static_cast<cl_float>(params.re));
            kernel_momentum_f.setArg(6, static_cast<cl_float>(params.dt));
            kernel_momentum_f.setArg(7, h);

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel_momentum_f, cl::NullRange, range, cl::NullRange);
        }

        {   // calculate preliminary velocities: f
            cl_float2 h = {{ static_cast<cl_float>(geom.mesh().x), static_cast<cl_float>(geom.mesh().y) }};

            cl::Kernel kernel_momentum_g{cl_momentum_program, "momentum_eq_g"};
            kernel_momentum_g.setArg(0, buf_u);
            kernel_momentum_g.setArg(1, buf_v);
            kernel_momentum_g.setArg(2, buf_g);
            kernel_momentum_g.setArg(3, buf_boundary);
            kernel_momentum_g.setArg(4, static_cast<cl_float>(params.alpha));
            kernel_momentum_g.setArg(5, static_cast<cl_float>(params.re));
            kernel_momentum_g.setArg(6, static_cast<cl_float>(params.dt));
            kernel_momentum_g.setArg(7, h);

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel_momentum_g, cl::NullRange, range, cl::NullRange);
        }

        {   // calculate rhs
            cl_float2 h = {{ static_cast<cl_float>(geom.mesh().x), static_cast<cl_float>(geom.mesh().y) }};

            cl::Kernel kernel_rhs{cl_rhs_program, "compute_rhs"};
            kernel_rhs.setArg(0, buf_f);
            kernel_rhs.setArg(1, buf_g);
            kernel_rhs.setArg(2, buf_rhs);
            kernel_rhs.setArg(3, buf_boundary);
            kernel_rhs.setArg(4, static_cast<cl_float>(params.dt));
            kernel_rhs.setArg(5, h);

            auto range = cl::NDRange(SIMULATION_SIZE.x - 2, SIMULATION_SIZE.y - 2);
            cl_queue.enqueueNDRangeKernel(kernel_rhs, cl::NullRange, range, cl::NullRange);
        }

        // TODO: run solver

        // TODO: calculate new velocities

        // write to texture via OpenCL
        glFlush();
        cl_queue.enqueueAcquireGLObjects(&cl_req);

        {
            // // visualize boundary types
            // cl::Kernel kernel_vis{cl_visualize_program, "visualize_boundaries"};
            // kernel_vis.setArg(0, cl_image);
            // kernel_vis.setArg(1, buf_boundary);

            // // visualize pressure
            // cl::Kernel kernel_vis{cl_visualize_program, "visualize_p"};
            // kernel_vis.setArg(0, cl_image);
            // kernel_vis.setArg(1, buf_p);

            // // visualize velocity u
            // cl::Kernel kernel_vis{cl_visualize_program, "visualize_u_center"};
            // kernel_vis.setArg(0, cl_image);
            // kernel_vis.setArg(1, buf_u);

            // // visualize velocity v
            // cl::Kernel kernel_vis{cl_visualize_program, "visualize_v_center"};
            // kernel_vis.setArg(0, cl_image);
            // kernel_vis.setArg(1, buf_v);

            // // visualize absolute velocity
            // cl::Kernel kernel_vis{cl_visualize_program, "visualize_uv_abs_center"};
            // kernel_vis.setArg(0, cl_image);
            // kernel_vis.setArg(1, buf_u);
            // kernel_vis.setArg(2, buf_v);

            // // visualize preliminary velocity f
            // cl::Kernel kernel_vis{cl_visualize_program, "visualize_u"};
            // kernel_vis.setArg(0, cl_image);
            // kernel_vis.setArg(1, buf_f);

            // // visualize preliminary velocity g
            // cl::Kernel kernel_vis{cl_visualize_program, "visualize_v"};
            // kernel_vis.setArg(0, cl_image);
            // kernel_vis.setArg(1, buf_g);

            // visualize rhs
            cl::Kernel kernel_vis{cl_visualize_program, "visualize_rhs"};
            kernel_vis.setArg(0, cl_image);
            kernel_vis.setArg(1, buf_rhs);

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel_vis, cl::NullRange, range, cl::NullRange);
        }

        cl_queue.enqueueReleaseGLObjects(&cl_req);
        cl_queue.flush();


        // render via OpenGL
        glClear(GL_COLOR_BUFFER_BIT);
        visualizer.draw();

        window.swap_buffers();
        opengl::check_error();
    }


} catch (cl::BuildError const& err) {
    auto const& log = err.getBuildLog();
    std::cerr << "OpenCL Build Error: " << err.what() << "\n";
    for (auto const& entry : log) {
        std::cerr << "-- LOG -------------------------------------------------------------------------\n";
        std::cout << "-- Device: " << entry.first.getInfo<CL_DEVICE_NAME>() << "\n";
        std::cout << entry.second << "\n";
    }
    std::cerr << "--------------------------------------------------------------------------------\n";
    throw err;

} catch (opengl::CompileError const& err) {
    std::cerr << "OpenGL Shader Compile Error: " << err.what() << "\n";
    std::cerr << "-- LOG -------------------------------------------------------------------------\n";
    std::cerr << err.log();
    std::cerr << "--------------------------------------------------------------------------------\n";
    throw err;

} catch (opengl::LinkError const& err) {
    std::cerr << "OpenGL Shader Link Error: " << err.what() << "\n";
    std::cerr << "-- LOG -------------------------------------------------------------------------\n";
    std::cerr << err.log();
    std::cerr << "--------------------------------------------------------------------------------\n";
    throw err;
}
