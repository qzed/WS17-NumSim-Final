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

#include "utils/pad.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>


const std::string WINDOW_TITLE = "Numerical Simulations Course 2017/18";
const ivec2 INITIAL_SCREEN_SIZE = {800, 800};
const ivec2 SIMULATION_SIZE = {128, 128};

const char* OCL_COMPILER_OPTIONS =
    "-cl-single-precision-constant "
    "-cl-denorms-are-zero "
    "-cl-strict-aliasing "
    "-cl-fast-relaxed-math "
    "-Werror";


enum class VisualTarget {
    UVAbsCentered,
    UCentered,
    U,
    VCentered,
    V,
    P,
    BoundaryTypes,
    F,
    G,
    Rhs,
};


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
    cl_zero_program.build({device}, OCL_COMPILER_OPTIONS);

    // progam: visualize
    cl::Program::Sources cl_visualize_sources;
    cl_visualize_sources.push_back(core::kernel::resources::visualize_cl.to_string());

    cl::Program cl_visualize_program{cl_context, cl_visualize_sources};
    cl_visualize_program.build({device}, OCL_COMPILER_OPTIONS);

    // progam: boundaries
    cl::Program::Sources cl_boundaries_sources;
    cl_boundaries_sources.push_back(core::kernel::resources::boundaries_cl.to_string());

    cl::Program cl_boundaries_program{cl_context, cl_boundaries_sources};
    cl_boundaries_program.build({device}, OCL_COMPILER_OPTIONS);

    // progam: momentum (preliminary velocities)
    cl::Program::Sources cl_momentum_sources;
    cl_momentum_sources.push_back(core::kernel::resources::momentum_cl.to_string());

    cl::Program cl_momentum_program{cl_context, cl_momentum_sources};
    cl_momentum_program.build({device}, OCL_COMPILER_OPTIONS);

    // progam: rhs (right-hand-side of pressure equation)
    cl::Program::Sources cl_rhs_sources;
    cl_rhs_sources.push_back(core::kernel::resources::rhs_cl.to_string());

    cl::Program cl_rhs_program{cl_context, cl_rhs_sources};
    cl_rhs_program.build({device}, OCL_COMPILER_OPTIONS);

    // progam: solver
    cl::Program::Sources cl_solver_sources;
    cl_solver_sources.push_back(core::kernel::resources::solver_cl.to_string());

    cl::Program cl_solver_program{cl_context, cl_solver_sources};
    cl_solver_program.build({device}, OCL_COMPILER_OPTIONS);

    // progam: velocities (calculate updated velocities)
    cl::Program::Sources cl_velocities_sources;
    cl_velocities_sources.push_back(core::kernel::resources::velocities_cl.to_string());

    cl::Program cl_velocities_program{cl_context, cl_velocities_sources};
    cl_velocities_program.build({device}, OCL_COMPILER_OPTIONS);

    // progam: reduce (calculate updated reduce)
    cl::Program::Sources cl_reduce_sources;
    cl_reduce_sources.push_back(core::kernel::resources::reduce_cl.to_string());

    cl::Program cl_reduce_program{cl_context, cl_reduce_sources};
    cl_reduce_program.build({device}, OCL_COMPILER_OPTIONS);


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

    // initialize reduction stuff
    uint_t const reduce_u_size = (SIMULATION_SIZE.x + 1) * SIMULATION_SIZE.y;
    uint_t const reduce_v_size = SIMULATION_SIZE.x * (SIMULATION_SIZE.y + 1);
    uint_t const reduce_local_size = 128;

    uint_t const reduce_global_size_u = utils::pad_up(reduce_u_size, reduce_local_size);
    uint_t const reduce_global_size_v = utils::pad_up(reduce_v_size, reduce_local_size);

    uint_t const reduce_output_size_u = reduce_global_size_u / reduce_local_size;
    uint_t const reduce_output_size_v = reduce_global_size_v / reduce_local_size;

    auto buf_reduce_out_u = cl::Buffer{cl_context, CL_MEM_WRITE_ONLY, reduce_output_size_u * sizeof(cl_float)};
    auto buf_reduce_out_v = cl::Buffer{cl_context, CL_MEM_WRITE_ONLY, reduce_output_size_v * sizeof(cl_float)};

    auto vec_reduce_out_u = std::vector<cl_float>(reduce_output_size_u);
    auto vec_reduce_out_v = std::vector<cl_float>(reduce_output_size_v);

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

    real_t t = 0.0;
    real_t dt = params.dt;

    VisualTarget visual = VisualTarget::UVAbsCentered;

    bool running = true;
    bool cont = false;
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
                if (e.key.keysym.sym == SDLK_RETURN) {
                    cont = true;
                } else if (e.key.keysym.sym == SDLK_l) {
                    visualizer.set_sampler(vis::SamplerType::Linear);
                } else if (e.key.keysym.sym == SDLK_n) {
                    visualizer.set_sampler(vis::SamplerType::Nearest);
                } else if (e.key.keysym.sym == SDLK_q) {
                    running = false;

                } else if (e.key.keysym.sym == SDLK_1) {
                    visual = VisualTarget::UVAbsCentered;
                } else if (e.key.keysym.sym == SDLK_2) {
                    visual = VisualTarget::U;
                } else if (e.key.keysym.sym == SDLK_3) {
                    visual = VisualTarget::V;
                } else if (e.key.keysym.sym == SDLK_4) {
                    visual = VisualTarget::P;
                } else if (e.key.keysym.sym == SDLK_5) {
                    visual = VisualTarget::F;
                } else if (e.key.keysym.sym == SDLK_6) {
                    visual = VisualTarget::G;
                } else if (e.key.keysym.sym == SDLK_7) {
                    visual = VisualTarget::Rhs;
                } else if (e.key.keysym.sym == SDLK_8) {
                    visual = VisualTarget::BoundaryTypes;
                }
            }
        }

        for (int i = 0; i < 100; i++) {
        // if (cont) { cont = false;
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

        {   // set pressure boundary    // TODO: only required initially
            cl::Kernel kernel_boundary_p{cl_boundaries_program, "set_boundary_p"};
            kernel_boundary_p.setArg(0, buf_p);
            kernel_boundary_p.setArg(1, buf_boundary);
            kernel_boundary_p.setArg(2, static_cast<cl_float>(geom.boundary_pressure()));

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel_boundary_p, cl::NullRange, range, cl::NullRange);
        }

        {   // calculate new dt

            // calculate maximum absolutes for u and v
            cl::Kernel kernel_u{cl_reduce_program, "reduce_max_abs"};
            kernel_u.setArg(0, buf_u);
            kernel_u.setArg(1, buf_reduce_out_u);
            kernel_u.setArg(2, cl::Local(reduce_local_size * sizeof(cl_float)));
            kernel_u.setArg(3, static_cast<cl_uint>(reduce_u_size));

            cl_queue.enqueueNDRangeKernel(kernel_u, cl::NullRange, cl::NDRange(reduce_global_size_u), cl::NDRange(reduce_local_size));

            cl::Kernel kernel_v{cl_reduce_program, "reduce_max_abs"};
            kernel_v.setArg(0, buf_v);
            kernel_v.setArg(1, buf_reduce_out_v);
            kernel_v.setArg(2, cl::Local(reduce_local_size * sizeof(cl_float)));
            kernel_v.setArg(3, static_cast<cl_uint>(reduce_v_size));

            cl_queue.enqueueNDRangeKernel(kernel_v, cl::NullRange, cl::NDRange(reduce_global_size_v), cl::NDRange(reduce_local_size));

            cl::copy(cl_queue, buf_reduce_out_u, vec_reduce_out_u.begin(), vec_reduce_out_u.end());
            cl::copy(cl_queue, buf_reduce_out_v, vec_reduce_out_v.begin(), vec_reduce_out_v.end());

            real_t u_abs_max = static_cast<real_t>(*std::max_element(vec_reduce_out_u.begin(), vec_reduce_out_u.end()));
            real_t v_abs_max = static_cast<real_t>(*std::max_element(vec_reduce_out_v.begin(), vec_reduce_out_v.end()));

            rvec2 const d = geom.mesh();
            real_t const dt_diff = ((d.x*d.x * d.y*d.y) / (d.x*d.x + d.y*d.y)) * params.re * static_cast<real_t>(0.5);
            real_t const dt_conv = std::min(d.x / u_abs_max, d.y / v_abs_max);
            dt = std::min(params.dt, params.tau * std::min(dt_diff, dt_conv));
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
            kernel_momentum_f.setArg(6, static_cast<cl_float>(dt));
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
            kernel_momentum_g.setArg(6, static_cast<cl_float>(dt));
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
            kernel_rhs.setArg(4, static_cast<cl_float>(dt));
            kernel_rhs.setArg(5, h);

            auto range = cl::NDRange(SIMULATION_SIZE.x - 2, SIMULATION_SIZE.y - 2);
            cl_queue.enqueueNDRangeKernel(kernel_rhs, cl::NullRange, range, cl::NullRange);
        }

        {   // calculate rhs
            cl_float2 h = {{ static_cast<cl_float>(geom.mesh().x), static_cast<cl_float>(geom.mesh().y) }};

            cl::Kernel kernel_rhs{cl_rhs_program, "compute_rhs"};
            kernel_rhs.setArg(0, buf_f);
            kernel_rhs.setArg(1, buf_g);
            kernel_rhs.setArg(2, buf_rhs);
            kernel_rhs.setArg(3, buf_boundary);
            kernel_rhs.setArg(4, static_cast<cl_float>(dt));
            kernel_rhs.setArg(5, h);

            auto range = cl::NDRange(SIMULATION_SIZE.x - 2, SIMULATION_SIZE.y - 2);
            cl_queue.enqueueNDRangeKernel(kernel_rhs, cl::NullRange, range, cl::NullRange);
        }

        {   // run solver
            cl_float2 h = {{ static_cast<cl_float>(geom.mesh().x), static_cast<cl_float>(geom.mesh().y) }};

            cl::Kernel kernel_red{cl_solver_program, "cycle_red"};
            kernel_red.setArg(0, buf_p);
            kernel_red.setArg(1, buf_rhs);
            kernel_red.setArg(2, buf_boundary);
            kernel_red.setArg(3, h);
            kernel_red.setArg(4, static_cast<cl_float>(params.omega));

            cl::Kernel kernel_black{cl_solver_program, "cycle_black"};
            kernel_black.setArg(0, buf_p);
            kernel_black.setArg(1, buf_rhs);
            kernel_black.setArg(2, buf_boundary);
            kernel_black.setArg(3, h);
            kernel_black.setArg(4, static_cast<cl_float>(params.omega));

            int_t y_cells_black = (SIMULATION_SIZE.y - 2) / 2;
            auto range_red = cl::NDRange(SIMULATION_SIZE.x - 2, SIMULATION_SIZE.y - 2 - y_cells_black);
            auto range_black = cl::NDRange(SIMULATION_SIZE.x - 2, y_cells_black);

            for (int_t i = 0; i < params.itermax; i++) {
                cl_queue.enqueueNDRangeKernel(kernel_red, cl::NullRange, range_red, cl::NullRange);
                cl_queue.enqueueNDRangeKernel(kernel_black, cl::NullRange, range_black, cl::NullRange);
            }

            {   // set pressure boundary
                cl::Kernel kernel_boundary_p{cl_boundaries_program, "set_boundary_p"};
                kernel_boundary_p.setArg(0, buf_p);
                kernel_boundary_p.setArg(1, buf_boundary);
                kernel_boundary_p.setArg(2, static_cast<cl_float>(geom.boundary_pressure()));

                auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
                cl_queue.enqueueNDRangeKernel(kernel_boundary_p, cl::NullRange, range, cl::NullRange);
            }
        }

        {   // calculate new velocities
            cl_float2 h = {{ static_cast<cl_float>(geom.mesh().x), static_cast<cl_float>(geom.mesh().y) }};

            cl::Kernel kernel{cl_velocities_program, "new_velocities"};
            kernel.setArg(0, buf_p);
            kernel.setArg(1, buf_f);
            kernel.setArg(2, buf_g);
            kernel.setArg(3, buf_u);
            kernel.setArg(4, buf_v);
            kernel.setArg(5, buf_boundary);
            kernel.setArg(6, static_cast<cl_float>(dt));
            kernel.setArg(7, h);

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
        }

        t += dt;
        }
        std::cout << "time: " << t << "\n";
        std::cout << "dt:   " << dt << "\n";

        // write to texture via OpenCL
        glFinish();
        cl_queue.finish();
        cl_queue.enqueueAcquireGLObjects(&cl_req);

        {
            cl::Kernel kernel;

            if (visual == VisualTarget::BoundaryTypes) {
                kernel = {cl_visualize_program, "visualize_boundaries"};
                kernel.setArg(0, cl_image);
                kernel.setArg(1, buf_boundary);

            } else if (visual == VisualTarget::P) {
                kernel = {cl_visualize_program, "visualize_p"};
                kernel.setArg(0, cl_image);
                kernel.setArg(1, buf_p);

            } else if (visual == VisualTarget::U) {
                kernel = {cl_visualize_program, "visualize_u"};
                kernel.setArg(0, cl_image);
                kernel.setArg(1, buf_u);

            } else if (visual == VisualTarget::V) {
                kernel = {cl_visualize_program, "visualize_v"};
                kernel.setArg(0, cl_image);
                kernel.setArg(1, buf_v);

            } else if (visual == VisualTarget::UVAbsCentered) {
                kernel = {cl_visualize_program, "visualize_uv_abs_center"};
                kernel.setArg(0, cl_image);
                kernel.setArg(1, buf_u);
                kernel.setArg(2, buf_v);

            } else if (visual == VisualTarget::F) {
                kernel = {cl_visualize_program, "visualize_u"};
                kernel.setArg(0, cl_image);
                kernel.setArg(1, buf_f);

            } else if (visual == VisualTarget::G) {
                kernel = {cl_visualize_program, "visualize_v"};
                kernel.setArg(0, cl_image);
                kernel.setArg(1, buf_g);

            } else if (visual == VisualTarget::Rhs) {
                kernel = {cl_visualize_program, "visualize_rhs"};
                kernel.setArg(0, cl_image);
                kernel.setArg(1, buf_rhs);
            }

            auto range = cl::NDRange(SIMULATION_SIZE.x, SIMULATION_SIZE.y);
            cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
        }

        cl_queue.finish();
        cl_queue.enqueueReleaseGLObjects(&cl_req);
        cl_queue.finish();

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

} catch (cl::Error const& err) {
    std::cerr << "OpenCL Error:\n";
    std::cerr << "  What: " << err.what() << "\n";
    std::cerr << "  Code: " << err.err() << "\n";
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
