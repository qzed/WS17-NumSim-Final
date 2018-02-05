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

#include "core/kernel/sources/resources.hpp"
#include "core/parameters.hpp"
#include "core/geometry.hpp"

#include "utils/pad.hpp"
#include "utils/perf.hpp"

#include "json/json.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>


const std::string WINDOW_TITLE = "Numerical Simulations Course 2017/18";
const ivec2 INITIAL_SCREEN_SIZE = {800, 800};

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
    Vorticity,
    Stream,
};

struct Environment {
    char const* params;
    char const* geom;
    char const* json;
};


auto parse_cmdline(int argc, char** argv) -> Environment;
void write_perf_stats(char const* json);


int main(int argc, char** argv) try {
    auto perf_tts_full = utils::perf::Record::start("tts::full");

    // parse arguments
    Environment env = parse_cmdline(argc, argv);

    auto params = core::Parameters{};
    if (env.params) params.load(env.params);
    auto geom = core::Geometry::lid_driven_cavity({128, 128});
    if (env.geom) geom.load(env.geom);

    auto n_fluid_cells = geom.num_fluid_cells();

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
    visualizer.initialize(INITIAL_SCREEN_SIZE, geom.size());

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

    // program: zero
    cl::Program::Sources cl_zero_sources;
    cl_zero_sources.push_back(core::kernel::resources::zero_cl.to_string());

    cl::Program cl_zero_program{cl_context, cl_zero_sources};
    cl_zero_program.build({device}, OCL_COMPILER_OPTIONS);

    // program: visualize
    cl::Program::Sources cl_visualize_sources;
    cl_visualize_sources.push_back(core::kernel::resources::visualize_cl.to_string());

    cl::Program cl_visualize_program{cl_context, cl_visualize_sources};
    cl_visualize_program.build({device}, OCL_COMPILER_OPTIONS);

    // program: boundaries
    cl::Program::Sources cl_boundaries_sources;
    cl_boundaries_sources.push_back(core::kernel::resources::boundaries_cl.to_string());

    cl::Program cl_boundaries_program{cl_context, cl_boundaries_sources};
    cl_boundaries_program.build({device}, OCL_COMPILER_OPTIONS);

    // program: momentum (preliminary velocities)
    cl::Program::Sources cl_momentum_sources;
    cl_momentum_sources.push_back(core::kernel::resources::momentum_cl.to_string());

    cl::Program cl_momentum_program{cl_context, cl_momentum_sources};
    cl_momentum_program.build({device}, OCL_COMPILER_OPTIONS);

    // program: rhs (right-hand-side of pressure equation)
    cl::Program::Sources cl_rhs_sources;
    cl_rhs_sources.push_back(core::kernel::resources::rhs_cl.to_string());

    cl::Program cl_rhs_program{cl_context, cl_rhs_sources};
    cl_rhs_program.build({device}, OCL_COMPILER_OPTIONS);

    // program: solver
    cl::Program::Sources cl_solver_sources;
    cl_solver_sources.push_back(core::kernel::resources::solver_cl.to_string());

    cl::Program cl_solver_program{cl_context, cl_solver_sources};
    cl_solver_program.build({device}, OCL_COMPILER_OPTIONS);

    // program: velocities (calculate updated velocities)
    cl::Program::Sources cl_velocities_sources;
    cl_velocities_sources.push_back(core::kernel::resources::velocities_cl.to_string());

    cl::Program cl_velocities_program{cl_context, cl_velocities_sources};
    cl_velocities_program.build({device}, OCL_COMPILER_OPTIONS);

    // program: reduce (calculate updated reduce)
    cl::Program::Sources cl_reduce_sources;
    cl_reduce_sources.push_back(core::kernel::resources::reduce_cl.to_string());

    cl::Program cl_reduce_program{cl_context, cl_reduce_sources};
    cl_reduce_program.build({device}, OCL_COMPILER_OPTIONS);

    // program: copy (calculate updated copy)
    cl::Program::Sources cl_copy_sources;
    cl_copy_sources.push_back(core::kernel::resources::copy_cl.to_string());

    cl::Program cl_copy_program{cl_context, cl_copy_sources};
    cl_copy_program.build({device}, OCL_COMPILER_OPTIONS);


    cl::CommandQueue cl_queue{cl_context, device, CL_QUEUE_PROFILING_ENABLE};

    // set boundary buffer
    auto buf_boundary = cl::Buffer{cl_context, CL_MEM_READ_ONLY, geom.data().size() * sizeof(cl_uchar)};
    cl::copy(cl_queue, geom.data().begin(), geom.data().end(), buf_boundary);

    // create component buffers
    auto buf_u_size = (geom.size().x + 1) * geom.size().y * sizeof(cl_float);
    auto buf_u = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_u_size};
    auto buf_f = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_u_size};

    auto buf_v_size = geom.size().x * (geom.size().y + 1) * sizeof(cl_float);
    auto buf_v = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_v_size};
    auto buf_g = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_v_size};

    auto buf_p_size = geom.size().x * geom.size().y * sizeof(cl_float);
    auto buf_p = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_p_size};

    auto buf_rhs_size = (geom.size().x - 2) * (geom.size().y - 2) * sizeof(cl_float);
    auto buf_rhs = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_rhs_size};

    // buffers for local residual
    auto buf_res_size = (geom.size().x - 2) * (geom.size().y - 2) * sizeof(cl_float);
    auto buf_res = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_res_size};

    // buffer vor visualization
    auto buf_vis_size = geom.size().x * geom.size().y * sizeof(cl_float);
    auto buf_vis = cl::Buffer{cl_context, CL_MEM_READ_WRITE, buf_vis_size};

    // initialize reduction stuff
    uint_t const reduce_res_size = (geom.size().x - 2) * (geom.size().y - 2);
    uint_t const reduce_vis_size = geom.size().x * geom.size().y;
    uint_t const reduce_u_size = (geom.size().x + 1) * geom.size().y;
    uint_t const reduce_v_size = geom.size().x * (geom.size().y + 1);
    uint_t const reduce_local_size = 128;

    uint_t const reduce_global_size_res = utils::pad_up(reduce_res_size, reduce_local_size);
    uint_t const reduce_global_size_vis = utils::pad_up(reduce_vis_size, reduce_local_size);
    uint_t const reduce_global_size_u = utils::pad_up(reduce_u_size, reduce_local_size);
    uint_t const reduce_global_size_v = utils::pad_up(reduce_v_size, reduce_local_size);

    uint_t const reduce_output_size_res = reduce_global_size_res / reduce_local_size;
    uint_t const reduce_output_size_vis = 2 * reduce_global_size_vis / reduce_local_size;
    uint_t const reduce_output_size_u = reduce_global_size_u / reduce_local_size;
    uint_t const reduce_output_size_v = reduce_global_size_v / reduce_local_size;

    auto buf_reduce_out_res = cl::Buffer{cl_context, CL_MEM_WRITE_ONLY, reduce_output_size_res * sizeof(cl_float)};
    auto buf_reduce_out_vis = cl::Buffer{cl_context, CL_MEM_WRITE_ONLY, reduce_output_size_vis * sizeof(cl_float)};
    auto buf_reduce_out_u = cl::Buffer{cl_context, CL_MEM_WRITE_ONLY, reduce_output_size_u * sizeof(cl_float)};
    auto buf_reduce_out_v = cl::Buffer{cl_context, CL_MEM_WRITE_ONLY, reduce_output_size_v * sizeof(cl_float)};

    auto vec_reduce_out_res = std::vector<cl_float>(reduce_output_size_res);
    auto vec_reduce_out_vis = std::vector<cl_float>(reduce_output_size_vis);
    auto vec_reduce_out_u = std::vector<cl_float>(reduce_output_size_u);
    auto vec_reduce_out_v = std::vector<cl_float>(reduce_output_size_v);

    {   // initialize u
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_u);

        auto range = cl::NDRange((geom.size().x + 1) * geom.size().y);
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize v
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_v);

        auto range = cl::NDRange(geom.size().x * (geom.size().y + 1));
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize f
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_f);

        auto range = cl::NDRange((geom.size().x + 1) * geom.size().y);
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize g
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_g);

        auto range = cl::NDRange(geom.size().x * (geom.size().y + 1));
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize p
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_p);

        auto range = cl::NDRange(geom.size().x * geom.size().y);
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // initialize rhs
        cl::Kernel kernel{cl_zero_program, "zero_float"};
        kernel.setArg(0, buf_rhs);

        auto range = cl::NDRange((geom.size().x - 2) * (geom.size().y - 2));
        cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
    }

    {   // set u boundary
        cl::Kernel kernel_boundary_u{cl_boundaries_program, "set_boundary_u"};
        kernel_boundary_u.setArg(0, buf_u);
        kernel_boundary_u.setArg(1, buf_boundary);
        kernel_boundary_u.setArg(2, static_cast<cl_float>(geom.boundary_velocity().x));

        auto range = cl::NDRange(geom.size().x, geom.size().y);
        cl_queue.enqueueNDRangeKernel(kernel_boundary_u, cl::NullRange, range, cl::NullRange);
    }

    {   // set v boundary
        cl::Kernel kernel_boundary_v{cl_boundaries_program, "set_boundary_v"};
        kernel_boundary_v.setArg(0, buf_v);
        kernel_boundary_v.setArg(1, buf_boundary);
        kernel_boundary_v.setArg(2, static_cast<cl_float>(geom.boundary_velocity().y));

        auto range = cl::NDRange(geom.size().x, geom.size().y);
        cl_queue.enqueueNDRangeKernel(kernel_boundary_v, cl::NullRange, range, cl::NullRange);
    }

    {   // set pressure boundary
        cl::Kernel kernel_boundary_p{cl_boundaries_program, "set_boundary_p"};
        kernel_boundary_p.setArg(0, buf_p);
        kernel_boundary_p.setArg(1, buf_boundary);
        kernel_boundary_p.setArg(2, static_cast<cl_float>(geom.boundary_pressure()));

        auto range = cl::NDRange(geom.size().x, geom.size().y);
        cl_queue.enqueueNDRangeKernel(kernel_boundary_p, cl::NullRange, range, cl::NullRange);
    }


    glClearColor(0.0, 0.0, 0.0, 1.0);

    // create OpenCL reference to OpenGL texture
    auto const& texture = visualizer.get_cl_target_texture();
    auto cl_image = cl::ImageGL{cl_context, CL_MEM_WRITE_ONLY, texture.target(), 0, texture.handle()};
    auto cl_req = std::vector<cl::Memory>{cl_image};

    real_t t = 0.0;
    real_t dt = params.dt;

    VisualTarget visual = VisualTarget::UVAbsCentered;

    auto perf_tts_noinit = utils::perf::Record::start("tts::noinit");

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
                } else if (e.key.keysym.sym == SDLK_ESCAPE) {
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
                } else if (e.key.keysym.sym == SDLK_9) {
                    visual = VisualTarget::Vorticity;
                } else if (e.key.keysym.sym == SDLK_0) {
                    visual = VisualTarget::Stream;
                }
            }
        }

        for (int i = 0; i < 100; i++) {
        // if (cont) { cont = false;
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

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel_momentum_f, cl::NullRange, range, cl::NullRange);
        }

        {   // calculate preliminary velocities: g
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

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel_momentum_g, cl::NullRange, range, cl::NullRange);
        }

        {   // set f boundary
            cl::Kernel kernel_boundary_u{cl_boundaries_program, "set_boundary_u"};
            kernel_boundary_u.setArg(0, buf_f);
            kernel_boundary_u.setArg(1, buf_boundary);
            kernel_boundary_u.setArg(2, static_cast<cl_float>(geom.boundary_velocity().x));

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel_boundary_u, cl::NullRange, range, cl::NullRange);
        }

        {   // set g boundary
            cl::Kernel kernel_boundary_v{cl_boundaries_program, "set_boundary_v"};
            kernel_boundary_v.setArg(0, buf_g);
            kernel_boundary_v.setArg(1, buf_boundary);
            kernel_boundary_v.setArg(2, static_cast<cl_float>(geom.boundary_velocity().y));

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel_boundary_v, cl::NullRange, range, cl::NullRange);
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

            auto range = cl::NDRange(geom.size().x - 2, geom.size().y - 2);
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

            cl::Kernel kernel_boundary_p{cl_boundaries_program, "set_boundary_p"};
            kernel_boundary_p.setArg(0, buf_p);
            kernel_boundary_p.setArg(1, buf_boundary);
            kernel_boundary_p.setArg(2, static_cast<cl_float>(geom.boundary_pressure()));

            cl::Kernel kernel_residual{cl_solver_program, "residual"};
            kernel_residual.setArg(0, buf_p);
            kernel_residual.setArg(1, buf_rhs);
            kernel_residual.setArg(2, buf_boundary);
            kernel_residual.setArg(3, buf_res);
            kernel_residual.setArg(4, h);

            cl::Kernel kernel_reduce{cl_reduce_program, "reduce_sum"};
            kernel_reduce.setArg(0, buf_res);
            kernel_reduce.setArg(1, buf_reduce_out_res);
            kernel_reduce.setArg(2, cl::Local(reduce_local_size * sizeof(cl_float)));
            kernel_reduce.setArg(3, reduce_res_size);

            int_t y_cells_black = (geom.size().y - 2) / 2;
            auto range_red = cl::NDRange(geom.size().x - 2, geom.size().y - 2 - y_cells_black);
            auto range_black = cl::NDRange(geom.size().x - 2, y_cells_black);
            auto range_bounds = cl::NDRange(geom.size().x, geom.size().y);
            auto range_residual = cl::NDRange(geom.size().x - 2, geom.size().y - 2);

            cl_float residual = std::numeric_limits<cl_float>::infinity();
            int_t iter = 0;
            for (; iter < params.itermax && residual > params.eps; iter++) {
                cl::Event perf_evt_start;
                cl::Event perf_evt_end;

                // solver cycles
                cl_queue.enqueueNDRangeKernel(kernel_red, cl::NullRange, range_red, cl::NullRange, nullptr, &perf_evt_start);
                cl_queue.enqueueNDRangeKernel(kernel_black, cl::NullRange, range_black, cl::NullRange);

                // update boundaries
                cl_queue.enqueueNDRangeKernel(kernel_boundary_p, cl::NullRange, range_bounds, cl::NullRange);

                {   // calculate residual   // TODO: only do once in k solver-iterations
                    cl_queue.enqueueNDRangeKernel(kernel_residual, cl::NullRange, range_residual, cl::NullRange);

                    // reduce residual
                    cl_queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(reduce_global_size_res), cl::NDRange(reduce_local_size));
                    cl_queue.enqueueReadBuffer(buf_reduce_out_res, true, 0, sizeof(cl_float) * vec_reduce_out_res.size(), vec_reduce_out_res.data(), nullptr, &perf_evt_end);

                    auto reduce_cpu_start = std::chrono::high_resolution_clock::now();
                    residual = std::accumulate(vec_reduce_out_res.begin(), vec_reduce_out_res.end(), static_cast<cl_float>(0.0));
                    residual = residual / n_fluid_cells;
                    auto reduce_cpu_end = std::chrono::high_resolution_clock::now();

                    auto solve_gpu_start = perf_evt_start.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                    auto solve_gpu_end = perf_evt_end.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                    auto dt_solve_gpu_nano = std::chrono::nanoseconds{solve_gpu_end - solve_gpu_start};

                    auto dt_solve_gpu = std::chrono::duration_cast<utils::perf::Registry::duration_type>(dt_solve_gpu_nano);
                    auto dt_reduce_cpu = std::chrono::duration_cast<utils::perf::Registry::duration_type>(reduce_cpu_start - reduce_cpu_end);

                    auto dt_all = dt_solve_gpu + dt_reduce_cpu;
                    utils::perf::add_cl_event_record("solver::iteration::full", dt_all);
                }
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

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
        }

        {   // set u boundary
            cl::Kernel kernel_boundary_u{cl_boundaries_program, "set_boundary_u"};
            kernel_boundary_u.setArg(0, buf_u);
            kernel_boundary_u.setArg(1, buf_boundary);
            kernel_boundary_u.setArg(2, static_cast<cl_float>(geom.boundary_velocity().x));

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel_boundary_u, cl::NullRange, range, cl::NullRange);
        }

        {   // set v boundary
            cl::Kernel kernel_boundary_v{cl_boundaries_program, "set_boundary_v"};
            kernel_boundary_v.setArg(0, buf_v);
            kernel_boundary_v.setArg(1, buf_boundary);
            kernel_boundary_v.setArg(2, static_cast<cl_float>(geom.boundary_velocity().y));

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel_boundary_v, cl::NullRange, range, cl::NullRange);
        }

        t += dt;
        }
        std::cout << "time: " << t << "\n";
        std::cout << "dt:   " << dt << "\n";

        {   // visualize: write visualization data to intermediate buffer
            cl::Kernel kernel;

            if (visual == VisualTarget::BoundaryTypes) {
                kernel = {cl_visualize_program, "visualize_boundaries"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_boundary);

            } else if (visual == VisualTarget::P) {
                kernel = {cl_visualize_program, "visualize_p"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_p);

            } else if (visual == VisualTarget::U) {
                kernel = {cl_visualize_program, "visualize_u"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_u);

            } else if (visual == VisualTarget::V) {
                kernel = {cl_visualize_program, "visualize_v"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_v);

            } else if (visual == VisualTarget::UVAbsCentered) {
                kernel = {cl_visualize_program, "visualize_uv_abs_center"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_u);
                kernel.setArg(2, buf_v);

            } else if (visual == VisualTarget::F) {
                kernel = {cl_visualize_program, "visualize_u"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_f);

            } else if (visual == VisualTarget::G) {
                kernel = {cl_visualize_program, "visualize_v"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_g);

            } else if (visual == VisualTarget::Rhs) {
                kernel = {cl_visualize_program, "visualize_rhs"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_rhs);

            } else if (visual == VisualTarget::Vorticity) {
                cl_float2 h = {{ static_cast<cl_float>(geom.mesh().x), static_cast<cl_float>(geom.mesh().y) }};

                kernel = {cl_visualize_program, "visualize_vorticity"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_u);
                kernel.setArg(2, buf_v);
                kernel.setArg(3, h);

            } else if (visual == VisualTarget::Stream) {
                cl_float2 h = {{ static_cast<cl_float>(geom.mesh().x), static_cast<cl_float>(geom.mesh().y) }};

                kernel = {cl_visualize_program, "visualize_stream"};
                kernel.setArg(0, buf_vis);
                kernel.setArg(1, buf_u);
                kernel.setArg(2, buf_v);
                kernel.setArg(3, h);
            }

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);

            // get min/max values
            cl::Kernel kernel_reduce{cl_reduce_program, "reduce_minmax"};
            kernel_reduce.setArg(0, buf_vis);
            kernel_reduce.setArg(1, buf_reduce_out_vis);
            kernel_reduce.setArg(2, cl::Local(2 * reduce_local_size * sizeof(cl_float)));
            kernel_reduce.setArg(3, reduce_vis_size);

            cl_queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(reduce_global_size_vis), cl::NDRange(reduce_local_size));
            cl::copy(cl_queue, buf_reduce_out_vis, vec_reduce_out_vis.begin(), vec_reduce_out_vis.end());

            std::size_t center = vec_reduce_out_vis.size() / 2;
            cl_float min = *std::min_element(vec_reduce_out_vis.begin(), vec_reduce_out_vis.begin() + center);
            cl_float max = *std::max_element(vec_reduce_out_vis.begin() + center, vec_reduce_out_vis.end());

            visualizer.set_data_range(min, max);
        }

        // copy visualization data to OpenGL texture via OpenCL
        glFinish();
        cl_queue.enqueueAcquireGLObjects(&cl_req);

        {
            cl::Kernel kernel{cl_copy_program, "copy_buf_to_img"};
            kernel.setArg(0, cl_image);
            kernel.setArg(1, buf_vis);

            auto range = cl::NDRange(geom.size().x, geom.size().y);
            cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
        }

        cl_queue.enqueueReleaseGLObjects(&cl_req);
        cl_queue.finish();

        // render via OpenGL
        glClear(GL_COLOR_BUFFER_BIT);
        visualizer.draw();

        window.swap_buffers();
        opengl::check_error();

        if (params.t_end > 0 && t >= params.t_end) {
            break;
        }
    }

    perf_tts_noinit.stop();
    perf_tts_full.stop();
    write_perf_stats(env.json);


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


auto parse_cmdline(int argc, char** argv) -> Environment {
    auto print_usage_and_exit = [&](int status, std::string msg = "") {
        if (!msg.empty()) std::cout << msg << "\n\n";
        std::cout <<
            "Usage:\n"
            "  " << argv[0] << " [options]\n"
            "\n"
            "Options:\n"
            "  -h --help                 Show this help message\n"
            "  -g --geometry <file>      Load geometry file (*.geom)\n"
            "  -p --parameters <file>    Load simulation parameters (*.param)\n"
            "  -j --json <file>          JSON output file (*.json)\n"
            "                            If not set, no file is created.\n";
        std::cout << std::endl;
        std::exit(status);
    };

    Environment env{nullptr, nullptr, nullptr};
    for (int i = 1; i < argc; i++) {
        char* arg = argv[i];

        if (  std::strcmp("-h", arg) == 0
           || std::strcmp("--help", arg) == 0
        ) {
            print_usage_and_exit(0, WINDOW_TITLE);
        }

        else if (  std::strcmp("-p", arg) == 0
                || std::strcmp("--params", arg) == 0
                || std::strcmp("--parameters", arg) == 0
        ) {
            if (i++ < argc) {
                env.params = argv[i];
            } else {
                print_usage_and_exit(1, "Error: Missing argument for '--parameters'.");
            }
        }

        else if (  std::strcmp("-g", arg) == 0
                || std::strcmp("--geom", arg) == 0
                || std::strcmp("--geometry", arg) == 0
        ) {
            if (++i < argc) {
                env.geom = argv[i];
            } else {
                print_usage_and_exit(1, "Error: Missing argument for '--geometry'.");
            }
        }

        else if (  std::strcmp("-j", arg) == 0
                || std::strcmp("--json", arg) == 0
        ) {
            if (++i < argc) {
                env.json = argv[i];
            } else {
                print_usage_and_exit(1, "Error: Missing argument for '--json'.");
            }
        }

        else {
            std::stringstream msg;
            msg << "Error: Unknown argument '" << arg << "'.";
            print_usage_and_exit(1, msg.str());
        }
    }

    return env;
}

void write_perf_stats(char const* json_file) {
    if (json_file) {
        nlohmann::json data = utils::perf::Registry::get();

        std::ofstream o(json_file);
        o << std::setw(4) << data << std::endl;
    }
}
