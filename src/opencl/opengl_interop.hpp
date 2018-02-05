#pragma once

#include "opencl/opencl.hpp"
#include "opengl/opengl.hpp"

#include "sdl/sdl.hpp"
#include "sdl/errors.hpp"
#include "sdl/opengl/window.hpp"

#include <SDL2/SDL_syswm.h>

#include <array>
#include <stdexcept>


#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#include <OpenGL/OpenGL.h>
#endif


namespace opencl {
namespace opengl {

struct ContextPropertyKv {
    cl_context_properties type;
    cl_context_properties value;
};


#if defined(__APPLE__)

char const* const EXT_CL_GL_SHARING = "cl_APPLE_gl_sharing";


inline auto get_context_share_properties(sdl::opengl::Window const&)
    -> std::array<ContextPropertyKv, 1>
{
    CGLContextObj ctx = CGLGetCurrentContext();
    CGLShareGroupObj grp = CGLGetShareGroup(ctx);

    return {{
        ContextPropertyKv{CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties) grp}
    }};
}


#elif defined(_WIN32) || defined(__linux__)

char const* const EXT_CL_GL_SHARING = "cl_khr_gl_sharing";


inline auto get_context_share_properties(sdl::opengl::Window const& window)
    -> std::array<ContextPropertyKv, 2>
{
    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    if (!SDL_GetWindowWMInfo(window.handle(), &info)) {
        sdl::except(-1);
    }

#if defined(SDL_VIDEO_DRIVER_WINDOWS)
    if (info.subsystem == SDL_SYSWM_WINDOWS) {
        return {{
            ContextPropertyKv{CL_GL_CONTEXT_KHR, (cl_context_properties) SDL_GL_GetCurrentContext()},
            ContextPropertyKv{CL_EGL_DISPLAY_KHR, (cl_context_properties) info.info.win.hdc}
        }};
    }
#endif

#if defined(SDL_VIDEO_DRIVER_X11)
    if (info.subsystem == SDL_SYSWM_X11) {
        return {{
            ContextPropertyKv{CL_GL_CONTEXT_KHR, (cl_context_properties) SDL_GL_GetCurrentContext()},
            ContextPropertyKv{CL_GLX_DISPLAY_KHR, (cl_context_properties) info.info.x11.display}
        }};
    }
#endif

#if defined(SDL_VIDEO_DRIVER_WAYLAND)
    if (info.subsystem == SDL_SYSWM_WAYLAND) {
        return {{
            ContextPropertyKv{CL_GL_CONTEXT_KHR, (cl_context_properties) SDL_GL_GetCurrentContext()},
            ContextPropertyKv{CL_EGL_DISPLAY_KHR, (cl_context_properties) info.info.wl.display}
        }};
    }
#endif

    throw std::runtime_error{"Platform not supported"};
}

#endif

}   /* namespace opengl */
}   /* namespace opencl */
