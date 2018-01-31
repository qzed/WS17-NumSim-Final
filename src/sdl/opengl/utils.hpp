#pragma once

#include "sdl/sdl.hpp"
#include "sdl/errors.hpp"


namespace sdl {
namespace opengl {

inline void set_swap_interval(int interval) {
    sdl::except(SDL_GL_SetSwapInterval(interval));
} 

}   /* namespace opengl */
}   /* namespace sdl */
