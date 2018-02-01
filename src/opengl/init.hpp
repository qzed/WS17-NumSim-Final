#pragma once

#include "opengl/opengl.hpp"
#include "opengl/errors.hpp"


namespace opengl {

inline void init(bool experimental = true) {
    glewExperimental = experimental;
    glew::except(glewInit());
} 

}   /* namespace opengl */ 