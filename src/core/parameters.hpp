#pragma once

#include "types.hpp"


namespace core {

//! Simulation parameters.
struct Parameters {
    real_t re       = 1000.0;
    real_t omega    = 1.7;
    real_t alpha    = 0.9;
    real_t dt       = 0.2;
    real_t t_end    = 16.4;
    real_t eps      = 0.001;
    real_t tau      = 0.5;
    int_t  itermax  = 100;

    //! Load parameters from file.
    void load(char const* file);
};

}   /* namespace core */
