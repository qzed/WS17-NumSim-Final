#pragma once

#include "utils/resource.hpp"


namespace core {
namespace kernel {
namespace resources {

extern const utils::Resource boundaries_cl;
extern const utils::Resource momentum_cl;
extern const utils::Resource rhs_cl;
extern const utils::Resource velocities_cl;

extern const utils::Resource visualize_cl;

extern const utils::Resource debug_cl;

}   /* namespace resources */
}   /* namespace kernel */
}   /* namespace core */
