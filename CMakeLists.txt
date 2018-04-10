cmake_minimum_required(VERSION 3.7.2)
project(numsim)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCL REQUIRED)

find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
endif()


add_executable(embed_resource "tools/embed_resource.cpp")

add_custom_command(
    OUTPUT "resources_shader.cpp"
    COMMAND embed_resource -o resources_shader.cpp
        "vis::shader::resources::fullscreen_vs"  "${CMAKE_CURRENT_SOURCE_DIR}/src/vis/shader/fullscreen.vs"
        "vis::shader::resources::cubehelix_glsl" "${CMAKE_CURRENT_SOURCE_DIR}/src/vis/shader/cubehelix.glsl"
        "vis::shader::resources::map_fs"         "${CMAKE_CURRENT_SOURCE_DIR}/src/vis/shader/map.fs"
    DEPENDS
        "src/vis/shader/fullscreen.vs"
        "src/vis/shader/cubehelix.glsl"
        "src/vis/shader/map.fs"
)

add_custom_command(
    OUTPUT "resources_kernel.cpp"
    COMMAND embed_resource -o resources_kernel.cpp
        "core::kernel::resources::boundaries_cl" "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/boundaries.cl"
        "core::kernel::resources::momentum_cl"   "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/momentum.cl"
        "core::kernel::resources::rhs_cl"        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/rhs.cl"
        "core::kernel::resources::velocities_cl" "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/velocities.cl"
        "core::kernel::resources::solver_cl"     "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/solver.cl"
        "core::kernel::resources::visualize_cl"  "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/visualize.cl"
        "core::kernel::resources::reduce_cl"     "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/reduce.cl"
        "core::kernel::resources::zero_cl"       "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/zero.cl"
        "core::kernel::resources::copy_cl"       "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/sources/copy.cl"
    DEPENDS
        "src/core/kernel/sources/boundaries.cl"
        "src/core/kernel/sources/momentum.cl"
        "src/core/kernel/sources/rhs.cl"
        "src/core/kernel/sources/velocities.cl"
        "src/core/kernel/sources/solver.cl"
        "src/core/kernel/sources/reduce.cl"
        "src/core/kernel/sources/zero.cl"
        "src/core/kernel/sources/visualize.cl"
        "src/core/kernel/sources/copy.cl"
)


include_directories("src" "thirdparty")

set(src_main
    "src/main.cpp"
    "src/core/geometry.cpp"
    "src/core/parameters.cpp"
    "resources_shader.cpp"
    "resources_kernel.cpp"
)

add_executable(main ${src_main})
target_include_directories(main PRIVATE OpenCL::OpenCL GLEW::GLEW OpenGL::GL ${SDL2_INCLUDE_DIR})
target_link_libraries(main OpenCL::OpenCL GLEW::GLEW OpenGL::GL ${SDL2_LIBRARIES})
