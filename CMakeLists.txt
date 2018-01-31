cmake_minimum_required(VERSION 3.7.2)
project(numsim)


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
        "vis::shader::resources::fullscreen_vs" "${CMAKE_CURRENT_SOURCE_DIR}/src/vis/shader/fullscreen.vs"
        "vis::shader::resources::debug_fs"      "${CMAKE_CURRENT_SOURCE_DIR}/src/vis/shader/debug.fs"
    DEPENDS
        "src/vis/shader/fullscreen.vs"
        "src/vis/shader/debug.fs"
)

add_custom_command(
    OUTPUT "resources_kernel.cpp"
    COMMAND embed_resource -o resources_kernel.cpp
        "core::kernel::resources::debug_cl" "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel/debug.cl"
    DEPENDS
        "src/core/kernel/debug.cl"
)


include_directories("src" "thirdparty")

set(src_main
    "src/main.cpp"
    "resources_shader.cpp"
    "resources_kernel.cpp"
)

add_executable(main ${src_main})
target_include_directories(main PRIVATE OpenCL::OpenCL GLEW::GLEW OpenGL::GL ${SDL2_INCLUDE_DIR})
target_link_libraries(main OpenCL::OpenCL GLEW::GLEW OpenGL::GL ${SDL2_LIBRARIES})
