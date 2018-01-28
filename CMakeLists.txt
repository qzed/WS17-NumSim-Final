cmake_minimum_required(VERSION 3.7.2)
project(numsim)


find_package(OpenCL REQUIRED)

find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)


set(main_sources
    "src/main.cpp"
)

add_executable(main ${main_sources})
target_include_directories(main PRIVATE OpenCL::OpenCL GLEW::GLEW OpenGL::GL ${SDL2_INCLUDE_DIR} src)
target_link_libraries(main OpenCL::OpenCL GLEW::GLEW OpenGL::GL ${SDL2_LIBRARIES})
