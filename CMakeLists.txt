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


include_directories("src" "thirdparty")

set(src_main
    "src/main.cpp"
)

add_executable(main ${src_main})
target_include_directories(main PRIVATE OpenCL::OpenCL GLEW::GLEW OpenGL::GL ${SDL2_INCLUDE_DIR})
target_link_libraries(main OpenCL::OpenCL GLEW::GLEW OpenGL::GL ${SDL2_LIBRARIES})
