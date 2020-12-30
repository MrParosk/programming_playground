project(neural_network)
cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Wextra -pedantic -Werror)

file(GLOB SOURCES
    src/*.hpp
    src/*.cpp
    ./main.cpp
)

add_executable(neural_network ${SOURCES})
