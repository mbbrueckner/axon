cmake_minimum_required(VERSION 3.20)
project(axon VERSION 0.1.0 LANGUAGES CXX)

# Library
add_library(axon STATIC
        src/tensor.cpp
        src/utils.cpp
)

target_include_directories(axon PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(axon PUBLIC cxx_std_20)

target_compile_options(axon PRIVATE
        -Wall -Wextra -Wpedantic
)
# Catch2
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# Tests
add_subdirectory(tests)