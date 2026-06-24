cmake_minimum_required(VERSION 3.20)
project(axon VERSION 0.1.0 LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library
add_library(axon STATIC
        src/tensor.cpp
        src/autograd/autograd.cpp
        src/modules/linear.cpp
        src/autograd/autograd.hpp
        src/modules/relu.cpp
        src/modules/sequential.cpp
        src/functional.cpp
        include/axon/modules/module.hpp
        include/axon/modules/linear.hpp
        include/axon/modules/relu.hpp
        include/axon/modules/sequential.hpp
        include/axon/functional.hpp
        src/functional.cpp
)

target_include_directories(axon
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(axon PUBLIC cxx_std_20)

target_compile_options(axon PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined>
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

target_link_options(axon PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
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
include(CTest)
add_subdirectory(tests)
