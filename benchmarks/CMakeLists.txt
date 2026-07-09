add_executable(mnist_benchmark
        mnist/mnist_benchmark.cpp
)
target_link_libraries(mnist_benchmark PRIVATE
        axon
)

execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_compile_definitions(mnist_benchmark PRIVATE
        GIT_SHA="${GIT_SHA}"
        BUILD_TYPE="$<CONFIG>"
)