add_executable(mnist_benchmark
        mnist/mnist_benchmark.cpp
)
target_link_libraries(mnist_benchmark PRIVATE
        axon
)