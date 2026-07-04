add_executable(xor
        xor_demo.cpp
)

target_link_libraries(xor PRIVATE
        axon
)

add_executable(mnist
        mnist_mlp.cpp
)
target_link_libraries(mnist PRIVATE
        axon
)