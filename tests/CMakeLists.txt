add_executable(axon_tests
        test_tensor.cpp
)

target_link_libraries(axon_tests PRIVATE
        axon
        Catch2::Catch2WithMain
)