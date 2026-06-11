add_executable(axon_tests
        tensor_test.cpp
        autograd_test.cpp
)

target_link_libraries(axon_tests PRIVATE
        axon
        Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(axon_tests)