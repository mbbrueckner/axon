#include "axon/functional.hpp"

#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("Softmax produces expected output", "[Functional]") {
  const axon::Tensor t =
      axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  const axon::Tensor t_softmax = axon::softmax(t);
  SECTION("Input shape equals output shape") {
    REQUIRE(t.shape() == t_softmax.shape());
  }

  SECTION("Output elements as expected") {
    REQUIRE(t_softmax.at({0, 0}) == Catch::Approx(0.0900f).margin(0.001));
    REQUIRE(t_softmax.at({0, 1}) == Catch::Approx(0.2447f).margin(0.001));
    REQUIRE(t_softmax.at({0, 2}) == Catch::Approx(0.6652f).margin(0.001));
    REQUIRE(t_softmax.at({1, 0}) == Catch::Approx(0.0900f).margin(0.001));
    REQUIRE(t_softmax.at({1, 1}) == Catch::Approx(0.2447f).margin(0.001));
    REQUIRE(t_softmax.at({1, 2}) == Catch::Approx(0.6652f).margin(0.001));
  }
}