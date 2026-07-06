/**
 * @file relu_test.cpp
 * @brief Unit tests for the axon::nn::ReLU activation layer module.
 * @author Mika Brückner
 * @date 2026-06-19
 */

#include "axon/modules/relu.hpp"

#include <cmath>

#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("ReLU forward produces the expected output ", "[ReLU]") {
  const std::vector<axon::idx_t> input_shape{4};
  axon::nn::ReLU layer{};

  const axon::Tensor input =
      axon::Tensor::from_data({-1.0f, 2.0f, -3.0f, 4.0f}, input_shape);

  axon::Tensor output = layer.forward(input);

  REQUIRE(output.shape() == input_shape);
  REQUIRE(output.at({0}) == 0.0f);
  REQUIRE(output.at({1}) == 2.0f);
  REQUIRE(output.at({2}) == 0.0f);
  REQUIRE(output.at({3}) == 4.0f);
}

TEST_CASE("ReLU set_parameters accepts an empty vector", "[ReLU]") {
  axon::nn::ReLU layer{};

  REQUIRE_NOTHROW(layer.set_parameters({}));
  REQUIRE(layer.parameters().empty());
}

TEST_CASE("ReLU set_parameters throws when given parameters", "[ReLU]") {
  axon::nn::ReLU layer{};

  axon::Tensor tensor = axon::Tensor::zeros({2});

  REQUIRE_THROWS_AS(layer.set_parameters({tensor}), std::logic_error);
}
