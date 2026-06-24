/**
 * @file relu_test.cpp
 * @brief Unit tests for the axon::nn::ReLU activation layer module.
 * @author Mika Brückner
 * @date 2026-06-19
 */

#include "../include/axon/modules/relu.hpp"

#include <cmath>

#include "../include/axon/tensor.hpp"
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
