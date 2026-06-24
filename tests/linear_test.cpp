/**
 * @file linear_test.cpp
 * @brief Unit tests for the axon::nn::Linear fully connected module.
 * @author Mika Brückner
 * @date 2026-06-17
 */

#include "../include/axon/linear.hpp"

#include <cmath>

#include "../include/axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("Linear parameters have correct shapes", "[Linear]") {
  axon::nn::Linear layer(3, 2);

  std::vector<axon::Tensor> params = layer.parameters();

  REQUIRE(params.size() == 2);

  const axon::Tensor& weights = params[0];
  const axon::Tensor& bias = params[1];

  REQUIRE(weights.shape() == std::vector<axon::idx_t>{3, 2});
  REQUIRE(bias.shape() == std::vector<axon::idx_t>{2});
}

TEST_CASE("Linear parameters track gradients", "[Linear]") {
  axon::nn::Linear layer(4, 5);

  std::vector<axon::Tensor> params = layer.parameters();

  REQUIRE(params[0].requires_grad());
  REQUIRE(params[1].requires_grad());
}

TEST_CASE("Linear bias is zero-initialized", "[Linear]") {
  axon::nn::Linear layer(3, 4);

  axon::Tensor bias = layer.parameters()[1];

  for (axon::idx_t i = 0; i < bias.shape()[0]; ++i) {
    REQUIRE(bias.at({i}) == 0.0f);
  }
}

TEST_CASE("Linear weights are within the init bound", "[Linear]") {
  constexpr axon::idx_t in_features = 4;
  constexpr axon::idx_t out_features = 6;
  axon::nn::Linear layer(in_features, out_features);

  const float bound = 1.0f / std::sqrt(static_cast<float>(in_features));

  axon::Tensor weights = layer.parameters()[0];

  REQUIRE(weights.abs().max() <= bound);
}

TEST_CASE("Linear weights are randomly initialized", "[Linear]") {
  axon::nn::Linear layer(8, 8);

  axon::Tensor weights = layer.parameters()[0];

  REQUIRE(weights.abs().max() > 0.0f);
}

TEST_CASE("Linear forward produces the expected output shape", "[Linear]") {
  constexpr axon::idx_t batch = 4;
  constexpr axon::idx_t in_features = 3;
  constexpr axon::idx_t out_features = 2;
  axon::nn::Linear layer(in_features, out_features);

  axon::Tensor input = axon::Tensor::from_data({1.0f,
                                                2.0f,
                                                3.0f,
                                                4.0f,
                                                5.0f,
                                                6.0f,
                                                7.0f,
                                                8.0f,
                                                9.0f,
                                                10.0f,
                                                11.0f,
                                                12.0f},
                                               {batch, in_features});

  axon::Tensor output = layer.forward(input);

  REQUIRE(output.shape() == std::vector<axon::idx_t>{batch, out_features});
}

TEST_CASE("Linear forward applies the affine transformation", "[Linear]") {
  constexpr axon::idx_t batch = 2;
  constexpr axon::idx_t in_features = 3;
  constexpr axon::idx_t out_features = 2;
  axon::nn::Linear layer(in_features, out_features);

  axon::Tensor weights = layer.parameters()[0];
  axon::Tensor bias = layer.parameters()[1];

  axon::Tensor input = axon::Tensor::from_data(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {batch, in_features});

  axon::Tensor output = layer.forward(input);

  for (axon::idx_t i = 0; i < batch; ++i) {
    for (axon::idx_t j = 0; j < out_features; ++j) {
      float expected = bias.at({j});
      for (axon::idx_t k = 0; k < in_features; ++k) {
        expected += input.at({i, k}) * weights.at({k, j});
      }
      REQUIRE(output.at({i, j}) == Catch::Approx(expected));
    }
  }
}
