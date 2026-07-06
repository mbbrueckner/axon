/**
 * @file linear_test.cpp
 * @brief Unit tests for the axon::nn::Linear fully connected module.
 * @author Mika Brückner
 * @date 2026-06-17
 */

#include "axon/modules/linear.hpp"

#include <cmath>

#include "axon/tensor.hpp"
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

TEST_CASE("Linear set_parameters replaces weights and bias",
          "[Linear]") {
  axon::nn::Linear layer(3, 2);

  axon::Tensor new_weights =
      axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  axon::Tensor new_bias = axon::Tensor::from_data({7.0f, 8.0f}, {2});

  layer.set_parameters({new_weights, new_bias});

  std::vector<axon::Tensor> params = layer.parameters();
  REQUIRE(params[0] == new_weights);
  REQUIRE(params[1] == new_bias);
}

TEST_CASE("Linear set_parameters throws on wrong parameter count",
          "[Linear]") {
  axon::nn::Linear layer(3, 2);

  axon::Tensor weights = axon::Tensor::zeros({3, 2});
  axon::Tensor bias = axon::Tensor::zeros({2});

  REQUIRE_THROWS_AS(layer.set_parameters({weights}), std::out_of_range);
  REQUIRE_THROWS_AS(layer.set_parameters({weights, bias, bias}),
                     std::out_of_range);
  REQUIRE_THROWS_AS(layer.set_parameters({}), std::out_of_range);
}

TEST_CASE("Linear set_parameters throws on wrong weight shape",
          "[Linear]") {
  axon::nn::Linear layer(3, 2);

  axon::Tensor wrong_weights = axon::Tensor::zeros({2, 2});
  axon::Tensor bias = axon::Tensor::zeros({2});

  REQUIRE_THROWS_AS(layer.set_parameters({wrong_weights, bias}),
                     std::out_of_range);
}

TEST_CASE("Linear set_parameters throws on wrong bias shape",
          "[Linear]") {
  axon::nn::Linear layer(3, 2);

  axon::Tensor weights = axon::Tensor::zeros({3, 2});
  axon::Tensor wrong_bias = axon::Tensor::zeros({3});

  REQUIRE_THROWS_AS(layer.set_parameters({weights, wrong_bias}),
                     std::out_of_range);
}

TEST_CASE("Linear set_parameters is reflected in forward",
          "[Linear]") {
  constexpr axon::idx_t batch = 1;
  constexpr axon::idx_t in_features = 2;
  constexpr axon::idx_t out_features = 1;
  axon::nn::Linear layer(in_features, out_features);

  axon::Tensor weights =
      axon::Tensor::from_data({1.0f, 2.0f}, {in_features, out_features});
  axon::Tensor bias = axon::Tensor::from_data({0.5f}, {out_features});
  layer.set_parameters({weights, bias});

  axon::Tensor input =
      axon::Tensor::from_data({3.0f, 4.0f}, {batch, in_features});

  axon::Tensor output = layer.forward(input);

  REQUIRE(output.at({0, 0}) == Catch::Approx(3.0f * 1.0f + 4.0f * 2.0f + 0.5f));
}
