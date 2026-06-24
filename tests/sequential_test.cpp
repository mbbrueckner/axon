/**
 * @file sequential_test.cpp
 * @brief Unit tests for the axon::nn::Sequential module container.
 * @author Mika Brückner
 * @date 2026-06-19
 */

#include "axon/modules/sequential.hpp"

#include "axon/modules/linear.hpp"
#include "axon/modules/relu.hpp"
#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("Sequential forward produces the expected output shape",
          "[Sequential]") {
  std::vector<std::unique_ptr<axon::nn::Module>> modules;
  modules.push_back(std::make_unique<axon::nn::Linear>(2, 4));
  modules.push_back(std::make_unique<axon::nn::ReLU>());
  modules.push_back(std::make_unique<axon::nn::Linear>(4, 1));
  axon::nn::Sequential sequential(std::move(modules));

  const std::vector<axon::idx_t> input_shape{2, 2};
  axon::Tensor input =
      axon::Tensor::from_data({-1.0f, 2.0f, -3.0f, 4.0f}, input_shape);

  const axon::Tensor output = sequential.forward(input);
  const std::vector<axon::idx_t> expected_output_shape{2, 1};
  REQUIRE(output.shape() == expected_output_shape);
}

TEST_CASE("Sequential forward produces the expected output", "[Sequential]") {
  axon::nn::Linear linear_first{2, 4};
  axon::nn::Linear linear_second{4, 1};
  axon::nn::ReLU relu{};
  std::vector<std::unique_ptr<axon::nn::Module>> modules;
  modules.push_back(std::make_unique<axon::nn::Linear>(linear_first));
  modules.push_back(std::make_unique<axon::nn::ReLU>(relu));
  modules.push_back(std::make_unique<axon::nn::Linear>(linear_second));
  axon::nn::Sequential sequential(std::move(modules));

  const std::vector<axon::idx_t> input_shape{2, 2};
  axon::Tensor input =
      axon::Tensor::from_data({-1.0f, 2.0f, -3.0f, 4.0f}, input_shape);

  const axon::Tensor output = sequential.forward(input);

  const axon::Tensor linear_first_output = linear_first.forward(input);
  const axon::Tensor relu_output = relu.forward(linear_first_output);
  const axon::Tensor linear_second_output = linear_second.forward(relu_output);

  REQUIRE(output == linear_second_output);
}

TEST_CASE("Sequential parameters produces the expected output size",
          "[Sequential]") {
  std::vector<std::unique_ptr<axon::nn::Module>> modules;
  modules.push_back(std::make_unique<axon::nn::Linear>(2, 4));
  modules.push_back(std::make_unique<axon::nn::Linear>(4, 1));
  axon::nn::Sequential sequential(std::move(modules));

  const std::vector<axon::idx_t> input_shape{2, 2};
  axon::Tensor input =
      axon::Tensor::from_data({-1.0f, 2.0f, -3.0f, 4.0f}, input_shape);

  const std::vector<axon::Tensor> output = sequential.parameters();
  const size_t expected_output_size{2 * 2};
  REQUIRE(output.size() == expected_output_size);
}

TEST_CASE(
    "Sequential without modules produces input on forward, empty vector on "
    "paramters",
    "[Sequential]") {
  std::vector<std::unique_ptr<axon::nn::Module>> modules;
  axon::nn::Sequential sequential(std::move(modules));

  const std::vector<axon::idx_t> input_shape{2, 2};
  axon::Tensor input =
      axon::Tensor::from_data({-1.0f, 2.0f, -3.0f, 4.0f}, input_shape);

  const axon::Tensor output_forward = sequential.forward(input);
  const std::vector<axon::Tensor> output_parameters = sequential.parameters();
  REQUIRE(output_forward == input);
  REQUIRE(output_parameters.empty());
}

TEST_CASE("Sequential backward propagates gradients through all modules",
          "[Sequential]") {
  auto linear_first_owned = std::make_unique<axon::nn::Linear>(2, 4, 42);
  auto linear_second_owned = std::make_unique<axon::nn::Linear>(4, 1, 42);
  axon::nn::Linear* linear_first = linear_first_owned.get();
  axon::nn::Linear* linear_second = linear_second_owned.get();

  std::vector<std::unique_ptr<axon::nn::Module>> modules;
  modules.push_back(std::move(linear_first_owned));
  modules.push_back(std::make_unique<axon::nn::ReLU>());
  modules.push_back(std::move(linear_second_owned));
  axon::nn::Sequential sequential(std::move(modules));

  const std::vector<axon::idx_t> input_shape{2, 2};
  axon::Tensor input =
      axon::Tensor::from_data({-1.0f, 2.0f, -3.0f, 4.0f}, input_shape);

  axon::Tensor output = sequential.forward(input);
  output.backward();

  axon::Tensor linear_first_w_grad = linear_first->parameters()[0].grad();
  const axon::Tensor comp_first =
      axon::Tensor::zeros(linear_first_w_grad.shape());
  REQUIRE(linear_first_w_grad != comp_first);

  axon::Tensor linear_second_w_grad = linear_second->parameters()[0].grad();
  const axon::Tensor comp_second =
      axon::Tensor::zeros(linear_second_w_grad.shape());
  REQUIRE(linear_second_w_grad != comp_second);
}