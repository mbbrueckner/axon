/**
 * @file autograd_test.cpp
 * @brief Unit tests for the axon autograd computation graph and backward pass.
 * @author Mika Brückner
 * @date 2026-06-11
 */

#include "../include/axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("Autograd mul backward", "[AutoGradMul]") {
  axon::Tensor x(std::vector<float>{2.0f}, std::vector<axon::idx_t>{1});
  axon::Tensor y(std::vector<float>{3.0f}, std::vector<axon::idx_t>{1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x * y;
  z.backward();

  REQUIRE(x.grad().at({0}) == 3.0f);
  REQUIRE(y.grad().at({0}) == 2.0f);
}

TEST_CASE("Autograd add backward", "[AutoGradAdd]") {
  axon::Tensor x(std::vector<float>{2.0f}, std::vector<axon::idx_t>{1});
  axon::Tensor y(std::vector<float>{3.0f}, std::vector<axon::idx_t>{1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x + y;
  z.backward();

  REQUIRE(x.grad().at({0}) == 1.0f);
  REQUIRE(y.grad().at({0}) == 1.0f);
}

TEST_CASE("Autograd subtract backward", "[AutoGradSub]") {
  axon::Tensor x(std::vector<float>{2.0f}, std::vector<axon::idx_t>{1});
  axon::Tensor y(std::vector<float>{3.0f}, std::vector<axon::idx_t>{1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x - y;
  z.backward();

  REQUIRE(x.grad().at({0}) == -1.0f);
  REQUIRE(y.grad().at({0}) == -1.0f);
}

TEST_CASE("Autograd divide backward", "[AutoGradDiv]") {
  axon::Tensor x(std::vector<float>{2.0f}, std::vector<axon::idx_t>{1});
  axon::Tensor y(std::vector<float>{3.0f}, std::vector<axon::idx_t>{1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x / y;
  z.backward();

  REQUIRE(x.grad().at({0}) == Catch::Approx(1.0f / 3.0f));
  REQUIRE(y.grad().at({0}) == Catch::Approx(-2.0f / 9.0f));
}