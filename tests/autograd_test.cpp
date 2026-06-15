/**
 * @file autograd_test.cpp
 * @brief Unit tests for the axon autograd computation graph and backward pass.
 * @author Mika Brückner
 * @date 2026-06-11
 */

#include "../include/axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("Autograd mul backward", "[AutoGradMul]") {
  axon::Tensor x = axon::Tensor::from_data(std::vector<float>{2.0f},
                                           std::vector<axon::idx_t>{1});
  axon::Tensor y = axon::Tensor::from_data(std::vector<float>{3.0f},
                                           std::vector<axon::idx_t>{1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x * y;
  z.backward();

  REQUIRE(x.grad().at({0}) == 3.0f);
  REQUIRE(y.grad().at({0}) == 2.0f);
}

TEST_CASE("Autograd add backward", "[AutoGradAdd]") {
  axon::Tensor x = axon::Tensor::from_data({2.0f}, {1});
  axon::Tensor y = axon::Tensor::from_data({3.0f}, {1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x + y;
  z.backward();

  REQUIRE(x.grad().at({0}) == 1.0f);
  REQUIRE(y.grad().at({0}) == 1.0f);
}

TEST_CASE("Autograd subtract backward", "[AutoGradSub]") {
  axon::Tensor x = axon::Tensor::from_data({2.0f}, {1});
  axon::Tensor y = axon::Tensor::from_data({3.0f}, {1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x - y;
  z.backward();

  REQUIRE(x.grad().at({0}) == -1.0f);
  REQUIRE(y.grad().at({0}) == -1.0f);
}

TEST_CASE("Autograd divide backward", "[AutoGradDiv]") {
  axon::Tensor x = axon::Tensor::from_data({2.0f}, {1});
  axon::Tensor y = axon::Tensor::from_data({3.0f}, {1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x / y;
  z.backward();

  REQUIRE(x.grad().at({0}) == Catch::Approx(1.0f / 3.0f));
  REQUIRE(y.grad().at({0}) == Catch::Approx(-2.0f / 9.0f));
}

TEST_CASE("Autograd sum backward", "[AutoGradSum]") {
  axon::Tensor t = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  t.requires_grad_(true);

  axon::Tensor z = t.sum();
  z.backward();

  REQUIRE(t.grad().at({0, 0}) == 1.0f);
  REQUIRE(t.grad().at({0, 1}) == 1.0f);
  REQUIRE(t.grad().at({1, 0}) == 1.0f);
  REQUIRE(t.grad().at({1, 1}) == 1.0f);
}

TEST_CASE("Autograd exp backward", "[AutoGradExp]") {
  axon::Tensor t = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  t.requires_grad_(true);

  axon::Tensor z = t.exp();
  z.backward();

  REQUIRE(t.grad().at({0, 0}) == Catch::Approx(std::exp(1.0f)));
  REQUIRE(t.grad().at({0, 1}) == Catch::Approx(std::exp(2.0f)));
  REQUIRE(t.grad().at({1, 0}) == Catch::Approx(std::exp(3.0f)));
  REQUIRE(t.grad().at({1, 1}) == Catch::Approx(std::exp(4.0f)));
}

TEST_CASE("Autograd log backward", "[AutoGradLog]") {
  axon::Tensor t = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  t.requires_grad_(true);

  axon::Tensor z = t.log();
  z.backward();

  REQUIRE(t.grad().at({0, 0}) == Catch::Approx(1.0f / 1.0f));
  REQUIRE(t.grad().at({0, 1}) == Catch::Approx(1.0f / 2.0f));
  REQUIRE(t.grad().at({1, 0}) == Catch::Approx(1.0f / 3.0f));
  REQUIRE(t.grad().at({1, 1}) == Catch::Approx(1.0f / 4.0f));
}

TEST_CASE("Autograd matmul backward", "[AutoGradMatMul]") {
  axon::Tensor x = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  axon::Tensor y = axon::Tensor::from_data({4.0f, 3.0f, 2.0f, 1.0f}, {2, 2});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x.matmul(y);
  axon::Tensor loss = z.sum();
  loss.backward();

  REQUIRE(x.grad().at({0, 0}) == Catch::Approx(7.0f));
  REQUIRE(x.grad().at({0, 1}) == Catch::Approx(3.0f));
  REQUIRE(x.grad().at({1, 0}) == Catch::Approx(7.0f));
  REQUIRE(x.grad().at({1, 1}) == Catch::Approx(3.0f));

  REQUIRE(y.grad().at({0, 0}) == Catch::Approx(4.0f));
  REQUIRE(y.grad().at({0, 1}) == Catch::Approx(4.0f));
  REQUIRE(y.grad().at({1, 0}) == Catch::Approx(6.0f));
  REQUIRE(y.grad().at({1, 1}) == Catch::Approx(6.0f));
}