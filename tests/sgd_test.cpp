/**
 * @file sgd_test.cpp
 * @brief Unit tests for the axon::optimizer::SGD optimizer.
 * @author Mika Brückner
 * @date 2026-06-25
 */

#include "axon/optimizers/sgd.hpp"

#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("SGD step updates parameters along the negative gradient", "[SGD]") {
  axon::Tensor p = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  p.requires_grad_(true);

  axon::Tensor loss = p.sum();
  loss.backward();  // grad is all ones

  axon::optimizer::SGD optimizer({p}, 0.1f);
  optimizer.step();

  REQUIRE(p.at({0, 0}) == Catch::Approx(0.9f));
  REQUIRE(p.at({0, 1}) == Catch::Approx(1.9f));
  REQUIRE(p.at({1, 0}) == Catch::Approx(2.9f));
  REQUIRE(p.at({1, 1}) == Catch::Approx(3.9f));
}

TEST_CASE("SGD step scales the update by the learning rate", "[SGD]") {
  axon::Tensor p = axon::Tensor::from_data({1.0f, 2.0f}, {2});
  p.requires_grad_(true);

  // loss = sum(p * p) => grad = 2 * p = {2, 4}
  axon::Tensor loss = (p * p).sum();
  loss.backward();

  axon::optimizer::SGD optimizer({p}, 0.5f);
  optimizer.step();

  // p - 0.5 * 2p = p - p == {0, 0}
  REQUIRE(p.at({0}) == Catch::Approx(0.0f));
  REQUIRE(p.at({1}) == Catch::Approx(0.0f));
}

TEST_CASE("SGD uses a default learning rate of 0.01", "[SGD]") {
  axon::Tensor p = axon::Tensor::from_data({1.0f, 2.0f}, {2});
  p.requires_grad_(true);

  axon::Tensor loss = p.sum();
  loss.backward();  // grad is all ones

  axon::optimizer::SGD optimizer({p});
  optimizer.step();

  REQUIRE(p.at({0}) == Catch::Approx(0.99f));
  REQUIRE(p.at({1}) == Catch::Approx(1.99f));
}

TEST_CASE("SGD step applies repeatedly", "[SGD]") {
  axon::Tensor p = axon::Tensor::from_data({1.0f, 2.0f}, {2});
  p.requires_grad_(true);

  axon::Tensor loss = p.sum();
  loss.backward();  // grad is all ones, unchanged across steps

  axon::optimizer::SGD optimizer({p}, 0.1f);
  optimizer.step();
  optimizer.step();

  REQUIRE(p.at({0}) == Catch::Approx(0.8f));
  REQUIRE(p.at({1}) == Catch::Approx(1.8f));
}

TEST_CASE("SGD optimizes multiple parameters", "[SGD]") {
  axon::Tensor a = axon::Tensor::from_data({1.0f, 2.0f}, {2});
  axon::Tensor b = axon::Tensor::from_data({3.0f, 4.0f}, {2});
  a.requires_grad_(true);
  b.requires_grad_(true);

  a.sum().backward();
  b.sum().backward();

  axon::optimizer::SGD optimizer({a, b}, 0.1f);
  optimizer.step();

  REQUIRE(a.at({0}) == Catch::Approx(0.9f));
  REQUIRE(a.at({1}) == Catch::Approx(1.9f));
  REQUIRE(b.at({0}) == Catch::Approx(2.9f));
  REQUIRE(b.at({1}) == Catch::Approx(3.9f));
}

TEST_CASE("SGD zero_grad resets gradients to zero", "[SGD]") {
  axon::Tensor p = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  p.requires_grad_(true);

  axon::Tensor loss = p.sum();
  loss.backward();  // grad is all ones

  axon::optimizer::SGD optimizer({p}, 0.1f);
  optimizer.zero_grad();

  axon::Tensor grad = p.grad();
  REQUIRE(grad.at({0, 0}) == 0.0f);
  REQUIRE(grad.at({0, 1}) == 0.0f);
  REQUIRE(grad.at({1, 0}) == 0.0f);
  REQUIRE(grad.at({1, 1}) == 0.0f);
}