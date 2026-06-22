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

  SECTION("Row-sum = 1") {
    const axon::Tensor sum = t_softmax.sum(1, false);
    for (axon::idx_t i = 0; i < sum.num_elements(); i++) {
      REQUIRE(sum[i].item() == Catch::Approx(1.0f));
    }
  }
}

TEST_CASE("Softmax remains numerically stable for large logits",
          "[Functional]") {
  const axon::Tensor t =
      axon::Tensor::from_data({1000.0f, 1001.0f, 1002.0f}, {1, 3});
  const axon::Tensor t_softmax = axon::softmax(t);

  for (axon::idx_t i = 0; i < t_softmax.num_elements(); i++) {
    float val = t_softmax.at({0, i});
    REQUIRE(std::isfinite(val));
    REQUIRE(val >= 0.0f);
    REQUIRE(val <= 1.0f);
  }
}

TEST_CASE("Log softmax produces expected output", "[Functional]") {
  const axon::Tensor t =
      axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  const axon::Tensor t_log_softmax = axon::log_softmax(t);

  SECTION("Input shape equals output shape") {
    REQUIRE(t.shape() == t_log_softmax.shape());
  }

  SECTION("Output elements as expected") {
    REQUIRE(t_log_softmax.at({0, 0}) == Catch::Approx(-2.4076f).margin(0.001));
    REQUIRE(t_log_softmax.at({0, 1}) == Catch::Approx(-1.4076f).margin(0.001));
    REQUIRE(t_log_softmax.at({0, 2}) == Catch::Approx(-0.4076f).margin(0.001));
    REQUIRE(t_log_softmax.at({1, 0}) == Catch::Approx(-2.4076f).margin(0.001));
    REQUIRE(t_log_softmax.at({1, 1}) == Catch::Approx(-1.4076f).margin(0.001));
    REQUIRE(t_log_softmax.at({1, 2}) == Catch::Approx(-0.4076f).margin(0.001));
  }

  SECTION("exp(log_softmax) row-sum = 1") {
    const axon::Tensor sum = t_log_softmax.exp().sum(1, false);
    for (axon::idx_t i = 0; i < sum.num_elements(); i++) {
      REQUIRE(sum[i].item() == Catch::Approx(1.0f).margin(1e-5f));
    }
  }
}

TEST_CASE("Log softmax remains numerically stable for large logits",
          "[Functional]") {
  const axon::Tensor t =
      axon::Tensor::from_data({1000.0f, 1001.0f, 1002.0f}, {1, 3});
  const axon::Tensor t_log_softmax = axon::log_softmax(t);

  for (axon::idx_t i = 0; i < t_log_softmax.num_elements(); i++) {
    float val = t_log_softmax.at({0, i});
    REQUIRE(std::isfinite(val));
  }
}

TEST_CASE("Cross entropy loss", "[Functional]") {
  SECTION("Output is scalar") {
    axon::Tensor logits =
        axon::Tensor::from_data({2.0f, 1.0f, 0.1f, 0.5f, 2.0f, 0.3f}, {2, 3});
    axon::Tensor targets =
        axon::Tensor::from_data({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {2, 3});
    axon::Tensor loss = axon::cross_entropy_loss(logits, targets);
    REQUIRE(loss.num_elements() == 1);
  }

  SECTION("Known value") {
    // log_softmax([2, 1, 0.1]) ≈ [-0.4170, -1.4170, -2.3170]
    // targets = [1, 0, 0] → loss = 0.4170
    axon::Tensor logits = axon::Tensor::from_data({2.0f, 1.0f, 0.1f}, {1, 3});
    axon::Tensor targets = axon::Tensor::from_data({1.0f, 0.0f, 0.0f}, {1, 3});
    axon::Tensor loss = axon::cross_entropy_loss(logits, targets);
    REQUIRE(loss.item() == Catch::Approx(0.4170f).margin(0.001f));
  }

  SECTION("Uniform logits → loss = log(num_classes)") {
    axon::Tensor logits = axon::Tensor::from_data({0.0f, 0.0f, 0.0f}, {1, 3});
    axon::Tensor targets = axon::Tensor::from_data({1.0f, 0.0f, 0.0f}, {1, 3});
    axon::Tensor loss = axon::cross_entropy_loss(logits, targets);
    REQUIRE(loss.item() == Catch::Approx(std::log(3.0f)).margin(0.001f));
  }
}
