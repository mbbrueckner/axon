#include "axon/tensor.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Tensor with given data", "[TensorData]") {
  const axon::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

  REQUIRE(t.num_elements() == 6);
  REQUIRE(t.num_dim() == 2);

  REQUIRE(t.shape().at(0) == 2);
  REQUIRE(t.shape().at(1) == 3);

  REQUIRE(t.stride().size() == 2);
  REQUIRE(t.stride().at(0) == 3);
  REQUIRE(t.stride().at(1) == 1);

  REQUIRE(t.at({0, 0}) == 1.0f);
  REQUIRE(t.at({0, 1}) == 2.0f);
  REQUIRE(t.at({0, 2}) == 3.0f);
  REQUIRE(t.at({1, 0}) == 4.0f);
  REQUIRE(t.at({1, 1}) == 5.0f);
  REQUIRE(t.at({1, 2}) == 6.0f);

  REQUIRE_THROWS_AS(t.at({5, 0}), std::out_of_range);
  REQUIRE_THROWS_AS(t.at({0}), std::out_of_range);
}

TEST_CASE("Tensor without given data", "[TensorZeros]") {
  const axon::Tensor t({2, 3});

  REQUIRE(t.num_elements() == 6);
  REQUIRE(t.num_dim() == 2);

  REQUIRE(t.shape().at(0) == 2);
  REQUIRE(t.shape().at(1) == 3);

  REQUIRE(t.stride().size() == 2);
  REQUIRE(t.stride().at(0) == 3);
  REQUIRE(t.stride().at(1) == 1);
  REQUIRE(t.at({0, 0}) == 0.0f);
  REQUIRE(t.at({0, 1}) == 0.0f);
  REQUIRE(t.at({0, 2}) == 0.0f);
  REQUIRE(t.at({1, 0}) == 0.0f);
  REQUIRE(t.at({1, 1}) == 0.0f);
  REQUIRE(t.at({1, 2}) == 0.0f);

  REQUIRE_THROWS_AS(t.at({5, 0}), std::out_of_range);
  REQUIRE_THROWS_AS(t.at({0}), std::out_of_range);
}

TEST_CASE("Transpose Tensor", "[TensorTranspose]") {
  const axon::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
  const axon::Tensor t_transposed = t.transpose();

  REQUIRE(t.num_elements() == t_transposed.num_elements());
  REQUIRE(t.num_dim() == t_transposed.num_dim());

  REQUIRE(t.shape().at(0) == t_transposed.shape().at(1));
  REQUIRE(t.shape().at(1) == t_transposed.shape().at(0));

  REQUIRE(t.stride().size() == 2);
  REQUIRE(t.stride().at(0) == t_transposed.stride().at(1));
  REQUIRE(t.stride().at(1) == t_transposed.stride().at(0));
  REQUIRE(t_transposed.at({0, 0}) == 1.0f);
  REQUIRE(t_transposed.at({0, 1}) == 4.0f);
  REQUIRE(t_transposed.at({1, 0}) == 2.0f);
  REQUIRE(t_transposed.at({1, 1}) == 5.0f);
  REQUIRE(t_transposed.at({2, 0}) == 3.0f);
  REQUIRE(t_transposed.at({2, 1}) == 6.0f);

  REQUIRE_THROWS_AS(t.at({5, 0}), std::out_of_range);
  REQUIRE_THROWS_AS(t.at({0}), std::out_of_range);
}