#include <catch2/catch_test_macros.hpp>
#include "axon/tensor.hpp"

TEST_CASE("Tensor with given data", "[TensorData]") {
  const axon::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

  REQUIRE(t.num_elements() == 6);
  REQUIRE(t.num_dim() == 2);

  REQUIRE(t.shape().at(0) == 2);
  REQUIRE(t.shape().at(1) == 3);

  REQUIRE(t.stride().size()== 2);
  REQUIRE(t.stride().at(0) == 3);
  REQUIRE(t.stride().at(1) == 1);

  REQUIRE(t.at({0,0})== 1.0f);
  REQUIRE(t.at({0,1})== 2.0f);
  REQUIRE(t.at({0,2})== 3.0f);
  REQUIRE(t.at({1,0})== 4.0f);
  REQUIRE(t.at({1,1})== 5.0f);
  REQUIRE(t.at({1,2})== 6.0f);

  REQUIRE_THROWS_AS(t.at({5, 0}), std::out_of_range);
  REQUIRE_THROWS_AS(t.at({0}), std::out_of_range);
}

TEST_CASE("Tensor without given data", "[TensorZeros]") {
  const axon::Tensor t({2, 3});

  REQUIRE(t.num_elements() == 6);
  REQUIRE(t.num_dim() == 2);

  REQUIRE(t.shape().at(0) == 2);
  REQUIRE(t.shape().at(1) == 3);

  REQUIRE(t.stride().size()== 2);
  REQUIRE(t.stride().at(0) == 3);
  REQUIRE(t.stride().at(1) == 1);
  REQUIRE(t.at({0,0})== 0.0f);
  REQUIRE(t.at({0,1})== 0.0f);
  REQUIRE(t.at({0,2})== 0.0f);
  REQUIRE(t.at({1,0})== 0.0f);
  REQUIRE(t.at({1,1})== 0.0f);
  REQUIRE(t.at({1,2})== 0.0f);

  REQUIRE_THROWS_AS(t.at({5, 0}), std::out_of_range);
  REQUIRE_THROWS_AS(t.at({0}), std::out_of_range);
}
