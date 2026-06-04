#include "axon/tensor.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Tensor with given data", "[TensorData]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});

  SECTION("size") {
    REQUIRE(t.num_elements() == 6);
    REQUIRE(t.num_dim() == 2);
  }
  SECTION("shape") {
    REQUIRE(t.shape().at(0) == 2);
    REQUIRE(t.shape().at(1) == 3);
  }

  SECTION("stride") {
    REQUIRE(t.stride().size() == 2);
    REQUIRE(t.stride().at(0) == 3);
    REQUIRE(t.stride().at(1) == 1);
  }

  SECTION("at") {
    REQUIRE(t.at({0, 0}) == 1.0f);
    REQUIRE(t.at({0, 1}) == 2.0f);
    REQUIRE(t.at({0, 2}) == 3.0f);
    REQUIRE(t.at({1, 0}) == 4.0f);
    REQUIRE(t.at({1, 1}) == 5.0f);
    REQUIRE(t.at({1, 2}) == 6.0f);
  }
  SECTION("at out of range ") {
    REQUIRE_THROWS_AS(t.at({5, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(t.at({0}), std::out_of_range);
  }
}

TEST_CASE("Tensor without given data", "[TensorZeros]") {
  const axon::Tensor t({2, 3});
  SECTION("size") {
    REQUIRE(t.num_elements() == 6);
    REQUIRE(t.num_dim() == 2);
  }
  SECTION("stride") {
    REQUIRE(t.shape().at(0) == 2);
    REQUIRE(t.shape().at(1) == 3);
  }
  SECTION("at") {
    REQUIRE(t.stride().size() == 2);
    REQUIRE(t.stride().at(0) == 3);
    REQUIRE(t.stride().at(1) == 1);
    REQUIRE(t.at({0, 0}) == 0.0f);
    REQUIRE(t.at({0, 1}) == 0.0f);
    REQUIRE(t.at({0, 2}) == 0.0f);
    REQUIRE(t.at({1, 0}) == 0.0f);
    REQUIRE(t.at({1, 1}) == 0.0f);
    REQUIRE(t.at({1, 2}) == 0.0f);
  }
  SECTION("at out of range ") {
    REQUIRE_THROWS_AS(t.at({5, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(t.at({0}), std::out_of_range);
  }
}

TEST_CASE("Transpose Tensor", "[TensorTranspose]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_transposed = t.transpose();
  SECTION("size") {
    REQUIRE(t.num_elements() == t_transposed.num_elements());
    REQUIRE(t.num_dim() == t_transposed.num_dim());
  }
  SECTION("stride") {
    REQUIRE(t.shape().at(0) == t_transposed.shape().at(1));
    REQUIRE(t.shape().at(1) == t_transposed.shape().at(0));
  }

  SECTION("at") {
    REQUIRE(t.stride().size() == 2);
    REQUIRE(t.stride().at(0) == t_transposed.stride().at(1));
    REQUIRE(t.stride().at(1) == t_transposed.stride().at(0));
    REQUIRE(t_transposed.at({0, 0}) == 1.0f);
    REQUIRE(t_transposed.at({0, 1}) == 4.0f);
    REQUIRE(t_transposed.at({1, 0}) == 2.0f);
    REQUIRE(t_transposed.at({1, 1}) == 5.0f);
    REQUIRE(t_transposed.at({2, 0}) == 3.0f);
    REQUIRE(t_transposed.at({2, 1}) == 6.0f);
  }

  SECTION("at out of range ") {
    REQUIRE_THROWS_AS(t.at({5, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(t.at({0}), std::out_of_range);
  }
}

TEST_CASE("Reshape a Tensor", "[TensorReshape]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_reshaped = t.reshape({3, 2});

  SECTION("size") {
    REQUIRE(t_reshaped.num_elements() == 6);
    REQUIRE(t_reshaped.num_dim() == 2);
  }
  SECTION("shape") {
    REQUIRE(t_reshaped.shape().at(0) == 3);
    REQUIRE(t_reshaped.shape().at(1) == 2);
  }

  SECTION("stride") {
    REQUIRE(t_reshaped.stride().size() == 2);
    REQUIRE(t_reshaped.stride().at(0) == 2);
    REQUIRE(t_reshaped.stride().at(1) == 1);
  }

  SECTION("at") {
    REQUIRE(t_reshaped.at({0, 0}) == 1.0f);
    REQUIRE(t_reshaped.at({0, 1}) == 2.0f);
    REQUIRE(t_reshaped.at({1, 0}) == 3.0f);
    REQUIRE(t_reshaped.at({1, 1}) == 4.0f);
    REQUIRE(t_reshaped.at({2, 0}) == 5.0f);
    REQUIRE(t_reshaped.at({2, 1}) == 6.0f);
  }
  SECTION("at out of range ") {
    REQUIRE_THROWS_AS(t_reshaped.at({5, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(t_reshaped.at({0}), std::out_of_range);
  }
}

TEST_CASE("Flatten Tensor", "[TensorFlatten]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_flat = t.flatten();

  SECTION("size") {
    REQUIRE(t_flat.num_elements() == 6);
    REQUIRE(t_flat.num_dim() == 1);
  }
  SECTION("shape") { REQUIRE(t_flat.shape().at(0) == 6); }

  SECTION("stride") {
    REQUIRE(t_flat.stride().size() == 1);
    REQUIRE(t_flat.stride().at(0) == 1);
  }

  SECTION("at") {
    REQUIRE(t_flat.at({0}) == 1.0f);
    REQUIRE(t_flat.at({1}) == 2.0f);
    REQUIRE(t_flat.at({2}) == 3.0f);
    REQUIRE(t_flat.at({3}) == 4.0f);
    REQUIRE(t_flat.at({4}) == 5.0f);
    REQUIRE(t_flat.at({5}) == 6.0f);
  }
  SECTION("at out of range ") {
    REQUIRE_THROWS_AS(t_flat.at({5, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(t_flat.at({}), std::out_of_range);
  }
}

TEST_CASE("Reshape non-contiguous Tensor", "[TensorRsNC]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE_THROWS_AS(t.transpose().reshape({6}), std::logic_error);
}

TEST_CASE("Reshape Tensor with non-matching size", "[TensorRsNMS]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE_THROWS_AS(t.reshape({7}), std::out_of_range);
}
