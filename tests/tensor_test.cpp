#include "axon/tensor.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

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

TEST_CASE("Tensor with no metching data and shape size", "[TensorSize]") {
  REQUIRE_THROWS_AS(axon::Tensor({1, 2, 3, 4}, {1, 5}), std::out_of_range);
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

TEST_CASE("Elementwise Tensor addition", "[TensorAddEw]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({6, 5, 4, 3, 2, 1}, {2, 3});

  const axon::Tensor t = t_l + t_r;

  REQUIRE(t.at({0, 0}) == Catch::Approx(7.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(7.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(7.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(7.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(7.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(7.0f));
}

TEST_CASE("Elementwise Tensor addition with no matching shapes",
          "[TensorAddEwNMS]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});

  REQUIRE_THROWS_AS(t_l + t_r, std::out_of_range);
}

TEST_CASE("Scalar Tensor addition", "[TensorAddSc]") {
  const axon::Tensor t_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
  constexpr float scalar = 1.0f;

  const axon::Tensor t = t_tensor + scalar;

  REQUIRE(t.at({0, 0}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(3.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(4.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(5.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(6.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(7.0f));
}

TEST_CASE("Elementwise Tensor substraction", "[TensorSubEw]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({6, 5, 4, 3, 2, 1}, {2, 3});

  const axon::Tensor t = t_l - t_r;

  REQUIRE(t.at({0, 0}) == Catch::Approx(-5.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(-3.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(-1.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(1.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(5.0f));
}

TEST_CASE("Elementwise Tensor substraction with no matching shapes",
          "[TensorSubEwNMS]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});

  REQUIRE_THROWS_AS(t_l - t_r, std::out_of_range);
}

TEST_CASE("Scalar Tensor substraction", "[TensorSubSc]") {
  const axon::Tensor t_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
  constexpr float scalar = 1.0f;

  const axon::Tensor t = t_tensor - scalar;

  REQUIRE(t.at({0, 0}) == Catch::Approx(0.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(1.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(2.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(4.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(5.0f));
}

TEST_CASE("Elementwise Tensor multiplication", "[TensorMulEw]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({6, 5, 4, 3, 2, 1}, {2, 3});

  const axon::Tensor t = t_l * t_r;

  REQUIRE(t.at({0, 0}) == Catch::Approx(6.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(10.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(12.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(12.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(10.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
}

TEST_CASE("Elementwise Tensor multiplication with no matching shapes",
          "[TensorMulEwNMS]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});

  REQUIRE_THROWS_AS(t_l * t_r, std::out_of_range);
}

TEST_CASE("Scalar Tensor multiplication", "[TensorMulSc]") {
  const axon::Tensor t_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
  constexpr float scalar = 2.0f;

  const axon::Tensor t = t_tensor * scalar;

  REQUIRE(t.at({0, 0}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(4.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(6.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(8.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(10.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(12.0f));
}

TEST_CASE("Elementwise Tensor division", "[TensorDivEw]") {
  const axon::Tensor t_l({6, 12, 9, 8, 10, 6}, {2, 3});
  const axon::Tensor t_r({6, 6, 3, 4, 2, 1}, {2, 3});

  const axon::Tensor t = t_l / t_r;

  REQUIRE(t.at({0, 0}) == Catch::Approx(1.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(2.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(5.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
}

TEST_CASE("Elementwise Tensor division with no matching shapes",
          "[TensorDivEwNMS]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});

  REQUIRE_THROWS_AS(t_l / t_r, std::out_of_range);
}

TEST_CASE("Scalar Tensor division", "[TensorDivSc]") {
  const axon::Tensor t_tensor({2, 4, 6, 8, 10, 12}, {2, 3});
  constexpr float scalar = 2.0f;

  const axon::Tensor t = t_tensor / scalar;

  REQUIRE(t.at({0, 0}) == Catch::Approx(1.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(4.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(5.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
}

TEST_CASE("Tensor matrix multiplication", "[TensorMatmul]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({7, 8, 9, 10, 11, 12}, {3, 2});

  const axon::Tensor t = t_l.matmul(t_r);

  SECTION("shape") {
    REQUIRE(t.num_dim() == 2);
    REQUIRE(t.shape().at(0) == 2);
    REQUIRE(t.shape().at(1) == 2);
  }

  SECTION("at") {
    REQUIRE(t.at({0, 0}) == Catch::Approx(58.0f));
    REQUIRE(t.at({0, 1}) == Catch::Approx(64.0f));
    REQUIRE(t.at({1, 0}) == Catch::Approx(139.0f));
    REQUIRE(t.at({1, 1}) == Catch::Approx(154.0f));
  }
}

TEST_CASE("Tensor matrix multiplication with non-matching inner dimensions",
          "[TensorMatmulNMS]") {
  const axon::Tensor t_l({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r({1, 2, 3, 4, 5, 6}, {2, 3});

  REQUIRE_THROWS_AS(t_l.matmul(t_r), std::out_of_range);
}

TEST_CASE("Tensor log", "[TensorLog]") {
  const axon::Tensor t_tensor({1, 2, 3, 4, 5, 6}, {2, 3});

  const axon::Tensor t = t_tensor.log();

  SECTION("shape") {
    REQUIRE(t.num_dim() == 2);
    REQUIRE(t.shape().at(0) == 2);
    REQUIRE(t.shape().at(1) == 3);
  }

  SECTION("at") {
    REQUIRE(t.at({0, 0}) == Catch::Approx(std::log(1.0f)));
    REQUIRE(t.at({0, 1}) == Catch::Approx(std::log(2.0f)));
    REQUIRE(t.at({0, 2}) == Catch::Approx(std::log(3.0f)));
    REQUIRE(t.at({1, 0}) == Catch::Approx(std::log(4.0f)));
    REQUIRE(t.at({1, 1}) == Catch::Approx(std::log(5.0f)));
    REQUIRE(t.at({1, 2}) == Catch::Approx(std::log(6.0f)));
  }
}

TEST_CASE("Tensor exp", "[TensorExp]") {
  const axon::Tensor t_tensor({1, 2, 3, 4, 5, 6}, {2, 3});

  const axon::Tensor t = t_tensor.exp();

  SECTION("shape") {
    REQUIRE(t.num_dim() == 2);
    REQUIRE(t.shape().at(0) == 2);
    REQUIRE(t.shape().at(1) == 3);
  }

  SECTION("at") {
    REQUIRE(t.at({0, 0}) == Catch::Approx(std::exp(1.0f)));
    REQUIRE(t.at({0, 1}) == Catch::Approx(std::exp(2.0f)));
    REQUIRE(t.at({0, 2}) == Catch::Approx(std::exp(3.0f)));
    REQUIRE(t.at({1, 0}) == Catch::Approx(std::exp(4.0f)));
    REQUIRE(t.at({1, 1}) == Catch::Approx(std::exp(5.0f)));
    REQUIRE(t.at({1, 2}) == Catch::Approx(std::exp(6.0f)));
  }
}

TEST_CASE("Tensor log and exp are inverse", "[TensorLogExp]") {
  const axon::Tensor t_tensor({1, 2, 3, 4, 5, 6}, {2, 3});

  const axon::Tensor t = t_tensor.log().exp();

  REQUIRE(t.at({0, 0}) == Catch::Approx(1.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(4.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(5.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
}

TEST_CASE("Tensor min", "[TensorMin]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE(t.min() == 1.0f);
}

TEST_CASE("Tensor max", "[TensorMax]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE(t.max() == 6.0f);
}

TEST_CASE("Tensor sum", "[TensorSum]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE(t.sum() == 21.0f);
}

TEST_CASE("Tensor mean", "[TensorMean]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE(t.mean() == 3.5f);
}

TEST_CASE("Tensor subscript operator", "[TensorSubscript]") {
  const axon::Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});

  SECTION("Matrix slice") {
    const axon::Tensor t_0 = t[0];
    REQUIRE(t_0.at({0}) == 1.0f);
    REQUIRE(t_0.at({1}) == 2.0f);
    REQUIRE(t_0.at({2}) == 3.0f);
  }

  SECTION("Multiple subscripts") {
    const axon::Tensor t_0_0 = t[0][0];
    REQUIRE(t_0_0.at({}) == 1.0f);
  }

  SECTION("Index out of Bounds") { REQUIRE_THROWS_AS(t[2], std::out_of_range); }

  SECTION("Too many subscripts") {
    REQUIRE_THROWS_AS(t[0][0][0], std::out_of_range);
  }
}