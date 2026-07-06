/**
 * @file tensor_test.cpp
 * @brief Unit tests for the axon::Tensor class.
 * @author Mika Brückner
 * @date 2026-06-02
 */

#include "axon/tensor.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>

TEST_CASE("Tensor with given data", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

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

TEST_CASE("Tensor with no matching data and shape size", "[Tensor]") {
  REQUIRE_THROWS_AS(axon::Tensor::from_data({1, 2, 3, 4}, {1, 5}),
                    std::out_of_range);
}

TEST_CASE("Tensor without given data", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::zeros({2, 3});
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

TEST_CASE("Transpose Tensor", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
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

TEST_CASE("Reshape a Tensor", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
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

TEST_CASE("Flatten Tensor", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
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

TEST_CASE("Reshape non-contiguous Tensor", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE_THROWS_AS(t.transpose().reshape({6}), std::logic_error);
}

TEST_CASE("Reshape Tensor with non-matching size", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE_THROWS_AS(t.reshape({7}), std::out_of_range);
}

TEST_CASE("Unsqueeze produces expected output", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1.0f, 2.0f, 3.0f}, {3});

  const axon::Tensor t_unsqueezed_0 = t.unsqueeze(0);
  std::vector<axon::idx_t> expected_shape = {1, 3};
  REQUIRE(t_unsqueezed_0.shape() == expected_shape);
  REQUIRE(t_unsqueezed_0[0][0].item() == 1.0f);
  REQUIRE(t_unsqueezed_0[0][1].item() == 2.0f);
  REQUIRE(t_unsqueezed_0[0][2].item() == 3.0f);

  const axon::Tensor t_unsqueezed_1 = t.unsqueeze(1);
  expected_shape = {3, 1};
  REQUIRE(t_unsqueezed_1.shape() == expected_shape);
  REQUIRE(t_unsqueezed_1[0][0].item() == 1.0f);
  REQUIRE(t_unsqueezed_1[1][0].item() == 2.0f);
  REQUIRE(t_unsqueezed_1[2][0].item() == 3.0f);
}

TEST_CASE("Unsqueeze out of range throws", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1.0f, 2.0f, 3.0f}, {3});
  REQUIRE_THROWS_AS(t.unsqueeze(-1), std::out_of_range);
  REQUIRE_THROWS_AS(t.unsqueeze(2), std::out_of_range);
}

TEST_CASE("Elementwise Tensor addition", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r = axon::Tensor::from_data({6, 5, 4, 3, 2, 1}, {2, 3});

  const axon::Tensor t = t_l + t_r;

  REQUIRE(t.at({0, 0}) == Catch::Approx(7.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(7.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(7.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(7.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(7.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(7.0f));
}

TEST_CASE("Elementwise Tensor addition with no matching shapes", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r =
      axon::Tensor::from_data({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});

  REQUIRE_THROWS_AS(t_l + t_r, std::out_of_range);
}

TEST_CASE("Scalar Tensor addition", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  constexpr float scalar = 1.0f;

  const axon::Tensor t = t_tensor + scalar;

  REQUIRE(t.at({0, 0}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(3.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(4.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(5.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(6.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(7.0f));
}

TEST_CASE("Elementwise Tensor substraction", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r = axon::Tensor::from_data({6, 5, 4, 3, 2, 1}, {2, 3});

  const axon::Tensor t = t_l - t_r;

  REQUIRE(t.at({0, 0}) == Catch::Approx(-5.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(-3.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(-1.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(1.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(5.0f));
}

TEST_CASE("Elementwise Tensor substraction with no matching shapes",
          "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r =
      axon::Tensor::from_data({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});

  REQUIRE_THROWS_AS(t_l - t_r, std::out_of_range);
}

TEST_CASE("Scalar Tensor substraction", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  constexpr float scalar = 1.0f;

  const axon::Tensor t = t_tensor - scalar;

  REQUIRE(t.at({0, 0}) == Catch::Approx(0.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(1.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(2.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(4.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(5.0f));
}

TEST_CASE("Elementwise Tensor multiplication", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r = axon::Tensor::from_data({6, 5, 4, 3, 2, 1}, {2, 3});

  const axon::Tensor t = t_l * t_r;

  REQUIRE(t.at({0, 0}) == Catch::Approx(6.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(10.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(12.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(12.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(10.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
}

TEST_CASE("Elementwise Tensor multiplication with no matching shapes",
          "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r =
      axon::Tensor::from_data({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});

  REQUIRE_THROWS_AS(t_l * t_r, std::out_of_range);
}

TEST_CASE("Scalar Tensor multiplication", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  constexpr float scalar = 2.0f;

  const axon::Tensor t = t_tensor * scalar;

  REQUIRE(t.at({0, 0}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(4.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(6.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(8.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(10.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(12.0f));
}

TEST_CASE("Elementwise Tensor division", "[Tensor]") {
  const axon::Tensor t_l =
      axon::Tensor::from_data({6, 12, 9, 8, 10, 6}, {2, 3});
  const axon::Tensor t_r = axon::Tensor::from_data({6, 6, 3, 4, 2, 1}, {2, 3});

  const axon::Tensor t = t_l / t_r;

  REQUIRE(t.at({0, 0}) == Catch::Approx(1.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(2.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(5.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
}

TEST_CASE("Elementwise Tensor division with no matching shapes", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r =
      axon::Tensor::from_data({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});

  REQUIRE_THROWS_AS(t_l / t_r, std::out_of_range);
}

TEST_CASE("Scalar Tensor division", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({2, 4, 6, 8, 10, 12}, {2, 3});
  constexpr float scalar = 2.0f;

  const axon::Tensor t = t_tensor / scalar;

  REQUIRE(t.at({0, 0}) == Catch::Approx(1.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(4.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(5.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
}

TEST_CASE("Tensor matrix multiplication", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r =
      axon::Tensor::from_data({7, 8, 9, 10, 11, 12}, {3, 2});

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
          "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

  REQUIRE_THROWS_AS(t_l.matmul(t_r), std::out_of_range);
}

TEST_CASE("Tensor log", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

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

TEST_CASE("Tensor exp", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

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

TEST_CASE("Tensor log and exp are inverse", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

  const axon::Tensor t = t_tensor.log().exp();

  REQUIRE(t.at({0, 0}) == Catch::Approx(1.0f));
  REQUIRE(t.at({0, 1}) == Catch::Approx(2.0f));
  REQUIRE(t.at({0, 2}) == Catch::Approx(3.0f));
  REQUIRE(t.at({1, 0}) == Catch::Approx(4.0f));
  REQUIRE(t.at({1, 1}) == Catch::Approx(5.0f));
  REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
}

TEST_CASE("Tensor min", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE(t.min() == 1.0f);
}

TEST_CASE("Tensor max", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE(t.max() == 6.0f);
}

TEST_CASE("Tensor sum", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE(t.sum().at({0}) == 21.0f);
}

TEST_CASE("Tensor sum(dim, keep_dim) returns expected output", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  SECTION("keep_dim = true") {
    axon::Tensor result = t.sum(1, true);
    std::vector<axon::idx_t> expected_shape{2, 1};
    REQUIRE(result.shape() == expected_shape);
    REQUIRE(result[0][0].item() == 6.0f);
    REQUIRE(result[1][0].item() == 15.0f);
  }
  SECTION("keep_dim = false") {
    axon::Tensor result = t.sum(1, false);
    std::vector<axon::idx_t> expected_shape{2};
    REQUIRE(result.shape() == expected_shape);
    REQUIRE(result[0].item() == 6.0f);
    REQUIRE(result[1].item() == 15.0f);
  }
}

TEST_CASE("Tensor argmax(dim, keep_dim) returns expected output", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 5, 3, 4, 2, 6}, {2, 3});

  SECTION("dim = 1, keep_dim = true") {
    const axon::Tensor result = t.argmax(1, true);
    const std::vector<axon::idx_t> expected_shape{2, 1};
    REQUIRE(result.shape() == expected_shape);
    REQUIRE(result.at({0, 0}) == 1.0f);
    REQUIRE(result.at({1, 0}) == 2.0f);
  }

  SECTION("dim = 1, keep_dim = false") {
    const axon::Tensor result = t.argmax(1, false);
    const std::vector<axon::idx_t> expected_shape{2};
    REQUIRE(result.shape() == expected_shape);
    REQUIRE(result.at({0}) == 1.0f);
    REQUIRE(result.at({1}) == 2.0f);
  }

  SECTION("dim = 0, keep_dim = true") {
    const axon::Tensor result = t.argmax(0, true);
    const std::vector<axon::idx_t> expected_shape{1, 3};
    REQUIRE(result.shape() == expected_shape);
    REQUIRE(result.at({0, 0}) == 1.0f);
    REQUIRE(result.at({0, 1}) == 0.0f);
    REQUIRE(result.at({0, 2}) == 1.0f);
  }

  SECTION("dim = 0, keep_dim = false") {
    const axon::Tensor result = t.argmax(0, false);
    const std::vector<axon::idx_t> expected_shape{3};
    REQUIRE(result.shape() == expected_shape);
    REQUIRE(result.at({0}) == 1.0f);
    REQUIRE(result.at({1}) == 0.0f);
    REQUIRE(result.at({2}) == 1.0f);
  }
}

TEST_CASE("Tensor argmax defaults to keep_dim = true", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 5, 3, 4, 2, 6}, {2, 3});

  const axon::Tensor result = t.argmax(1);
  const std::vector<axon::idx_t> expected_shape{2, 1};
  REQUIRE(result.shape() == expected_shape);
  REQUIRE(result.at({0, 0}) == 1.0f);
  REQUIRE(result.at({1, 0}) == 2.0f);
}

TEST_CASE("Tensor argmax with negative values", "[Tensor]") {
  const axon::Tensor t =
      axon::Tensor::from_data({-1, -5, -3, -4, -2, -6}, {2, 3});

  const axon::Tensor result = t.argmax(1, false);
  const std::vector<axon::idx_t> expected_shape{2};
  REQUIRE(result.shape() == expected_shape);
  REQUIRE(result.at({0}) == 0.0f);
  REQUIRE(result.at({1}) == 1.0f);
}

TEST_CASE("Tensor argmax breaks ties with the first occurrence", "[Tensor]") {
  SECTION("tie at the start") {
    const axon::Tensor t = axon::Tensor::from_data({3, 3, 1}, {3});
    REQUIRE(t.argmax(0, false).at({}) == 0.0f);
  }
  SECTION("tie later in the tensor") {
    const axon::Tensor t = axon::Tensor::from_data({1, 3, 3}, {3});
    REQUIRE(t.argmax(0, false).at({}) == 1.0f);
  }
}

TEST_CASE("Tensor argmax on a 1D tensor reduces to a scalar", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 5, 3, 4, 2, 6}, {6});

  SECTION("keep_dim = false") {
    const axon::Tensor result = t.argmax(0, false);
    REQUIRE(result.num_dim() == 0);
    REQUIRE(result.item() == 5.0f);
  }
  SECTION("keep_dim = true") {
    const axon::Tensor result = t.argmax(0, true);
    const std::vector<axon::idx_t> expected_shape{1};
    REQUIRE(result.shape() == expected_shape);
    REQUIRE(result.at({0}) == 5.0f);
  }
}

TEST_CASE("Tensor argmax with dim out of range throws", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE_THROWS_AS(t.argmax(2), std::out_of_range);
  REQUIRE_THROWS_AS(t.argmax(-1), std::out_of_range);
}

TEST_CASE("Tensor mean", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  REQUIRE(t.mean().at({0}) == 3.5f);
}

TEST_CASE("Tensor subscript operator", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

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

TEST_CASE("Tensor abs", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({-1, 2, -3, 4, -5, 6}, {2, 3});

  const axon::Tensor t = t_tensor.abs();

  SECTION("shape") {
    REQUIRE(t.num_dim() == 2);
    REQUIRE(t.shape().at(0) == 2);
    REQUIRE(t.shape().at(1) == 3);
  }

  SECTION("at") {
    REQUIRE(t.at({0, 0}) == Catch::Approx(1.0f));
    REQUIRE(t.at({0, 1}) == Catch::Approx(2.0f));
    REQUIRE(t.at({0, 2}) == Catch::Approx(3.0f));
    REQUIRE(t.at({1, 0}) == Catch::Approx(4.0f));
    REQUIRE(t.at({1, 1}) == Catch::Approx(5.0f));
    REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
  }
}

TEST_CASE("Tensor ReLU", "[Tensor]") {
  const axon::Tensor t_tensor =
      axon::Tensor::from_data({-1, 2, -3, 4, -5, 6}, {2, 3});

  const axon::Tensor t = t_tensor.relu();

  SECTION("shape") {
    REQUIRE(t.num_dim() == 2);
    REQUIRE(t.shape().at(0) == 2);
    REQUIRE(t.shape().at(1) == 3);
  }

  SECTION("at") {
    REQUIRE(t.at({0, 0}) == Catch::Approx(0.0f));
    REQUIRE(t.at({0, 1}) == Catch::Approx(2.0f));
    REQUIRE(t.at({0, 2}) == Catch::Approx(0.0f));
    REQUIRE(t.at({1, 0}) == Catch::Approx(4.0f));
    REQUIRE(t.at({1, 1}) == Catch::Approx(0.0f));
    REQUIRE(t.at({1, 2}) == Catch::Approx(6.0f));
  }
}

TEST_CASE("Tensor less-than comparison", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_greater =
      axon::Tensor::from_data({2, 3, 4, 5, 6, 7}, {2, 3});
  const axon::Tensor t_mixed =
      axon::Tensor::from_data({2, 3, 3, 5, 6, 7}, {2, 3});

  SECTION("all elements strictly less") { REQUIRE(t < t_greater); }
  SECTION("an equal element is not strictly less") {
    REQUIRE_FALSE(t < t_mixed);
  }
  SECTION("equal tensors are not strictly less") { REQUIRE_FALSE(t < t); }
}

TEST_CASE("Tensor less-than-or-equal comparison", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_greater_eq =
      axon::Tensor::from_data({1, 3, 3, 5, 5, 7}, {2, 3});
  const axon::Tensor t_smaller =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 5}, {2, 3});

  SECTION("equal tensors compare less-or-equal") { REQUIRE(t <= t); }
  SECTION("greater-or-equal elements compare less-or-equal") {
    REQUIRE(t <= t_greater_eq);
  }
  SECTION("a single greater element breaks less-or-equal") {
    REQUIRE_FALSE(t <= t_smaller);
  }
}

TEST_CASE("Tensor greater-than comparison", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({2, 3, 4, 5, 6, 7}, {2, 3});
  const axon::Tensor t_smaller =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_mixed =
      axon::Tensor::from_data({1, 3, 3, 4, 5, 6}, {2, 3});

  SECTION("all elements strictly greater") { REQUIRE(t > t_smaller); }
  SECTION("an equal element is not strictly greater") {
    REQUIRE_FALSE(t > t_mixed);
  }
  SECTION("equal tensors are not strictly greater") { REQUIRE_FALSE(t > t); }
}

TEST_CASE("Tensor greater-than-or-equal comparison", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({2, 3, 4, 5, 6, 7}, {2, 3});
  const axon::Tensor t_smaller_eq =
      axon::Tensor::from_data({2, 2, 4, 4, 6, 6}, {2, 3});
  const axon::Tensor t_greater =
      axon::Tensor::from_data({2, 3, 4, 5, 6, 8}, {2, 3});

  SECTION("equal tensors compare greater-or-equal") { REQUIRE(t >= t); }
  SECTION("smaller-or-equal elements compare greater-or-equal") {
    REQUIRE(t >= t_smaller_eq);
  }
  SECTION("a single smaller element breaks greater-or-equal") {
    REQUIRE_FALSE(t >= t_greater);
  }
}

TEST_CASE("Tensor equality comparison", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_equal =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_different =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 7}, {2, 3});

  SECTION("identical data compares equal") { REQUIRE(t == t_equal); }
  SECTION("a single differing element breaks equality") {
    REQUIRE_FALSE(t == t_different);
  }
}

TEST_CASE("Tensor eq produces an elementwise mask", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r = axon::Tensor::from_data({1, 9, 3, 9, 5, 9}, {2, 3});

  const axon::Tensor mask = t_l.eq(t_r);

  SECTION("shape matches the input shape") {
    REQUIRE(mask.shape() == t_l.shape());
  }

  SECTION("mask holds 1.0 where equal and 0.0 elsewhere") {
    REQUIRE(mask.at({0, 0}) == 1.0f);
    REQUIRE(mask.at({0, 1}) == 0.0f);
    REQUIRE(mask.at({0, 2}) == 1.0f);
    REQUIRE(mask.at({1, 0}) == 0.0f);
    REQUIRE(mask.at({1, 1}) == 1.0f);
    REQUIRE(mask.at({1, 2}) == 0.0f);
  }
}

TEST_CASE("Tensor eq broadcasts against a smaller operand", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 1, 2, 9}, {2, 3});
  const axon::Tensor row = axon::Tensor::from_data({1, 2, 3}, {1, 3});

  const axon::Tensor mask = t.eq(row);

  const std::vector<axon::idx_t> expected_shape{2, 3};
  REQUIRE(mask.shape() == expected_shape);
  REQUIRE(mask.at({0, 0}) == 1.0f);
  REQUIRE(mask.at({0, 1}) == 1.0f);
  REQUIRE(mask.at({0, 2}) == 1.0f);
  REQUIRE(mask.at({1, 0}) == 1.0f);
  REQUIRE(mask.at({1, 1}) == 1.0f);
  REQUIRE(mask.at({1, 2}) == 0.0f);
}

TEST_CASE("Tensor eq with non-broadcastable shapes throws", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r =
      axon::Tensor::from_data({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});

  REQUIRE_THROWS_AS(t_l.eq(t_r), std::out_of_range);
}

TEST_CASE("Tensor comparison with non-matching shapes", "[Tensor]") {
  const axon::Tensor t_l = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
  const axon::Tensor t_r = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {3, 2});

  REQUIRE_THROWS_AS(t_l < t_r, std::out_of_range);
  REQUIRE_THROWS_AS(t_l <= t_r, std::out_of_range);
  REQUIRE_THROWS_AS(t_l > t_r, std::out_of_range);
  REQUIRE_THROWS_AS(t_l >= t_r, std::out_of_range);
  REQUIRE_THROWS_AS(t_l == t_r, std::out_of_range);
}

TEST_CASE("Tensor item method", "[Tensor]") {
  SECTION(".item() on 0D Tensor") {
    const axon::Tensor t = axon::Tensor::from_data({1.0f}, {1});
    REQUIRE(t.item() == 1.0f);
  }
  SECTION(".item() on >0D Tensor") {
    const axon::Tensor t = axon::Tensor::from_data({1.0f, 2.0f}, {2, 1});
    REQUIRE_THROWS_AS(t.item(), std::runtime_error);
  }
}

TEST_CASE("Tensor data", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

  SECTION("returns a flat copy of all elements") {
    const std::vector<float> expected = {1, 2, 3, 4, 5, 6};
    REQUIRE(t.data() == expected);
    REQUIRE(t.data().size() == t.num_elements());
  }

  SECTION("returns a copy that does not alias the storage") {
    std::vector<float> copy = t.data();
    copy.at(0) = 42.0f;
    REQUIRE(t.at({0, 0}) == 1.0f);
  }
}

TEST_CASE("Tensor set_data", "[Tensor]") {
  const axon::Tensor t = axon::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

  SECTION("overwrites the elements in-place") {
    t.set_data({6, 5, 4, 3, 2, 1});
    REQUIRE(t.at({0, 0}) == 6.0f);
    REQUIRE(t.at({0, 1}) == 5.0f);
    REQUIRE(t.at({0, 2}) == 4.0f);
    REQUIRE(t.at({1, 0}) == 3.0f);
    REQUIRE(t.at({1, 1}) == 2.0f);
    REQUIRE(t.at({1, 2}) == 1.0f);

    const std::vector<float> expected = {6, 5, 4, 3, 2, 1};
    REQUIRE(t.data() == expected);
  }

  SECTION("does not change shape or rank") {
    t.set_data({6, 5, 4, 3, 2, 1});
    REQUIRE(t.num_elements() == 6);
    REQUIRE(t.num_dim() == 2);
    const std::vector<axon::idx_t> expected_shape = {2, 3};
    REQUIRE(t.shape() == expected_shape);
  }

  SECTION("is visible through copies sharing the same storage") {
    const axon::Tensor copy = t.shared_autograd_copy();
    t.set_data({6, 5, 4, 3, 2, 1});
    const std::vector<float> expected = {6, 5, 4, 3, 2, 1};
    REQUIRE(copy.data() == expected);
  }

  SECTION("throws on too few elements") {
    REQUIRE_THROWS_AS(t.set_data({1, 2, 3}), std::out_of_range);
  }

  SECTION("throws on too many elements") {
    REQUIRE_THROWS_AS(t.set_data({1, 2, 3, 4, 5, 6, 7}), std::out_of_range);
  }
}