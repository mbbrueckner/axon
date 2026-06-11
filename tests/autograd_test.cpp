//
// Created by Mika Brückner on 11.06.26.
//

#include "../include/axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("Autograd mul backward", "[AutoGMul]") {
  axon::Tensor x(std::vector<float>{2.0f}, std::vector<axon::idx_t>{1});
  axon::Tensor y(std::vector<float>{3.0f}, std::vector<axon::idx_t>{1});

  x.requires_grad_(true);
  y.requires_grad_(true);

  axon::Tensor z = x * y;
  z.backward();

  REQUIRE(x.grad().at({0}) == 3.0f);
  REQUIRE(y.grad().at({0}) == 2.0f);
}