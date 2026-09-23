/**
 * @file tensor_iterator_test.cpp
 * @brief Unit tests for axon::internal::TensorIterator.
 * @author Mika Brückner
 * @date 2026-09-23
 */

#include "../src/core/tensor_iterator.hpp"

#include <vector>

#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("unary_op exposes the common shape and element count",
          "[TensorIterator]") {
  const std::vector<axon::idx_t> shape{2, 3};
  const axon::Tensor input =
      axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, shape);
  axon::Tensor output = axon::Tensor::zeros(shape);

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);

  REQUIRE(iter.shape() == shape);
  REQUIRE(iter.num_dim() == 2);
  REQUIRE(iter.num_elements() == 6);
}

TEST_CASE("unary_op registers the output as operand 0", "[TensorIterator]") {
  const std::vector<axon::idx_t> shape{2, 2};
  const axon::Tensor input =
      axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, shape);
  axon::Tensor output = axon::Tensor::zeros(shape);

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);

  REQUIRE(iter.num_operands() == 2);
  REQUIRE(iter.data(1)[0] == 1.0f);

  SECTION("Writing through operand 0 has to land in the output tensor") {
    iter.data(0)[0] = 42.0f;
    REQUIRE(output.at({0, 0}) == 42.0f);
  }
}

TEST_CASE("unary_op keeps the strides of a transposed input",
          "[TensorIterator]") {
  const axon::Tensor base =
      axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  const axon::Tensor input = base.transpose();
  axon::Tensor output = axon::Tensor::zeros({3, 2});

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);

  const std::vector<axon::idx_t> expected_shape{3, 2};
  const std::vector<axon::idx_t> transposed_strides{1, 3};
  const std::vector<axon::idx_t> contiguous_strides{2, 1};

  REQUIRE(iter.shape() == expected_shape);

  REQUIRE(iter.stride(1) == transposed_strides);
  REQUIRE(iter.stride(0) == contiguous_strides);
}

TEST_CASE("unary_op reports contiguous strides for the output",
          "[TensorIterator]") {
  const std::vector<axon::idx_t> shape{2, 3, 4};
  const axon::Tensor input = axon::Tensor::ones(shape);
  axon::Tensor output = axon::Tensor::zeros(shape);

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);

  const std::vector<axon::idx_t> contiguous_strides{12, 4, 1};

  REQUIRE(iter.num_dim() == 3);
  REQUIRE(iter.num_elements() == 24);
  REQUIRE(iter.stride(0) == contiguous_strides);
}

TEST_CASE("unary_op folds the tensor offset into the base pointer",
          "[TensorIterator]") {
  const axon::Tensor base =
      axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  const axon::Tensor input = base[1];  // second row {4, 5, 6}, offset 3
  axon::Tensor output = axon::Tensor::zeros({3});

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);

  REQUIRE(iter.num_elements() == 3);
  REQUIRE(iter.data(1)[0] == 4.0f);
  REQUIRE(iter.data(1)[2] == 6.0f);
}

TEST_CASE("unary_op reports zero elements for an empty tensor",
          "[TensorIterator]") {
  const axon::Tensor input = axon::Tensor::zeros({0});
  axon::Tensor output = axon::Tensor::zeros({0});

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);

  REQUIRE(iter.num_elements() == 0);
  REQUIRE(iter.num_operands() == 2);
}
