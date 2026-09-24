/**
 * @file cpu_kernel_test.cpp
 * @brief Unit tests for axon::internal::cpu_kernel().
 * @author Mika Brückner
 * @date 2026-09-24
 */

#include <vector>

#include "../src/ops/cpu/loops.hpp"
#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("cpu_kernel applies the function to every element", "[CpuKernel]") {
  const axon::Tensor input =
      axon::Tensor::from_data({-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f}, {2, 3});
  axon::Tensor output = axon::Tensor::zeros({2, 3});

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);
  axon::internal::cpu_kernel(iter, [](float x) { return x > 0 ? x : 0.0f; });

  const std::vector<float> expected{0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f};
  REQUIRE(output.data() == expected);
}

TEST_CASE("cpu_kernel reads a transposed input in the right order",
          "[CpuKernel]") {
  const axon::Tensor base =
      axon::Tensor::from_data({-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f}, {2, 3});
  const axon::Tensor input = base.transpose();
  axon::Tensor output = axon::Tensor::zeros({3, 2});

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);
  axon::internal::cpu_kernel(iter, [](float x) { return x > 0 ? x : 0.0f; });

  const std::vector<float> expected{0.0f, 4.0f, 2.0f, 0.0f, 0.0f, 6.0f};
  REQUIRE(output.data() == expected);
}

TEST_CASE("cpu_kernel respects the input's offset", "[CpuKernel]") {
  const axon::Tensor base =
      axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  const axon::Tensor input = base[1];  // second row {4, 5, 6}, offset 3
  axon::Tensor output = axon::Tensor::zeros({3});

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);
  axon::internal::cpu_kernel(iter, [](float x) { return x * 10.0f; });

  const std::vector<float> expected{40.0f, 50.0f, 60.0f};
  REQUIRE(output.data() == expected);
}

TEST_CASE("cpu_kernel carries across several dimensions", "[CpuKernel]") {
  const std::vector<axon::idx_t> shape{2, 2, 2};
  const axon::Tensor base = axon::Tensor::from_data(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, shape);
  const axon::Tensor input = base.transpose();
  axon::Tensor output = axon::Tensor::zeros(shape);

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);
  axon::internal::cpu_kernel(iter, [](float x) { return x * x; });

  const std::vector<float> expected{
      1.0f, 25.0f, 9.0f, 49.0f, 4.0f, 36.0f, 16.0f, 64.0f};
  REQUIRE(output.data() == expected);
}

TEST_CASE("cpu_kernel does nothing for an empty tensor", "[CpuKernel]") {
  const axon::Tensor input = axon::Tensor::zeros({0});
  axon::Tensor output = axon::Tensor::zeros({0});

  const auto iter = axon::internal::TensorIterator::unary_op(output, input);

  int calls = 0;
  axon::internal::cpu_kernel(iter, [&calls](float x) {
    calls++;
    return x + 1.0f;
  });

  REQUIRE(calls == 0);
  REQUIRE(output.num_elements() == 0);
}
