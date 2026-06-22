/**
 * @file grad_check.cpp
 * @brief  Sanity check (finite difference check) for the autograd backwards
 * pass.
 * @author Mika Brückner
 * @date 2026-06-14
 */

#include <functional>
#include <iostream>

#include "axon/functional.hpp"
#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

namespace {

/// @brief Local helper: converts a flat index into multi-dimensional
/// indices for a given shape. Duplicated here since axon::utils lives in a
/// private header not exposed to tests.
std::vector<axon::idx_t> flat_to_indices(
    axon::idx_t flat, const std::vector<axon::idx_t>& shape) {
  std::vector<axon::idx_t> indices(shape.size());
  for (axon::idx_t i = static_cast<axon::idx_t>(shape.size()) - 1; i >= 0;
       i--) {
    indices[i] = flat % shape[i];
    flat /= shape[i];
  }
  return indices;
}

}  // namespace

/**
 * @brief Finite difference gradient check for a tensor-valued function f.
 *
 * Perturbs each element of x individually by +/- epsilon, evaluates
 * sum(f(x)) for each perturbation, and compares the resulting numeric
 * gradient against the analytic gradient computed via backward(). This
 * works for both elementwise functions and functions with cross-element
 * dependencies (e.g. sum(dim), matmul).
 *
 * @param f         Function to check.
 * @param x         Input tensor (must have requires_grad_(true)).
 * @param epsilon   Step size for finite differences.
 * @param tolerance Maximum allowed absolute difference.
 * @return true if all gradients match within tolerance.
 */
bool grad_check(std::function<axon::Tensor(axon::Tensor)> f,
                axon::Tensor& x,
                float epsilon = 1e-4f,
                float tolerance = 1e-3f) {
  x.requires_grad_(true);
  axon::Tensor y = f(x.shared_autograd_copy());
  y.backward();
  const axon::Tensor analytic = x.grad();

  const axon::idx_t n = x.num_elements();
  std::vector<float> original_data(n);
  for (axon::idx_t i = 0; i < n; ++i) {
    original_data[i] = x.at(flat_to_indices(i, x.shape()));
  }

  std::vector<float> numeric_data(n);
  for (axon::idx_t i = 0; i < n; ++i) {
    std::vector<float> plus_data = original_data;
    std::vector<float> minus_data = original_data;
    plus_data[i] += epsilon;
    minus_data[i] -= epsilon;

    axon::Tensor x_plus = axon::Tensor::from_data(plus_data, x.shape());
    axon::Tensor x_minus = axon::Tensor::from_data(minus_data, x.shape());

    const float loss_plus = f(x_plus).sum().item();
    const float loss_minus = f(x_minus).sum().item();

    numeric_data[i] = (loss_plus - loss_minus) / (2.0f * epsilon);
  }

  const axon::Tensor numeric_solution =
      axon::Tensor::from_data(numeric_data, x.shape());
  const axon::Tensor difference = (numeric_solution - analytic).abs();
  const axon::Tensor tolerance_tensor =
      axon::Tensor::ones(difference.shape()) * tolerance;

  return difference <= tolerance_tensor;
}

TEST_CASE("Gradient check", "[GradCheck]") {
  axon::Tensor x = axon::Tensor::from_data({1.0f, 2.0f}, {2});

  SECTION("mul") {
    auto f = [](axon::Tensor x) { return x * x; };
    REQUIRE(grad_check(f, x, 1e-4f, 1e-2f));
  }
  SECTION("add") {
    auto f = [](axon::Tensor x) { return x + x; };
    REQUIRE(grad_check(f, x, 1e-4f, 1e-2f));
  }
  SECTION("log") {
    auto f = [](axon::Tensor x) { return x.log(); };
    REQUIRE(grad_check(f, x, 1e-4f, 1e-2f));
  }
  SECTION("exp") {
    auto f = [](axon::Tensor x) { return x.exp(); };
    REQUIRE(grad_check(f, x, 1e-4f, 1e-2f));
  }
  SECTION("relu") {
    auto f = [](axon::Tensor x) { return x.relu(); };
    REQUIRE(grad_check(f, x, 1e-4f, 1e-2f));
  }
  SECTION("sum(dim)") {
    axon::Tensor x2 = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto f = [](axon::Tensor x) { return x.sum(1, true); };
    REQUIRE(grad_check(f, x2, 1e-4f, 1e-2f));
  }
  SECTION("matmul") {
    const axon::Tensor b =
        axon::Tensor::from_data({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    axon::Tensor x2 = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto f = [b](axon::Tensor x) { return x.matmul(b); };

    REQUIRE(grad_check(f, x2, 1e-2f, 1e-2f));
  }
  SECTION("matmul self") {
    axon::Tensor x2 = axon::Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto f = [](axon::Tensor x) { return x.matmul(x); };
    REQUIRE(grad_check(f, x2, 1e-2f, 1e-2f));
  }

  SECTION("softmax") {
    axon::Tensor x2 = axon::Tensor::from_data({1.0f, 2.0f, 3.0f}, {1, 3});
    auto f = [](axon::Tensor x) { return axon::softmax(x); };
    REQUIRE(grad_check(f, x2, 1e-3f, 1e-2f));
  }

  SECTION("log_softmax") {
    axon::Tensor x = axon::Tensor::from_data({1.0f, 2.0f, 3.0f}, {1, 3});
    auto f = [](axon::Tensor x) { return axon::log_softmax(x); };
    REQUIRE(grad_check(f, x, 1e-3f, 1e-2f));
  }

  SECTION("cross_entropy_loss") {
    axon::Tensor logits =
        axon::Tensor::from_data({2.0f, 1.0f, 0.1f, 0.5f, 2.0f, 0.3f}, {2, 3});
    axon::Tensor targets =
        axon::Tensor::from_data({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {2, 3});
    auto f = [&targets](axon::Tensor x) {
      return axon::cross_entropy_loss(x, targets);
    };
    REQUIRE(grad_check(f, logits, 1e-3f, 1e-2f));
  }
}