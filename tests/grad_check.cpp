/**
 * @file grad_check.cpp
 * @brief  Sanity check (finite difference check) for the autograd backwards
 * pass.
 * @author Mika Brückner
 * @date 2026-06-14
 */

#include <functional>
#include <iostream>

#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

/**
 * @brief Finite difference gradient check for a scalar function f.
 *
 * Compares the analytically computed gradient (via backward()) against
 * a numerical approximation using the central difference formula:
 *   (f(x + \epsilon) - f(x - \epsilon)) / 2 \epsilon
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
  const axon::Tensor numeric_solution =
      (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon);

  x.requires_grad_(true);
  axon::Tensor y = f(x.shared_autograd_copy());
  y.backward();
  const axon::Tensor analytic_solution = x.grad();

  const axon::Tensor difference = (numeric_solution - analytic_solution).abs();
  const axon::Tensor tolerance_tensor =
      axon::Tensor::ones(difference.shape()) * tolerance;

  return difference <= tolerance_tensor;
}

TEST_CASE("Gradient check", "[GradCheck]") {
  axon::Tensor x = axon::Tensor::from_data({1.0f, 2.0f}, {2});
  axon::Tensor y(x);
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
  SECTION("ReLU") {
    auto f = [](axon::Tensor x) { return x.relu(); };
    REQUIRE(grad_check(f, x, 1e-4f, 1e-2f));
  }
  // TODO matmul
}
