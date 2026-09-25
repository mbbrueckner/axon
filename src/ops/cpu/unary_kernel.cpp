/**
 * @file unary_kernel.cpp
 * @brief CPU implementations of the elementwise unary kernels.
 * @author Mika Brückner
 * @date 2026-09-25
 */

#include "ops/unary_kernel.hpp"

#include <cmath>

#include "ops/cpu/loops.hpp"

namespace axon::internal {

void relu_kernel(const TensorIterator& iter) {
  cpu_kernel(iter, [](float x) { return x > 0 ? x : 0; });
}

void log_kernel(const TensorIterator& iter) {
  cpu_kernel(iter, [](float x) { return std::log(x); });
}

void exp_kernel(const TensorIterator& iter) {
  cpu_kernel(iter, [](float x) { return std::exp(x); });
}

void abs_kernel(const TensorIterator& iter) {
  cpu_kernel(iter, [](float x) { return std::abs(x); });
}

}  // namespace axon::internal
