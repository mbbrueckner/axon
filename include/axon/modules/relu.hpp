/**
 * @file relu.hpp
 * @brief Declaration of the axon::nn::ReLU module, an activation layer
 *        applying the Rectified Linear Unit (ReLU).
 * @author Mika Brückner
 * @date 2026-06-17
 */

#pragma once

#include "module.hpp"

namespace axon::nn {
class ReLU;
}

/**
 * @brief ReLU activation layer.
 *
 * Applies the Rectified Linear Unit elementwise: @c f(x) = max(0, x).
 * Has no learnable parameters.
 */
class axon::nn::ReLU : public Module {
 public:
  /**
   * @brief Applies ReLU elementwise to the input.
   * @param input The input tensor.
   * @return A tensor of the same shape with negative values clamped to zero.
   */
  Tensor forward(const Tensor& input) override;

  /// @return An empty vector, since ReLU has no learnable parameters.
  std::vector<Tensor> parameters() override;
};