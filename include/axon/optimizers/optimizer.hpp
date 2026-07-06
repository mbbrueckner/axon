/**
 * @file optimizer.hpp
 * @brief Declaration of the axon::Optimizer abstract base class, the common
 *        interface for gradient-based parameter optimizers.
 * @author Mika Brückner
 * @date 2026-06-24
 */

#pragma once
#include <algorithm>
#include <vector>

#include "../tensor.hpp"

namespace axon::optimizer {
class Optimizer;
}

/**
 * @brief Abstract base class for all parameter optimizers.
 *
 * An Optimizer holds references to a model's trainable parameters and
 * updates them after each backward pass. Derived classes implement the
 * concrete update rule (e.g. SGD, Adam).
 *
 * Typical usage in a training loop:
 * @code
 * optimizer.zero_grad();
 * auto loss = criterion(model.forward(input), target);
 * loss.backward();
 * optimizer.step();
 * @endcode
 */
class axon::optimizer::Optimizer {
 protected:
  /// @brief The trainable parameters managed by this optimizer.
  std::vector<Tensor> params_;

 public:
  /**
   * @brief Constructs an optimizer for the given parameters.
   * @param params Trainable tensors to optimize, typically from
   *        @c model.parameters().
   */
  explicit Optimizer(const std::vector<Tensor>& params) {
    params_.reserve(params.size());
    std::ranges::transform(
        params, std::back_inserter(params_), [](const Tensor& p) {
          return p.shared_autograd_copy();
        });
  }

  /**
   * @brief Updates all parameters according to the optimizer's update rule.
   *
   * Called after @c loss.backward() to apply the accumulated gradients.
   */
  virtual void step() = 0;

  /**
   * @brief Resets all parameter gradients to zero.
   *
   * Must be called before each forward pass to prevent gradient
   * accumulation across iterations.
   */
  virtual void zero_grad() = 0;

  /**
   * @brief Returns a human-readable description of the optimizer.
   * @return A string describing the optimizer's type and hyperparameters.
   */
  virtual std::string to_string() const = 0;

  /// @brief Virtual destructor to allow safe polymorphic deletion.
  virtual ~Optimizer() = default;
};