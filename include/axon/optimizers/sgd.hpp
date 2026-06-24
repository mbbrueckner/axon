/**
 * @file sgd.hpp
 * @brief Declaration of the axon::optimizer::SGD optimizer, which implements
 *        vanilla stochastic gradient descent.
 * @author Mika Brückner
 * @date 2026-06-24
 */

#pragma once

#include "optimizer.hpp"

namespace axon::optimizer {
class SGD;
}

/**
 * @brief Stochastic gradient descent optimizer.
 *
 * Updates each parameter by stepping against its gradient, scaled by a
 * fixed learning rate:
 * @f[ \theta \leftarrow \theta - \eta \, \nabla_\theta @f]
 *
 * @see axon::optimizer::Optimizer for the base interface and typical usage
 *      within a training loop.
 */
class axon::optimizer::SGD : public Optimizer {
 private:
  /// @brief The step size @f$\eta@f$ applied to each gradient update.
  float learning_rate_;

 public:
  /**
   * @brief Constructs an SGD optimizer for the given parameters.
   * @param params Trainable tensors to optimize, typically from
   *        @c model.parameters().
   * @param learning_rate The step size applied to each gradient update.
   */
  explicit SGD(const std::vector<Tensor>& params,
               const float learning_rate = 0.01f)
      : Optimizer(params), learning_rate_(learning_rate) {};

  /**
   * @brief Applies one SGD update to every managed parameter.
   *
   * For each parameter @p p, computes @c p - learning_rate * p.grad() and
   * writes the result back in-place. Call after @c loss.backward().
   */
  void step() override;

  /// @copydoc axon::optimizer::Optimizer::zero_grad
  void zero_grad() override;
};