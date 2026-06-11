/**
 * @file autograd.hpp
 * @brief Declaration of the axon::GradFn and axon::AutogradMeta structs that
 *        form the nodes of the reverse-mode autograd computation graph.
 * @author Mika Brückner
 * @date 2026-06-11
 */

#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "axon/constants.hpp"

namespace axon {
class Tensor;
struct AutogradMeta;
struct GradFn;

}  // namespace axon

/**
 * @struct axon::GradFn
 * @brief The backward operation of a single node in the autograd graph.
 *
 * A GradFn bundles the closure that propagates the incoming gradient to the
 * operation's operands together with the autograd metadata of those operands,
 * which are the node's parents in the computation graph.
 */
struct axon::GradFn {
  /// @brief Propagates the gradient flowing into this node to its inputs.
  /// @param grad The gradient with respect to this node's output.
  std::function<void(const axon::Tensor&)> backward;
  /// @brief Autograd metadata of the operands this operation depends on.
  std::vector<std::shared_ptr<axon::AutogradMeta>> inputs;
};

/**
 * @struct axon::AutogradMeta
 * @brief Per-tensor autograd state attached to tensors that require gradients.
 *
 * Holds the accumulated gradient for the associated tensor and, for tensors
 * produced by a differentiable operation, the GradFn used to traverse the
 * computation graph during the backward pass.
 */
struct axon::AutogradMeta {
  /**
   * @brief Constructs autograd metadata with a zero-initialized gradient.
   * @param shape Shape of the associated tensor; the gradient takes the same
   *        shape.
   */
  explicit AutogradMeta(const std::vector<idx_t>& shape);
  /// @brief Accumulated gradient of the associated tensor (held by pointer so
  /// it can be shared and reassigned).
  std::shared_ptr<axon::Tensor> grad;
  /// @brief Backward operation that produced the associated tensor, or @c
  /// nullptr for leaf tensors.
  std::shared_ptr<axon::GradFn> grad_fn_;
};
