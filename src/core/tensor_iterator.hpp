/**
 * @file tensor_iterator.hpp
 * @brief Iteration plan describing how an operation traverses its operands.
 * @author Mika Brückner
 * @date 2026-09-18
 */

#pragma once
#include <vector>

#include "axon/constants.hpp"

namespace axon {
class Tensor;
namespace internal {
class TensorIterator;
}  // namespace internal
}  // namespace axon

/**
 * @class axon::internal::TensorIterator
 * @brief Describes how an operation traverses its operands.
 *
 * Holds the shape shared by all operands together with, per operand, a base
 * pointer and the strides needed to walk it. Operand 0 is the output, the
 * remaining operands are inputs.
 */
class axon::internal::TensorIterator {
  /// @brief One participant of the operation: where its data starts and how
  ///        to step through it.
  struct Operand {
    /// Base pointer, with the tensor's offset already applied.
    float* data;
    /// Step per dimension, in elements, in terms of shape_.
    std::vector<idx_t> strides;
  };

 private:
  /// Shape shared by all operands.
  std::vector<idx_t> shape_;
  /// Output first, then the inputs.
  std::vector<Operand> operands_;
  /// Product of shape_.
  idx_t num_elements_;

 public:
  /**
   * @brief Builds the iteration plan for an elementwise unary operation.
   * @param output Destination tensor; becomes operand 0.
   * @param input Source tensor; becomes operand 1.
   * @return The plan describing the traversal of @p output and @p input.
   * @pre @p output and @p input have the same shape.
   */
  static TensorIterator unary_op(Tensor& output, const Tensor& input);

  /// @brief Total number of elements to traverse.
  [[nodiscard]] idx_t num_elements() const;

  /// @brief Number of dimensions of the shared shape.
  [[nodiscard]] idx_t num_dim() const;

  /// @brief Number of participating operands, outputs included.
  [[nodiscard]] idx_t num_operands() const;

  /// @brief The shape shared by all operands.
  [[nodiscard]] const std::vector<idx_t>& shape() const;

  /// @brief Step per dimension for @p operand, in elements.
  [[nodiscard]] const std::vector<idx_t>& stride(idx_t operand) const;

  /// @brief Base pointer of @p operand, with its tensor's offset applied.
  /// @note Non-const on purpose: operand 0 is written to.
  [[nodiscard]] float* data(idx_t operand) const;
};
