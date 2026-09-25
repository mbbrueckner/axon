/**
 * @file unary_kernel.hpp
 * @brief Backend entry points for the elementwise unary operations.
 * @author Mika Brückner
 * @date 2026-09-24
 */

#pragma once

namespace axon::internal {

class TensorIterator;

/**
 * @brief Writes the rectified linear unit of operand 1 into operand 0.
 *
 *
 * @param iter Plan describing the output and the input.
 */
void relu_kernel(const TensorIterator& iter);

/**
 * @brief Writes the log of operand 1 into operand 0.
 *
 * @param iter Plan describing the output and the input.
 */
void log_kernel(const TensorIterator& iter);

/**
 * @brief Writes the exp of operand 1 into operand 0.
 *
 * @param iter Plan describing the output and the input.
 */
void exp_kernel(const TensorIterator& iter);

/**
 * @brief Writes the exp of operand 1 into operand 0.
 *
 * @param iter Plan describing the output and the input.
 */
void abs_kernel(const TensorIterator& iter);

}  // namespace axon::internal
