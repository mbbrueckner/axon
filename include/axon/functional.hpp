/**
 * @file functional.hpp
 * @brief Stateless functional operations: activation and loss functions.
 * @author Mika Brückner
 * @date 2026-06-20
 */

#pragma once
#include "tensor.hpp"

namespace axon {

/**
 * @brief Computes the softmax of @p logits along the last dimension.
 *
 * Normalizes each row into a probability distribution:
 * @f[ \mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} @f]
 *
 * The computation is shifted by the per-row maximum for numerical
 * stability, so large logits do not overflow @c exp.
 *
 * @param logits Input tensor of shape @c {batch, classes}.
 * @return Tensor of the same shape whose rows sum to 1.
 */
Tensor softmax(const Tensor& logits);

/**
 * @brief Computes the natural logarithm of the softmax of @p logits.
 *
 * Equivalent to @c softmax(logits).log() but evaluated in the log domain
 * for numerical stability, which avoids underflow from small
 * probabilities. Preferred as the input to @ref cross_entropy_loss.
 *
 * @param logits Input tensor of shape @c {batch, classes}.
 * @return Tensor of the same shape holding the log-probabilities.
 */
Tensor log_softmax(const Tensor& logits);

/**
 * @brief Computes the mean cross-entropy loss over a batch.
 *
 * Combines @ref log_softmax with the negative log-likelihood, averaged
 * across the batch:
 * @f[ \mathcal{L} = -\frac{1}{N} \sum_{n} \sum_{c} t_{nc}
 *     \log \mathrm{softmax}(x_n)_c @f]
 *
 * @param logits Unnormalized scores of shape @c {batch, classes}.
 * @param targets One-hot target distribution of the same shape as
 *        @p logits.
 * @return Scalar tensor holding the mean loss.
 */
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);

/**
 * @brief Computes the sparse cross-entropy loss between logits and integer
 *        class indices.
 *
 * Equivalent to @c cross_entropy_loss(logits, one_hot(targets)) but more
 * memory-efficient. For each sample, only the log-probability of the true
 * class is used.
 *
 * @param logits Unnormalized class scores of shape @c [batch, num_classes].
 * @param targets Integer class indices of length @c batch, where each value
 *        is in @c [0, num_classes).
 * @return A scalar tensor containing the mean cross-entropy loss.
 * @throws std::out_of_range if @p targets.size() != logits.shape()[0] or
 *         any index is out of @c [0, num_classes).
 */
Tensor cross_entropy_loss(const Tensor& logits,
                          const std::vector<idx_t>& targets);

/**
 * @brief Computes the mean squared error between @p logits and @p targets.
 *
 * @f[ \mathcal{L} = \big(\mathrm{mean}(x - t)\big)^2 @f]
 *
 * @param logits Predicted values.
 * @param targets Ground-truth values broadcastable to @p logits.
 * @return Scalar tensor holding the loss.
 */
Tensor mse_loss(const Tensor& logits, const Tensor& targets);

/**
 * @brief Computes the fraction of predictions that match their targets.
 *
 * A prediction counts as correct when it lies within 0.5 of the target
 * value, i.e. @c |prediction - target| <= 0.5. The result is the mean
 * over all elements:
 * @f[ \mathrm{acc} = 1 - \frac{1}{N}
 *     \sum_{i} \mathbb{1}\big[\,|p_i - t_i| > 0.5\,\big] @f]
 *
 * @param logits Predicted values.
 * @param targets Ground-truth values of the same shape as @p logits.
 * @return Scalar tensor in @c [0, 1] holding the accuracy.
 */
Tensor accuracy(const Tensor& predictions, const Tensor& targets);

}  // namespace axon