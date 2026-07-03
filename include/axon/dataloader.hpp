/**
 * @file dataloader.hpp
 * @brief Declaration of the axon::DataLoader class, which wraps a Dataset
 *        and provides batched, optionally shuffled iteration.
 * @author Mika Brückner
 * @date 2026-07-01
 */

#pragma once

#include <random>
#include <vector>

#include "axon/constants.hpp"
#include "axon/datasets/dataset.hpp"
#include "axon/tensor.hpp"

namespace axon {
/**
 * @brief A batch of stacked (input, target) tensors.
 *
 * Unlike @c datasets::Sample, which holds a single example, a Batch holds
 * @c batch_size stacked examples: the input has shape
 * @c [batch_size, ...] and the target has shape @c [batch_size, ...].
 * Produced by @c DataLoader::Iterator::operator*() via @c stack().
 */
using Batch = std::pair<Tensor, Tensor>;

class DataLoader;
}  // namespace axon

/**
 * @brief Wraps a Dataset and provides batched, optionally shuffled iteration.
 *
 * DataLoader groups individual samples from a Dataset into mini-batches
 * and optionally shuffles the sample order at the start of each epoch.
 * Iteration follows standard C++ range semantics:
 *
 * @code
 * axon::DataLoader loader(dataset, 32, true);
 * for (auto [input, target] : loader) {
 *     // input:  Tensor of shape [batch_size, 784]
 *     // target: Tensor of shape [batch_size]
 * }
 * @endcode
 */
class axon::DataLoader {
 public:
  /**
   * @brief Constructs a DataLoader for the given dataset.
   *
   * @param dataset    The dataset to iterate over. Must outlive the
   *                   DataLoader.
   * @param batch_size Number of samples per batch.
   * @param shuffle    If @c true, the sample order is randomized at the
   *                   start of each epoch via @c begin().
   */
  DataLoader(const datasets::Dataset& dataset,
             idx_t batch_size,
             bool shuffle = true);

  /**
   * @brief A forward iterator over batches.
   *
   * Each increment advances by one batch. Dereferencing yields a
   * @c std::pair<Tensor, Tensor> whose first element contains the
   * stacked inputs and whose second element contains the stacked targets
   * for the current batch.
   */
  struct Iterator {
    /**
     * @brief Returns the current batch as a (input, target) pair.
     * @return Pair of tensors with shape @c [batch_size, ...].
     */
    Batch operator*() const;

    /// @brief Advances to the next batch.
    Iterator& operator++();

    /**
     * @brief Tests whether two iterators refer to different positions.
     * @param other The iterator to compare against.
     * @return @c true if the iterators are at different batch indices.
     */
    bool operator!=(const Iterator& other) const;

    /// @brief Pointer to the owning DataLoader.
    DataLoader* loader_;
    /// @brief Index of the current batch.
    idx_t batch_idx_;
  };

  /**
   * @brief Returns an iterator to the first batch.
   *
   * If shuffling is enabled, the sample indices are reshuffled here,
   * so each call to @c begin() starts a new randomized epoch.
   *
   * @return Iterator pointing to the first batch.
   */
  Iterator begin();

  /// @brief Returns a sentinel iterator marking the end of the dataset.
  Iterator end();

 private:
  /// @brief The dataset being iterated over.
  const datasets::Dataset& dataset_;
  /// @brief Number of samples per batch.
  idx_t batch_size_;
  /// @brief Whether to shuffle sample order each epoch.
  bool shuffle_;
  /// @brief Permuted sample indices used for shuffling.
  std::vector<idx_t> indices_;
  /// @brief Mersenne Twister random number generator for shuffling.
  ///        Stored as a member so the sequence differs across epochs.
  std::mt19937 rng_;

  /**
   * @brief Randomly permutes @c indices_ using @c rng_.
   *
   * Called by @c begin() at the start of each epoch when shuffling is
   * enabled.
   */
  void shuffle_indices();
};