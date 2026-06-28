/**
 * @file dataset.hpp
 * @brief Declaration of the axon::datasets::Dataset abstract base class,
 *        the common interface for indexed datasets.
 * @author Mika Brückner
 * @date 2026-06-27
 */

#pragma once

#include <utility>

#include "axon/constants.hpp"
#include "axon/tensor.hpp"

namespace axon::datasets {

/// @brief A single (input, target) pair returned by a dataset.
using Sample = std::pair<axon::Tensor, axon::Tensor>;

class Dataset;
}  // namespace axon::datasets

/**
 * @brief Abstract base class for all datasets.
 *
 * A Dataset provides indexed access to individual (input, target) samples.
 * Derived classes implement the concrete data loading and preprocessing
 * logic (e.g. reading MNIST IDX files).
 *
 * Typical usage via a DataLoader:
 * @code
 * MnistDataset dataset("data/train-images.idx3-ubyte",
 *                      "data/train-labels.idx1-ubyte");
 * for (idx_t i = 0; i < dataset.size(); i++) {
 *     auto [input, target] = dataset[i];
 * }
 * @endcode
 */
class axon::datasets::Dataset {
 public:
  /**
   * @brief Returns the sample at the given index.
   * @param idx Zero-based index of the sample to retrieve.
   * @return The (input, target) pair at @p idx.
   * @throws std::out_of_range if @p idx >= size().
   */
  virtual Sample operator[](axon::idx_t idx) const = 0;

  /**
   * @brief Returns the total number of samples in the dataset.
   * @return The number of samples.
   */
  [[nodiscard]] virtual axon::idx_t size() const = 0;

  /// @brief Virtual destructor to allow safe polymorphic deletion.
  virtual ~Dataset() = default;
};