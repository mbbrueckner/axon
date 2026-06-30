/**
 * @file mnist.hpp
 * @brief Declaration of the axon::datasets::MNIST dataset class,
 *        which loads and provides access to the MNIST handwritten digit
 *        dataset from IDX binary files.
 * @author Mika Brückner
 * @date 2026-06-28
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "axon/constants.hpp"
#include "axon/tensor.hpp"
#include "dataset.hpp"

namespace axon::datasets {
class MNIST;
}

/**
 * @brief MNIST handwritten digit dataset.
 *
 * Loads images and labels from IDX binary files into memory on construction.
 * Images are normalized to [0, 1] and flattened to shape [n, 784].
 * Labels are kept sparse, shape [n].
 *
 * Typical usage:
 * @code
 * axon::datasets::MNIST dataset("data/train-images-idx3-ubyte",
 *                               "data/train-labels-idx1-ubyte");
 * auto [input, target] = dataset[0];
 * @endcode
 */
class axon::datasets::MNIST : public Dataset {
 private:
  /// @brief All images, shape [n, 784], normalized to [0, 1].
  std::optional<Tensor> images_;

  /// @brief All labels, shape [n], sparse.
  std::optional<Tensor> labels_;

 public:
  /**
   * @brief Loads MNIST images and labels from IDX binary files.
   *
   * Reads both files completely into memory, normalizes pixel values
   * to [0, 1] and keeps the labels sparse.
   *
   * @param images_path Path to the IDX image file (e.g.
   *        @c train-images-idx3-ubyte).
   * @param labels_path Path to the IDX label file (e.g.
   *        @c train-labels-idx1-ubyte).
   * @throws std::runtime_error if either file cannot be opened or the
   *         magic number is invalid.
   */
  explicit MNIST(const std::string& images_path,
                 const std::string& labels_path);

  /// @copydoc axon::datasets::Dataset::operator[]()
  Sample operator[](axon::idx_t idx) const override;

  /// @copydoc axon::datasets::Dataset::size()
  [[nodiscard]] axon::idx_t size() const override;
};