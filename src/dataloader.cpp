/**
 * @file dataloader.cpp
 * @brief Implementation of the axon::DataLoader class, which wraps a Dataset
 *        and provides batched, optionally shuffled iteration.
 * @author Mika Brückner
 * @date 2026-07-01
 */

#include "axon/dataloader.hpp"

#include <algorithm>

namespace axon {

DataLoader::DataLoader(const datasets::Dataset& dataset,
                       idx_t batch_size,
                       bool shuffle)
    : dataset_(dataset),
      batch_size_(batch_size),
      shuffle_(shuffle),
      indices_(dataset.size()) {
  std::iota(indices_.begin(), indices_.end(), 0);
}

DataLoader::Iterator DataLoader::begin() {
  if (shuffle_) shuffle_indices();

  return Iterator{this, 0};
}
void DataLoader::shuffle_indices() {
  std::shuffle(indices_.begin(), indices_.end(), rng_);
}

}  // namespace axon
