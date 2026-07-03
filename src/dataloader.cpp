/**
 * @file dataloader.cpp
 * @brief Implementation of the axon::DataLoader class, which wraps a Dataset
 *        and provides batched, optionally shuffled iteration.
 * @author Mika Brückner
 * @date 2026-07-01
 */

#include "axon/dataloader.hpp"

#include <algorithm>
#include <span>

#include "axon/functional.hpp"
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

void DataLoader::shuffle_indices() { std::ranges::shuffle(indices_, rng_); }

Batch DataLoader::Iterator::operator*() const {
  const idx_t batch_size = loader_->batch_size_;
  const auto batch_begin = loader_->indices_.begin() + batch_idx_ * batch_size;
  const std::span<idx_t> sample_indices(batch_begin, batch_begin + batch_size);

  std::vector<Tensor> feat_tensors, label_tensors;
  feat_tensors.reserve(batch_size);
  label_tensors.reserve(batch_size);

  for (idx_t idx : sample_indices) {
    feat_tensors.push_back(loader_->dataset_[idx].first);
    label_tensors.push_back(loader_->dataset_[idx].second);
  }

  return {stack(feat_tensors), stack(label_tensors)};
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
  batch_idx_++;
  return *this;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
  return batch_idx_ != other.batch_idx_;
}

}  // namespace axon
