/**
 * @file functional.cpp
 * @brief Implementation of axon's stateless activation and loss functions.
 * @author Mika Brückner
 * @date 2026-06-20
 */

#include "axon/functional.hpp"

namespace axon {
Tensor softmax(const Tensor& logits) {
  const idx_t batches = logits.shape()[0];
  std::vector<float> sample_maxes;

  for (idx_t i = 0; i < batches; i++) {
    Tensor sample = logits[i];
    sample_maxes.push_back(sample.max());
  }

  Tensor max_tensor = Tensor::from_data(sample_maxes, {batches, 1});
  Tensor shifted = logits - max_tensor;
  Tensor exp_shifted = shifted.exp();

  Tensor sum_tensor = exp_shifted.sum(1, true);
  return exp_shifted / sum_tensor;
}

Tensor log_softmax(const Tensor& logits) {
  const idx_t batches = logits.shape()[0];
  std::vector<float> sample_maxes;

  for (idx_t i = 0; i < batches; i++) {
    Tensor sample = logits[i];
    sample_maxes.push_back(sample.max());
  }

  Tensor max_tensor = Tensor::from_data(sample_maxes, {batches, 1});
  Tensor shifted = logits - max_tensor;

  Tensor sum_tensor = shifted.exp().sum(1, true);
  return shifted - sum_tensor.log();
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
  return -(targets * log_softmax(logits)).sum(1, false).mean();
}

Tensor cross_entropy_loss(const Tensor& logits,
                          const std::vector<idx_t>& targets) {
  const idx_t n_samples = logits.shape()[0];
  const idx_t n_classes = logits.shape()[1];

  // TODO find better way to keep gradient
  std::vector<float> one_hot_data(n_samples * n_classes, 0.0f);
  for (idx_t i = 0; i < n_samples; i++) {
    one_hot_data[i * n_classes + targets[i]] = 1.0f;
  }
  Tensor one_hot = Tensor::from_data(one_hot_data, {n_samples, n_classes});

  return cross_entropy_loss(logits, one_hot);
}

Tensor mse_loss(const Tensor& logits, const Tensor& targets) {
  const Tensor error = (logits - targets);
  const Tensor squared_error = error * error;
  return squared_error.mean();
}

Tensor accuracy(const Tensor& predictions, const Tensor& targets) {
  return 1.0f - (predictions - targets).abs().gt(0.5f).sum() /
                    static_cast<float>(predictions.num_elements());
}
}  // namespace axon
