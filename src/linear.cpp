/**
 * @file linear.cpp
 * @brief Implementation of the axon::Linear module.
 * @author Mika Brückner
 * @date 2026-06-17
 */

#include "axon/linear.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace axon {
Linear::Linear(idx_t in_features, idx_t out_features)
    : weights_(Tensor::zeros({in_features, out_features})),
      bias_(Tensor::zeros({out_features})) {
  float bound = 1.0f / std::sqrt(static_cast<float>(in_features));

  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-bound, bound);

  std::vector<float> w_data(in_features * out_features);
  std::ranges::generate(w_data, [&] { return dist(rng); });

  weights_ = Tensor::from_data(w_data, {in_features, out_features});
  weights_.requires_grad_(true);

  bias_ = Tensor::zeros({out_features});
  bias_.requires_grad_(true);
}

Tensor Linear::forward(const Tensor& input) {
  return input.matmul(weights_) + bias_;
}

std::vector<Tensor> Linear::parameters() {
  return {weights_.shared_autograd_copy(), bias_.shared_autograd_copy()};
}

}  // namespace axon