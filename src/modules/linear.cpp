/**
 * @file linear.cpp
 * @brief Implementation of the axon::nn::Linear module.
 * @author Mika Brückner
 * @date 2026-06-17
 */

#include "axon/modules/linear.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <random>

#include "utils.hpp"

namespace axon::nn {
Linear::Linear(idx_t in_features, idx_t out_features, unsigned seed)
    : weights_(Tensor::zeros({in_features, out_features})),
      bias_(Tensor::zeros({out_features})) {
  float bound = 1.0f / std::sqrt(static_cast<float>(in_features));

  std::mt19937 rng(seed);
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
  std::vector<Tensor> params;
  params.reserve(2);
  params.push_back(weights_.shared_autograd_copy());
  params.push_back(bias_.shared_autograd_copy());
  return params;
}

void Linear::set_parameters(std::vector<Tensor> params) {
  const idx_t params_size = params.size();
  if (params_size != 2) {
    throw std::out_of_range(std::format(
        "cannot set parameters because too many parameter-tensors were "
        "given: expected: 2 (weights, bias), got {}",
        params_size));
  }

  const std::vector<idx_t> first_shape = params[0].shape();
  const std::vector<idx_t> weight_shape = weights_.shape();

  if (first_shape != weight_shape) {
    throw std::out_of_range(
        std::format("cannot set parameters because weight shape"
                    "does not fit: expected: {}, got {}",
                    utils::vector_to_string(weight_shape),
                    utils::vector_to_string(first_shape)));
  }
  weights_ = params[0];

  const std::vector<idx_t> second_shape = params[1].shape();
  const std::vector<idx_t> bias_shape = bias_.shape();

  if (second_shape != bias_shape) {
    throw std::out_of_range(
        std::format("cannot set parameters because bias shape"
                    "does not fit: expected: {}, got {}",
                    utils::vector_to_string(bias_shape),
                    utils::vector_to_string(second_shape)));
  }
  bias_ = params[1];
}
std::string Linear::to_string() const {
  const std::vector<idx_t> weights_shape = weights_.shape();
  const idx_t in_features = weights_shape[0];
  const idx_t out_features = weights_shape[1];
  return std::format("Linear(in_features = {},  out_features = {})",
                     in_features,
                     out_features);
}

}  // namespace axon::nn