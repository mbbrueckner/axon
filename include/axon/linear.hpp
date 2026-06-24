/**
 * @file linear.hpp
 * @brief Declaration of the axon::nn::Linear module, a fully connected layer
 *        applying an affine transformation @c y = x * W + b.
 * @author Mika Brückner
 * @date 2026-06-17
 */

#pragma once

#include <random>

#include "axon/constants.hpp"
#include "axon/module.hpp"
#include "axon/tensor.hpp"

namespace axon::nn {
class Linear;
}

/**
 * @brief Fully connected (dense) layer.
 *
 * Applies an affine transformation to the input tensor: @c y = x * W + b,
 * where @c W is a learnable weight matrix and @c b a learnable bias vector.
 */
class axon::nn::Linear : public Module {
 public:
  /**
   * @brief Constructs a linear layer with the given dimensions.
   *
   * @param in_features Size of each input sample.
   * @param out_features Size of each output sample.
   * @param seed Optional randomness seed.
   */
  explicit Linear(idx_t in_features,
                  idx_t out_features,
                  unsigned seed = std::random_device{}());

  /**
   * @brief Applies the affine transformation to the input.
   *
   * @param input The input tensor of shape @c [..., in_features].
   * @return The transformed tensor of shape @c [..., out_features].
   */
  Tensor forward(const Tensor& input) override;

  /**
   * @brief Collects the layer's trainable parameters.
   *
   * @return The weight and bias tensors.
   */
  std::vector<Tensor> parameters() override;

 private:
  Tensor weights_;  ///< Learnable weight matrix of shape [in_features,
                    ///< out_features].
  Tensor bias_;     ///< Learnable bias vector of shape [out_features].
};