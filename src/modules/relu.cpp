/**
 * @file relu.cpp
 * @brief Implementation of the axon::nn::ReLU module.
 * @author Mika Brückner
 * @date 2026-06-19
 */

#include "axon/modules/relu.hpp"

namespace axon::nn {

Tensor ReLU::forward(const Tensor& input) { return input.relu(); }

std::vector<Tensor> ReLU::parameters() { return {}; }

void ReLU::set_parameters(std::vector<Tensor> params) {
  if (!params.empty()) {
    throw std::logic_error("ReLU has no parameters to be set");
  }
}

}  // namespace axon::nn