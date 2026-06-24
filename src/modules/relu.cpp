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

}  // namespace axon::nn