/**
 * @file relu.cpp
 * @brief Implementation of the axon::ReLU module.
 * @author Mika Brückner
 * @date 2026-06-19
 */

#include "axon/relu.hpp"

namespace axon {

Tensor ReLU::forward(const Tensor& input) { return input.relu(); }

std::vector<Tensor> ReLU::parameters() { return {}; }

}  // namespace axon