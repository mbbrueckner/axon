/**
 * @file autograd.cpp
 * @brief Implementation of the axon::AutogradMeta struct.
 * @author Mika Brückner
 * @date 2026-06-09
 */

#include "autograd.hpp"

#include "axon/tensor.hpp"

namespace axon {

AutogradMeta::AutogradMeta(const std::vector<idx_t>& shape)
    : grad(std::make_shared<Tensor>(Tensor::zeros(shape))) {}
}  // namespace axon
