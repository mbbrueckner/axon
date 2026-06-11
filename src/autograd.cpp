//
// Created by Mika Brückner on 09.06.26.
//

#include "autograd.hpp"

#include "axon/tensor.hpp"

namespace axon {

AutogradMeta::AutogradMeta(const std::vector<idx_t>& shape)
    : grad(std::make_shared<Tensor>(shape)) {}
}  // namespace axon
