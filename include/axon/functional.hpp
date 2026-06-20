#pragma once
#include "tensor.hpp"

namespace axon {
Tensor softmax(const Tensor& logits);
// TODO Tensor log_softmax(const Tensor& logits);
// TODO Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);
}  // namespace axon