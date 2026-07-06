/**
 * @file sgd.cpp
 * @brief Implementation of the axon::optimizer::SGD optimizer.
 * @author Mika Brückner
 * @date 2026-06-24
 */

#include "axon/optimizers/sgd.hpp"

#include <format>

namespace axon::optimizer {
void SGD::step() {
  for (Tensor& param : params_) {
    Tensor update = param - learning_rate_ * param.grad();
    param.set_data(update.data());
  }
}
std::string SGD::to_string() const {
  return std::format("SGD(lr={})", learning_rate_);
}

void SGD::zero_grad() {
  for (Tensor& param : params_) {
    param.grad_ref().set_data(Tensor::zeros(param.shape()).data());
  }
}

}  // namespace axon::optimizer