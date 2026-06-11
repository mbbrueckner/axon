//
// Created by Mika Brückner on 09.06.26.
//

#pragma once
#include <functional>
#include <memory>

#include "axon/tensor.hpp"

namespace axon {

struct GradFn {
  std::function<void(const Tensor&)> backward;
};

struct AutogradMeta {
  explicit AutogradMeta(const std::vector<idx_t>& shape)
      : grad(shape) {}

  Tensor grad;
  std::shared_ptr<GradFn> grad_fn_;
};

}  // namespace axon
