//
// Created by Mika Brückner on 09.06.26.
//

#pragma once
#include <memory>

#include "axon/tensor.hpp"

namespace axon {
struct AutogradMeta {
  Tensor grad;
  // TODO GradFn std::shared_ptr<GradFn> grad_fn_;
};

}  // namespace axon
