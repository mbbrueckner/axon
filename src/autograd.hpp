//
// Created by Mika Brückner on 09.06.26.
//

#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "../include/axon/constants.hpp"

namespace axon {
class Tensor;
struct AutogradMeta;
struct GradFn;

}  // namespace axon

struct axon::GradFn {
  std::function<void(const axon::Tensor&)> backward;
  std::vector<std::shared_ptr<axon::AutogradMeta>> inputs;
};

struct axon::AutogradMeta {
  explicit AutogradMeta(const std::vector<idx_t>& shape);
  std::shared_ptr<axon::Tensor> grad;  // Pointer, nicht direkter Wert
  std::shared_ptr<axon::GradFn> grad_fn_;
};
