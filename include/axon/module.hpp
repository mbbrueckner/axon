//
// Created by Mika Brückner on 17.06.26.
//

#pragma once
#include "tensor.hpp"
namespace axon {
class Module;
}

class axon::Module {
 public:
  virtual Tensor forward(const Tensor& input) = 0;
  virtual std::vector<Tensor> parameters() = 0;
  virtual ~Module() = default;
};