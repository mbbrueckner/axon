

#pragma once

#include "axon/constants.hpp"
#include "axon/module.hpp"
#include "axon/tensor.hpp"

namespace axon {
class Linear;
}

class axon::Linear : public Module {
 public:
  Linear(idx_t in_features, idx_t out_features);
  Tensor forward(const Tensor& input) override;
  std::vector<Tensor> parameters() override;

 private:
  Tensor weights_;
  Tensor bias_;
};