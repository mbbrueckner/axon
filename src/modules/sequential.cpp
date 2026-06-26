
#include "axon/modules/sequential.hpp"

namespace axon::nn {

Sequential::Sequential(std::vector<std::unique_ptr<Module>> modules)
    : modules_(std::move(modules)) {}

Tensor Sequential::forward(const Tensor& input) {
  Tensor output = input;
  for (const std::unique_ptr<Module>& module : modules_) {
    output = module->forward(output);
  }
  return output;
}
std::vector<Tensor> Sequential::parameters() {
  std::vector<Tensor> result;
  for (const std::unique_ptr<Module>& module : modules_) {
    std::vector<Tensor> module_params = module->parameters();

    result.insert(result.end(),
                  std::make_move_iterator(module_params.begin()),
                  std::make_move_iterator(module_params.end()));
  }
  return result;
}
}  // namespace axon::nn