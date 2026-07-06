
#include "axon/modules/sequential.hpp"

#include <format>

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

void Sequential::set_parameters(std::vector<Tensor> params) {
  idx_t offset = 0;
  for (const std::unique_ptr<Module>& module : modules_) {
    std::vector<Tensor> module_params = module->parameters();
    idx_t module_params_size = module_params.size();

    module->set_parameters(std::vector<Tensor>(
        params.begin() + offset, params.begin() + offset + module_params_size));

    offset += module_params_size;
  }

  if (offset != static_cast<idx_t>(params.size())) {
    throw std::out_of_range(
        "set_parameters: number of parameters does not match");
  }
}
}  // namespace axon::nn