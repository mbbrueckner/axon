//
// Created by Mika Brückner on 26.06.26.
//

#include <iostream>
#include <ostream>

#include "../include/axon/functional.hpp"
#include "axon/functional.hpp"
#include "axon/modules/linear.hpp"
#include "axon/modules/relu.hpp"
#include "axon/modules/sequential.hpp"
#include "axon/optimizers/sgd.hpp"
#include "axon/tensor.hpp"

int main(int argc, char *argv[]) {
  const axon::Tensor xor_input =
      axon::Tensor::from_data({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});

  const axon::Tensor xor_target = axon::Tensor::from_data({0, 1, 1, 0}, {4, 1});

  std::vector<std::unique_ptr<axon::nn::Module>> modules;
  modules.push_back(std::make_unique<axon::nn::Linear>(2, 4));
  modules.push_back(std::make_unique<axon::nn::ReLU>());
  modules.push_back(std::make_unique<axon::nn::Linear>(4, 1));
  axon::nn::Sequential model(std::move(modules));

  axon::optimizer::SGD optimizer(model.parameters(), 0.1f);

  for (int epoch = 0; epoch < 1000; epoch++) {
    optimizer.zero_grad();
    axon::Tensor output = model.forward(xor_input);
    axon::Tensor loss = axon::mse_loss(output, xor_target);
    axon::Tensor acc = axon::accuracy(output, xor_target);
    loss.backward();
    optimizer.step();

    if (epoch % 100 == 0)
      std::cout << "Epoch: " << epoch << "; loss:" << loss.item()
                << "; acc.:" << acc.item() << std::endl;
  }
}
