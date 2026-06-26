/**
 * @file xor_demo.cpp
 * @brief End-to-end example: training a small MLP to learn the XOR function.
 *
 * Demonstrates the core axon workflow — building a model from @ref
 * axon::nn::Sequential, optimizing it with @ref axon::optimizer::SGD, and
 * running a manual training loop (zero_grad, forward, loss, backward, step).
 * XOR is the classic non-linearly-separable problem, so a hidden ReLU layer
 * is required for the network to fit it.
 *
 * @author Mika Brückner
 * @date 2026-06-26
 */

#include <iostream>
#include <ostream>

#include "axon/functional.hpp"
#include "axon/modules/linear.hpp"
#include "axon/modules/relu.hpp"
#include "axon/modules/sequential.hpp"
#include "axon/optimizers/sgd.hpp"
#include "axon/tensor.hpp"

/**
 * @brief Trains a 2-4-1 MLP on the four XOR samples and logs progress.
 *
 * Builds the network @c Linear(2,4) -> ReLU -> Linear(4,1), then runs 1000
 * epochs of gradient descent against the mean squared error, printing the
 * loss and accuracy every 100 epochs.
 *
 * @return @c 0 on success.
 */
int main() {
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
      std::cout << "Epoch: " << epoch << "; loss: " << loss.item()
                << "; acc.: " << acc.item() << std::endl;
  }
}
