/**
 * @file mnist_profile.cpp
 * @brief Profiling harness: runs the MNIST training loop with no timing or
 *        validation overhead, for use under a sampling profiler (e.g. Xcode
 *        Instruments' Time Profiler, or `sample`).
 *
 * Loads the MNIST dataset via @ref axon::datasets::MNIST, batches it with @ref
 * axon::DataLoader, and repeatedly trains a Linear(784,128)->ReLU->
 * Linear(128,10) model using @ref axon::optimizer::SGD for a fixed, small
 * number of epochs. Unlike @c mnist_benchmark.cpp, this does not measure wall
 * -clock time or write CSVs; it exists purely to give a profiler enough
 * samples of the hot path to attribute time to individual functions.
 *
 * @author Mika Brückner
 * @date 2026-07-18
 */

#include <iostream>

#include "axon/dataloader.hpp"
#include "axon/datasets/mnist.hpp"
#include "axon/functional.hpp"
#include "axon/modules/linear.hpp"
#include "axon/modules/module.hpp"
#include "axon/modules/relu.hpp"
#include "axon/modules/sequential.hpp"
#include "axon/optimizers/sgd.hpp"

constexpr std::string_view TRAIN_IMAGES_PATH =
    "data/mnist/train-images.idx3-ubyte";
constexpr std::string_view TRAIN_LABELS_PATH =
    "data/mnist/train-labels.idx1-ubyte";

// number of training epochs
constexpr axon::idx_t EPOCHS = 5;

int main(int argc, char *argv[]) {
  std::cout << "#################################" << std::endl;
  std::cout << "#                               #" << std::endl;
  std::cout << "#            AXON               #" << std::endl;
  std::cout << "#-------------------------------#" << std::endl;
  std::cout << "#       MNIST - profile         #" << std::endl;
  std::cout << "#                               #" << std::endl;
  std::cout << "#################################" << std::endl;

  std::cout << "> loading data..." << std::endl;

  // initialize datasets
  axon::datasets::MNIST training_data((TRAIN_IMAGES_PATH.data()),
                                      TRAIN_LABELS_PATH.data());

  std::cout << "> finished loading data" << std::endl;

  // initialize data loader
  axon::DataLoader train_loader(training_data, 32);

  std::cout << "> initialized data-loader" << std::endl;

  // build model: Linear(784, 128) -> ReLU -> Linear(128,10)
  std::vector<std::unique_ptr<axon::nn::Module>> modules;
  modules.push_back(std::make_unique<axon::nn::Linear>(784, 128));
  modules.push_back(std::make_unique<axon::nn::ReLU>());
  modules.push_back(std::make_unique<axon::nn::Linear>(128, 10));
  axon::nn::Sequential model(std::move(modules));

  // initialize SGD optimizer
  axon::optimizer::SGD optimizer(model.parameters(), 0.1f);

  std::cout << "> start running training-loop" << std::endl;

  // Training loop
  for (axon::idx_t epoch = 0; epoch < EPOCHS; epoch++) {
    // training loop
    for (auto [input, target] : train_loader) {
      optimizer.zero_grad();
      axon::Tensor output = model.forward(input);
      axon::Tensor loss = axon::cross_entropy_loss(output, target);
      loss.backward();
      optimizer.step();
    }
  }

  std::cout << "> finished running training-loop" << std::endl;

  return 0;
}
