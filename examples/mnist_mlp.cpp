/**
 * @file mnist_mlp.cpp
 * @brief End-to-end example: training an MLP to classify MNIST digits.
 *
 * Loads the MNIST dataset via @ref axon::datasets::MNIST, batches it with @ref
 * axon::DataLoader, and trains a Linear(784,128)->ReLU->Linear(128,10) model
 * using @ref axon::optimizer::SGD. Uses @ref axon::EarlyStopping to halt
 * training once the validation loss stops improving.
 *
 * @author Mika Brückner
 * @date 2026-07-04
 */

#include <iostream>
#include <string>

#include "axon/dataloader.hpp"
#include "axon/datasets/mnist.hpp"
#include "axon/early_stopping.hpp"
#include "axon/functional.hpp"
#include "axon/modules/linear.hpp"
#include "axon/modules/module.hpp"
#include "axon/modules/relu.hpp"
#include "axon/modules/sequential.hpp"
#include "axon/optimizers/sgd.hpp"
#include "axon/summary.hpp"

constexpr std::string_view TRAIN_IMAGES_PATH =
    "data/mnist/train-images.idx3-ubyte";
constexpr std::string_view TRAIN_LABELS_PATH =
    "data/mnist/train-labels.idx1-ubyte";
constexpr std::string_view VAL_IMAGES_PATH =
    "data/mnist/t10k-images.idx3-ubyte";
constexpr std::string_view VAL_LABELS_PATH =
    "data/mnist/t10k-labels.idx1-ubyte";

// number of training epochs
constexpr axon::idx_t EPOCHS = 1000;

int main(int argc, char *argv[]) {
  std::cout << "#################################" << std::endl;
  std::cout << "#                               #" << std::endl;
  std::cout << "#            AXON               #" << std::endl;
  std::cout << "#-------------------------------#" << std::endl;
  std::cout << "# MNIST - multilayer perceptron #" << std::endl;
  std::cout << "#                               #" << std::endl;
  std::cout << "#################################" << std::endl;

  std::cout << "> loading data..." << std::endl;

  // initialize datasets
  axon::datasets::MNIST training_data((TRAIN_IMAGES_PATH.data()),
                                      TRAIN_LABELS_PATH.data());

  axon::datasets::MNIST val_data(VAL_IMAGES_PATH.data(),
                                 VAL_LABELS_PATH.data());

  std::cout << "> finished loading data" << std::endl;

  // initialize data loaders
  axon::DataLoader train_loader(training_data, 32);
  axon::DataLoader val_loader(val_data, 32);

  std::cout << "> initialized data-loaders" << std::endl;

  // build model: Linear(784, 128) -> ReLU -> Linear(128,10)
  std::vector<std::unique_ptr<axon::nn::Module>> modules;
  modules.push_back(std::make_unique<axon::nn::Linear>(784, 128));
  modules.push_back(std::make_unique<axon::nn::ReLU>());
  modules.push_back(std::make_unique<axon::nn::Linear>(128, 10));
  axon::nn::Sequential model(std::move(modules));

  // initialize SGD optimizer
  axon::optimizer::SGD optimizer(model.parameters(), 0.1f);

  // initialize early stop with patience 5 and shuffle
  axon::EarlyStopping early_stop(5);

  std::cout << "> start running training-loop with Setup:" << std::endl;
  std::cout << axon::summary(model, optimizer, early_stop);

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

    // validation
    float val_loss_sum = 0.0f;
    float val_acc_sum = 0.0f;
    axon::idx_t n_val_batches = 0;
    for (auto [input, target] : val_loader) {
      axon::Tensor output = model.forward(input);
      axon::Tensor loss = axon::cross_entropy_loss(output, target);
      axon::Tensor acc = axon::accuracy(output, target);
      val_loss_sum += loss.item();
      val_acc_sum += acc.item();
      n_val_batches++;
    }
    float val_loss = val_loss_sum / static_cast<float>(n_val_batches);
    float val_acc = val_acc_sum / static_cast<float>(n_val_batches);
    // if (epoch % 100 == 0)
    std::cout << "Epoch: " << epoch << "; val_loss: " << val_loss
              << "; val_acc: " << val_acc << std::endl;

    // early stop if no improvement for 5 epochs
    if (early_stop.should_stop(val_loss, model.parameters())) {
      std::cout << "Early stop at epoch " << epoch << "; val_loss: " << val_loss
                << "; val_acc: " << val_acc << std::endl;
      break;
    }
  }

  // restore best params after training
  auto best = early_stop.best_params();
  return 0;
}
