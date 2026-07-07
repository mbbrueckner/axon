/**
 * @file mnist_benchmark.cpp
 * @brief Timing harness: measures per-epoch training time on MNIST
 *        across multiple independent runs, exported as CSV.
 *
 * Runs one untimed warmup epoch, then NUM_RUNS independent training
 * runs (fresh model + optimizer per run, distinct reproducible seeds)
 * of NUM_EPOCHS epochs each, recording wall-clock time per epoch.
 *
 * @author Mika Brückner
 * @date 2026-07-07
 */

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "axon/constants.hpp"
#include "axon/dataloader.hpp"
#include "axon/datasets/mnist.hpp"
#include "axon/functional.hpp"
#include "axon/modules/linear.hpp"
#include "axon/modules/relu.hpp"
#include "axon/modules/sequential.hpp"
#include "axon/optimizers/sgd.hpp"

constexpr std::string_view TRAIN_IMAGES_PATH =
    "data/mnist/train-images.idx3-ubyte";
constexpr std::string_view TRAIN_LABELS_PATH =
    "data/mnist/train-labels.idx1-ubyte";

constexpr axon::idx_t NUM_RUNS = 10;
constexpr axon::idx_t NUM_EPOCHS = 10;
constexpr axon::idx_t SEED = 42;

int main() {
  std::cout << "#################################" << std::endl;
  std::cout << "#                               #" << std::endl;
  std::cout << "#            AXON               #" << std::endl;
  std::cout << "#-------------------------------#" << std::endl;
  std::cout << "#      MNIST - Benchmark        #" << std::endl;
  std::cout << "#                               #" << std::endl;
  std::cout << "#################################" << std::endl;

  std::cout << "> loading data..." << std::endl;

  axon::datasets::MNIST training_data((TRAIN_IMAGES_PATH.data()),
                                      TRAIN_LABELS_PATH.data());

  std::cout << "> finished loading data" << std::endl;
  // initialize data loaders
  axon::DataLoader train_loader(training_data, 32);
  std::cout << "> initialized data-loaders" << std::endl;
  std::cout << "> running warmup " << std::endl;
  // warmup
  {
    // build warmup  model: Linear(784, 128) -> ReLU -> Linear(128,10)
    std::vector<std::unique_ptr<axon::nn::Module>> modules;
    modules.push_back(std::make_unique<axon::nn::Linear>(784, 128, SEED));
    modules.push_back(std::make_unique<axon::nn::ReLU>());
    modules.push_back(std::make_unique<axon::nn::Linear>(128, 10, SEED));
    axon::nn::Sequential model(std::move(modules));

    // initialize warmup SGD optimizer
    axon::optimizer::SGD optimizer(model.parameters(), 0.1f);

    // run warmup epoch
    for (auto [input, target] : train_loader) {
      optimizer.zero_grad();
      axon::Tensor output = model.forward(input);
      axon::Tensor loss = axon::cross_entropy_loss(output, target);
      loss.backward();
      optimizer.step();
    }
  }
  std::cout << "> finished warmup " << std::endl;

  const std::string csv_path{
      std::format("results/mnist_benchmark_{:%FT%TZ}.csv",
                  std::chrono::system_clock::now())};

  const std::filesystem::path path_obj{csv_path};

  if (const std::filesystem::path dir = path_obj.parent_path();
      !dir.empty() && !std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }

  std::ofstream csv_file(csv_path);
  if (!csv_file.is_open()) {
    throw std::runtime_error(std::format("Cannot open file: '{}'", csv_path));
  }
  csv_file << "run,epoch,time_ms\n";

  std::cout << "> running measurements " << std::endl;
  // measurement loop
  for (axon::idx_t run = 0; run < NUM_RUNS; run++) {
    // generate new seed
    axon::idx_t seed = SEED + run;

    // build new model: Linear(784, 128) -> ReLU -> Linear(128,10)
    std::vector<std::unique_ptr<axon::nn::Module>> modules;
    modules.push_back(std::make_unique<axon::nn::Linear>(784, 128, seed));
    modules.push_back(std::make_unique<axon::nn::ReLU>());
    modules.push_back(std::make_unique<axon::nn::Linear>(128, 10, seed));
    axon::nn::Sequential model(std::move(modules));

    // initialize new optimizer
    axon::optimizer::SGD optimizer(model.parameters(), 0.1f);

    for (axon::idx_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
      // start measurement
      auto start = std::chrono::steady_clock::now();
      // training loop
      for (auto [input, target] : train_loader) {
        optimizer.zero_grad();
        axon::Tensor output = model.forward(input);
        axon::Tensor loss = axon::cross_entropy_loss(output, target);
        loss.backward();
        optimizer.step();
      }
      // end measurement
      auto end = std::chrono::steady_clock::now();

      // write to CSV
      const std::chrono::duration<double, std::milli> time = end - start;
      csv_file << run << "," << epoch << "," << time.count() << "\n";
      csv_file.flush();

      std::cout << "run: " << run << ", epoch: " << epoch
                << ", time (ms): " << time.count() << std::endl;
    }
  }
  std::cout << "> finished measurements " << std::endl;
  csv_file.close();
  std::cout << "> wrote results to: " << csv_path << std::endl;
  return 0;
}
