/**
 * @file mnist_benchmark.cpp
 * @brief Timing harness: measures per-epoch training time on MNIST
 *        across multiple independent runs, exported as CSV.
 *
 * Runs an adaptive warmup phase (trains until WARMUP_WINDOW consecutive
 * epoch times land within WARMUP_TOLERANCE of each other, capped at
 * MAX_WARMUP_EPOCHS, logged to warmup.csv), then NUM_RUNS independent
 * training runs (fresh model + optimizer per run, distinct reproducible
 * seeds) of NUM_EPOCHS epochs each, recording wall-clock time per epoch.
 *
 * @author Mika Brückner
 * @date 2026-07-07
 */

#include <algorithm>
#include <chrono>
#include <cmath>
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
constexpr axon::idx_t BASE_SEED = 42;

constexpr axon::idx_t WARMUP_WINDOW = 3;
constexpr double WARMUP_TOLERANCE = 0.02;
constexpr axon::idx_t MAX_WARMUP_EPOCHS = 30;

/// A helper method that returns the mean of a double vector
double mean(std::vector<double> v) {
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

/// A helper method that returns the median of a double vector
double median(std::vector<double> v) {
  if (v.empty()) return 0.0;

  size_t n = v.size() / 2;

  std::nth_element(v.begin(), v.begin() + n, v.end());

  if (v.size() % 2 != 0) {
    return v[n];
  } else {
    auto max_it = std::max_element(v.begin(), v.begin() + n);
    return (*max_it + v[n]) / 2.0;
  }
}

/// A helper method that returns the maximum element of a double vector
double max(std::vector<double> v) {
  return *std::ranges::max_element(v.begin(), v.end());
}

/// A helper method that returns the minimum element of a double vector
double min(std::vector<double> v) {
  return *std::ranges::min_element(v.begin(), v.end());
}

/// A helper method that returns the (sample) standard deviation of a double
/// vector
double stddev(const std::vector<double>& v) {
  if (v.size() < 2) return 0.0;
  const double m = mean(v);
  const double sq_sum =
      std::accumulate(v.begin(), v.end(), 0.0, [m](double acc, double x) {
        return acc + (x - m) * (x - m);
      });
  return std::sqrt(sq_sum / static_cast<double>(v.size() - 1));
}

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

  const auto timestamp = std::chrono::floor<std::chrono::seconds>(
      std::chrono::system_clock::now());

  const std::string base_path{std::format(
      "benchmarks/mnist/results/mnist_benchmark_{0}_{1:%F}T{1:%H:%M:%S}Z",
      GIT_SHA,
      timestamp)};

  if (const std::filesystem::path dir{base_path};
      !std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }

  const std::string warmup_csv_path{std::format("{}/warmup.csv", base_path)};
  std::ofstream warmup_file(warmup_csv_path);
  if (!warmup_file.is_open()) {
    throw std::runtime_error(
        std::format("Cannot open file: '{}'", warmup_csv_path));
  }
  warmup_file << "epoch,time_ms\n";

  const std::string raw_csv_path{std::format("{}/raw.csv", base_path)};
  std::ofstream raw_file(raw_csv_path);
  if (!raw_file.is_open()) {
    throw std::runtime_error(
        std::format("Cannot open file: '{}'", raw_csv_path));
  }
  raw_file << "run,epoch,time_ms\n";

  const std::string summary_csv_path{std::format("{}/summary.csv", base_path)};
  std::ofstream summary_file(summary_csv_path);
  if (!summary_file.is_open()) {
    throw std::runtime_error(
        std::format("Cannot open file: '{}'", summary_csv_path));
  }
  summary_file << "scope,n,mean_ms,median_ms,min_ms,max_ms,stddev_ms\n";

  std::cout << "> running warmup " << std::endl;
  // keep training until last WARMUP_WINDOW epoch times land within
  // WARMUP_TOLERANCE of each other, capped at MAX_WARMUP_EPOCHS
  {
    // build warmup model: Linear(784, 128) -> ReLU -> Linear(128,10)
    std::vector<std::unique_ptr<axon::nn::Module>> modules;
    modules.push_back(std::make_unique<axon::nn::Linear>(784, 128, BASE_SEED));
    modules.push_back(std::make_unique<axon::nn::ReLU>());
    modules.push_back(std::make_unique<axon::nn::Linear>(128, 10, BASE_SEED));
    axon::nn::Sequential model(std::move(modules));

    // initialize warmup SGD optimizer
    axon::optimizer::SGD optimizer(model.parameters(), 0.1f);

    std::vector<double> warmup_times;

    for (axon::idx_t epoch = 0; epoch < MAX_WARMUP_EPOCHS; epoch++) {
      auto start = std::chrono::steady_clock::now();
      for (auto [input, target] : train_loader) {
        optimizer.zero_grad();
        axon::Tensor output = model.forward(input);
        axon::Tensor loss = axon::cross_entropy_loss(output, target);
        loss.backward();
        optimizer.step();
      }
      auto end = std::chrono::steady_clock::now();

      const std::chrono::duration<double, std::milli> time = end - start;
      warmup_times.push_back(time.count());

      warmup_file << epoch << "," << time.count() << "\n";
      warmup_file.flush();

      std::cout << "warmup epoch: " << epoch << ", time (ms): " << time.count()
                << std::endl;

      if (static_cast<axon::idx_t>(warmup_times.size()) >= WARMUP_WINDOW) {
        const std::vector<double> window(warmup_times.end() - WARMUP_WINDOW,
                                         warmup_times.end());
        const double spread = (max(window) - min(window)) / mean(window);

        if (spread <= WARMUP_TOLERANCE) {
          std::cout
              << std::format(
                     "> warmup converged after {} epochs (spread: {:.4f})",
                     warmup_times.size(),
                     spread)
              << std::endl;
          break;
        }
      }

      if (epoch == MAX_WARMUP_EPOCHS - 1) {
        std::cout << std::format(
                         "> warmup did not converge within {} epochs, "
                         "proceeding anyway",
                         MAX_WARMUP_EPOCHS)
                  << std::endl;
      }
    }
  }
  warmup_file.close();
  std::cout << "> finished warmup " << std::endl;

  std::cout << "> running measurements " << std::endl;
  // measurement loop
  for (axon::idx_t run = 0; run < NUM_RUNS; run++) {
    // generate new seed
    axon::idx_t seed = BASE_SEED + run;

    // build new model: Linear(784, 128) -> ReLU -> Linear(128,10)
    std::vector<std::unique_ptr<axon::nn::Module>> modules;
    modules.push_back(std::make_unique<axon::nn::Linear>(784, 128, seed));
    modules.push_back(std::make_unique<axon::nn::ReLU>());
    modules.push_back(std::make_unique<axon::nn::Linear>(128, 10, seed));
    axon::nn::Sequential model(std::move(modules));

    // initialize new optimizer
    axon::optimizer::SGD optimizer(model.parameters(), 0.1f);

    // initialize measurements-vector for summary
    std::vector<double> measurements(NUM_EPOCHS);

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

      raw_file << run << "," << epoch << "," << time.count() << "\n";
      raw_file.flush();

      std::cout << "run: " << run << ", epoch: " << epoch
                << ", time (ms): " << time.count() << std::endl;

      measurements[epoch] = time.count();
    }

    double run_mean = mean(measurements);
    double run_median = median(measurements);
    double run_min = min(measurements);
    double run_max = max(measurements);
    double run_stddev = stddev(measurements);

    // write run summary
    summary_file << std::format("run_{},{},{},{},{},{},{}\n",
                                run,
                                NUM_EPOCHS,
                                run_mean,
                                run_median,
                                run_min,
                                run_max,
                                run_stddev);

    summary_file.flush();

    std::cout << std::format(
                     "Run-{} summary: mean(ms): {},  median(ms): {}, min(ms): "
                     "{}, max(ms): "
                     "{}, stddev: {}",
                     run,
                     run_mean,
                     run_median,
                     run_min,
                     run_max,
                     run_stddev)
              << std::endl;
  }
  std::cout << "> finished measurements " << std::endl;
  raw_file.close();
  std::cout << "> wrote raw results to: " << raw_csv_path << std::endl;

  summary_file.close();
  std::cout << "> wrote summary results to: " << summary_csv_path << std::endl;
  return 0;
}
