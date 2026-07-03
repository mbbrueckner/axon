/**
 * @file dataloader_test.cpp
 * @brief Unit tests for the axon::DataLoader class.
 * @author Mika Brückner
 * @date 2026-07-03
 */

#include "axon/dataloader.hpp"

#include <set>

#include "axon/datasets/dataset.hpp"
#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

namespace {

class RangeDataset final : public axon::datasets::Dataset {
 public:
  explicit RangeDataset(axon::idx_t n) : n_(n) {}

  axon::datasets::Sample operator[](axon::idx_t idx) const override {
    const auto value = static_cast<float>(idx);
    return {axon::Tensor::from_data({value, value}, {2}),
            axon::Tensor::from_data({value}, {1})};
  }

  [[nodiscard]] axon::idx_t size() const override { return n_; }

 private:
  axon::idx_t n_;
};

}  // namespace

TEST_CASE("DataLoader yields the expected number of batches", "[DataLoader]") {
  RangeDataset dataset(10);

  SECTION("drop_last keeps only full batches") {
    axon::DataLoader loader(dataset, 3, false, true);
    axon::idx_t n_batches = 0;
    for (auto _ : loader) {
      (void)_;
      n_batches++;
    }
    REQUIRE(n_batches == 3);
  }

  SECTION("keeping the last batch includes the partial batch") {
    axon::DataLoader loader(dataset, 3, false, false);
    axon::idx_t n_batches = 0;
    for (auto _ : loader) {
      (void)_;
      n_batches++;
    }
    REQUIRE(n_batches == 4);
  }
}

TEST_CASE("DataLoader stacks samples into batched tensors", "[DataLoader]") {
  RangeDataset dataset(6);
  axon::DataLoader loader(dataset, 2, false, true);

  auto [input, target] = *loader.begin();

  REQUIRE(input.shape() == std::vector<axon::idx_t>{2, 2});
  REQUIRE(target.shape() == std::vector<axon::idx_t>{2, 1});
}

TEST_CASE("DataLoader without shuffle preserves sample order", "[DataLoader]") {
  RangeDataset dataset(6);
  axon::DataLoader loader(dataset, 2, false, true);

  float expected = 0.0f;
  for (auto [input, target] : loader) {
    for (axon::idx_t row = 0; row < input.shape()[0]; row++) {
      REQUIRE(target.at({row, 0}) == Catch::Approx(expected));
      REQUIRE(input.at({row, 0}) == Catch::Approx(expected));
      expected += 1.0f;
    }
  }
  REQUIRE(expected == Catch::Approx(6.0f));
}

TEST_CASE("DataLoader final partial batch has the remaining samples",
          "[DataLoader]") {
  RangeDataset dataset(7);
  axon::DataLoader loader(dataset, 3, false, false);

  axon::Tensor last_input = axon::Tensor::zeros({1});
  for (auto [input, target] : loader) {
    last_input = input;
  }
  REQUIRE(last_input.shape() == std::vector<axon::idx_t>{1, 2});
  REQUIRE(last_input.at({0, 0}) == Catch::Approx(6.0f));
}

TEST_CASE("DataLoader with shuffle visits every sample exactly once",
          "[DataLoader]") {
  RangeDataset dataset(8);
  axon::DataLoader loader(dataset, 2, true, false);

  std::multiset<int> seen;
  for (auto [input, target] : loader) {
    for (axon::idx_t row = 0; row < target.shape()[0]; row++) {
      seen.insert(static_cast<int>(target.at({row, 0})));
    }
  }

  REQUIRE(seen.size() == 8);
  for (int i = 0; i < 8; i++) {
    REQUIRE(seen.count(i) == 1);
  }
}
