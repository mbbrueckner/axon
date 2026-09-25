/**
 * @file parallel_test.cpp
 * @brief Unit tests for axon::internal::parallel_for().
 * @author Mika Brückner
 * @date 2026-09-17
 */

#include "../src/core/parallel.hpp"

#include <vector>

#include "axon/constants.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("parallel_for covers every index exactly once", "[Parallel]") {
  constexpr axon::idx_t n{10};
  constexpr axon::idx_t grain_size{2};

  std::vector<float> data(n, 0);

  auto increment = [&](const axon::idx_t begin, const axon::idx_t end) {
    for (axon::idx_t i{begin}; i < end; i++) {
      data[i]++;
    }
  };

  axon::internal::parallel_for(0, n, grain_size, increment);

  for (axon::idx_t i = 0; i < n; i++) REQUIRE(data[i] == 1);
}

TEST_CASE("parallel_for leaves indices outside the range untouched",
          "[Parallel]") {
  constexpr axon::idx_t n{10};
  constexpr axon::idx_t grain_size{2};

  std::vector<float> data(n, 0);

  auto increment = [&](const axon::idx_t begin, const axon::idx_t end) {
    for (axon::idx_t i{begin}; i < end; i++) {
      data[i]++;
    }
  };

  axon::internal::parallel_for(7, n, grain_size, increment);

  for (axon::idx_t i{0}; i < 7; i++) REQUIRE(data[i] == 0);
  for (axon::idx_t i{7}; i < n; i++) REQUIRE(data[i] == 1);
}

TEST_CASE("parallel_for does not invoke the operation on an empty range",
          "[Parallel]") {
  constexpr axon::idx_t n{10};
  constexpr axon::idx_t grain_size{2};
  std::vector<float> data(n, 0);
  axon::idx_t counter{0};

  auto increment = [&](const axon::idx_t begin, const axon::idx_t end) {
    counter++;
    for (axon::idx_t i{begin}; i < end; i++) {
      data[i]++;
    }
  };

  axon::internal::parallel_for(n, n, grain_size, increment);

  REQUIRE(counter == 0);
}
