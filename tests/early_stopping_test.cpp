/**
 * @file early_stopping_test.cpp
 * @brief Unit tests for the axon::EarlyStopping helper.
 * @author Mika Brückner
 * @date 2026-07-03
 */

#include "axon/early_stopping.hpp"

#include "axon/tensor.hpp"
#include "catch2/catch_all.hpp"

namespace {

std::vector<axon::Tensor> make_params(float value) {
  return {axon::Tensor::from_data({value, value}, {2})};
}

}  // namespace

TEST_CASE("EarlyStopping does not stop while the metric keeps improving",
          "[EarlyStopping]") {
  axon::EarlyStopping stopper(2);

  REQUIRE_FALSE(stopper.should_stop(1.0f, make_params(1.0f)));
  REQUIRE_FALSE(stopper.should_stop(0.9f, make_params(2.0f)));
  REQUIRE_FALSE(stopper.should_stop(0.8f, make_params(3.0f)));
  REQUIRE_FALSE(stopper.should_stop(0.7f, make_params(4.0f)));
}

TEST_CASE("EarlyStopping stops after patience non-improving steps",
          "[EarlyStopping]") {
  axon::EarlyStopping stopper(2);

  REQUIRE_FALSE(stopper.should_stop(1.0f, make_params(1.0f)));
  REQUIRE_FALSE(stopper.should_stop(2.0f, make_params(2.0f)));  // counter 1
  REQUIRE_FALSE(stopper.should_stop(2.0f, make_params(3.0f)));  // counter 2
  REQUIRE(stopper.should_stop(2.0f, make_params(4.0f)));        // counter 3
}

TEST_CASE("EarlyStopping resets its patience counter on improvement",
          "[EarlyStopping]") {
  axon::EarlyStopping stopper(1);

  REQUIRE_FALSE(stopper.should_stop(1.0f, make_params(1.0f)));
  REQUIRE_FALSE(stopper.should_stop(2.0f, make_params(2.0f)));  // counter 1
  REQUIRE_FALSE(stopper.should_stop(0.5f, make_params(3.0f)));  // improves
  REQUIRE_FALSE(stopper.should_stop(2.0f, make_params(4.0f)));  // counter 1
  REQUIRE(stopper.should_stop(2.0f, make_params(5.0f)));        // counter 2
}

TEST_CASE("EarlyStopping treats sub-min_delta gains as non-improvements",
          "[EarlyStopping]") {
  axon::EarlyStopping stopper(1, 0.5f);

  REQUIRE_FALSE(stopper.should_stop(1.0f, make_params(1.0f)));
  REQUIRE_FALSE(stopper.should_stop(0.9f, make_params(2.0f)));  // < min_delta
  REQUIRE(stopper.should_stop(0.8f, make_params(3.0f)));        // < min_delta
}

TEST_CASE("EarlyStopping supports maximize mode", "[EarlyStopping]") {
  axon::EarlyStopping stopper(1, 0.0f, axon::Mode::Maximize);

  REQUIRE_FALSE(stopper.should_stop(0.5f, make_params(1.0f)));
  REQUIRE_FALSE(stopper.should_stop(0.7f, make_params(2.0f)));  // improves
  REQUIRE_FALSE(stopper.should_stop(0.6f, make_params(3.0f)));  // counter 1
  REQUIRE(stopper.should_stop(0.6f, make_params(4.0f)));        // counter 2
}

TEST_CASE("EarlyStopping keeps the parameters from the best step",
          "[EarlyStopping]") {
  axon::EarlyStopping stopper(2);

  stopper.should_stop(1.0f, make_params(10.0f));
  stopper.should_stop(0.5f, make_params(20.0f));  // best
  stopper.should_stop(0.8f, make_params(30.0f));  // worse, ignored

  std::vector<axon::Tensor> best = stopper.best_params();
  REQUIRE(best.size() == 1);
  REQUIRE(best[0].at({0}) == Catch::Approx(20.0f));
  REQUIRE(best[0].at({1}) == Catch::Approx(20.0f));
}