/**
 * @file early_stopping.cpp
 * @brief Implementation of the axon::EarlyStopping class, which monitors a
 *        training metric and signals when training should stop.
 * @author Mika Brückner
 * @date 2026-07-03
 */

#include "axon/early_stopping.hpp"

#include <cfloat>
#include <format>

#include "axon/tensor.hpp"

axon::EarlyStopping::EarlyStopping(const idx_t patience,
                                   const float min_delta,
                                   const Mode mode)
    : patience_(patience),
      min_delta_(min_delta),
      mode_(mode),
      best_metric_(mode == Mode::Minimize ? FLT_MAX : -FLT_MAX),
      patience_counter_(0) {}

bool axon::EarlyStopping::should_stop(
    const float current_metric, const std::vector<Tensor>& current_params) {
  bool is_better = mode_ == Mode::Minimize
                       ? current_metric < best_metric_ - min_delta_
                       : current_metric > best_metric_ + min_delta_;

  if (is_better) {
    patience_counter_ = 0;
    best_metric_ = current_metric;

    best_params_.clear();
    best_params_.reserve(current_params.size());
    for (const Tensor& param : current_params) {
      best_params_.push_back(Tensor::from_data(param.data(), param.shape()));
    }
    return false;

  } else {
    patience_counter_++;
    return patience_counter_ > patience_;
  }
}
std::vector<axon::Tensor> axon::EarlyStopping::best_params() {
  return best_params_;
}
std::string axon::EarlyStopping::to_string() const {
  return std::format("EarlyStopping(patience={}, min_delta={}, mode={})",
                     patience_,
                     min_delta_,
                     mode_ == Mode::Minimize ? "min" : "max");
}