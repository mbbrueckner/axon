/**
 * @file early_stopping.hpp
 * @brief Declaration of the axon::EarlyStopping class, which monitors a
 *        training metric and signals when training should stop.
 * @author Mika Brückner
 * @date 2026-07-03
 */

#pragma once
#include <vector>

#include "constants.hpp"
#include "tensor.hpp"

namespace axon {
class EarlyStopping;

/// @brief Direction in which the monitored metric should improve.
enum class Mode { Minimize, Maximize };
}  // namespace axon

/**
 * @brief Monitors a metric across epochs and signals when training should
 *        stop due to lack of improvement.
 *
 * Tracks the best observed metric value and a snapshot of the model
 * parameters at the point of the best value, so training can be reverted
 * to its best-performing state after stopping.
 *
 * Typical usage:
 * @code
 * EarlyStopping stopper(5, 0.0f, Mode::Minimize);
 * for (int epoch = 0; epoch < max_epochs; epoch++) {
 *     float val_loss = ...;
 *     if (stopper.should_stop(val_loss, model.parameters())) break;
 * }
 * // restore best weights:
 * auto best = stopper.best_params();
 * @endcode
 */
class axon::EarlyStopping {
 public:
  /**
   * @brief Constructs an EarlyStopping monitor.
   *
   * @param patience  Number of consecutive non-improving calls to
   *        @c should_stop() tolerated before signaling a stop.
   * @param min_delta Minimum change in the metric to qualify as an
   *        improvement. Defaults to 0, i.e. any improvement counts.
   * @param mode      Whether the metric should be minimized (e.g. loss)
   *        or maximized (e.g. accuracy). Defaults to @c Mode::Minimize.
   */
  EarlyStopping(idx_t patience,
                float min_delta = 0.0f,
                Mode mode = Mode::Minimize);

  /**
   * @brief Records the current metric and reports whether training should
   *        stop.
   *
   * If @p current_metric improves on the best value seen so far (by more
   * than @c min_delta_, in the direction given by @c mode_), the internal
   * best metric and a deep copy of @p current_params are stored and the
   * patience counter resets. Otherwise the patience counter increments.
   *
   * @param current_metric The metric value for the current epoch.
   * @param current_params The current model parameters, snapshotted via
   *        deep copy if this is a new best.
   * @return @c true if the patience threshold has been exceeded and
   *         training should stop, @c false otherwise.
   */
  bool should_stop(float current_metric,
                   const std::vector<Tensor>& current_params);

  /**
   * @brief Returns a deep copy of the parameters at the best observed
   *        metric value.
   * @return The snapshotted best parameters, or an empty vector if
   *         @c should_stop() has not yet recorded an improvement.
   */
  std::vector<Tensor> best_params();

  /**
   * @brief Returns a human-readable description of the early-stopping config.
   * @return A string describing patience, min_delta, and mode.
   */
  [[nodiscard]] std::string to_string() const;

 private:
  /// @brief Epochs to tolerate without improvement before stopping.
  idx_t patience_;
  /// @brief Minimum change required to count as an improvement.
  float min_delta_;
  /// @brief Direction of improvement (minimize or maximize).
  Mode mode_;
  /// @brief Best metric value observed so far.
  float best_metric_;
  /// @brief Consecutive non-improving calls since the last improvement.
  idx_t patience_counter_;
  /// @brief Deep copy of model parameters at the best metric value.
  std::vector<Tensor> best_params_;
};