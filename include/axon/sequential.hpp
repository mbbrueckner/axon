/**
 * @file sequential.hpp
 * @brief Declaration of the axon::nn::Sequential module, a container that
 * chains a sequence of sub-modules into a single forward pass.
 * @author Mika Brückner
 * @date 2026-06-19
 */

#pragma once

#include <memory>
#include <vector>

#include "axon/module.hpp"
#include "axon/tensor.hpp"

namespace axon::nn {
class Sequential;
}

/**
 * @brief A container module that runs its sub-modules in sequence.
 *
 * Sequential owns a list of sub-modules and feeds the output of each one as
 * the input to the next, starting from the input passed to forward(). It has
 * no parameters of its own; parameters() returns the concatenated parameters
 * of all sub-modules in order.
 */
class axon::nn::Sequential : public Module {
 public:
  /**
   * @brief Constructs a Sequential module from an ordered list of sub-modules.
   *
   * @param modules Sub-modules to run in order, from first to last.
   */
  explicit Sequential(std::vector<std::unique_ptr<Module>> modules);

  /**
   * @brief Runs the input through each sub-module in order.
   * @param input The input tensor.
   * @return The output of the final sub-module.
   */
  Tensor forward(const Tensor& input) override;

  /// @return The concatenated parameters of all sub-modules, in order.
  std::vector<Tensor> parameters() override;

 private:
  /// Sub-modules executed in order during forward().
  std::vector<std::unique_ptr<Module>> modules_;
};