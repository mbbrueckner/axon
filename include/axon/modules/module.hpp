/**
 * @file module.hpp
 * @brief Declaration of the axon::nn::Module abstract base class, the common
 *        interface for composable neural network building blocks.
 * @author Mika Brückner
 * @date 2026-06-17
 */

#pragma once
#include <vector>

#include "axon/tensor.hpp"
namespace axon::nn {
class Module;
}

/**
 * @brief Abstract base class for all neural network modules.
 *
 * A Module represents a composable building block of a neural network, such as
 * a layer, an activation function, or an entire model. Derived classes
 * implement the forward pass and expose their trainable parameters.
 */
class axon::nn::Module {
 public:
  /**
   * @brief Computes the forward pass of the module.
   *
   * @param input The input tensor.
   * @return The result of applying the module to @p input.
   */
  virtual Tensor forward(const Tensor& input) = 0;

  /**
   * @brief Collects the module's trainable parameters.
   *
   * @return A list of tensors representing the module's parameters, e.g. for
   *         use by an optimizer.
   */
  virtual std::vector<Tensor> parameters() = 0;

  /**
   * @brief Sets the module's trainable parameters.
   *
   * @param params A list of tensors to replace the module's parameters,
   *        in the same order as returned by parameters().
   */
  virtual void set_parameters(std::vector<Tensor> params) = 0;

  /**
   * @brief Returns a human-readable description of the module.
   *
   * @return A string describing the module's type and configuration,
   *         e.g. for printing a model's architecture.
   */
  virtual std::string to_string() const = 0;

  /// @brief Virtual destructor to allow safe polymorphic deletion.
  virtual ~Module() = default;
};