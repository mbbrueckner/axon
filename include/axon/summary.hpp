/**
 * @file summary.hpp
 * @brief Declaration of axon::summary, which builds a human-readable
 *        overview of a model's training setup.
 * @author Mika Brückner
 * @date 2026-07-06
 */

#pragma once
#include <string>

#include "axon/early_stopping.hpp"
#include "axon/modules/module.hpp"
#include "axon/optimizers/optimizer.hpp"

namespace axon {

/**
 * @brief Builds a boxed summary of a model and optimizer setup.
 * @param model Described via Module::to_string().
 * @param optimizer Described via Optimizer::to_string().
 * @return A multi-line string containing the boxed summary.
 */
std::string summary(const nn::Module& model,
                    const optimizer::Optimizer& optimizer);

/**
 * @brief Builds a boxed summary including early-stopping configuration.
 * @param model Described via Module::to_string().
 * @param optimizer Described via Optimizer::to_string().
 * @param early_stop Described via EarlyStopping::to_string().
 * @return A multi-line string containing the boxed summary.
 */
std::string summary(const nn::Module& model,
                    const optimizer::Optimizer& optimizer,
                    const EarlyStopping& early_stop);
}  // namespace axon