/**
 * @file utils.hpp
 * @brief Small general-purpose helper utilities used across axon.
 * @author Mika Brückner
 * @date 2026-06-04
 */

#pragma once
#include <string>
#include <vector>

#include "axon/constants.hpp"

/// @brief General-purpose helper utilities.
namespace axon::utils {

/**
 * @brief Formats a vector as a bracketed, comma-separated string.
 *
 * Produces output of the form @c "[a,b,c]"; an empty vector yields @c "[]".
 *
 * @tparam T Element type; must be supported by @c std::to_string.
 * @param vec The vector to format.
 * @return The string representation of @p vec.
 */
template <typename T>
std::string vector_to_string(const std::vector<T>& vec) {
  std::string vec_string = "[";

  for (T entry : vec) {
    vec_string.append(std::to_string(entry));
    vec_string.append(",");
  }
  if (!vec.empty()) vec_string.pop_back();
  vec_string.append("]");
  return vec_string;
}

inline std::vector<idx_t> flat_to_indices(idx_t flat,
                                          const std::vector<idx_t>& shape) {
  std::vector<idx_t> indices(shape.size());
  for (idx_t i = shape.size() - 1; i >= 0; i--) {
    indices[i] = flat % shape[i];
    flat /= shape[i];
  }
  return indices;
}

}  // namespace axon::utils
