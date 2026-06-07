/**
 * @file utils.hpp
 * @brief Small general-purpose helper utilities used across axon.
 * @author Mika Brückner
 * @date 2026-06-04
 */

#pragma once
#include <string>
#include <vector>

namespace axon {
/// @brief General-purpose helper utilities.
namespace utils {

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

}  // namespace utils
}  // namespace axon