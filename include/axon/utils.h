//
// Created by Mika Brückner on 04.06.26.
//

#pragma once
#include <string>
#include <vector>

namespace axon {
namespace utils {

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