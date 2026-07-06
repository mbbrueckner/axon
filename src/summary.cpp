/**
 * @file summary.cpp
 * @brief Implementation of axon::summary.
 * @author Mika Brückner
 * @date 2026-07-06
 */

#include "axon/summary.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

namespace {

axon::idx_t utf8_display_width(const std::string& s) {
  axon::idx_t count = 0;
  for (unsigned char c : s) {
    if ((c & 0xC0) != 0x80) count++;
  }
  return count;
}

std::string build_box(std::vector<std::string> lines) {
  axon::idx_t max_width = 0;
  for (const auto& l : lines) {
    max_width = std::max(max_width, utf8_display_width(l));
  }

  std::string box;
  box += "+" + std::string(max_width + 2, '-') + "+\n";
  for (const auto& l : lines) {
    axon::idx_t pad = max_width - utf8_display_width(l);
    box += "| " + l + std::string(pad, ' ') + " |\n";
  }
  box += "+" + std::string(max_width + 2, '-') + "+\n";
  return box;
}

std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> lines;
  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) lines.push_back(line);
  return lines;
}

}  // namespace

namespace axon {

std::string summary(const nn::Module& model,
                    const optimizer::Optimizer& optimizer) {
  std::vector<std::string> lines = split_lines(model.to_string());
  lines.push_back(optimizer.to_string());
  return build_box(lines);
}

std::string summary(const nn::Module& model,
                    const optimizer::Optimizer& optimizer,
                    const EarlyStopping& early_stop) {
  std::vector<std::string> lines = split_lines(model.to_string());
  lines.push_back(optimizer.to_string());
  lines.push_back(early_stop.to_string());
  return build_box(lines);
}

}  // namespace axon