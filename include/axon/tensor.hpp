//
// Created by Mika Brückner on 02.06.26.
//

#pragma once

#include <memory>
#include <numeric>
#include <vector>

namespace axon {
class Tensor;
}

class axon::Tensor {
 private:
  std::vector<int64_t> shape_;
  std::vector<int64_t> stride_;
  size_t offset_;
  std::shared_ptr<std::vector<float>> data_;

  Tensor(std::shared_ptr<std::vector<float>> data,
         std::vector<int64_t> shape,
         std::vector<int64_t> stride,
         size_t offset);

  void calculate_strides();

 public:
  Tensor(const std::vector<int64_t>& shape, const std::vector<float>& data);

  explicit Tensor(const std::vector<int64_t>& shape);

  // TODO [] operator

  [[nodiscard]] const std::vector<int64_t>& shape() const { return shape_; };
  [[nodiscard]] const std::vector<int64_t>& stride() const { return stride_; };
  [[nodiscard]] size_t offset() const { return offset_; };

  [[nodiscard]] size_t num_dim() const { return shape_.size(); };

  [[nodiscard]] size_t num_elements() const {
    return std::accumulate(
        shape_.begin(), shape_.end(), 1, std::multiplies<>());
  };

  [[nodiscard]] float at(std::initializer_list<int64_t> indices) const;

  [[nodiscard]] Tensor transpose() const;
};
