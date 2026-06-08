/**
 * @file tensor.cpp
 * @brief Implementation of the axon::Tensor class.
 * @author Mika Brückner
 * @date 2026-06-02
 */

#include "axon/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

#include "axon/constants.hpp"
#include "axon/utils.hpp"

namespace axon {
Tensor::Tensor(std::shared_ptr<std::vector<float>> data,
               std::vector<int64_t> shape,
               std::vector<int64_t> stride,
               size_t offset)
    : shape_(std::move(shape)),
      stride_(std::move(stride)),
      offset_(offset),
      data_(std::move(data)) {}

Tensor::Tensor(const std::vector<float>& data,
               const std::vector<int64_t>& shape)
    : shape_(shape),
      stride_(calculate_strides(shape)),
      offset_(0),
      data_(std::make_shared<std::vector<float>>(data)) {
  if (num_elements() != data.size()) {
    throw std::out_of_range(
        std::format("Number of elements does not match shape: got {} "
                    "expected {} for shape {}",
                    data.size(),
                    num_elements(),
                    axon::utils::vector_to_string(shape)));
  }
}

Tensor::Tensor(const std::vector<int64_t>& shape)
    : shape_(shape),
      stride_(calculate_strides(shape)),
      offset_(0),
      data_(std::make_shared<std::vector<float>>(std::accumulate(
          shape_.begin(), shape_.end(), 1, std::multiplies<>()))) {
  ;
}

std::vector<int64_t> Tensor::calculate_strides(
    const std::vector<int64_t>& shape) {
  const size_t dim = shape.size();
  std::vector<int64_t> stride;
  stride.resize(dim, 1);

  if (dim >= 2) {
    // check if tensor or is 0D/1D -> no stride needed
    for (idx_t i = static_cast<int64_t>(dim) - 2; i >= 0; i--) {
      // stride[i] = shape[i+1] * stride[i+1]; stride[dim-1] = 1
      stride[i] = shape[i + 1] * stride[i + 1];
    }
  }

  return stride;
}

bool Tensor::is_contiguous() const {
  const size_t dim = shape_.size();

  if (dim == 0) return true;

  if (stride_.back() != 1) return false;

  for (idx_t i{}; i < dim - 1; i++) {
    if (stride_[i] != shape_[i + 1] * stride_[i + 1]) return false;
  }
  return true;
}

Tensor Tensor::operator[](size_t idx) const {
  if (shape_.empty()) {
    throw std::out_of_range(
        "Cannot subscript a 0-dimensional tensor (too many subscripts)");
  }
  if (idx >= static_cast<size_t>(shape_[0])) {
    throw std::out_of_range(
        std::format("Index out of bounds: expected index to be < {}, got{}",
                    shape_[0],
                    idx));
  }

  const size_t new_offset = idx * (stride_[0]);
  const std::vector<int64_t> new_shape(shape_.begin() + 1, shape_.end());
  const std::vector<int64_t> new_stride(stride_.begin() + 1, stride_.end());
  return {data_, new_shape, new_stride, new_offset};
}

float Tensor::at(std::initializer_list<int64_t> indices) const {
  const size_t num_dim = shape_.size();
  const size_t num_indices = indices.size();

  if (num_indices != num_dim) {
    throw std::out_of_range(
        std::format("Number of indices does not match dimensions: got {} "
                    "indices for a {}-dimensional tensor",
                    num_indices,
                    num_dim));
  }

  int64_t dim = 0;
  size_t flat = offset_;
  for (int64_t idx : indices) {
    if (idx >= shape_[dim]) {
      throw std::out_of_range(
          std::format("Index {} out of range for dimension {} with size {}",
                      idx,
                      dim,
                      shape_[dim]));
    }

    flat += idx * stride_[dim];
    dim++;
  }
  return (*data_)[flat];
}

Tensor Tensor::transpose() const {
  std::vector shape_t(shape_);
  std::ranges::reverse(shape_t);
  std::vector stride_t(stride_);
  std::ranges::reverse(stride_t);

  return {data_, shape_t, stride_t, offset_};
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
  if (!is_contiguous()) {
    throw std::logic_error("Non-contiguous Tensor can not be reshaped");
  }

  const size_t old_num_elements = num_elements();
  const size_t new_num_elements = std::accumulate<>(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<>());

  if (old_num_elements != new_num_elements) {
    throw std::out_of_range(
        std::format("New number of elements does not match available: got {} "
                    "elements for reshaped Tensor with {} elements",
                    new_num_elements,
                    old_num_elements));
  }

  std::vector<int64_t> new_stride = calculate_strides(new_shape);
  return {data_, new_shape, new_stride, offset_};
}

Tensor Tensor::flatten() const {
  const std::vector<int64_t> flat_shape = {
      static_cast<int64_t>(num_elements())};
  return reshape(flat_shape);
}

Tensor Tensor::matmul(const Tensor& other) const {
  if (num_dim() != 2 && other.num_dim() != 2) {
    throw std::out_of_range(std::format(
        "Cannot perform matrix multiplication on tensor with degree ≠ 2,"
        "got dimensions: {}; {}",
        num_dim(),
        other.num_dim()));
  }
  if (shape_[1] != other.shape()[0]) {
    throw std::out_of_range(
        std::format("Cannot perform matrix multiplication on matrices with "
                    "unmatching row x column sizes:"
                    "got rows left: {}; columns right {}",
                    shape_[1],
                    other.shape()[0]));
  }
  const int64_t rows = shape_[0];
  const int64_t inner = shape_[1];
  const int64_t cols = other.shape()[1];
  Tensor result({rows, cols});
  for (idx_t i{}; i < rows; i++) {
    for (idx_t j{}; j < cols; j++) {
      for (idx_t k{}; k < inner; k++) {
        (*result.data_)[i * cols + j] += at({i, k}) * other.at({k, j});
      }
    }
  }
  return result;
}

Tensor Tensor::log() const {
  std::vector<float> new_data = (*data_);

  std::ranges::transform(
      new_data, new_data.begin(), [](float x) { return std::log(x); });
  return {new_data, shape_};
}

Tensor Tensor::exp() const {
  std::vector<float> new_data = (*data_);

  std::ranges::transform(
      new_data, new_data.begin(), [](float x) { return std::exp(x); });
  return {new_data, shape_};
}

float Tensor::min() const { return *std::ranges::min_element(*data_); }

float Tensor::max() const { return *std::ranges::max_element(*data_); }

float Tensor::sum() const {
  return std::accumulate(data_->begin(), data_->end(), 0.0f);
}

float Tensor::mean() const {
  return sum() / static_cast<float>(num_elements());
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
  const std::vector<int64_t>& lhs_shape = lhs.shape();
  const std::vector<int64_t>& rhs_shape = rhs.shape();
  if (lhs_shape != rhs_shape) {
    throw std::out_of_range(
        std::format("Cannot perform elementwise addition for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs_shape),
                    utils::vector_to_string(rhs_shape)));
  }

  const size_t num_elements = std::accumulate(
      lhs_shape.begin(), lhs_shape.end(), 1, std::multiplies<>());

  std::vector<float> new_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);

  for (idx_t i{}; i < num_elements; i++) {
    new_data[i] += rhs_data[i];
  }
  return {new_data, lhs_shape};
}

Tensor operator+(const float sclr, const Tensor& tnsr) {
  std::vector<float> new_data((*tnsr.data_));
  std::ranges::transform(
      new_data, new_data.begin(), [sclr](float x) { return x + sclr; });
  return {new_data, tnsr.shape_};
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
  const std::vector<int64_t>& lhs_shape = lhs.shape();
  const std::vector<int64_t>& rhs_shape = rhs.shape();
  if (lhs_shape != rhs_shape) {
    throw std::out_of_range(
        std::format("Cannot perform elementwise addition for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs_shape),
                    utils::vector_to_string(rhs_shape)));
  }

  const size_t num_elements = std::accumulate(
      lhs_shape.begin(), lhs_shape.end(), 1, std::multiplies<>());

  std::vector<float> new_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);

  for (idx_t i{}; i < num_elements; i++) {
    new_data[i] -= rhs_data[i];
  }
  return {new_data, lhs_shape};
}

Tensor operator-(const float sclr, const Tensor& tnsr) {
  std::vector<float> new_data((*tnsr.data_));
  std::ranges::transform(
      new_data, new_data.begin(), [sclr](float x) { return x - sclr; });
  return {new_data, tnsr.shape_};
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
  const std::vector<int64_t>& lhs_shape = lhs.shape();
  const std::vector<int64_t>& rhs_shape = rhs.shape();
  if (lhs_shape != rhs_shape) {
    throw std::out_of_range(
        std::format("Cannot perform elementwise addition for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs_shape),
                    utils::vector_to_string(rhs_shape)));
  }

  const size_t num_elements = std::accumulate(
      lhs_shape.begin(), lhs_shape.end(), 1, std::multiplies<>());

  std::vector<float> new_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);

  for (idx_t i{}; i < num_elements; i++) {
    new_data[i] *= rhs_data[i];
  }
  return {new_data, lhs_shape};
}

Tensor operator*(const float sclr, const Tensor& tnsr) {
  std::vector<float> new_data((*tnsr.data_));
  std::ranges::transform(
      new_data, new_data.begin(), [sclr](float x) { return x * sclr; });
  return {new_data, tnsr.shape_};
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
  const std::vector<int64_t>& lhs_shape = lhs.shape();
  const std::vector<int64_t>& rhs_shape = rhs.shape();
  if (lhs_shape != rhs_shape) {
    throw std::out_of_range(
        std::format("Cannot perform elementwise addition for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs_shape),
                    utils::vector_to_string(rhs_shape)));
  }

  const size_t num_elements = std::accumulate(
      lhs_shape.begin(), lhs_shape.end(), 1, std::multiplies<>());

  std::vector<float> new_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);

  for (idx_t i{}; i < num_elements; i++) {
    new_data[i] /= rhs_data[i];
  }
  return {new_data, lhs_shape};
}

Tensor operator/(const float sclr, const Tensor& tnsr) {
  std::vector<float> new_data((*tnsr.data_));
  std::ranges::transform(
      new_data.begin(), new_data.end(), new_data.begin(), [sclr](float x) {
        return x / sclr;
      });
  return {new_data, tnsr.shape_};
}

}  // namespace axon