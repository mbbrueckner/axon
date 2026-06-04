#include "axon/tensor.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>

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
      data_(std::make_shared<std::vector<float>>(data)) {}

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
    for (int64_t i = static_cast<int64_t>(dim) - 2; i >= 0; i--) {
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

  for (size_t i = 0; i < dim - 1; i++) {
    if (stride_[i] != shape_[i + 1] * stride_[i + 1]) return false;
  }
  return true;
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