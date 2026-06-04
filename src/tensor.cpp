#include "axon/tensor.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>

axon::Tensor::Tensor(std::shared_ptr<std::vector<float>> data,
                     std::vector<int64_t> shape,
                     std::vector<int64_t> stride,
                     size_t offset)
    : shape_(std::move(shape)),
      stride_(std::move(stride)),
      offset_(offset),
      data_(std::move(data)) {}

axon::Tensor::Tensor(const std::vector<int64_t>& shape,
                     const std::vector<float>& data)
    : shape_(shape),
      offset_(0),
      data_(std::make_shared<std::vector<float>>(data)) {
  calculate_strides();
}

axon::Tensor::Tensor(const std::vector<int64_t>& shape)
    : shape_(shape),
      offset_(0),
      data_(std::make_shared<std::vector<float>>(std::accumulate(
          shape_.begin(), shape_.end(), 1, std::multiplies<>()))) {
  calculate_strides();
}

void axon::Tensor::calculate_strides() {
  const size_t dim = shape_.size();
  stride_.resize(dim, 1);

  if (dim < 2) return;  // check if tensor or is 0D/1D -> no stride needed

  for (int64_t i = static_cast<int64_t>(dim) - 2; i >= 0; i--) {
    // stride[i] = shape[i+1] * stride[i+1]; stride[dim-1] = 1
    stride_[i] = shape_[i + 1] * stride_[i + 1];
  }
}

float axon::Tensor::at(std::initializer_list<int64_t> indices) const {
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

axon::Tensor axon::Tensor::transpose() const {
  std::vector shape_t(shape_);
  std::ranges::reverse(shape_t);
  std::vector stride_t(stride_);
  std::ranges::reverse(stride_t);

  return {data_, shape_t, stride_t, offset_};
}
