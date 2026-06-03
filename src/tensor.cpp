#include "axon/tensor.hpp"
#include <stdexcept>
#include <format>


axon::Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data)
  : shape_(shape),
    offset_(0),
    data_(std::make_shared<std::vector<float> >(data)) {
  calculate_strides();
}

void axon::Tensor::calculate_strides() {
  const size_t dim = shape_.size();
  stride_.resize(dim, 1);

  if (dim < 2) return; // check if tensor or is 0D/1D -> no stride needed

  for (int i = dim - 2; i >= 0; i--) {
    stride_[i] = shape_[i + 1] * stride_[i + 1]; // stride[i] = shape[i+1] * stride[i+1]; stride[dim-1] = 1
  }
}

float axon::Tensor::at(std::initializer_list<size_t> indices) const {
  const size_t num_dim = shape_.size();
  const size_t num_indices = indices.size();

  if (num_indices != num_dim) {
    throw std::out_of_range(
      std::format("Number of indices does not match dimensions: got {} indices for a {}-dimensional tensor",
                  num_indices,
                  num_dim));
  }

  size_t dim = 0;
  size_t flat = offset_;
  for (size_t idx: indices) {
    if (idx >= shape_[dim]) {
      throw std::out_of_range(
        std::format("Index {} out of range for dimension {} with size {}", idx, dim, shape_[dim]));
    }

    flat += idx * stride_[dim];
    dim++;
  }
  return (*data_)[flat];
}
