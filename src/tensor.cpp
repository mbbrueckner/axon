#include "axon/tensor.hpp"


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


