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
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "autograd/autograd.hpp"
#include "axon/constants.hpp"
#include "utils.hpp"

namespace axon {
Tensor::Tensor(std::shared_ptr<std::vector<float>> data,
               std::vector<idx_t> shape,
               std::vector<idx_t> stride,
               idx_t offset)
    : shape_(std::move(shape)),
      stride_(std::move(stride)),
      offset_(offset),
      data_(std::move(data)) {}

Tensor::Tensor(const std::vector<float>& data, const std::vector<idx_t>& shape)
    : shape_(shape),
      stride_(calculate_strides(shape)),
      offset_(0),
      data_(std::make_shared<std::vector<float>>(data)) {
  if (num_elements() != static_cast<idx_t>(data.size())) {
    throw std::out_of_range(
        std::format("Number of elements does not match shape: got {} "
                    "expected {} for shape {}",
                    data.size(),
                    num_elements(),
                    axon::utils::vector_to_string(shape)));
  }
}

Tensor::Tensor(const std::vector<idx_t>& shape)
    : shape_(shape),
      stride_(calculate_strides(shape)),
      offset_(0),
      data_(std::make_shared<std::vector<float>>(std::accumulate(
          shape_.begin(), shape_.end(), 1, std::multiplies<>()))) {}

Tensor::Tensor(const std::vector<idx_t>& shape, float fill_value)
    : shape_(shape),
      stride_(calculate_strides(shape)),
      offset_(0),
      data_(std::make_shared<std::vector<float>>(std::accumulate(
          shape_.begin(), shape_.end(), 1, std::multiplies<>()))) {
  std::ranges::fill(*data_, fill_value);
}

Tensor::~Tensor() = default;

Tensor::Tensor(Tensor&&) noexcept = default;

Tensor& Tensor::operator=(Tensor&&) noexcept = default;

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      stride_(other.stride_),
      offset_(other.offset_),
      data_(other.data_),
      autograd_meta_(nullptr) {}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    shape_ = other.shape_;
    stride_ = other.stride_;
    offset_ = other.offset_;
    data_ = other.data_;
    autograd_meta_ = nullptr;
  }
  return *this;
}

std::vector<idx_t> Tensor::calculate_strides(const std::vector<idx_t>& shape) {
  const idx_t dim = shape.size();
  std::vector<idx_t> stride;
  stride.resize(dim, 1);

  if (dim >= 2) {
    // check if tensor or is 0D/1D -> no stride needed
    for (idx_t i = dim - 2; i >= 0; i--) {
      // stride[i] = shape[i+1] * stride[i+1]; stride[dim-1] = 1
      stride[i] = shape[i + 1] * stride[i + 1];
    }
  }

  return stride;
}

bool Tensor::is_contiguous() const {
  const idx_t dim = shape_.size();

  if (dim == 0) return true;

  if (stride_.back() != 1) return false;

  for (idx_t i{}; i < dim - 1; i++) {
    if (stride_[i] != shape_[i + 1] * stride_[i + 1]) return false;
  }
  return true;
}

void Tensor::requires_grad_(bool requires_grad) {
  if (requires_grad) {
    autograd_meta_ = std::make_shared<AutogradMeta>(shape_);
  } else {
    autograd_meta_ = nullptr;
  }
}
Tensor Tensor::grad() const {
  if (!autograd_meta_) throw std::runtime_error("Tensor has no gradient");
  return *autograd_meta_->grad;
}

void Tensor::backward() {
  autograd_meta_->grad = std::make_shared<Tensor>(shape_, 1.0f);

  std::vector<AutogradMeta*> topo;
  std::unordered_set<AutogradMeta*> visited;

  std::function<void(AutogradMeta*)> build_topo = [&](AutogradMeta* node) {
    if (visited.contains(node)) return;
    visited.insert(node);

    if (node->grad_fn_) {
      for (auto& input_meta : node->grad_fn_->inputs) {
        build_topo(input_meta.get());
      }
    }

    topo.push_back(node);
  };

  build_topo(autograd_meta_.get());

  for (const auto& it : std::views::reverse(topo)) {
    if (it->grad_fn_) {
      it->grad_fn_->backward(*it->grad);
    }
  }
}

Tensor Tensor::operator[](idx_t idx) const {
  if (shape_.empty()) {
    throw std::out_of_range(
        "Cannot subscript a 0-dimensional tensor (too many subscripts)");
  }
  if (idx >= shape_[0]) {
    throw std::out_of_range(
        std::format("Index out of bounds: expected index to be < {}, got{}",
                    shape_[0],
                    idx));
  }

  const idx_t new_offset = idx * (stride_[0]);
  const std::vector<idx_t> new_shape(shape_.begin() + 1, shape_.end());
  const std::vector<idx_t> new_stride(stride_.begin() + 1, stride_.end());
  return {data_, new_shape, new_stride, new_offset};
}

float Tensor::at(std::initializer_list<idx_t> indices) const {
  const idx_t num_dim = shape_.size();
  const idx_t num_indices = indices.size();

  if (num_indices != num_dim) {
    throw std::out_of_range(
        std::format("Number of indices does not match dimensions: got {} "
                    "indices for a {}-dimensional tensor",
                    num_indices,
                    num_dim));
  }

  idx_t dim = 0;
  idx_t flat = offset_;
  for (idx_t idx : indices) {
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

Tensor Tensor::reshape(const std::vector<idx_t>& new_shape) const {
  if (!is_contiguous()) {
    throw std::logic_error("Non-contiguous Tensor can not be reshaped");
  }

  const idx_t old_num_elements = num_elements();
  const idx_t new_num_elements = std::accumulate<>(
      new_shape.begin(), new_shape.end(), idx_t{1}, std::multiplies<>());

  if (old_num_elements != new_num_elements) {
    throw std::out_of_range(
        std::format("New number of elements does not match available: got {} "
                    "elements for reshaped Tensor with {} elements",
                    new_num_elements,
                    old_num_elements));
  }

  std::vector<idx_t> new_stride = calculate_strides(new_shape);
  return {data_, new_shape, new_stride, offset_};
}

Tensor Tensor::flatten() const {
  const std::vector<idx_t> flat_shape = {num_elements()};
  return reshape(flat_shape);
}

Tensor Tensor::matmul(const Tensor& other) const {
  if (num_dim() != 2 || other.num_dim() != 2) {
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
  const idx_t rows = shape_[0];
  const idx_t inner = shape_[1];
  const idx_t cols = other.shape()[1];
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
  const std::vector<idx_t>& lhs_shape = lhs.shape();
  const std::vector<idx_t>& rhs_shape = rhs.shape();
  if (lhs_shape != rhs_shape) {
    throw std::out_of_range(
        std::format("Cannot perform elementwise addition for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs_shape),
                    utils::vector_to_string(rhs_shape)));
  }

  const idx_t num_elements = std::accumulate(
      lhs_shape.begin(), lhs_shape.end(), idx_t{1}, std::multiplies<>());

  std::vector<float> new_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);

  for (idx_t i{}; i < num_elements; i++) {
    new_data[i] += rhs_data[i];
  }

  Tensor result{new_data, lhs_shape};
  auto lhs_meta = lhs.autograd_meta_;
  auto rhs_meta = rhs.autograd_meta_;
  if (lhs_meta != nullptr || rhs_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward =
        [lhs_meta, rhs_meta, lhs, rhs](const Tensor& grad_output) {
          if (lhs_meta) *lhs_meta->grad += grad_output;
          if (rhs_meta) *rhs_meta->grad += grad_output;
        };
    meta->grad_fn_->inputs = {lhs_meta, rhs_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor operator+(const float sclr, const Tensor& tnsr) {
  std::vector<float> new_data((*tnsr.data_));
  std::ranges::transform(
      new_data, new_data.begin(), [sclr](float x) { return x + sclr; });
  return {new_data, tnsr.shape_};
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
  const std::vector<idx_t>& lhs_shape = lhs.shape();
  const std::vector<idx_t>& rhs_shape = rhs.shape();
  if (lhs_shape != rhs_shape) {
    throw std::out_of_range(
        std::format("Cannot perform elementwise addition for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs_shape),
                    utils::vector_to_string(rhs_shape)));
  }

  const idx_t num_elements = std::accumulate(
      lhs_shape.begin(), lhs_shape.end(), idx_t{1}, std::multiplies<>());

  std::vector<float> new_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);

  for (idx_t i{}; i < num_elements; i++) {
    new_data[i] -= rhs_data[i];
  }
  Tensor result{new_data, lhs_shape};
  auto lhs_meta = lhs.autograd_meta_;
  auto rhs_meta = rhs.autograd_meta_;
  if (lhs_meta != nullptr || rhs_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward =
        [lhs_meta, rhs_meta, lhs, rhs](const Tensor& grad_output) {
          if (lhs_meta) *lhs_meta->grad -= grad_output;
          if (rhs_meta) *rhs_meta->grad -= grad_output;
        };
    meta->grad_fn_->inputs = {lhs_meta, rhs_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor operator-(const float sclr, const Tensor& tnsr) {
  std::vector<float> new_data((*tnsr.data_));
  std::ranges::transform(
      new_data, new_data.begin(), [sclr](float x) { return x - sclr; });
  return {new_data, tnsr.shape_};
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
  const std::vector<idx_t>& lhs_shape = lhs.shape();
  const std::vector<idx_t>& rhs_shape = rhs.shape();
  if (lhs_shape != rhs_shape) {
    throw std::out_of_range(
        std::format("Cannot perform elementwise addition for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs_shape),
                    utils::vector_to_string(rhs_shape)));
  }

  const idx_t num_elements = std::accumulate(
      lhs_shape.begin(), lhs_shape.end(), idx_t{1}, std::multiplies<>());

  std::vector<float> new_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);

  for (idx_t i{}; i < num_elements; i++) {
    new_data[i] *= rhs_data[i];
  }

  Tensor result{new_data, lhs_shape};
  auto lhs_meta = lhs.autograd_meta_;
  auto rhs_meta = rhs.autograd_meta_;
  if (lhs_meta != nullptr || rhs_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward =
        [lhs_meta, rhs_meta, lhs, rhs](const Tensor& grad_output) {
          if (lhs_meta) *lhs_meta->grad += rhs * grad_output;
          if (rhs_meta) *rhs_meta->grad += lhs * grad_output;
        };
    meta->grad_fn_->inputs = {lhs_meta, rhs_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor operator*(const float sclr, const Tensor& tnsr) {
  std::vector<float> new_data((*tnsr.data_));
  std::ranges::transform(
      new_data, new_data.begin(), [sclr](float x) { return x * sclr; });
  return {new_data, tnsr.shape_};
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
  const std::vector<idx_t>& lhs_shape = lhs.shape();
  const std::vector<idx_t>& rhs_shape = rhs.shape();
  if (lhs_shape != rhs_shape) {
    throw std::out_of_range(
        std::format("Cannot perform elementwise addition for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs_shape),
                    utils::vector_to_string(rhs_shape)));
  }

  const idx_t num_elements = std::accumulate(
      lhs_shape.begin(), lhs_shape.end(), idx_t{1}, std::multiplies<>());

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