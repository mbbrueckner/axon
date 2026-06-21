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

Tensor Tensor::zeros(std::vector<idx_t> shape) {
  return Tensor{std::vector<float>(std::accumulate(
                    shape.begin(), shape.end(), 1, std::multiplies<>())),
                shape};
}

Tensor Tensor::ones(std::vector<idx_t> shape) {
  std::vector<float> data = std::vector<float>(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
  std::ranges::fill(data, 1.0f);
  return {data, shape};
}
Tensor Tensor::from_data(std::vector<float> data, std::vector<idx_t> shape) {
  return {data, shape};
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

Tensor Tensor::shared_autograd_copy() const {
  Tensor copy = *this;
  copy.autograd_meta_ = autograd_meta_;
  return copy;
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
  if (!autograd_meta_)
    throw std::runtime_error("backward() called on tensor without grad");

  autograd_meta_->grad = std::make_shared<Tensor>(ones(shape_));

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

  const idx_t new_offset = offset_ + idx * stride_[0];
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

float Tensor::at(const std::vector<idx_t>& indices) const {
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

Tensor Tensor::unsqueeze(idx_t dim) const {
  if (dim < 0 || dim > num_dim()) {
    throw std::out_of_range(
        std::format("Cannot unsqueeze: dimension {} out of range for tensor "
                    "with {} dimensions (valid range: 0 to {})",
                    dim,
                    num_dim(),
                    num_dim()));
  }

  std::vector<idx_t> new_shape = shape_;
  std::vector<idx_t> new_stride = stride_;

  new_shape.insert(new_shape.begin() + dim, 1);
  const idx_t inserted_stride =
      (dim < num_dim()) ? shape_[dim] * stride_[dim] : 1;
  new_stride.insert(new_stride.begin() + dim, inserted_stride);

  return {data_, new_shape, new_stride, offset_};
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
  Tensor result = zeros({rows, cols});
  for (idx_t i{}; i < rows; i++) {
    for (idx_t j{}; j < cols; j++) {
      for (idx_t k{}; k < inner; k++) {
        (*result.data_)[i * cols + j] += at({i, k}) * other.at({k, j});
      }
    }
  }

  auto lhs_meta = autograd_meta_;
  auto rhs_meta = other.autograd_meta_;
  if (lhs_meta != nullptr || rhs_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward = [lhs_meta, rhs_meta, lhs = *this, rhs = other](
                                   const Tensor& grad_output) {
      if (lhs_meta) *lhs_meta->grad += grad_output.matmul(rhs.transpose());
      if (rhs_meta) *rhs_meta->grad += lhs.transpose().matmul(grad_output);
    };
    std::vector<std::shared_ptr<AutogradMeta>> inputs;
    if (lhs_meta) inputs.push_back(lhs_meta);
    if (rhs_meta) inputs.push_back(rhs_meta);
    meta->grad_fn_->inputs = inputs;
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor Tensor::log() const {
  std::vector<float> new_data = (*data_);

  std::ranges::transform(
      new_data, new_data.begin(), [](float x) { return std::log(x); });
  Tensor result{new_data, shape_};
  auto input_meta = autograd_meta_;
  if (input_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward = [input_meta,
                                input = *this](const Tensor& grad_output) {
      if (input_meta) *input_meta->grad += grad_output / input;
    };
    meta->grad_fn_->inputs = {input_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor Tensor::exp() const {
  std::vector<float> new_data = (*data_);

  std::ranges::transform(
      new_data, new_data.begin(), [](float x) { return std::exp(x); });
  Tensor result{new_data, shape_};
  auto input_meta = autograd_meta_;
  if (input_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward = [input_meta, result](const Tensor& grad_output) {
      if (input_meta) *input_meta->grad += result * grad_output;
    };
    meta->grad_fn_->inputs = {input_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor Tensor::abs() const {
  std::vector<float> new_data = (*data_);

  std::ranges::transform(
      new_data, new_data.begin(), [](float x) { return std::abs(x); });
  return {new_data, shape_};
}

Tensor Tensor::relu() const {
  std::vector<float> new_data = (*data_);

  std::ranges::transform(
      new_data, new_data.begin(), [](float x) { return x > 0 ? x : 0; });
  Tensor result{new_data, shape_};
  auto input_meta = autograd_meta_;
  if (input_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward = [input_meta,
                                input = *this](const Tensor& grad_output) {
      if (input_meta) {
        std::vector<float> mask = (*input.data_);
        std::ranges::transform(
            mask, mask.begin(), [](float x) { return x > 0.0f ? 1.0f : 0.0f; });
        Tensor mask_tensor{mask, input.shape_};
        *input_meta->grad += mask_tensor * grad_output;
      }
    };
    meta->grad_fn_->inputs = {input_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

float Tensor::min() const {
  if (is_contiguous()) {
    return *std::ranges::min_element(data_->begin() + offset_,
                                     data_->begin() + offset_ + num_elements());
  }
  float result = at(utils::flat_to_indices(0, shape_));
  for (idx_t i = 1; i < num_elements(); i++) {
    result = std::min(result, at(utils::flat_to_indices(i, shape_)));
  }
  return result;
}

float Tensor::max() const {
  if (is_contiguous()) {
    return *std::ranges::max_element(data_->begin() + offset_,
                                     data_->begin() + offset_ + num_elements());
  }
  float result = at(utils::flat_to_indices(0, shape_));
  for (idx_t i = 1; i < num_elements(); i++) {
    result = std::max(result, at(utils::flat_to_indices(i, shape_)));
  }
  return result;
}

Tensor Tensor::sum() const {
  float total;
  if (is_contiguous()) {
    total = std::accumulate(data_->begin() + offset_,
                            data_->begin() + offset_ + num_elements(),
                            0.0f);
  } else {
    total = 0.0f;
    for (idx_t i = 0; i < num_elements(); i++) {
      total += at(utils::flat_to_indices(i, shape_));
    }
  }
  Tensor result{std::vector<float>{total}, {1}};
  auto input_meta = autograd_meta_;
  if (input_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward =
        [input_meta, input_shape = shape_](const Tensor& grad_output) {
          if (input_meta)
            *input_meta->grad += ones(input_shape) * grad_output.item();
        };
    meta->grad_fn_->inputs = {input_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor Tensor::sum(idx_t dim, bool keep_dim) const {
  std::vector<idx_t> output_shape = shape_;

  if (keep_dim)
    output_shape[dim] = 1;
  else
    output_shape.erase(output_shape.begin() + dim);

  const std::vector<idx_t> output_stride = calculate_strides(output_shape);
  const idx_t output_elements = std::accumulate(
      output_shape.begin(), output_shape.end(), idx_t{1}, std::multiplies<>());
  std::vector<float> new_data(output_elements, 0.0f);

  for (idx_t i = 0; i < num_elements(); i++) {
    std::vector<idx_t> idx = utils::flat_to_indices(i, shape_);

    std::vector<idx_t> output_idx = idx;
    if (keep_dim) {
      output_idx[dim] = 0;
    } else {
      output_idx.erase(output_idx.begin() + dim);
    }

    idx_t output_flat = std::inner_product(
        output_idx.begin(), output_idx.end(), output_stride.begin(), 0.0);
    new_data[output_flat] += at(idx);
  }

  Tensor result(new_data, output_shape);
  auto input_meta = autograd_meta_;
  if (input_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward = [input_meta,
                                input_shape = shape_,
                                dim,
                                keep_dim](const Tensor& grad_output) {
      if (input_meta) {
        const Tensor grad = keep_dim ? grad_output : grad_output.unsqueeze(dim);
        *input_meta->grad += ones(input_shape) * grad;
      }
    };
    meta->grad_fn_->inputs = {input_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor Tensor::mean() const {
  return sum() / static_cast<float>(num_elements());
}

float Tensor::item() const {
  if (num_elements() != 1)
    throw std::runtime_error("item() only works on single-element tensors");
  return (*data_)[offset_];
}

bool operator<(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::out_of_range(
        std::format("Cannot compare for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs.shape()),
                    utils::vector_to_string(rhs.shape())));
  }
  std::vector<float> lhs_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);
  for (idx_t i = 0; i < lhs.num_elements(); i++) {
    if (lhs_data[i] >= rhs_data[i]) return false;
  }
  return true;
}
bool operator<=(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::out_of_range(
        std::format("Cannot compare for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs.shape()),
                    utils::vector_to_string(rhs.shape())));
  }
  std::vector<float> lhs_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);
  for (idx_t i = 0; i < lhs.num_elements(); i++) {
    if (lhs_data[i] > rhs_data[i]) return false;
  }
  return true;
}
bool operator>(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::out_of_range(
        std::format("Cannot compare for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs.shape()),
                    utils::vector_to_string(rhs.shape())));
  }
  std::vector<float> lhs_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);
  for (idx_t i = 0; i < lhs.num_elements(); i++) {
    if (lhs_data[i] <= rhs_data[i]) return false;
  }
  return true;
}

bool operator>=(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::out_of_range(
        std::format("Cannot compare for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs.shape()),
                    utils::vector_to_string(rhs.shape())));
  }

  std::vector<float> lhs_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);
  for (idx_t i = 0; i < lhs.num_elements(); i++) {
    if (lhs_data[i] < rhs_data[i]) return false;
  }
  return true;
}
bool operator==(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::out_of_range(
        std::format("Cannot compare for tensors with "
                    "different shapes: {}; {}",
                    utils::vector_to_string(lhs.shape()),
                    utils::vector_to_string(rhs.shape())));
  }

  std::vector<float> lhs_data = (*lhs.data_);
  std::vector<float> rhs_data = (*rhs.data_);
  for (idx_t i = 0; i < lhs.num_elements(); i++) {
    if (lhs_data[i] != rhs_data[i]) return false;
  }
  return true;
}

std::pair<Tensor, Tensor> broadcast(const Tensor& a, const Tensor& b) {
  const idx_t ndim = std::max(a.num_dim(), b.num_dim());

  std::vector<idx_t> shape_a(ndim, 1), shape_b(ndim, 1);
  std::vector<idx_t> stride_a(ndim, 0), stride_b(ndim, 0);

  for (idx_t i = 0; i < a.num_dim(); i++) {
    shape_a[ndim - a.num_dim() + i] = a.shape()[i];
    stride_a[ndim - a.num_dim() + i] = a.stride()[i];
  }

  for (idx_t i = 0; i < b.num_dim(); i++) {
    shape_b[ndim - b.num_dim() + i] = b.shape()[i];
    stride_b[ndim - b.num_dim() + i] = b.stride()[i];
  }

  std::vector<idx_t> out_shape(ndim);
  for (idx_t i = 0; i < ndim; i++) {
    if (shape_a[i] == shape_b[i]) {
      out_shape[i] = shape_a[i];
    } else if (shape_a[i] == 1) {
      out_shape[i] = shape_b[i];
      stride_a[i] = 0;
    } else if (shape_b[i] == 1) {
      out_shape[i] = shape_a[i];
      stride_b[i] = 0;
    } else {
      throw std::out_of_range("Shapes are not broadcastable");
    }
  }

  return {Tensor(a.data_, out_shape, stride_a, a.offset_),
          Tensor(b.data_, out_shape, stride_b, b.offset_)};
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
  auto [a, b] = broadcast(lhs, rhs);

  const idx_t num_elements = a.num_elements();
  const std::vector<idx_t> out_shape = a.shape();
  std::vector<float> new_data(num_elements);

  for (idx_t i = 0; i < num_elements; i++) {
    std::vector<idx_t> idx = utils::flat_to_indices(i, out_shape);
    new_data[i] = a.at(idx) + b.at(idx);
  }

  Tensor result{new_data, out_shape};
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
    std::vector<std::shared_ptr<AutogradMeta>> inputs;
    if (lhs_meta) inputs.push_back(lhs_meta);
    if (rhs_meta) inputs.push_back(rhs_meta);
    meta->grad_fn_->inputs = inputs;
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
  auto [a, b] = broadcast(lhs, rhs);

  const idx_t num_elements = a.num_elements();
  const std::vector<idx_t> out_shape = a.shape();
  std::vector<float> new_data(num_elements);

  for (idx_t i = 0; i < num_elements; i++) {
    std::vector<idx_t> idx = utils::flat_to_indices(i, out_shape);
    new_data[i] = a.at(idx) - b.at(idx);
  }

  Tensor result{new_data, out_shape};
  auto lhs_meta = lhs.autograd_meta_;
  auto rhs_meta = rhs.autograd_meta_;
  if (lhs_meta != nullptr || rhs_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward =
        [lhs_meta, rhs_meta, lhs, rhs](const Tensor& grad_output) {
          if (lhs_meta) *lhs_meta->grad += grad_output;
          if (rhs_meta) *rhs_meta->grad -= grad_output;
        };
    std::vector<std::shared_ptr<AutogradMeta>> inputs;
    if (lhs_meta) inputs.push_back(lhs_meta);
    if (rhs_meta) inputs.push_back(rhs_meta);
    meta->grad_fn_->inputs = inputs;
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
  auto [a, b] = broadcast(lhs, rhs);

  const idx_t num_elements = a.num_elements();
  const std::vector<idx_t> out_shape = a.shape();
  std::vector<float> new_data(num_elements);

  for (idx_t i = 0; i < num_elements; i++) {
    std::vector<idx_t> idx = utils::flat_to_indices(i, out_shape);
    new_data[i] = a.at(idx) * b.at(idx);
  }

  Tensor result{new_data, out_shape};
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
    std::vector<std::shared_ptr<AutogradMeta>> inputs;
    if (lhs_meta) inputs.push_back(lhs_meta);
    if (rhs_meta) inputs.push_back(rhs_meta);
    meta->grad_fn_->inputs = inputs;
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
  auto [a, b] = broadcast(lhs, rhs);

  const idx_t num_elements = a.num_elements();
  const std::vector<idx_t> out_shape = a.shape();
  std::vector<float> new_data(num_elements);

  for (idx_t i = 0; i < num_elements; i++) {
    std::vector<idx_t> idx = utils::flat_to_indices(i, out_shape);
    new_data[i] = a.at(idx) / b.at(idx);
  }

  Tensor result{new_data, out_shape};
  auto lhs_meta = lhs.autograd_meta_;
  auto rhs_meta = rhs.autograd_meta_;
  if (lhs_meta != nullptr || rhs_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward =
        [lhs_meta, rhs_meta, lhs, rhs](const Tensor& grad_output) {
          if (lhs_meta) *lhs_meta->grad += grad_output / rhs;
          if (rhs_meta) *rhs_meta->grad += -lhs / (rhs * rhs) * grad_output;
        };
    std::vector<std::shared_ptr<AutogradMeta>> inputs;
    if (lhs_meta) inputs.push_back(lhs_meta);
    if (rhs_meta) inputs.push_back(rhs_meta);
    meta->grad_fn_->inputs = inputs;
    result.autograd_meta_ = meta;
  }
  return result;
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
