/**
 * @file unary.cpp
 * @brief Elementwise unary operations on axon::Tensor.
 * @author Mika Brückner
 * @date 2026-09-25
 */

#include <ranges>
#include <vector>

#include "autograd/autograd.hpp"
#include "axon/tensor.hpp"
#include "core/tensor_iterator.hpp"
#include "ops/unary_kernel.hpp"
#include "utils.hpp"

namespace axon {
Tensor Tensor::relu() const {
  Tensor result = Tensor::zeros(shape_);
  const internal::TensorIterator iter =
      internal::TensorIterator::unary_op(result, *this);
  internal::relu_kernel(iter);

  auto input_meta = autograd_meta_;
  if (input_meta != nullptr) {
    auto meta = std::make_shared<AutogradMeta>(result.shape());
    meta->grad_fn_ = std::make_shared<GradFn>();
    meta->grad_fn_->backward = [input_meta,
                                input = *this](const Tensor& grad_output) {
      if (input_meta) {
        std::vector<float> mask(input.num_elements());

        if (input.is_contiguous()) {
          std::ranges::transform(
              input.data_->begin() + input.offset_,
              input.data_->begin() + input.offset_ + input.num_elements(),
              mask.begin(),
              [](float x) { return x > 0.0f ? 1.0f : 0.0f; });
        } else {
          for (idx_t i = 0; i < input.num_elements(); i++) {
            const float val = input.at(utils::flat_to_indices(i, input.shape_));
            mask[i] = val > 0.0f ? 1.0f : 0.0f;
          }
        }

        Tensor mask_tensor{mask, input.shape_};
        *input_meta->grad += mask_tensor * grad_output;
      }
    };
    meta->grad_fn_->inputs = {input_meta};
    result.autograd_meta_ = meta;
  }
  return result;
}

Tensor Tensor::log() const {
  Tensor result = Tensor::zeros(shape_);
  const internal::TensorIterator iter =
      internal::TensorIterator::unary_op(result, *this);
  internal::log_kernel(iter);

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
  Tensor result = Tensor::zeros(shape_);
  const internal::TensorIterator iter =
      internal::TensorIterator::unary_op(result, *this);
  internal::exp_kernel(iter);

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
  Tensor result = Tensor::zeros(shape_);
  const internal::TensorIterator iter =
      internal::TensorIterator::unary_op(result, *this);
  internal::abs_kernel(iter);

  return result;
}

}  // namespace axon