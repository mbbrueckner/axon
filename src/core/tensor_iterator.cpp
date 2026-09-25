#include "tensor_iterator.hpp"

#include <cassert>
#include <functional>
#include <numeric>
#include <utility>

#include "axon/tensor.hpp"

namespace axon::internal {
TensorIterator::TensorIterator(std::vector<idx_t> shape,
                               std::vector<Operand> operands)
    : shape_(std::move(shape)),
      operands_(std::move(operands)),
      num_elements_(std::accumulate(
          shape_.begin(), shape_.end(), idx_t{1}, std::multiplies<>())) {}

TensorIterator TensorIterator::unary_op(Tensor& output, const Tensor& input) {
  assert(output.shape() == input.shape());

  const Operand out_op{output.data_->data() + output.offset(), output.stride()};
  const Operand in_op{input.data_->data() + input.offset(), input.stride()};

  return TensorIterator{output.shape(), std::vector<Operand>{out_op, in_op}};
}

idx_t TensorIterator::num_elements() const { return num_elements_; }

idx_t TensorIterator::num_dim() const { return shape_.size(); }

idx_t TensorIterator::num_operands() const { return operands_.size(); }

const std::vector<idx_t>& TensorIterator::shape() const { return shape_; }

const std::vector<idx_t>& TensorIterator::stride(idx_t operand) const {
  assert(operand >= 0 && operand < num_operands());
  return operands_[operand].strides;
}

float* TensorIterator::data(idx_t operand) const {
  assert(operand >= 0 && operand < num_operands());
  return operands_[operand].data;
}

}  // namespace axon::internal