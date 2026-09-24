/**
 * @file loops.hpp
 * @brief CPU loop drivers that run an elementwise operation over a plan.
 * @author Mika Brückner
 * @date 2026-09-24
 */

#pragma once
#include <vector>

#include "axon/constants.hpp"
#include "core/parallel.hpp"
#include "core/tensor_iterator.hpp"

namespace axon::internal {

/**
 * @brief Advances @p index by one element and keeps @p offsets in sync.
 *
 * Counts odometer-style, innermost dimension first, carrying into the next
 * outer dimension on overflow. @p offsets is updated incrementally rather
 * than recomputed, which costs one addition per operand per element.
 *
 * @param shape Size along each dimension.
 * @param num_dim Number of dimensions.
 * @param strides Strides per operand, indexed as @c strides[operand][dim].
 * @param index Per-dimension index, advanced in place.
 * @param offsets Flat offset per operand, advanced in place.
 */
inline void advance_offsets(const idx_t* shape,
                            const idx_t num_dim,
                            const std::vector<const idx_t*>& strides,
                            std::vector<idx_t>& index,
                            std::vector<idx_t>& offsets) {
  const idx_t num_ops = static_cast<idx_t>(strides.size());

  for (idx_t d = num_dim - 1; d >= 0; d--) {
    index[d]++;
    for (idx_t k = 0; k < num_ops; k++) offsets[k] += strides[k][d];

    if (index[d] < shape[d]) return;

    index[d] = 0;
    for (idx_t k = 0; k < num_ops; k++) offsets[k] -= shape[d] * strides[k][d];
  }
}

/**
 * @brief Applies @p fn to every element described by @p iter.
 *
 * Reduces the plan to a flat index range, hands that range to parallel_for()
 * and walks each sub-range, mapping indices to addresses through the plan's
 * strides. Operand 0 is written, operand 1 is read.
 *
 * @tparam Fn Callable with the signature @c float(float).
 * @param iter Plan describing shape, strides and base pointers.
 * @param fn Applied to each input element; the result goes to the output.
 *
 */
template <typename Fn>
void cpu_kernel(const TensorIterator& iter, const Fn& fn) {
  const idx_t num_elements = iter.num_elements();
  const idx_t num_dim = iter.num_dim();
  const idx_t num_ops = iter.num_operands();
  const idx_t* shape = iter.shape().data();

  std::vector<const idx_t*> strides(num_ops);
  for (idx_t k = 0; k < num_ops; k++) strides[k] = iter.stride(k).data();

  float* out_data = iter.data(0);
  const float* in_data = iter.data(1);

  parallel_for(
      0, num_elements, GRAIN_SIZE, [&](const idx_t begin, const idx_t end) {
        std::vector<idx_t> index(num_dim, 0);
        std::vector<idx_t> offsets(num_ops, 0);

        idx_t rest = begin;
        for (idx_t d = num_dim - 1; d >= 0; d--) {
          index[d] = rest % shape[d];
          rest /= shape[d];
          for (idx_t k = 0; k < num_ops; k++)
            offsets[k] += index[d] * strides[k][d];
        }

        for (idx_t i = begin; i < end; i++) {
          out_data[offsets[0]] = fn(in_data[offsets[1]]);
          advance_offsets(shape, num_dim, strides, index, offsets);
        }
      });
}

}  // namespace axon::internal
