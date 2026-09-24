/**
 * @file parallel.hpp
 * @brief Index-range iteration primitive shared by all CPU kernels.
 * @author Mika Brückner
 * @date 2026-09-17
 */

#pragma once
#include <cassert>

#include "axon/constants.hpp"

/// @brief Internal implementation utilities; not part of the public API.
namespace axon::internal {

constexpr idx_t GRAIN_SIZE = 32768;

/**
 * @brief Applies @p operation to sub-ranges covering [begin, end).
 *
 * The sub-ranges are disjoint, cover the whole range, and may be processed
 * concurrently; @p operation must therefore write only to memory determined
 * by the indices it receives. It is never invoked with an empty sub-range.
 *
 * @pre @c begin <= @c end
 * @pre @c grain_size > 0
 *
 * @tparam Op Callable with the signature @c void(idx_t,idx_t).
 * @param begin First index of the range.
 * @param end One past the last index of the range.
 * @param grain_size Smallest number of indices a sub-range should span for
 *        splitting to pay off.
 * @param operation Invoked as @c operation(sub_begin,sub_end) per
 * sub-range.
 */
template <typename Op>
void parallel_for(const idx_t begin,
                  const idx_t end,
                  [[maybe_unused]] const idx_t grain_size,
                  const Op& operation) {
  assert(begin <= end);
  assert(grain_size > 0);

  if (begin == end) return;

  operation(begin, end);
}

}  // namespace axon::internal
