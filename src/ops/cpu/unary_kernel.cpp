#include "../unary_kernel.hpp"

#include "loops.hpp"

namespace axon::internal {
void relu_kernel(const TensorIterator& iter) {
  cpu_kernel(iter, [](float x) { return x > 0 ? x : 0; });
}
}  // namespace axon::internal