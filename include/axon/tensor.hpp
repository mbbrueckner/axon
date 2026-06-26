/**
 * @file tensor.hpp
 * @brief Declaration of the axon::Tensor class, an N-dimensional array of
 *        @c float values with strided, shared-storage views.
 * @author Mika Brückner
 * @date 2026-06-02
 */

#pragma once

#include <memory>
#include <numeric>
#include <vector>

#include "axon/constants.hpp"

namespace axon {
struct AutogradMeta;
class Tensor;
}  // namespace axon

/**
 * @class axon::Tensor
 * @brief A multi-dimensional array of @c float values.
 *
 * A Tensor describes a view over a contiguous block of @c float data through a
 * shape, a per-dimension stride, and a flat offset. The underlying data buffer
 * is reference-counted (@c std::shared_ptr), so views such as those produced by
 * transpose(), reshape() or operator[]() share storage with the original tensor
 * instead of copying it.
 *
 * Indices follow row-major ordering by default.
 */
class axon::Tensor {
 private:
  /// Size of the tensor along each dimension.
  std::vector<idx_t> shape_;
  /// Number of flat elements to step over per index in each dimension.
  std::vector<idx_t> stride_;
  /// Flat index of the first element within the shared data buffer.
  idx_t offset_;
  /// Reference-counted backing storage shared between views.
  std::shared_ptr<std::vector<float>> data_;
  /// Autograd meta data.
  std::shared_ptr<AutogradMeta> autograd_meta_;
  /**
   * @brief Constructs a tensor view from explicit storage parameters.
   *
   * Private low-level constructor used internally to create views that share
   * the given data buffer (e.g. by transpose(), reshape() and operator[]()).
   *
   * @param data   Shared backing storage.
   * @param shape  Size along each dimension.
   * @param stride Step size along each dimension.
   * @param offset Flat index of the first element within @p data.
   */
  Tensor(std::shared_ptr<std::vector<float>> data,
         std::vector<idx_t> shape,
         std::vector<idx_t> stride,
         idx_t offset);

  /**
   * @brief Constructs a tensor from a flat data buffer and a shape.
   * @param data  Flat element values in row-major order.
   * @param shape Size along each dimension.
   * @throws std::out_of_range if @p data.size() does not equal the product of
   *         @p shape.
   */
  Tensor(const std::vector<float>& data, const std::vector<idx_t>& shape);

  /**
   * @brief Computes the row-major (contiguous) strides for a given shape.
   * @param shape Size along each dimension.
   * @return The stride for each dimension.
   */
  static std::vector<idx_t> calculate_strides(const std::vector<idx_t>& shape);

  /**
   * @brief Checks whether the tensor's elements are laid out contiguously.
   * @return @c true if the current strides match a dense row-major layout.
   */
  [[nodiscard]] bool is_contiguous() const;

 public:
  /**
   * @brief Constructs a zero-initialized tensor of the given shape.
   * @param shape Size along each dimension.
   */
  static Tensor zeros(std::vector<idx_t> shape);

  /**
   * @brief Constructs a one-initialized tensor of the given shape.
   * @param shape Size along each dimension.
   */
  static Tensor ones(std::vector<idx_t> shape);

  /**
   * @brief Constructs a zero-initialized tensor of the given shape.
   * @param data Flat element values in row-major order.
   * @param shape Size along each dimension.
   */
  static Tensor from_data(std::vector<float> data, std::vector<idx_t> shape);

  /// @brief Destroys the Tensor and releases its resources.
  ~Tensor();

  /// @brief Move-constructs a Tensor, transferring ownership of all resources.
  Tensor(Tensor&&) noexcept;

  /// @brief Move-assigns a Tensor, transferring ownership of all resources.
  Tensor& operator=(Tensor&&) noexcept;

  /// @brief Copy-constructs a Tensor, sharing the underlying storage.
  /// @note The new Tensor has no autograd history.
  Tensor(const Tensor&);

  /// @brief Copy-assigns a Tensor, sharing the underlying storage.
  /// @note The new Tensor has no autograd history.
  Tensor& operator=(const Tensor&);

  ///@brief Copy-constructs a Tensor, sharing the underlying storage.
  /// @note The new Tensor shares the autograd history.
  [[nodiscard]] Tensor shared_autograd_copy() const;

  /// @return The size of the tensor along each dimension.
  [[nodiscard]] const std::vector<idx_t>& shape() const { return shape_; };
  /// @return The stride of the tensor along each dimension.
  [[nodiscard]] const std::vector<idx_t>& stride() const { return stride_; };
  /// @return The flat offset of the first element within the data buffer.
  [[nodiscard]] idx_t offset() const { return offset_; };

  /**
   * @brief Returns a copy of the tensor's elements as a flat vector.
   *
   * The returned vector contains exactly @c num_elements() values in
   * row-major order, regardless of whether the tensor is contiguous.
   *
   * @return A flat copy of the tensor's data.
   */
  [[nodiscard]] std::vector<float> data() const;

  /**
   * @brief Overwrites the tensor's elements in-place without replacing
   *        the underlying storage.
   *
   * Writes @p new_data directly into the shared data buffer, so all views
   * and copies that share the same storage (e.g. optimizer parameter copies
   * created via @c shared_autograd_copy()) see the updated values immediately.
   *
   * @param new_data Flat replacement values in row-major order.
   * @throws std::out_of_range if @p new_data.size() != num_elements().
   */
  void set_data(const std::vector<float>& new_data) const;

  /// @return The number of dimensions (rank) of the tensor.
  [[nodiscard]] idx_t num_dim() const { return shape_.size(); };

  /// @return The total number of elements (product of the shape).
  [[nodiscard]] idx_t num_elements() const {
    return std::accumulate(
        shape_.begin(), shape_.end(), idx_t{1}, std::multiplies<>());
  };

  /**
   * @brief Enables or disables gradient tracking for this tensor.
   *
   * When enabled, fresh autograd metadata with a zero-initialized gradient is
   * attached; when disabled, any existing autograd history is dropped.
   *
   * @param requires_grad @c true to start tracking gradients, @c false to stop.
   */
  void requires_grad_(bool requires_grad);

  /// @return @c true if this tensor tracks gradients for autograd.
  [[nodiscard]] bool requires_grad() const { return autograd_meta_ != nullptr; }

  /**
   * @brief Returns the gradient accumulated for this tensor.
   * @return The gradient tensor, matching this tensor's shape.
   * @throws std::runtime_error if this tensor does not track gradients.
   */
  [[nodiscard]] Tensor grad() const;

  /**
   * @brief Returns a mutable reference to the gradient tensor.
   *
   * Unlike @c grad(), this returns a direct reference to the gradient held
   * inside the autograd metadata, allowing in-place modification (e.g. via
   * @c set_data()). Use this when you need to modify the gradient directly,
   * such as in @c zero_grad().
   *
   * @return A mutable reference to the gradient tensor.
   * @throws std::runtime_error if this tensor does not track gradients.
   */
  [[nodiscard]] Tensor& grad_ref();

  /**
   * @brief Runs the reverse-mode backward pass starting from this tensor.
   *
   * Seeds this tensor's gradient with ones, builds a topological ordering of
   * the computation graph reachable through its autograd history, and invokes
   * each node's GradFn in reverse order to accumulate gradients into the leaf
   * tensors.
   */
  void backward();

  /**
   * @brief Returns the sub-tensor obtained by indexing the first dimension.
   *
   * Produces a view with one fewer dimension that shares storage with this
   * tensor.
   *
   * @param idx Index along the first dimension.
   * @return The (rank - 1)-dimensional sub-tensor at @p idx.
   * @throws std::out_of_range if the tensor is 0-dimensional or @p idx is out
   *         of bounds.
   */
  [[nodiscard]] Tensor operator[](idx_t idx) const;

  /**
   * @brief Returns the scalar value at the given multi-dimensional index.
   * @param indices One index per dimension, in order.
   * @return The element at @p indices.
   * @throws std::out_of_range if the number of indices does not match the rank
   *         or any index is out of bounds.
   */
  [[nodiscard]] float at(std::initializer_list<idx_t> indices) const;

  /**
   * @brief Returns the scalar value at the given multi-dimensional index.
   * @param indices One index per dimension, in order.
   * @return The element at @p indices.
   * @throws std::out_of_range if the number of indices does not match the rank
   *         or any index is out of bounds.
   */
  [[nodiscard]] float at(const std::vector<idx_t>& indices) const;

  /**
   * @brief Returns a view with all dimensions reversed.
   *
   * Shares storage with this tensor; no data is copied. The result is generally
   * non-contiguous.
   *
   * @return The transposed tensor view.
   */
  [[nodiscard]] Tensor transpose() const;

  /**
   * @brief Returns a view of the data interpreted with a new shape.
   * @param new_shape The desired shape.
   * @return A reshaped view sharing storage with this tensor.
   * @throws std::logic_error if the tensor is not contiguous.
   * @throws std::out_of_range if @p new_shape has a different element count.
   */
  [[nodiscard]] Tensor reshape(const std::vector<idx_t>& new_shape) const;

  /**
   * @brief Returns a one-dimensional view of all elements.
   * @return A rank-1 tensor containing every element in row-major order.
   * @throws std::logic_error if the tensor is not contiguous.
   */
  [[nodiscard]] Tensor flatten() const;

  /**
   * @brief Returns a view with a new dimension of size 1 inserted at the
   *        given position.
   *
   * Shares storage with this tensor; no data is copied. The inserted
   * dimension's stride is chosen so that is_contiguous() still holds if this
   * tensor was already contiguous.
   *
   * @param dim Position at which to insert the new dimension. Must satisfy
   *        @c 0 <= dim <= num_dim() (insertion at the end is allowed).
   * @return A view with rank @c num_dim() + 1.
   * @throws std::out_of_range if @p dim is out of the valid range.
   */
  [[nodiscard]] Tensor unsqueeze(idx_t dim) const;

  /**
   * @brief Computes the matrix product of two 2-D tensors.
   * @param other The right-hand matrix operand.
   * @return The resulting matrix product.
   * @throws std::out_of_range if either operand is not 2-dimensional or the
   *         inner dimensions do not match.
   */
  [[nodiscard]] Tensor matmul(const Tensor& other) const;

  /**
   * @brief Returns a new tensor with the natural logarithm applied elementwise.
   * @return The elementwise natural logarithm.
   */
  [[nodiscard]] Tensor log() const;
  /**
   * @brief Returns a new tensor with the exponential applied elementwise.
   * @return The elementwise base-e exponential.
   */
  [[nodiscard]] Tensor exp() const;
  /**
   * @brief Returns a new tensor with the absolute value applied elementwise.
   * @return The elementwise absolute value.
   */
  [[nodiscard]] Tensor abs() const;
  /**
   * @brief Returns a new tensor with the Rectified Linear Unit function (ReLU)
   * applied elementwise.
   * @return The elementwise ReLU value.
   */
  [[nodiscard]] Tensor relu() const;

  /// @return The smallest element value in the tensor.
  [[nodiscard]] float min() const;
  /// @return The largest element value in the tensor.
  [[nodiscard]] float max() const;
  /// @return The sum of all element values.
  [[nodiscard]] Tensor sum() const;

  /**
   * @brief Sums elements along a single dimension.
   * @param dim The dimension to reduce.
   * @param keep_dim If true, the reduced dimension is kept with size 1;
   *        otherwise it is removed from the result shape.
   * @return A tensor with @p dim reduced, summing all elements along it.
   * @throws std::out_of_range if @p dim is out of range for this tensor's rank.
   */
  [[nodiscard]] Tensor sum(idx_t dim, bool keep_dim = true) const;
  /// @return The arithmetic mean of all element values.
  [[nodiscard]] Tensor mean() const;
  /// @return The float-value of zero-dimensional Tensor.
  [[nodiscard]] float item() const;

  [[nodiscard]] Tensor where() const;

  /**
   * @brief Returns a boolean mask tensor with 1.0 where elements are strictly
   *        greater than the corresponding elements of @p other, 0.0 elsewhere.
   *
   * Supports broadcasting between @c this and @p other.
   *
   * @param other The right-hand operand to compare against.
   * @return A tensor of the same shape as the broadcast result, containing
   *         1.0 or 0.0.
   * @throws std::out_of_range if the shapes are not broadcastable.
   */
  [[nodiscard]] Tensor gt(const Tensor& other) const;

  /**
   * @brief Returns a boolean mask tensor with 1.0 where elements are strictly
   *        greater than @p threshold, 0.0 elsewhere.
   *
   * @param threshold The scalar threshold to compare against.
   * @return A tensor of the same shape as @c this, containing 1.0 or 0.0.
   */
  [[nodiscard]] Tensor gt(float threshold) const {
    return gt(Tensor::from_data({threshold}, {1}));
  };

  /**
   * @brief Tests whether @p lhs is elementwise strictly less than @p rhs.
   * @param lhs The left-hand operand.
   * @param rhs The right-hand operand.
   * @return @c true if @c lhs[i] < @c rhs[i] holds for every element,
   * otherwise
   *         @c false.
   * @throws std::out_of_range if the shapes differ.
   */
  friend bool operator<(const Tensor& lhs, const Tensor& rhs);
  /**
   * @brief Tests whether @p lhs is elementwise less than or equal to @p rhs.
   * @param lhs The left-hand operand.
   * @param rhs The right-hand operand.
   * @return @c true if @c lhs[i] <= @c rhs[i] holds for every element,
   *         otherwise @c false.
   * @throws std::out_of_range if the shapes differ.
   */
  friend bool operator<=(const Tensor& lhs, const Tensor& rhs);
  /**
   * @brief Tests whether @p lhs is elementwise strictly greater than @p rhs.
   * @param lhs The left-hand operand.
   * @param rhs The right-hand operand.
   * @return @c true if @c lhs[i] > @c rhs[i] holds for every element,
   * otherwise
   *         @c false.
   * @throws std::out_of_range if the shapes differ.
   */
  friend bool operator>(const Tensor& lhs, const Tensor& rhs);
  /**
   * @brief Tests whether @p lhs is elementwise greater than or equal to
   *        @p rhs.
   * @param lhs The left-hand operand.
   * @param rhs The right-hand operand.
   * @return @c true if @c lhs[i] >= @c rhs[i] holds for every element,
   *         otherwise @c false.
   * @throws std::out_of_range if the shapes differ.
   */
  friend bool operator>=(const Tensor& lhs, const Tensor& rhs);
  /**
   * @brief Tests whether two tensors are elementwise equal.
   * @param lhs The left-hand operand.
   * @param rhs The right-hand operand.
   * @return @c true if @c lhs[i] == @c rhs[i] holds for every element,
   *         otherwise @c false.
   * @throws std::out_of_range if the shapes differ.
   */
  friend bool operator==(const Tensor& lhs, const Tensor& rhs);

  friend std::pair<Tensor, Tensor> broadcast(const Tensor& a, const Tensor& b);

  /**
   * @brief Elementwise addition of two tensors of identical shape.
   * @throws std::out_of_range if the shapes differ.
   */
  friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
  /// @brief Adds a scalar to every element of a tensor.
  friend Tensor operator+(const float sclr, const Tensor& tnsr) {
    return tnsr + Tensor::from_data({sclr}, {1});
  }
  /// @brief Adds a scalar to every element of a tensor.
  friend Tensor operator+(const Tensor& tnsr, const float sclr) {
    return Tensor::from_data({sclr}, {1}) + tnsr;
  }

  /**
   * @brief Elementwise in-place addition of another tensor of identical
   * shape.
   * @param other The tensor to add to this one.
   * @return Reference to this tensor after the addition.
   * @throws std::out_of_range if the shapes differ.
   */
  Tensor& operator+=(const Tensor& other) {
    *this = *this + other;
    return *this;
  }

  /**
   * @brief Elementwise subtraction of two tensors of identical shape.
   * @throws std::out_of_range if the shapes differ.
   */
  friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);
  /// @brief Subtracts a scalar from every element of a tensor.
  friend Tensor operator-(const float sclr, const Tensor& tnsr) {
    return Tensor::from_data({sclr}, {1}) - tnsr;
  }
  /// @brief Subtracts a scalar from every element of a tensor.
  friend Tensor operator-(const Tensor& tnsr, const float sclr) {
    return tnsr - Tensor::from_data({sclr}, {1});
  }
  /// @brief negates all data elements of a tensor.
  Tensor operator-() const;

  /**
   * @brief Elementwise in-place substraction of another tensor of identical
   * shape.
   * @param other The tensor to subtract to this one.
   * @return Reference to this tensor after the substraction.
   * @throws std::out_of_range if the shapes differ.
   */
  Tensor& operator-=(const Tensor& other) {
    *this = *this - other;
    return *this;
  }

  /**
   * @brief Elementwise multiplication of two tensors of identical shape.
   * @throws std::out_of_range if the shapes differ.
   */
  friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);
  /// @brief Multiplies every element of a tensor by a scalar.
  friend Tensor operator*(const float sclr, const Tensor& tnsr) {
    return tnsr * Tensor::from_data({sclr}, {1});
  }
  /// @brief Multiplies every element of a tensor by a scalar.
  friend Tensor operator*(const Tensor& tnsr, const float sclr) {
    return Tensor::from_data({sclr}, {1}) * tnsr;
  }

  /**
   * @brief Elementwise division of two tensors of identical shape.
   * @throws std::out_of_range if the shapes differ.
   */
  friend Tensor operator/(const Tensor& lhs, const Tensor& rhs);
  /// @brief Divides a scalar by every element of a tensor.
  friend Tensor operator/(const float sclr, const Tensor& tnsr) {
    return Tensor::from_data({sclr}, {1}) / tnsr;
  }
  /// @brief Divides every element of a tensor by a scalar.
  friend Tensor operator/(const Tensor& tnsr, const float sclr) {
    return tnsr / Tensor::from_data({sclr}, {1});
  }
};
