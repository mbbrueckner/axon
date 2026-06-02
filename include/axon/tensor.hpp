//
// Created by Mika Brückner on 02.06.26.
//

#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <vector>
#include <numeric>

namespace axon {
     class Tensor;
}


class axon::Tensor {
private:
     std::shared_ptr<std::vector<float> > data_;
     std::vector<size_t> shape_;
     std::vector<int64_t> stride_;

     size_t offset_;

public:
     Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);

     //TODO [] operator

     const std::vector<size_t> &shape() const { return shape_; };
     const std::vector<int64_t> &stride() const { return stride_; };
     size_t offset() const { return offset_; };

     template<typename... Idx>
     float at(Idx... indices) const;

     size_t num_dim() const { return shape_.size(); };

     size_t num_elements() const {
          return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
     };
};
#endif //AXON_TENSOR_HPP
