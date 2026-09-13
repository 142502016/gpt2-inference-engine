#include"gpt2/tensor.hpp"
#include <stdexcept>
#include <limits>

namespace gpt2{
 Tensor::Tensor(const std::vector<std::size_t>& shape):shape_(shape){
    if (shape_.empty()) {

        throw std::invalid_argument("Tensor shape cannot be empty");

    }

    for (const std::size_t dimension : shape_) {

        if (dimension == 0) {

            throw std::invalid_argument(

                "Tensor dimensions must be greater than zero"

            );

        }

    }

    std::size_t num_elements = 1;

for (const std::size_t dimension : shape_) {

    if (num_elements > std::numeric_limits<std::size_t>::max() / dimension) {

        throw std::overflow_error(

            "Tensor size exceeds maximum representable size"

        );

    }

    num_elements *= dimension;

}

    data_.resize(num_elements);
 }

 const std::vector<std::size_t>& Tensor::shape() const noexcept {

    return shape_;

}

std::size_t Tensor::size() const noexcept {

    return data_.size();

}

float*Tensor::data() noexcept{
    return data_.data();
}


const float*Tensor::data() const noexcept{
     return data_.data();
}

float& Tensor::at(std::size_t row, std::size_t column){
    if(shape_.size()!=2){
      throw std::invalid_argument(
        "Tensor::at(row,column) requires a 2D tensor"
      );
    }

    if(row>=shape_[0] || column>=shape_[1]){
        throw std::out_of_range(

            "Tensor index out of range"

        );
    }

    const std::size_t index=row*shape_[1]+column;

    return data_[index];


}

const float& Tensor::at(std::size_t row,std::size_t column) const
{
    if (shape_.size() != 2) {
        throw std::invalid_argument(
            "Tensor::at(row, column) requires a 2D tensor"
        );
    }

    if (row >= shape_[0] || column >= shape_[1]) {
        throw std::out_of_range(
            "Tensor index out of range"
        );
    }

    const std::size_t index =
        row * shape_[1] + column;

    return data_[index];
}

float&Tensor::at(const std::vector<std::size_t>&indices){

    if(indices.size()!=shape_.size()){
        throw std::invalid_argument(

            "Number of indices must match tensor dimensions"

        );
    }

    std::size_t offset = 0;

    std::size_t stride = 1;

for (std::size_t dimension = shape_.size();

     dimension-- > 0;) {

    if (indices[dimension] >= shape_[dimension]) {

        throw std::out_of_range(

            "Tensor index out of range"

        );

    }

    if (indices[dimension] >

        std::numeric_limits<std::size_t>::max() / stride) {

        throw std::overflow_error(

            "Tensor offset calculation overflow"

        );

    }

    const std::size_t contribution =

        indices[dimension] * stride;

    if (offset >

        std::numeric_limits<std::size_t>::max() - contribution) {

        throw std::overflow_error(

            "Tensor offset calculation overflow"

        );

    }

    offset += contribution;

    if (stride >

        std::numeric_limits<std::size_t>::max() /

            shape_[dimension]) {

        throw std::overflow_error(

            "Tensor stride calculation overflow"

        );

    }

    stride *= shape_[dimension];

}
 return data_[offset];

}


const float& Tensor::at(
    const std::vector<std::size_t>& indices) const
{
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument(
            "Number of indices must match tensor dimensions"
        );
    }

    std::size_t offset = 0;

std::size_t stride = 1;

for (std::size_t dimension = shape_.size();

     dimension-- > 0;) {

    if (indices[dimension] >= shape_[dimension]) {

        throw std::out_of_range(

            "Tensor index out of range"

        );

    }

    if (indices[dimension] >

        std::numeric_limits<std::size_t>::max() / stride) {

        throw std::overflow_error(

            "Tensor offset calculation overflow"

        );

    }

    const std::size_t contribution =

        indices[dimension] * stride;

    if (offset >

        std::numeric_limits<std::size_t>::max() - contribution) {

        throw std::overflow_error(

            "Tensor offset calculation overflow"

        );

    }

    offset += contribution;

    if (stride >

        std::numeric_limits<std::size_t>::max() /

            shape_[dimension]) {

        throw std::overflow_error(

            "Tensor stride calculation overflow"

        );

    }

    stride *= shape_[dimension];

}
 return data_[offset];
}
}
