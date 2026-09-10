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


}
