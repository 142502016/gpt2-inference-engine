#pragma once
#include <cstddef>
#include <vector>
namespace gpt2{

 class Tensor{
    public:
    Tensor()=default;
    explicit Tensor(const std::vector<std::size_t> &shape);

    const std::vector<std::size_t>& shape() const noexcept;

    std::size_t size() const noexcept;

    float* data() noexcept;

    const float* data() const noexcept;
    private:
    std::vector<std::size_t>shape_;
    std::vector<float>data_;


  };
}

