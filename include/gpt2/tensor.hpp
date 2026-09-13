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

    float& at(std::size_t row,std::size_t column);

    const float& at(std::size_t row,std::size_t column) const;


  float& at(const std::vector<std::size_t>& indices);

  const float& at(const std::vector<std::size_t>& indices) const;

    const float* data() const noexcept;
    private:
    std::vector<std::size_t>shape_;
    std::vector<float>data_;


  };
}

