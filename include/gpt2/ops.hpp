#pragma once

#include "gpt2/tensor.hpp"

namespace gpt2 {

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& input);
Tensor matmul_naive(const Tensor& a, const Tensor& b);
}