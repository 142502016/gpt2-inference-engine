#include "gpt2/ops.hpp"

namespace gpt2 {

Tensor matmul(const Tensor& a, const Tensor& b)
{
   if(a.shape().size()!=2 || b.shape().size()!=2){
    throw std::invalid_argument(

            "matmul requires 2D tensors"

        );
   }

   const std::size_t M=a.shape()[0];
   const std::size_t K=a.shape()[1];

   const std::size_t K2=b.shape()[0];
   const std::size_t N=b.shape()[1];

   if(K!=K2){
    throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
   }

   Tensor result({M, N});

Tensor b_transposed = transpose(b);

const float* a_data = a.data();

const float* b_data = b_transposed.data();

float* result_data = result.data();

for (std::size_t row = 0; row < M; ++row) {

    const std::size_t a_offset = row * K;

    for (std::size_t column = 0; column < N; ++column) {

        const std::size_t b_offset = column * K;

        float sum = 0.0f;

        for (std::size_t k = 0; k < K; ++k) {

            sum += a_data[a_offset + k] *

                   b_data[b_offset + k];

        }

        result_data[row * N + column] = sum;

    }

}
return result;

}

Tensor matmul_naive(const Tensor& a, const Tensor& b)
{
   if(a.shape().size()!=2 || b.shape().size()!=2){
    throw std::invalid_argument(

            "matmul requires 2D tensors"

        );
   }

   const std::size_t M=a.shape()[0];
   const std::size_t K=a.shape()[1];

   const std::size_t K2=b.shape()[0];
   const std::size_t N=b.shape()[1];

   if(K!=K2){
    throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
   }

   Tensor result({M,N});
   for(std::size_t row=0;row<M;++row){
     
    for(std::size_t column=0;column<N;++column){
          float sum=0.00;
        for(std::size_t k=0;k<K;++k){
            sum+=a.at(row,k)*b.at(column,k);
        }
        result.at(row,column)=sum;
    }
   }
  

   return result;

}


Tensor transpose(const Tensor& input){
    if (input.shape().size() != 2) {

        throw std::invalid_argument("transpose requires a 2D tensor");

    }

    const std::size_t rows = input.shape()[0];

    const std::size_t columns = input.shape()[1];

    Tensor result({columns, rows});

    for (std::size_t row = 0; row < rows; ++row) {

        for (std::size_t column = 0; column < columns; ++column) {

            result.at(column, row) = input.at(row, column);

        }

    }

    return result;

}
}