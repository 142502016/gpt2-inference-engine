#include <gtest/gtest.h>
#include <gpt2/tensor.hpp>
#include <limits>

TEST(TensorTest,CreatesTensorWithCorrectShape){
    const gpt2::Tensor tensor({2,3});
    EXPECT_EQ(tensor.shape().size(),2);
    EXPECT_EQ(tensor.shape()[0],2);
    EXPECT_EQ(tensor.shape()[1],3);
}

TEST(TensorTest, CalculatesCorrectNumberOfElements)
{
    const gpt2::Tensor tensor({2, 3, 4});

    EXPECT_EQ(tensor.size(), 24);
}

TEST(TensorTest, DataCanBeReadAndWritten)
{
    gpt2::Tensor tensor({2, 3});
    tensor.data()[0]=42.0f;
    tensor.data()[5]=-3.5f;
   EXPECT_FLOAT_EQ(tensor.data()[0], 42.0f);
   EXPECT_FLOAT_EQ(tensor.data()[5], -3.5f);
}

TEST(TensorTest, RejectsEmptyShape)
{
    EXPECT_THROW(
        gpt2::Tensor(std::vector<std::size_t>{}),
        std::invalid_argument
    );
}

TEST(TensorTest, RejectsZeroDimension)
{
    EXPECT_THROW(
        gpt2::Tensor({2, 0, 4}),
        std::invalid_argument
    );
}

TEST(TensorTest, RejectsSizeOverflow)

{

    const std::size_t max_size =

        std::numeric_limits<std::size_t>::max();

    EXPECT_THROW(

        gpt2::Tensor({max_size, 2}),

        std::overflow_error

    );

}

TEST(TensorTest, ConstTensorProvidesReadOnlyData)
{
    const gpt2::Tensor tensor({2, 3});

    const float* data = tensor.data();

    EXPECT_EQ(data, tensor.data());
}

TEST(TensorTest, Supports2DIndexing)
{
    gpt2::Tensor tensor({2, 3});

    tensor.at(0, 0) = 1.0f;
    tensor.at(0, 1) = 2.0f;
    tensor.at(1, 0) = 4.0f;
    tensor.at(1, 2) = 6.0f;

    EXPECT_FLOAT_EQ(tensor.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(tensor.at(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(tensor.at(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(tensor.at(1, 2), 6.0f);
}


TEST(TensorTest, RejectsOutOfBoundsIndex)
{
    gpt2::Tensor tensor({2, 3});

    EXPECT_THROW(
        tensor.at(2, 0),
        std::out_of_range
    );

    EXPECT_THROW(
        tensor.at(0, 3),
        std::out_of_range
    );
}

TEST(TensorTest, Rejects2DIndexingForNon2DTensor)
{
    gpt2::Tensor tensor({2, 3, 4});

    EXPECT_THROW(
        tensor.at(0, 0),
        std::invalid_argument
    );
}

TEST(TensorTest, Supports3DIndexing)
{
    gpt2::Tensor tensor({2, 3, 4});

    tensor.at({0, 0, 0}) = 1.0f;
    tensor.at({0, 1, 2}) = 12.0f;
    tensor.at({1, 2, 3}) = 42.0f;

    EXPECT_FLOAT_EQ(tensor.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(tensor.at({0, 1, 2}), 12.0f);
    EXPECT_FLOAT_EQ(tensor.at({1, 2, 3}), 42.0f);
}

TEST(TensorTest, RejectsWrongNumberOfIndices)
{
    gpt2::Tensor tensor({2, 3, 4});

    EXPECT_THROW(
        tensor.at({1, 2}),
        std::invalid_argument
    );

    EXPECT_THROW(
        tensor.at({1, 2, 3, 4}),
        std::invalid_argument
    );
}

TEST(TensorTest, RejectsOutOfBoundsNDIndex)
{
    gpt2::Tensor tensor({2, 3, 4});

    EXPECT_THROW(
        tensor.at({2, 0, 0}),
        std::out_of_range
    );

    EXPECT_THROW(
        tensor.at({0, 3, 0}),
        std::out_of_range
    );

    EXPECT_THROW(
        tensor.at({0, 0, 4}),
        std::out_of_range
    );
}


