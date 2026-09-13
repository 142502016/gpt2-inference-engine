#include<gtest/gtest.h>
#include <gpt2/ops.hpp>

TEST(MatMulTest, Multiplies2DMatrices)
{
    gpt2::Tensor a({2, 3});
    gpt2::Tensor b({3, 2});

    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;

    a.at(1, 0) = 4.0f;
    a.at(1, 1) = 5.0f;
    a.at(1, 2) = 6.0f;

    b.at(0, 0) = 7.0f;
    b.at(0, 1) = 8.0f;

    b.at(1, 0) = 9.0f;
    b.at(1, 1) = 10.0f;

    b.at(2, 0) = 11.0f;
    b.at(2, 1) = 12.0f;

    const gpt2::Tensor c = gpt2::matmul(a, b);

    EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 2}));

    EXPECT_FLOAT_EQ(c.at(0, 0), 58.0f);
    EXPECT_FLOAT_EQ(c.at(0, 1), 64.0f);
    EXPECT_FLOAT_EQ(c.at(1, 0), 139.0f);
    EXPECT_FLOAT_EQ(c.at(1, 1), 154.0f);
}

TEST(MatMulTest, RejectsIncompatibleDimensions)
{
    gpt2::Tensor a({2, 3});
    gpt2::Tensor b({4, 2});

    EXPECT_THROW(
        gpt2::matmul(a, b),
        std::invalid_argument
    );
}

TEST(MatMulTest, RejectsNon2DTensors)
{
    gpt2::Tensor a({2, 3, 4});
    gpt2::Tensor b({4, 2});

    EXPECT_THROW(
        gpt2::matmul(a, b),
        std::invalid_argument
    );
}

TEST(TransposeTest, TransposesMatrix)
{
    gpt2::Tensor input({2, 3});

    input.at(0, 0) = 1.0f;
    input.at(0, 1) = 2.0f;
    input.at(0, 2) = 3.0f;

    input.at(1, 0) = 4.0f;
    input.at(1, 1) = 5.0f;
    input.at(1, 2) = 6.0f;

    gpt2::Tensor result = gpt2::transpose(input);

    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 2);

    EXPECT_FLOAT_EQ(result.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result.at(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(result.at(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(result.at(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(result.at(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(result.at(2, 1), 6.0f);
}