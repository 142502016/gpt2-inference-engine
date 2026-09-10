#include <gtest/gtest.h>

#include <gpt2/config.hpp>

TEST(ConfigTest, DefaultValuesMatchGPT2Small) {
    const gpt2::Config config;

    EXPECT_EQ(config.vocab_size, 50257);
    EXPECT_EQ(config.context_length, 1024);
    EXPECT_EQ(config.embedding_dim, 768);
    EXPECT_EQ(config.num_layers, 12);
    EXPECT_EQ(config.num_heads, 12);
    EXPECT_EQ(config.feed_forward_dim, 3072);
}

TEST(ConfigTest, HeadDimensionIsCalculatedCorrectly) {
    const gpt2::Config config;

    EXPECT_EQ(config.head_dim(), 64);
}

