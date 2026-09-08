#pragma once

#include <cstddef>
#include <stdexcept>

namespace gpt2 {

struct Config {
    std::size_t vocab_size = 50257;
    std::size_t context_length = 1024;
    std::size_t embedding_dim = 768;
    std::size_t num_layers = 12;
    std::size_t num_heads = 12;
    std::size_t feed_forward_dim = 3072;

    std::size_t head_dim() const {
        if (embedding_dim % num_heads != 0) {
            throw std::invalid_argument(
                "embedding_dim must be divisible by num_heads"
            );
        }

        return embedding_dim / num_heads;
    }
};

}  // namespace gpt2