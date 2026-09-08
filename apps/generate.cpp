#include <iostream>

#include <gpt2/config.hpp>

int main() {
    const gpt2::Config config;

    std::cout << "GPT-2 Inference Engine\n";
    std::cout << "Vocabulary size: " << config.vocab_size << '\n';
    std::cout << "Context length: " << config.context_length << '\n';
    std::cout << "Embedding dimension: " << config.embedding_dim << '\n';
    std::cout << "Transformer layers: " << config.num_layers << '\n';
    std::cout << "Attention heads: " << config.num_heads << '\n';
    std::cout << "Head dimension: " << config.head_dim() << '\n';

    return 0;
}