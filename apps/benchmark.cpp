#include "gpt2/ops.hpp"

#include <chrono>
#include <iostream>

double benchmark_naive(const gpt2::Tensor& A,
                       const gpt2::Tensor& B,
                       int iterations)
{
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::steady_clock::now();

        gpt2::Tensor result = gpt2::matmul_naive(A, B);

        auto end = std::chrono::steady_clock::now();

        total_ms += std::chrono::duration<double, std::milli>(
            end - start).count();
    }

    return total_ms / iterations;
}

double benchmark_optimized(const gpt2::Tensor& A,
                           const gpt2::Tensor& B,
                           int iterations)
{
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::steady_clock::now();

        gpt2::Tensor result = gpt2::matmul(A, B);

        auto end = std::chrono::steady_clock::now();

        total_ms += std::chrono::duration<double, std::milli>(
            end - start).count();
    }

    return total_ms / iterations;
}

void run_benchmark(std::size_t size, int iterations)
{
    gpt2::Tensor A({size, size});
    gpt2::Tensor B({size, size});

    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            A.at(i, j) = 1.0f;
            B.at(i, j) = 1.0f;
        }
    }

    // Correctness check
    gpt2::Tensor reference = gpt2::matmul_naive(A, B);
    gpt2::Tensor optimized = gpt2::matmul(A, B);

    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            if (reference.at(i, j) != optimized.at(i, j)) {
                std::cerr << "ERROR: Results do not match!\n";
                return;
            }
        }
    }

    std::cout << "Correctness: PASS\n";

    // Warm-up
    gpt2::matmul_naive(A, B);
    gpt2::matmul(A, B);

    const double naive_ms =
        benchmark_naive(A, B, iterations);

    const double optimized_ms =
        benchmark_optimized(A, B, iterations);

    const double speedup = naive_ms / optimized_ms;

    std::cout << size << " x " << size << '\n';
    std::cout << "  Naive:     " << naive_ms << " ms\n";
    std::cout << "  Optimized: " << optimized_ms << " ms\n";
    std::cout << "  Speedup:   " << speedup << "x\n\n";
}
int main()
{
    constexpr int iterations = 10;

    run_benchmark(128, iterations);
    run_benchmark(256, iterations);
    run_benchmark(512, iterations);

    return 0;
}