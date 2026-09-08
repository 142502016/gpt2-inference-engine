# GPT-2 Inference Engine

A from-scratch C++ inference engine for GPT-2, built to understand and
optimize the core components of modern LLM inference systems.

## Project Goals

This project aims to build a clean, modular, and progressively optimized
LLM inference engine while developing a deep understanding of:

- Transformer architecture
- Causal self-attention
- Autoregressive text generation
- Model weight loading
- CPU inference
- Memory management
- Inference profiling
- KV caching
- GPU acceleration
- CUDA kernels
- Batching and request scheduling

The project is developed incrementally, with correctness validated before
performance optimizations are introduced.

## Architecture

The initial inference pipeline is:

    Prompt
      |
      v
    Tokenizer
      |
      v
    Token IDs
      |
      v
    Token + Position Embeddings
      |
      v
    Transformer Blocks
      |
      +-- LayerNorm
      +-- Causal Self-Attention
      +-- MLP
      +-- Residual Connections
      |
      v
    Language Model Head
      |
      v
    Logits
      |
      v
    Token Sampling
      |
      v
    Generated Token
      |
      +------> Repeat
     
## Project Roadmap

### Phase 1 — GPT-2 CPU Inference

Build a complete GPT-2 inference engine in C++ capable of:

- Loading GPT-2 configuration
- Loading tokenizer data
- Loading pretrained model weights
- Running the GPT-2 forward pass
- Performing autoregressive generation
- Supporting basic sampling strategies
- Validating numerical correctness
- Measuring inference latency and throughput

Status: In progress

### Phase 2 — CPU Optimization

Improve the Phase 1 engine through:

- Memory-layout optimization
- Profiling
- Multithreading
- SIMD/vectorization
- Reduced memory movement
- CPU performance benchmarking

Status: Planned

### Phase 3 — GPU Acceleration

Introduce GPU execution and low-level acceleration:

- CUDA programming
- GPU memory management
- Custom CUDA kernels
- GPU profiling
- Kernel optimization

Status: Planned

### Phase 4 — Efficient Decoding

Implement techniques used by modern LLM inference systems:

- Key-Value (KV) cache
- Prefill/decode separation
- Memory-efficient decoding
- Decode performance benchmarking

Status: Planned

### Phase 5 — Batching and Scheduling

Extend the engine toward multi-request inference:

- Static batching
- Dynamic batching
- Continuous batching
- Request management
- Basic scheduling

Status: Planned

### Phase 6 — Production-Oriented Inference

Build a more complete inference system:

- Inference server
- Request handling
- Performance monitoring
- Throughput benchmarking
- Latency analysis
- System-level optimization

Status: Planned

## Design Principles

The project follows these principles:

1. Correctness before optimization
2. Small and modular components
3. Explicit interfaces
4. Reproducible experiments
5. Automated testing
6. Measured performance
7. Clear documentation

## Technology

Initial implementation:

- C++20
- CMake
- LibTorch
- GoogleTest
- Google Benchmark
- YAML

Development environment:

- macOS
- Apple Silicon
- CPU-based inference for Phase 1

GPU acceleration will be developed on NVIDIA hardware when required.

## Repository Structure

```text
gpt2-inference-engine/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── LICENSE
├── .gitignore
├── configs/
├── include/
├── src/
├── apps/
├── tests/
├── benchmarks/
├── scripts/
└── docs/
