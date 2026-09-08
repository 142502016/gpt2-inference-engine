cmake_minimum_required(VERSION 3.20)

project(
    gpt2_inference_engine
    VERSION 0.1.0
    DESCRIPTION "A C++ inference engine for GPT-2"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
    gpt2_engine
    src/model.cpp
)

target_include_directories(
    gpt2_engine
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(
    gpt2_generate
    apps/generate.cpp
)

target_link_libraries(
    gpt2_generate
    PRIVATE
        gpt2_engine
)
