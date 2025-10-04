cmake_minimum_required(VERSION 3.24)
project(ledger_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

add_library(ledger_kernels STATIC
    kernels/null_stability.cu
)

target_include_directories(ledger_kernels PUBLIC include)
