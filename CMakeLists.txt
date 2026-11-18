cmake_minimum_required(VERSION 3.22)
project(disaggregated_kv_transport_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler optimization and SIMD flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 90)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Core library source files
set(LIB_SOURCES
    src/transport_channel.cpp
    src/kv_codec_kernels.cu
)

add_library(disagg_kv_codec STATIC ${LIB_SOURCES})
target_link_libraries(disagg_kv_codec PRIVATE pthread rt)

# Test executable
add_executable(disagg_kv_test src/disagg_test.cpp)
target_link_libraries(disagg_kv_test PRIVATE disagg_kv_codec)

# Benchmark executable
add_executable(disagg_kv_benchmark src/disagg_benchmark.cpp)
target_link_libraries(disagg_kv_benchmark PRIVATE disagg_kv_codec)
