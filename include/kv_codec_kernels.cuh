#ifndef KV_CODEC_KERNELS_CUH
#define KV_CODEC_KERNELS_CUH

#include "disagg_types.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace disagg {

// Asynchronous KV block compression host entrypoint
cudaError_t compress_kv_block_async(
    const __half* d_src_k,
    const __half* d_src_v,
    uint8_t* d_dst_compressed,
    float* d_dst_scales,
    QuantMode mode,
    size_t num_tokens,
    cudaStream_t stream
);

// Asynchronous KV block decompression host entrypoint
cudaError_t decompress_kv_block_async(
    const uint8_t* d_src_compressed,
    const float* d_src_scales,
    __half* d_dst_k,
    __half* d_dst_v,
    QuantMode mode,
    size_t num_tokens,
    cudaStream_t stream
);

// CUDA kernel prototypes
__global__ void quantize_fp16_to_int8_kernel(
    const __half* __restrict__ d_src_k,
    const __half* __restrict__ d_src_v,
    int8_t* __restrict__ d_dst_k,
    int8_t* __restrict__ d_dst_v,
    float* __restrict__ d_scales_k,
    float* __restrict__ d_scales_v,
    size_t num_tokens
);

__global__ void dequantize_int8_to_fp16_kernel(
    const int8_t* __restrict__ d_src_k,
    const int8_t* __restrict__ d_src_v,
    const float* __restrict__ d_scales_k,
    const float* __restrict__ d_scales_v,
    __half* __restrict__ d_dst_k,
    __half* __restrict__ d_dst_v,
    size_t num_tokens
);

__global__ void quantize_fp16_to_fp8_kernel(
    const __half* __restrict__ d_src_k,
    const __half* __restrict__ d_src_v,
    uint8_t* __restrict__ d_dst_k,
    uint8_t* __restrict__ d_dst_v,
    float* __restrict__ d_scales_k,
    float* __restrict__ d_scales_v,
    size_t num_tokens
);

__global__ void dequantize_fp8_to_fp16_kernel(
    const uint8_t* __restrict__ d_src_k,
    const uint8_t* __restrict__ d_src_v,
    const float* __restrict__ d_scales_k,
    const float* __restrict__ d_scales_v,
    __half* __restrict__ d_dst_k,
    __half* __restrict__ d_dst_v,
    size_t num_tokens
);

} // namespace disagg

#endif // KV_CODEC_KERNELS_CUH
