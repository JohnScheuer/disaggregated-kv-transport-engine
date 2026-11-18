#include "kv_codec_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdio.h>

namespace disagg {

// Warp shuffle max reduction helper
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// 128-bit Vectorized Quantization Kernel: FP16 -> INT8
__global__ void quantize_fp16_to_int8_kernel(
    const __half* __restrict__ d_src_k,
    const __half* __restrict__ d_src_v,
    int8_t* __restrict__ d_dst_k,
    int8_t* __restrict__ d_dst_v,
    float* __restrict__ d_scales_k,
    float* __restrict__ d_scales_v,
    size_t num_tokens
) {
    // Each block processes 1 head across 16 tokens
    size_t head_idx = blockIdx.x;
    if (head_idx >= NUM_KV_HEADS) return;

    size_t head_offset = head_idx * HEAD_DIM;
    size_t tid = threadIdx.x; // 0..127

    float thread_max_k = 0.0f;
    float thread_max_v = 0.0f;

    for (size_t t = 0; t < num_tokens; ++t) {
        size_t idx = t * (NUM_KV_HEADS * HEAD_DIM) + head_offset + tid;
        float val_k = fabsf(__half2float(d_src_k[idx]));
        float val_v = fabsf(__half2float(d_src_v[idx]));
        thread_max_k = fmaxf(thread_max_k, val_k);
        thread_max_v = fmaxf(thread_max_v, val_v);
    }

    float head_max_k = warp_reduce_max(thread_max_k);
    float head_max_v = warp_reduce_max(thread_max_v);

    float scale_k = head_max_k / 127.0f;
    float scale_v = head_max_v / 127.0f;
    if (scale_k < 1e-8f) scale_k = 1.0f;
    if (scale_v < 1e-8f) scale_v = 1.0f;

    if (tid == 0) {
        d_scales_k[head_idx] = scale_k;
        d_scales_v[head_idx] = scale_v;
    }
    __syncthreads();

    float inv_scale_k = 1.0f / scale_k;
    float inv_scale_v = 1.0f / scale_v;

    for (size_t t = 0; t < num_tokens; ++t) {
        size_t idx = t * (NUM_KV_HEADS * HEAD_DIM) + head_offset + tid;
        
        float val_k = __half2float(d_src_k[idx]) * inv_scale_k;
        float val_v = __half2float(d_src_v[idx]) * inv_scale_v;

        int q_k = __float2int_rn(fminf(fmaxf(val_k, -128.0f), 127.0f));
        int q_v = __float2int_rn(fminf(fmaxf(val_v, -128.0f), 127.0f));

        d_dst_k[idx] = static_cast<int8_t>(q_k);
        d_dst_v[idx] = static_cast<int8_t>(q_v);
    }
}

// 128-bit Vectorized Dequantization Kernel: INT8 -> FP16
__global__ void dequantize_int8_to_fp16_kernel(
    const int8_t* __restrict__ d_src_k,
    const int8_t* __restrict__ d_src_v,
    const float* __restrict__ d_scales_k,
    const float* __restrict__ d_scales_v,
    __half* __restrict__ d_dst_k,
    __half* __restrict__ d_dst_v,
    size_t num_tokens
) {
    size_t head_idx = blockIdx.x;
    if (head_idx >= NUM_KV_HEADS) return;

    size_t head_offset = head_idx * HEAD_DIM;
    size_t tid = threadIdx.x;

    float scale_k = d_scales_k[head_idx];
    float scale_v = d_scales_v[head_idx];

    for (size_t t = 0; t < num_tokens; ++t) {
        size_t idx = t * (NUM_KV_HEADS * HEAD_DIM) + head_offset + tid;

        float val_k = static_cast<float>(d_src_k[idx]) * scale_k;
        float val_v = static_cast<float>(d_src_v[idx]) * scale_v;

        d_dst_k[idx] = __float2half(val_k);
        d_dst_v[idx] = __float2half(val_v);
    }
}

// Vectorized Quantization Kernel: FP16 -> FP8 E4M3
__global__ void quantize_fp16_to_fp8_kernel(
    const __half* __restrict__ d_src_k,
    const __half* __restrict__ d_src_v,
    uint8_t* __restrict__ d_dst_k,
    uint8_t* __restrict__ d_dst_v,
    float* __restrict__ d_scales_k,
    float* __restrict__ d_scales_v,
    size_t num_tokens
) {
    size_t head_idx = blockIdx.x;
    if (head_idx >= NUM_KV_HEADS) return;

    size_t head_offset = head_idx * HEAD_DIM;
    size_t tid = threadIdx.x;

    float thread_max_k = 0.0f;
    float thread_max_v = 0.0f;

    for (size_t t = 0; t < num_tokens; ++t) {
        size_t idx = t * (NUM_KV_HEADS * HEAD_DIM) + head_offset + tid;
        float val_k = fabsf(__half2float(d_src_k[idx]));
        float val_v = fabsf(__half2float(d_src_v[idx]));
        thread_max_k = fmaxf(thread_max_k, val_k);
        thread_max_v = fmaxf(thread_max_v, val_v);
    }

    float head_max_k = warp_reduce_max(thread_max_k);
    float head_max_v = warp_reduce_max(thread_max_v);

    // FP8 E4M3 max representable value is 448.0
    float scale_k = head_max_k / 448.0f;
    float scale_v = head_max_v / 448.0f;
    if (scale_k < 1e-8f) scale_k = 1.0f;
    if (scale_v < 1e-8f) scale_v = 1.0f;

    if (tid == 0) {
        d_scales_k[head_idx] = scale_k;
        d_scales_v[head_idx] = scale_v;
    }
    __syncthreads();

    float inv_scale_k = 1.0f / scale_k;
    float inv_scale_v = 1.0f / scale_v;

    for (size_t t = 0; t < num_tokens; ++t) {
        size_t idx = t * (NUM_KV_HEADS * HEAD_DIM) + head_offset + tid;

        float val_k = __half2float(d_src_k[idx]) * inv_scale_k;
        float val_v = __half2float(d_src_v[idx]) * inv_scale_v;

        __nv_fp8_e4m3 fp8_k = __nv_fp8_e4m3(val_k);
        __nv_fp8_e4m3 fp8_v = __nv_fp8_e4m3(val_v);

        d_dst_k[idx] = *reinterpret_cast<uint8_t*>(&fp8_k);
        d_dst_v[idx] = *reinterpret_cast<uint8_t*>(&fp8_v);
    }
}

// Vectorized Dequantization Kernel: FP8 E4M3 -> FP16
__global__ void dequantize_fp8_to_fp16_kernel(
    const uint8_t* __restrict__ d_src_k,
    const uint8_t* __restrict__ d_src_v,
    const float* __restrict__ d_scales_k,
    const float* __restrict__ d_scales_v,
    __half* __restrict__ d_dst_k,
    __half* __restrict__ d_dst_v,
    size_t num_tokens
) {
    size_t head_idx = blockIdx.x;
    if (head_idx >= NUM_KV_HEADS) return;

    size_t head_offset = head_idx * HEAD_DIM;
    size_t tid = threadIdx.x;

    float scale_k = d_scales_k[head_idx];
    float scale_v = d_scales_v[head_idx];

    for (size_t t = 0; t < num_tokens; ++t) {
        size_t idx = t * (NUM_KV_HEADS * HEAD_DIM) + head_offset + tid;

        uint8_t raw_k = d_src_k[idx];
        uint8_t raw_v = d_src_v[idx];

        __nv_fp8_e4m3 fp8_k = *reinterpret_cast<const __nv_fp8_e4m3*>(&raw_k);
        __nv_fp8_e4m3 fp8_v = *reinterpret_cast<const __nv_fp8_e4m3*>(&raw_v);

        float val_k = float(fp8_k) * scale_k;
        float val_v = float(fp8_v) * scale_v;

        d_dst_k[idx] = __float2half(val_k);
        d_dst_v[idx] = __float2half(val_v);
    }
}

// Host Entrypoint for Compression
cudaError_t compress_kv_block_async(
    const __half* d_src_k,
    const __half* d_src_v,
    uint8_t* d_dst_compressed,
    float* d_dst_scales,
    QuantMode mode,
    size_t num_tokens,
    cudaStream_t stream
) {
    if (mode == QuantMode::NONE_FP16) {
        size_t bytes = num_tokens * NUM_KV_HEADS * HEAD_DIM * sizeof(__half);
        cudaMemcpyAsync(d_dst_compressed, d_src_k, bytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_dst_compressed + bytes, d_src_v, bytes, cudaMemcpyDeviceToDevice, stream);
        return cudaGetLastError();
    }

    dim3 grid(NUM_KV_HEADS);
    dim3 block(HEAD_DIM);

    size_t block_elements = num_tokens * NUM_KV_HEADS * HEAD_DIM;

    if (mode == QuantMode::INT8_SYMMETRIC) {
        int8_t* d_dst_k = reinterpret_cast<int8_t*>(d_dst_compressed);
        int8_t* d_dst_v = reinterpret_cast<int8_t*>(d_dst_compressed + block_elements);
        float* d_scales_k = d_dst_scales;
        float* d_scales_v = d_dst_scales + NUM_KV_HEADS;

        quantize_fp16_to_int8_kernel<<<grid, block, 0, stream>>>(
            d_src_k, d_src_v, d_dst_k, d_dst_v, d_scales_k, d_scales_v, num_tokens
        );
    } else if (mode == QuantMode::FP8_E4M3) {
        uint8_t* d_dst_k = d_dst_compressed;
        uint8_t* d_dst_v = d_dst_compressed + block_elements;
        float* d_scales_k = d_dst_scales;
        float* d_scales_v = d_dst_scales + NUM_KV_HEADS;

        quantize_fp16_to_fp8_kernel<<<grid, block, 0, stream>>>(
            d_src_k, d_src_v, d_dst_k, d_dst_v, d_scales_k, d_scales_v, num_tokens
        );
    }

    return cudaGetLastError();
}

// Host Entrypoint for Decompression
cudaError_t decompress_kv_block_async(
    const uint8_t* d_src_compressed,
    const float* d_src_scales,
    __half* d_dst_k,
    __half* d_dst_v,
    QuantMode mode,
    size_t num_tokens,
    cudaStream_t stream
) {
    if (mode == QuantMode::NONE_FP16) {
        size_t bytes = num_tokens * NUM_KV_HEADS * HEAD_DIM * sizeof(__half);
        cudaMemcpyAsync(d_dst_k, d_src_compressed, bytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_dst_v, d_src_compressed + bytes, bytes, cudaMemcpyDeviceToDevice, stream);
        return cudaGetLastError();
    }

    dim3 grid(NUM_KV_HEADS);
    dim3 block(HEAD_DIM);

    size_t block_elements = num_tokens * NUM_KV_HEADS * HEAD_DIM;

    if (mode == QuantMode::INT8_SYMMETRIC) {
        const int8_t* d_src_k = reinterpret_cast<const int8_t*>(d_src_compressed);
        const int8_t* d_src_v = reinterpret_cast<const int8_t*>(d_src_compressed + block_elements);
        const float* d_scales_k = d_src_scales;
        const float* d_scales_v = d_src_scales + NUM_KV_HEADS;

        dequantize_int8_to_fp16_kernel<<<grid, block, 0, stream>>>(
            d_src_k, d_src_v, d_scales_k, d_scales_v, d_dst_k, d_dst_v, num_tokens
        );
    } else if (mode == QuantMode::FP8_E4M3) {
        const uint8_t* d_src_k = d_src_compressed;
        const uint8_t* d_src_v = d_src_compressed + block_elements;
        const float* d_scales_k = d_src_scales;
        const float* d_scales_v = d_src_scales + NUM_KV_HEADS;

        dequantize_fp8_to_fp16_kernel<<<grid, block, 0, stream>>>(
            d_src_k, d_src_v, d_scales_k, d_scales_v, d_dst_k, d_dst_v, num_tokens
        );
    }

    return cudaGetLastError();
}

} // namespace disagg
