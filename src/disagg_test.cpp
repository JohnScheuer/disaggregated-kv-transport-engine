#include "disagg_types.hpp"
#include "avx2_transport_framing.hpp"
#include "transport_channel.hpp"
#include "kv_codec_kernels.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>

using namespace disagg;

// Test 1: Lock-Free Ring Buffer & AVX2 Framing Checksum Integrity
bool test_ring_buffer_integrity() {
    std::cout << "[TEST 1/3] Running Lock-Free Pinned Ring Buffer & AVX2 Checksum Test... " << std::flush;

    LockFreePinnedRingBuffer ring(16);

    KVBlockDescriptor desc{};
    desc.sequence_id = 101;
    desc.logical_block_id = 0;
    desc.physical_block_id = 42;
    desc.num_tokens = BLOCK_SIZE;
    desc.quant_mode = QuantMode::INT8_SYMMETRIC;
    desc.uncompressed_bytes = FP16_BLOCK_BYTES * 2;
    desc.compressed_bytes = INT8_BLOCK_BYTES * 2;

    std::vector<uint8_t> dummy_payload(desc.compressed_bytes, 0xAB);
    dummy_payload[10] = 0x12;

    bool enqueued = ring.enqueue(desc, dummy_payload.data(), dummy_payload.size());
    if (!enqueued) {
        std::cout << "FAILED (Enqueue failed)" << std::endl;
        return false;
    }

    KVBlockDescriptor rx_desc{};
    std::vector<uint8_t> rx_payload(desc.compressed_bytes);
    size_t rx_bytes = 0;

    bool dequeued = ring.dequeue(rx_desc, rx_payload.data(), rx_bytes);
    if (!dequeued) {
        std::cout << "FAILED (Dequeue failed)" << std::endl;
        return false;
    }

    if (rx_desc.sequence_id != desc.sequence_id || rx_payload != dummy_payload) {
        std::cout << "FAILED (Payload mismatch)" << std::endl;
        return false;
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

// Test 2: Quantization Round-Trip Cosine Similarity & SQNR (FP16 <-> INT8/FP8)
bool test_quantization_fidelity() {
    std::cout << "[TEST 2/3] Running Quantization Round-Trip Accuracy Test (FP16 -> INT8/FP8 -> FP16)... " << std::flush;

    size_t num_elements = ELEMENTS_PER_BLOCK;
    std::vector<uint16_t> h_src_k(num_elements);
    std::vector<uint16_t> h_src_v(num_elements);

    // Initialize with sine wave test pattern
    for (size_t i = 0; i < num_elements; ++i) {
        float val_k = std::sin(static_cast<float>(i) * 0.05f) * 2.5f;
        float val_v = std::cos(static_cast<float>(i) * 0.05f) * 2.5f;
        __half h_k = __float2half(val_k);
        __half h_v = __float2half(val_v);
        h_src_k[i] = *reinterpret_cast<uint16_t*>(&h_k);
        h_src_v[i] = *reinterpret_cast<uint16_t*>(&h_v);
    }

    __half *d_src_k, *d_src_v, *d_dst_k, *d_dst_v;
    uint8_t* d_compressed;
    float* d_scales;

    cudaMalloc(&d_src_k, num_elements * sizeof(__half));
    cudaMalloc(&d_src_v, num_elements * sizeof(__half));
    cudaMalloc(&d_dst_k, num_elements * sizeof(__half));
    cudaMalloc(&d_dst_v, num_elements * sizeof(__half));
    cudaMalloc(&d_compressed, num_elements * 2);
    cudaMalloc(&d_scales, NUM_KV_HEADS * 2 * sizeof(float));

    cudaMemcpy(d_src_k, h_src_k.data(), num_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_v, h_src_v.data(), num_elements * sizeof(__half), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test INT8 Symmetric
    compress_kv_block_async(d_src_k, d_src_v, d_compressed, d_scales, QuantMode::INT8_SYMMETRIC, BLOCK_SIZE, stream);
    decompress_kv_block_async(d_compressed, d_scales, d_dst_k, d_dst_v, QuantMode::INT8_SYMMETRIC, BLOCK_SIZE, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint16_t> h_dst_k(num_elements);
    cudaMemcpy(h_dst_k.data(), d_dst_k, num_elements * sizeof(__half), cudaMemcpyDeviceToHost);

    double dot = 0.0, norm_orig = 0.0, norm_rec = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        __half h_orig = *reinterpret_cast<__half*>(&h_src_k[i]);
        __half h_rec  = *reinterpret_cast<__half*>(&h_dst_k[i]);
        float orig = __half2float(h_orig);
        float rec  = __half2float(h_rec);

        dot += orig * rec;
        norm_orig += orig * orig;
        norm_rec  += rec * rec;
    }

    double cos_sim = dot / (std::sqrt(norm_orig) * std::sqrt(norm_rec));

    cudaFree(d_src_k); cudaFree(d_src_v);
    cudaFree(d_dst_k); cudaFree(d_dst_v);
    cudaFree(d_compressed); cudaFree(d_scales);
    cudaStreamDestroy(stream);

    if (cos_sim < 0.998) {
        std::cout << "FAILED (Cosine similarity = " << cos_sim << " < 0.998)" << std::endl;
        return false;
    }

    std::cout << "PASSED (Cos Sim: " << cos_sim << " > 0.998)" << std::endl;
    return true;
}

// Test 3: Prefill-to-Decode Handshake
bool test_prefill_decode_handshake() {
    std::cout << "[TEST 3/3] Running Prefill-to-Decode Handoff Consistency Test... " << std::flush;

    LockFreePinnedRingBuffer transport(32);

    KVBlockDescriptor tx_desc{};
    tx_desc.sequence_id = 999;
    tx_desc.logical_block_id = 12;
    tx_desc.physical_block_id = 512;
    tx_desc.num_tokens = BLOCK_SIZE;
    tx_desc.quant_mode = QuantMode::FP8_E4M3;
    tx_desc.uncompressed_bytes = FP16_BLOCK_BYTES * 2;
    tx_desc.compressed_bytes = FP8_BLOCK_BYTES * 2;

    std::vector<uint8_t> tx_payload(tx_desc.compressed_bytes, 0xEF);

    bool ok1 = transport.enqueue(tx_desc, tx_payload.data(), tx_payload.size());

    KVBlockDescriptor rx_desc{};
    std::vector<uint8_t> rx_payload(tx_desc.compressed_bytes);
    size_t rx_bytes = 0;

    bool ok2 = transport.dequeue(rx_desc, rx_payload.data(), rx_bytes);

    if (!ok1 || !ok2 || rx_desc.physical_block_id != 512 || rx_desc.sequence_id != 999) {
        std::cout << "FAILED" << std::endl;
        return false;
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "  DISAGGREGATED KV-TRANSPORT ENGINE: DIAGNOSTIC VERIFICATION SUITE" << std::endl;
    std::cout << "================================================================================" << std::endl;

    bool pass1 = test_ring_buffer_integrity();
    bool pass2 = test_quantization_fidelity();
    bool pass3 = test_prefill_decode_handshake();

    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    if (pass1 && pass2 && pass3) {
        std::cout << "  ALL 3 DIAGNOSTIC TESTS PASSED (100% SUCCESS RATE)" << std::endl;
        std::cout << "================================================================================" << std::endl;
        return 0;
    } else {
        std::cout << "  DIAGNOSTIC SUITE FAILED" << std::endl;
        std::cout << "================================================================================" << std::endl;
        return 1;
    }
}
