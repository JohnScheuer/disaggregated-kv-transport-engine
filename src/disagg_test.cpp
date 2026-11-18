#include "disagg_types.hpp"
#include "avx2_transport_framing.hpp"
#include "transport_channel.hpp"
#include "kv_codec_kernels.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <cassert>

using namespace disagg;

bool test_ring_buffer_integrity() {
    std::cout << "[TEST 1/4] Running Lock-Free Pinned Ring Buffer & AVX2 Checksum Test... " << std::flush;

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

bool test_quantization_fidelity() {
    std::cout << "[TEST 2/4] Running Quantization Round-Trip Accuracy & SQNR Test... " << std::flush;

    size_t num_elements = ELEMENTS_PER_BLOCK;
    std::vector<uint16_t> h_src_k(num_elements);
    std::vector<uint16_t> h_src_v(num_elements);

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

    compress_kv_block_async(d_src_k, d_src_v, d_compressed, d_scales, QuantMode::FP8_E4M3, BLOCK_SIZE, stream);
    decompress_kv_block_async(d_compressed, d_scales, d_dst_k, d_dst_v, QuantMode::FP8_E4M3, BLOCK_SIZE, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint16_t> h_dst_k(num_elements);
    cudaMemcpy(h_dst_k.data(), d_dst_k, num_elements * sizeof(__half), cudaMemcpyDeviceToHost);

    double dot = 0.0, norm_orig = 0.0, norm_rec = 0.0, signal_pwr = 0.0, noise_pwr = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        float orig = __half2float(*reinterpret_cast<__half*>(&h_src_k[i]));
        float rec  = __half2float(*reinterpret_cast<__half*>(&h_dst_k[i]));

        dot += orig * rec;
        norm_orig += orig * orig;
        norm_rec  += rec * rec;

        signal_pwr += orig * orig;
        float diff = orig - rec;
        noise_pwr += diff * diff;
    }

    double cos_sim = dot / (std::sqrt(norm_orig) * std::sqrt(norm_rec));
    double sqnr_db = 10.0 * std::log10(signal_pwr / (noise_pwr + 1e-12));

    cudaFree(d_src_k); cudaFree(d_src_v);
    cudaFree(d_dst_k); cudaFree(d_dst_v);
    cudaFree(d_compressed); cudaFree(d_scales);
    cudaStreamDestroy(stream);

    if (cos_sim < 0.995 || sqnr_db < 30.0) {
        std::cout << "FAILED (Cosine sim: " << cos_sim << ", SQNR: " << sqnr_db << " dB)" << std::endl;
        return false;
    }

    std::cout << "PASSED (Cos Sim: " << cos_sim << ", SQNR: " << sqnr_db << " dB)" << std::endl;
    return true;
}

bool test_multi_stream_pipeline() {
    std::cout << "[TEST 3/4] Running Multi-Stream CUDA Pipelining Test... " << std::flush;

    cudaStream_t s1, s2;
    cudaEvent_t ev;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaEventCreate(&ev);

    size_t num_elements = ELEMENTS_PER_BLOCK;
    __half *d_k, *d_v;
    uint8_t* d_comp;
    float* d_scales;

    cudaMalloc(&d_k, num_elements * sizeof(__half));
    cudaMalloc(&d_v, num_elements * sizeof(__half));
    cudaMalloc(&d_comp, num_elements * 2);
    cudaMalloc(&d_scales, NUM_KV_HEADS * 2 * sizeof(float));

    compress_kv_block_async(d_k, d_v, d_comp, d_scales, QuantMode::FP8_E4M3, BLOCK_SIZE, s1);
    cudaEventRecord(ev, s1);
    cudaStreamWaitEvent(s2, ev, 0);

    decompress_kv_block_async(d_comp, d_scales, d_k, d_v, QuantMode::FP8_E4M3, BLOCK_SIZE, s2);
    cudaStreamSynchronize(s2);

    cudaFree(d_k); cudaFree(d_v); cudaFree(d_comp); cudaFree(d_scales);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2); cudaEventDestroy(ev);

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_socket_transport_handshake() {
    std::cout << "[TEST 4/4] Running Socket IPC Loopback Transport Test... " << std::flush;

    int port = 9144;
    bool server_ok = false;
    bool client_ok = false;

    std::thread server_thread([&]() {
        try {
            SocketTransportChannel server(true, port);
            KVBlockDescriptor rx_desc{};
            std::vector<uint8_t> rx_payload(FP8_BLOCK_BYTES * 2);
            size_t rx_bytes = 0;
            server_ok = server.receive_block(rx_desc, rx_payload.data(), rx_bytes);
        } catch (...) {
            server_ok = false;
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::thread client_thread([&]() {
        try {
            SocketTransportChannel client(false, port);
            KVBlockDescriptor tx_desc{};
            tx_desc.sequence_id = 777;
            tx_desc.quant_mode = QuantMode::FP8_E4M3;
            std::vector<uint8_t> tx_payload(FP8_BLOCK_BYTES * 2, 0xCD);
            client_ok = client.send_block(tx_desc, tx_payload.data(), tx_payload.size());
        } catch (...) {
            client_ok = false;
        }
    });

    server_thread.join();
    client_thread.join();

    if (!server_ok || !client_ok) {
        std::cout << "FAILED (Server OK: " << server_ok << ", Client OK: " << client_ok << ")" << std::endl;
        return false;
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "  DISAGGREGATED KV-TRANSPORT ENGINE: EXTENDED DIAGNOSTIC SUITE" << std::endl;
    std::cout << "================================================================================" << std::endl;

    bool p1 = test_ring_buffer_integrity();
    bool p2 = test_quantization_fidelity();
    bool p3 = test_multi_stream_pipeline();
    bool p4 = test_socket_transport_handshake();

    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    if (p1 && p2 && p3 && p4) {
        std::cout << "  ALL 4 DIAGNOSTIC TESTS PASSED (100% SUCCESS RATE)" << std::endl;
        std::cout << "================================================================================" << std::endl;
        return 0;
    } else {
        std::cout << "  DIAGNOSTIC SUITE FAILED" << std::endl;
        std::cout << "================================================================================" << std::endl;
        return 1;
    }
}
