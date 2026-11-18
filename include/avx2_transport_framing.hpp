#ifndef AVX2_TRANSPORT_FRAMING_HPP
#define AVX2_TRANSPORT_FRAMING_HPP

#include "disagg_types.hpp"
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace disagg {

class AVX2FramingValidator {
public:
    // Computes ultra-fast 32-bit SIMD checksum using AVX2 vectorized XOR-folding + FNV-1a
    static uint32_t compute_checksum(const void* data, size_t length) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
        size_t chunks = length / 32;
        size_t remainder = length % 32;

        __m256i accum = _mm256_setzero_si256();

        for (size_t i = 0; i < chunks; ++i) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i * 32));
            accum = _mm256_xor_si256(accum, chunk);
        }

        // Extract 256-bit accumulator to 32-bit integers and XOR fold
        alignas(32) uint32_t temp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), accum);

        uint32_t crc = 0x811C9DC5; // FNV-1a basis
        for (int i = 0; i < 8; ++i) {
            crc ^= temp[i];
            crc *= 0x01000193; // FNV prime
        }

        // Process remaining scalar bytes
        const uint8_t* rem_ptr = ptr + (chunks * 32);
        for (size_t i = 0; i < remainder; ++i) {
            crc ^= rem_ptr[i];
            crc *= 0x01000193;
        }

        return crc;
    }

    static void frame_packet(TransportPacketHeader& header,
                             void* payload,
                             size_t payload_bytes,
                             const KVBlockDescriptor& desc,
                             uint64_t seq_num) {
        header.magic = TRANSPORT_MAGIC;
        header.version = 1;
        header.payload_bytes = static_cast<uint32_t>(payload_bytes);
        header.flags = 0;
        header.sequence_num = seq_num;
        header.descriptor = desc;

        // Compute payload checksum and assign to header
        header.descriptor.checksum = compute_checksum(payload, payload_bytes);
        header.checksum_crc32 = compute_checksum(&header.descriptor, sizeof(KVBlockDescriptor));
    }

    static bool validate_packet(const TransportPacketHeader& header, const void* payload) {
        if (header.magic != TRANSPORT_MAGIC) {
            return false;
        }
        if (header.version != 1) {
            return false;
        }
        uint32_t expected_desc_crc = compute_checksum(&header.descriptor, sizeof(KVBlockDescriptor));
        if (expected_desc_crc != header.checksum_crc32) {
            return false;
        }
        uint32_t expected_payload_crc = compute_checksum(payload, header.payload_bytes);
        if (expected_payload_crc != header.descriptor.checksum) {
            return false;
        }
        return true;
    }
};

} // namespace disagg

#endif // AVX2_TRANSPORT_FRAMING_HPP
