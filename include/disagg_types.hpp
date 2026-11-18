#ifndef DISAGG_TYPES_HPP
#define DISAGG_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <array>
#include <chrono>
#include <cuda_fp16.h>

namespace disagg {

constexpr size_t BLOCK_SIZE = 16;       // Tokens per KV physical block
constexpr size_t NUM_KV_HEADS = 8;      // Grouped Query Attention / Multi-Head Attention heads
constexpr size_t HEAD_DIM = 128;        // Dimension per head
constexpr size_t ELEMENTS_PER_HEAD = BLOCK_SIZE * HEAD_DIM; // 2048 elements
constexpr size_t ELEMENTS_PER_BLOCK = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM; // 16,384 elements

constexpr size_t FP16_BLOCK_BYTES = ELEMENTS_PER_BLOCK * sizeof(uint16_t); // 32,768 bytes (32 KB)
constexpr size_t INT8_BLOCK_BYTES = ELEMENTS_PER_BLOCK * sizeof(int8_t);   // 16,384 bytes (16 KB)
constexpr size_t FP8_BLOCK_BYTES  = ELEMENTS_PER_BLOCK * sizeof(uint8_t);  // 16,384 bytes (16 KB)
constexpr size_t SCALES_PER_BLOCK = NUM_KV_HEADS; // One scale factor per head per block
constexpr size_t SCALE_BYTES_PER_BLOCK = SCALES_PER_BLOCK * sizeof(float); // 32 bytes

constexpr uint32_t TRANSPORT_MAGIC = 0x444B5654; // "DKVT" in ASCII

enum class QuantMode : uint32_t {
    NONE_FP16 = 0,
    INT8_SYMMETRIC = 1,
    FP8_E4M3 = 2
};

struct alignas(64) KVBlockDescriptor {
    uint32_t sequence_id;
    uint32_t logical_block_id;
    uint32_t physical_block_id;
    uint32_t num_tokens;
    QuantMode quant_mode;
    uint32_t uncompressed_bytes;
    uint32_t compressed_bytes;
    uint32_t checksum;
    uint64_t timestamp_ns;
};

struct alignas(64) TransportPacketHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t payload_bytes;
    uint32_t flags;
    uint32_t checksum_crc32;
    uint64_t sequence_num;
    KVBlockDescriptor descriptor;
};

enum class RingSlotState : uint32_t {
    EMPTY = 0,
    PRODUCING = 1,
    READY = 2,
    CONSUMING = 3
};

constexpr size_t MAX_PAYLOAD_SIZE = (FP16_BLOCK_BYTES * 2) + 512; // K + V FP16 blocks + scales/metadata
constexpr size_t RING_BUFFER_CAPACITY = 64; // Power of 2 for fast modulo indexing

struct alignas(64) RingSlot {
    TransportPacketHeader header;
    alignas(64) uint8_t payload[MAX_PAYLOAD_SIZE];
    alignas(64) std::atomic<uint32_t> state{static_cast<uint32_t>(RingSlotState::EMPTY)};
};

inline uint64_t get_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

} // namespace disagg

#endif // DISAGG_TYPES_HPP
