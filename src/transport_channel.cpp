#include "transport_channel.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace disagg {

LockFreePinnedRingBuffer::LockFreePinnedRingBuffer(size_t capacity)
    : capacity_(capacity) {
    size_t total_bytes = sizeof(RingSlot) * capacity_;
    
    // Allocate Host Pinned Zero-Copy memory
    cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&slots_host_), total_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        slots_host_ = reinterpret_cast<RingSlot*>(malloc(total_bytes));
    }

    for (size_t i = 0; i < capacity_; ++i) {
        slots_host_[i].state.store(static_cast<uint32_t>(RingSlotState::EMPTY), std::memory_order_relaxed);
    }
}

LockFreePinnedRingBuffer::~LockFreePinnedRingBuffer() {
    if (slots_host_) {
        cudaFreeHost(slots_host_);
    }
}

bool LockFreePinnedRingBuffer::enqueue(const KVBlockDescriptor& desc, const void* payload, size_t payload_bytes) {
    uint64_t current_tail = tail_.load(std::memory_order_relaxed);
    size_t slot_idx = current_tail % capacity_;
    RingSlot& slot = slots_host_[slot_idx];

    uint32_t expected = static_cast<uint32_t>(RingSlotState::EMPTY);
    if (!slot.state.compare_exchange_strong(expected, static_cast<uint32_t>(RingSlotState::PRODUCING),
                                           std::memory_order_acquire)) {
        return false; // Ring buffer full
    }

    AVX2FramingValidator::frame_packet(slot.header, const_cast<void*>(payload), payload_bytes, desc, sequence_num_++);
    std::memcpy(slot.payload, payload, payload_bytes);

    slot.state.store(static_cast<uint32_t>(RingSlotState::READY), std::memory_order_release);
    tail_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool LockFreePinnedRingBuffer::dequeue(KVBlockDescriptor& out_desc, void* out_payload, size_t& out_payload_bytes) {
    uint64_t current_head = head_.load(std::memory_order_relaxed);
    size_t slot_idx = current_head % capacity_;
    RingSlot& slot = slots_host_[slot_idx];

    uint32_t expected = static_cast<uint32_t>(RingSlotState::READY);
    if (!slot.state.compare_exchange_strong(expected, static_cast<uint32_t>(RingSlotState::CONSUMING),
                                           std::memory_order_acquire)) {
        return false; // Ring buffer empty
    }

    if (!AVX2FramingValidator::validate_packet(slot.header, slot.payload)) {
        slot.state.store(static_cast<uint32_t>(RingSlotState::EMPTY), std::memory_order_release);
        return false;
    }

    out_desc = slot.header.descriptor;
    out_payload_bytes = slot.header.payload_bytes;
    std::memcpy(out_payload, slot.payload, out_payload_bytes);

    slot.state.store(static_cast<uint32_t>(RingSlotState::EMPTY), std::memory_order_release);
    head_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

size_t LockFreePinnedRingBuffer::get_occupancy() const {
    uint64_t h = head_.load(std::memory_order_relaxed);
    uint64_t t = tail_.load(std::memory_order_relaxed);
    return (t >= h) ? (t - h) : 0;
}

SocketTransportChannel::SocketTransportChannel(bool is_server, int port)
    : is_server_(is_server), port_(port) {
    if (is_server_) {
        setup_server();
    } else {
        setup_client();
    }
}

SocketTransportChannel::~SocketTransportChannel() {
    if (conn_fd_ >= 0) close(conn_fd_);
    if (socket_fd_ >= 0) close(socket_fd_);
}

void SocketTransportChannel::setup_server() {
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr));
    listen(socket_fd_, 1);
}

void SocketTransportChannel::setup_client() {
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr));
    conn_fd_ = socket_fd_;
}

bool SocketTransportChannel::send_block(const KVBlockDescriptor& desc, const void* payload, size_t payload_bytes) {
    TransportPacketHeader header;
    AVX2FramingValidator::frame_packet(header, const_cast<void*>(payload), payload_bytes, desc, seq_num_++);

    send(conn_fd_, &header, sizeof(header), 0);
    send(conn_fd_, payload, payload_bytes, 0);
    return true;
}

bool SocketTransportChannel::receive_block(KVBlockDescriptor& out_desc, void* out_payload, size_t& out_payload_bytes) {
    TransportPacketHeader header;
    ssize_t r1 = recv(conn_fd_, &header, sizeof(header), MSG_WAITALL);
    if (r1 <= 0) return false;

    out_payload_bytes = header.payload_bytes;
    ssize_t r2 = recv(conn_fd_, out_payload, out_payload_bytes, MSG_WAITALL);
    if (r2 <= 0) return false;

    if (!AVX2FramingValidator::validate_packet(header, out_payload)) {
        return false;
    }

    out_desc = header.descriptor;
    return true;
}

} // namespace disagg
