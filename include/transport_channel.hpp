#ifndef TRANSPORT_CHANNEL_HPP
#define TRANSPORT_CHANNEL_HPP

#include "disagg_types.hpp"
#include "avx2_transport_framing.hpp"
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>

namespace disagg {

class LockFreePinnedRingBuffer {
public:
    LockFreePinnedRingBuffer(size_t capacity = RING_BUFFER_CAPACITY);
    ~LockFreePinnedRingBuffer();

    bool enqueue(const KVBlockDescriptor& desc, const void* payload, size_t payload_bytes);
    bool dequeue(KVBlockDescriptor& out_desc, void* out_payload, size_t& out_payload_bytes);

    size_t get_capacity() const { return capacity_; }
    size_t get_occupancy() const;

private:
    size_t capacity_;
    RingSlot* slots_host_{nullptr};
    std::atomic<uint64_t> head_{0};
    std::atomic<uint64_t> tail_{0};
    uint64_t sequence_num_{0};
};

class SocketTransportChannel {
public:
    SocketTransportChannel(bool is_server, int port = 9099);
    ~SocketTransportChannel();

    bool send_block(const KVBlockDescriptor& desc, const void* payload, size_t payload_bytes);
    bool receive_block(KVBlockDescriptor& out_desc, void* out_payload, size_t& out_payload_bytes);

private:
    bool is_server_;
    int port_;
    int socket_fd_{-1};
    int conn_fd_{-1};
    uint64_t seq_num_{0};

    void setup_server();
    void setup_client();
};

} // namespace disagg

#endif // TRANSPORT_CHANNEL_HPP
