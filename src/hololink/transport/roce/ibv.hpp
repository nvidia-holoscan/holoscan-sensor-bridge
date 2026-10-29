/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_HOLOLINK_TRANSPORT_ROCE_IBV_HPP
#define SRC_HOLOLINK_TRANSPORT_ROCE_IBV_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <infiniband/verbs.h>

// Thin, non-logging RAII wrappers shared by CpuRoceTransceiver and
// RoceTransmitterOp.
namespace hololink::transport::roce::ibv {

struct Error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

::ibv_gid ipv4_to_gid(const std::string& ip);
std::string gid_to_ipv4(const ::ibv_gid& gid);
// IPv4-mapped RoCE v2 GID. Inspects network-order bytes; do not mask
// interface_id (little-endian-only).
bool is_roce_v2_ipv4_mapped(const ::ibv_gid_entry& entry);

struct SendWorkRequest : public ::ibv_send_wr {
    SendWorkRequest(uint64_t wr_id, ::ibv_sge* sge);
    // Toggle IBV_SEND_INLINE without changing the constructor's SIGNALED flag.
    SendWorkRequest& inline_data(bool enabled);
};

struct WriteWithImmWorkRequest : public ::ibv_send_wr {
    WriteWithImmWorkRequest(
        uint64_t wr_id,
        ::ibv_sge* sge,
        uint64_t remote_addr,
        uint32_t rkey,
        uint32_t imm_be);
    // Toggle IBV_SEND_INLINE without changing the constructor's SIGNALED flag.
    WriteWithImmWorkRequest& inline_data(bool enabled);
};

struct ReceiveWorkRequest : public ::ibv_recv_wr {
    // A null SGE creates a credit-only receive WQE (num_sge == 0), used for
    // RDMA Write-with-Immediate into an independently addressed DMA-BUF MR.
    ReceiveWorkRequest(uint64_t wr_id, ::ibv_sge* sge);
};

struct ScatterGatherElement : public ::ibv_sge {
    ScatterGatherElement(uint64_t addr, uint32_t length, uint32_t lkey);
};

struct QueuePair {
    struct Capacity : public ::ibv_qp_cap {
        Capacity();
        Capacity& max_send_write_read(uint32_t max_send_wr);
        Capacity& max_receive_write_read(uint32_t max_recv_wr);
        Capacity& max_send_scatter_gather_entry(uint32_t max_send_sge);
        Capacity& max_receive_scatter_gather_entry(uint32_t max_recv_sge);
        // Request ibv_qp_cap::max_inline_data bytes.
        Capacity& inline_data_size(uint32_t bytes);
    };

    struct InitAttr : public ::ibv_qp_init_attr {
        InitAttr();
        InitAttr& query_pair_type(ibv_qp_type qp_type);
        InitAttr& send_completion_queue(ibv_cq* send_cq);
        InitAttr& receive_completion_queue(ibv_cq* recv_cq);
        InitAttr& capacity(const QueuePair::Capacity& cap);
    };
    struct Attr : public ::ibv_qp_attr {
        Attr();
    };
    QueuePair() = default;
    explicit QueuePair(::ibv_qp* qp);
    QueuePair(const QueuePair&) = delete;
    QueuePair(QueuePair&& other) noexcept;
    QueuePair& operator=(const QueuePair&) = delete;
    QueuePair& operator=(QueuePair&& other) noexcept;
    ~QueuePair();
    explicit operator bool() const;

    // QP transition and post methods return the libibverbs error number
    // directly (zero on success). Pre-modify validation still throws Error.
    int reset_to_init(uint32_t device_port);
    int init_to_rtr(
        uint32_t device_port,
        ibv_port_attr& port_attr,
        const std::string& host_if_name,
        const ::ibv_gid& remote_gid,
        uint32_t remote_qp_num,
        int gid_index,
        const ::ibv_mtu* path_mtu = nullptr);
    int rtr_to_rts();
    bool move_to_error() noexcept;
    int post_send(ibv_send_wr& send_wr);
    int post_receive(ibv_recv_wr& receive_wr);

    friend void swap(QueuePair& lhs, QueuePair& rhs) noexcept;

    ibv_qp* ptr_ {};
};

struct MemoryRegion {
    MemoryRegion() = default;
    explicit MemoryRegion(::ibv_mr* mr);
    MemoryRegion(const MemoryRegion&) = delete;
    MemoryRegion(MemoryRegion&& other) noexcept;
    MemoryRegion& operator=(const MemoryRegion&) = delete;
    MemoryRegion& operator=(MemoryRegion&& other) noexcept;
    ~MemoryRegion();
    explicit operator bool() const;

    friend void swap(MemoryRegion& lhs, MemoryRegion& rhs) noexcept;
    ibv_mr* ptr_ {};
};

struct CompletionQueue {
    CompletionQueue() = default;
    explicit CompletionQueue(::ibv_cq* cq);
    CompletionQueue(const CompletionQueue&) = delete;
    CompletionQueue(CompletionQueue&& other) noexcept;
    CompletionQueue& operator=(const CompletionQueue&) = delete;
    CompletionQueue& operator=(CompletionQueue&& other) noexcept;
    ~CompletionQueue();
    explicit operator bool() const;

    // Poll up to max_entries completions into caller storage. Returns the
    // number written (zero if the CQ is empty). Throws Error if the CQ is
    // null, completions is null, max_entries is not positive, or ibv_poll_cq
    // fails.
    int poll(::ibv_wc* completions, int max_entries);
    // Poll one completion; nullopt if the CQ is empty.
    std::optional<::ibv_wc> poll();
    // Discard completions currently available on the CQ. Does not wait for
    // outstanding work, and does not destroy the CQ; the owner still decides
    // teardown order relative to the QP.
    void discard_pending() noexcept;

    friend void swap(CompletionQueue& lhs, CompletionQueue& rhs) noexcept;
    ::ibv_cq* ptr_ {};
};

struct ProtectionDomain {
    ProtectionDomain() = default;
    explicit ProtectionDomain(::ibv_pd* pd);
    ProtectionDomain(const ProtectionDomain&) = delete;
    ProtectionDomain(ProtectionDomain&& other) noexcept;
    ProtectionDomain& operator=(const ProtectionDomain&) = delete;
    ProtectionDomain& operator=(ProtectionDomain&& other) noexcept;
    ~ProtectionDomain();
    explicit operator bool() const;

    MemoryRegion register_memory_region(void* addr, size_t length, int access);
    MemoryRegion register_memory_region_iova(
        void* addr, size_t length, uint64_t iova, int access);
    // ibv_reg_dmabuf_mr; building the transport requires rdma-core >= 34.
    MemoryRegion register_dmabuf_memory_region(
        uint64_t offset, size_t length, uint64_t iova, int fd, int access);

    // Return an empty QP without logging so callers can retry a different
    // optional capability. On success, libibverbs writes the granted values
    // back to init_attr.cap. create_queue_pair() reports the terminal failure.
    QueuePair try_create_queue_pair(QueuePair::InitAttr& init_attr) noexcept;
    QueuePair create_queue_pair(QueuePair::InitAttr& init_attr);

    friend void swap(ProtectionDomain& lhs, ProtectionDomain& rhs) noexcept;
    ibv_pd* ptr_ {};
};

struct Context {
    Context() = default;
    explicit Context(::ibv_context* context);
    explicit Context(const std::string& dev_name);
    Context(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) noexcept;
    ~Context();
    explicit operator bool() const;

    ::ibv_device_attr query_device();
    // First network interface under the device's sysfs net directory; filesystem failures throw Error.
    std::string get_host_if_name();
    ::ibv_port_attr query_port(int port);
    ::ibv_gid query_gid(int port, int index);
    // Index of the first IPv4-mapped RoCE v2 GID in the device GID table,
    // which spans every port. query_gid_ex() is the per-port query.
    int query_gid_roce_v2_ip();
    // Empty slot (ENODATA) is nullopt. Other verbs failures throw Error.
    std::optional<::ibv_gid_entry> query_gid_ex(int port, int index);
    ProtectionDomain allocate_protection_domain();
    CompletionQueue create_completion_queue(int size);
    // One get that does not block, for any number of callers. Readiness first
    // (poll, timeout 0), so the empty path is one syscall and does not change
    // the fd flags; O_NONBLOCK is established only on the ready path, before
    // the get, so a caller that loses the event to a racing consumer sees
    // EAGAIN rather than blocking. Empty queue, a lost race, or EINTR is
    // nullopt. POLLERR, POLLHUP, or POLLNVAL throws instead of reading as
    // empty. Copies type and, for port events, port_num, then acknowledges so
    // the caller cannot leak an event. Fatality is the caller's policy.
    // Context construction does not change the async fd flags: opening a
    // Context, as RoceTransmitterOp does, does not imply async polling.
    struct AsyncEvent {
        ::ibv_event_type type {};
        std::uint32_t port_num = 0;
    };
    std::optional<AsyncEvent> poll_async_event();

    friend void swap(Context& lhs, Context& rhs) noexcept;
    ::ibv_context* ptr_ {};

private:
    // Ensure O_NONBLOCK on the context async fd. The fd flags are the source
    // of truth, so Context stays a pointer-sized handle with no cached state.
    // Called after poll() reports readiness. O_NONBLOCK is what keeps a racing
    // consumer from blocking this get; the ptr_ check only rejects a null
    // Context.
    void enable_nonblocking_async_events();
};

// Owns ibv_get_device_list / ibv_free_device_list. Names are copied, so they
// remain valid after this list is destroyed.
class DeviceList {
public:
    DeviceList();
    DeviceList(const DeviceList&) = delete;
    DeviceList& operator=(const DeviceList&) = delete;
    ~DeviceList();

    size_t size() const;
    std::string name(size_t index) const;
    // Copied name of device 0. nullopt when the host has no devices.
    std::optional<std::string> first_name() const;

private:
    std::optional<int> index_of(const std::string& device_name) const;
    Context open(const std::string& device_name) const;

    int num_devices_ = 0;
    ::ibv_device** ibv_devices_ = nullptr;

    friend struct Context;
};

} // namespace hololink::transport::roce::ibv

#endif /* SRC_HOLOLINK_TRANSPORT_ROCE_IBV_HPP */
