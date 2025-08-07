/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_IBV_HPP
#define SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_IBV_HPP

#include <optional>
#include <stdexcept>
#include <string>

#include <infiniband/verbs.h>

namespace hololink::operators::ibv {

::ibv_gid ipv4_to_gid(const std::string& ip);
std::string gid_to_ipv4(const ::ibv_gid& gid);
struct Error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct SendWorkRequest : public ::ibv_send_wr {
    SendWorkRequest(uint64_t wr_id, ::ibv_sge* sge);
};

struct ReceiveWorkRequest : public ::ibv_recv_wr {
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
        Capacity& max_receive_scallter_gather_entry(uint32_t max_recv_sge);
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
    QueuePair(QueuePair&& other);
    QueuePair& operator=(const QueuePair&) = delete;
    QueuePair& operator=(QueuePair&& other);
    ~QueuePair();
    explicit operator bool() const;

    bool reset_to_init(uint32_t device_port);
    bool init_to_rtr(
        uint32_t device_port,
        const ::ibv_gid& remote_gid,
        uint32_t remote_qp_num,
        int gid_index);
    bool rtr_to_rts();
    bool post_send(ibv_send_wr& send_wr);
    bool post_receive(ibv_recv_wr& receive_wr);

    friend void swap(QueuePair& lhs, QueuePair& rhs) noexcept;

    ibv_qp* ptr_ {};
};

struct MemoryRegion {
    MemoryRegion() = default;
    explicit MemoryRegion(::ibv_mr* mr);
    MemoryRegion(const MemoryRegion&) = delete;
    MemoryRegion(MemoryRegion&& other);
    MemoryRegion& operator=(const MemoryRegion&) = delete;
    MemoryRegion& operator=(MemoryRegion&& other);
    ~MemoryRegion();
    explicit operator bool() const;

    friend void swap(MemoryRegion& lhs, MemoryRegion& rhs) noexcept;
    ibv_mr* ptr_ {};
};

struct CompletionQueue {
    CompletionQueue() = default;
    explicit CompletionQueue(::ibv_cq* cq);
    CompletionQueue(const CompletionQueue&) = delete;
    CompletionQueue(CompletionQueue&& other);
    CompletionQueue& operator=(const CompletionQueue&) = delete;
    CompletionQueue& operator=(CompletionQueue&& other);
    ~CompletionQueue();
    explicit operator bool() const;

    std::optional<::ibv_wc> poll();

    friend void swap(CompletionQueue& lhs, CompletionQueue& rhs) noexcept;
    ::ibv_cq* ptr_ {};
};

struct ProtectionDomain {
    ProtectionDomain() = default;
    explicit ProtectionDomain(::ibv_pd* pd);
    ProtectionDomain(const ProtectionDomain&) = delete;
    ProtectionDomain(ProtectionDomain&& other);
    ProtectionDomain& operator=(const ProtectionDomain&) = delete;
    ProtectionDomain& operator=(ProtectionDomain&& other);
    ~ProtectionDomain();
    explicit operator bool() const;

    MemoryRegion register_memory_region(void* addr, size_t length, int access);

    QueuePair create_queue_pair(QueuePair::InitAttr& init_attr);

    friend void swap(ProtectionDomain& lhs, ProtectionDomain& rhs) noexcept;
    ibv_pd* ptr_ {};
};

struct Context {
    Context();
    explicit Context(::ibv_context* context);
    explicit Context(const std::string& dev_name);
    Context(const Context&) = delete;
    Context(Context&&);
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&);
    ~Context();
    explicit operator bool() const;

    ::ibv_device_attr query_device();
    ::ibv_port_attr query_port(int port);
    int query_gid();
    int query_gid_roce_v2_ip();
    ibv_gid query_gid(int port, int index);
    ProtectionDomain allocate_protection_domain();
    CompletionQueue create_completion_queue(int size);

    friend void swap(Context& lhs, Context& rhs) noexcept;
    ::ibv_context* ptr_ {};
};

struct Device {
    Device();
    explicit Device(::ibv_device* device);
    explicit operator bool() const;
    const char* name() const;
    Context get_context();

    ::ibv_device* ptr_ {};
};

class DeviceList {
public:
    DeviceList();
    DeviceList(const DeviceList&) = delete;
    DeviceList& operator=(const DeviceList&) = delete;
    ~DeviceList();

    size_t size() const;
    Device get(size_t index) const;
    std::optional<Device> find(const std::string& device_name) const;

private:
    int num_devices_;
    ::ibv_device** ibv_devices_;
};

} // namespace hololink::operators::ibv

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_IBV_HPP */