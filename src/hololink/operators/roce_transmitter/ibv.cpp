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
#include "ibv.hpp"

#include <sstream>
#include <thread>

#include <arpa/inet.h>

#include <hololink/core/logging_internal.hpp>

#define THROW_ERROR(streamable)                   \
    do {                                          \
        using namespace hololink::operators::ibv; \
        std::stringstream ss;                     \
        ss << streamable;                         \
        HSB_LOG_ERROR(ss.str());                  \
        throw Error(ss.str());                    \
    } while (0)

namespace hololink::operators::ibv {

// Based on the assigned IP to the Hololink board, it's GID is computed.
::ibv_gid ipv4_to_gid(const std::string& ip)
{
    ::ibv_gid ipv4_gid {};

    unsigned long binary_ip;
    if (inet_pton(AF_INET, ip.c_str(), &binary_ip) < 1)
        THROW_ERROR("Unable to convert IPv4 and IPv6 address (" << ip << ") from text to binary form");

    ipv4_gid.global.subnet_prefix = 0;
    ipv4_gid.global.interface_id = (binary_ip << 32) | 0xffff0000;

    return ipv4_gid;
}

std::string gid_to_ipv4(const ::ibv_gid& gid)
{
    // For RoCE v2 with IPv4, the IP address is in the upper 32 bits of interface_id
    ::in_addr ip_addr;
    ip_addr.s_addr = gid.global.interface_id >> 32;

    char ip_str[INET_ADDRSTRLEN];
    ::inet_ntop(AF_INET, &ip_addr, ip_str, INET_ADDRSTRLEN);

    return std::string(ip_str);
}

SendWorkRequest::SendWorkRequest(uint64_t wr_id, ::ibv_sge* sge)
{
    memset(this, 0, sizeof(*this));

    this->wr_id = wr_id;
    next = nullptr;
    sg_list = sge;
    num_sge = 1;
    send_flags = IBV_SEND_SIGNALED;
    opcode = IBV_WR_SEND;
}

ReceiveWorkRequest::ReceiveWorkRequest(uint64_t wr_id, ::ibv_sge* sge)
{
    memset(this, 0, sizeof(*this));

    this->wr_id = wr_id;
    next = nullptr;
    sg_list = sge;
    num_sge = 1;
}

ScatterGatherElement::ScatterGatherElement(uint64_t addr, uint32_t length, uint32_t lkey)
{
    this->addr = addr;
    this->length = length;
    this->lkey = lkey;
}

QueuePair::Capacity::Capacity()
{
    std::memset(this, 0, sizeof(Capacity));
}

QueuePair::Capacity& QueuePair::Capacity::max_send_write_read(uint32_t max_send_wr)
{
    this->max_send_wr = max_send_wr;
    return *this;
}

QueuePair::Capacity& QueuePair::Capacity::max_receive_write_read(uint32_t max_recv_wr)
{
    this->max_recv_wr = max_recv_wr;
    return *this;
}

QueuePair::Capacity& QueuePair::Capacity::max_send_scatter_gather_entry(uint32_t max_send_sge)
{
    this->max_send_sge = max_send_sge;
    return *this;
}

QueuePair::Capacity& QueuePair::Capacity::max_receive_scallter_gather_entry(uint32_t max_recv_sge)
{
    this->max_recv_sge = max_recv_sge;
    return *this;
}

QueuePair::InitAttr::InitAttr()
{
    std::memset(this, 0, sizeof(InitAttr));
}

QueuePair::InitAttr& QueuePair::InitAttr::query_pair_type(ibv_qp_type qp_type)
{
    this->qp_type = qp_type;
    return *this;
}

QueuePair::InitAttr& QueuePair::InitAttr::send_completion_queue(ibv_cq* send_cq)
{
    this->send_cq = send_cq;
    return *this;
}

QueuePair::InitAttr& QueuePair::InitAttr::receive_completion_queue(ibv_cq* recv_cq)
{
    this->recv_cq = recv_cq;
    return *this;
}

QueuePair::InitAttr& QueuePair::InitAttr::capacity(const QueuePair::Capacity& cap)
{
    this->cap = cap;
    return *this;
}

QueuePair::Attr::Attr()
{
    std::memset(this, 0, sizeof(Attr));
}

QueuePair::QueuePair(::ibv_qp* qp)
    : ptr_(qp)
{
}

QueuePair::QueuePair(QueuePair&& other)
    : QueuePair()
{
    swap(*this, other);
}

QueuePair& QueuePair::operator=(QueuePair&& other)
{
    swap(*this, other);
    return *this;
}
QueuePair::~QueuePair()
{
    if (ptr_ && ibv_destroy_qp(ptr_))
        HSB_LOG_ERROR("Failed to destroy QP");
}

QueuePair::operator bool() const
{
    return static_cast<bool>(ptr_);
}

bool QueuePair::reset_to_init(uint32_t device_port)
{
    Attr attr;
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = device_port;
    // SEND needs no flags, but to be general
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    auto flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    if (ibv_modify_qp(ptr_, &attr, flags)) {
        HSB_LOG_ERROR("Failed to modify QP state to INIT, {}", strerror(errno));
        return false;
    }
    return true;
}

bool QueuePair::init_to_rtr(
    uint32_t device_port,
    const ::ibv_gid& remote_gid,
    uint32_t remote_qp_num,
    int gid_index)
{
    Attr attr;
    // More generic values, e.g., UD only needs qp_state
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote_qp_num;
    attr.rq_psn = 0;

    attr.ah_attr.is_global = 1;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = device_port;
    attr.ah_attr.grh.dgid = remote_gid;
    attr.ah_attr.grh.hop_limit = 0xFF;
    attr.ah_attr.grh.sgid_index = gid_index;

    auto flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
    // We see occasional errno=110 (ETIMEDOUT); no idea
    // what causes this but it works on retry.
    for (int retry = 5; retry--;) {
        auto r = ibv_modify_qp(ptr_, &attr, flags);
        if (!r)
            break;
        if (!retry) {
            HSB_LOG_ERROR("Cannot modify queue pair to IBV_QPS_RTR, errno={}.", r);
            return false;
        }
        HSB_LOG_ERROR("Cannot modify queue pair to IBV_QPS_RTR, errno={}: \"{}\"; retrying.", r, strerror(r));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return true;
}

bool QueuePair::rtr_to_rts()
{
    Attr attr;
    attr.qp_state = IBV_QPS_RTS;

    auto flags = IBV_QP_STATE | IBV_QP_SQ_PSN;

    if (ibv_modify_qp(ptr_, &attr, flags)) {
        HSB_LOG_ERROR("Failed to modify QP state to RTS");
        return false;
    }

    return true;
}

bool QueuePair::post_send(ibv_send_wr& send_wr)
{
    ibv_send_wr* bad_wr {};
    auto err = ibv_post_send(ptr_, &send_wr, &bad_wr);
    switch (err) {
    case 0: // No error
        break;
    case EINVAL:
        HSB_LOG_ERROR("Invalid value provided in send_wr");
        break;
    case ENOMEM:
        HSB_LOG_ERROR("Send Queue is full or not enough resources to complete this operation");
        break;
    case EFAULT:
        HSB_LOG_ERROR("Invalid QueuePair");
        break;
    default:
        HSB_LOG_ERROR("Failed to post send, {}", strerror(errno));
    }
    return err == 0;
}

bool QueuePair::post_receive(ibv_recv_wr& receive_wr)
{
    ibv_recv_wr* bad_wr {};
    auto err = ibv_post_recv(ptr_, &receive_wr, &bad_wr);
    switch (err) {
    case 0: // No error
        break;
    case EINVAL:
        HSB_LOG_ERROR("Invalid value provided in send_wr");
        break;
    case ENOMEM:
        HSB_LOG_ERROR("Receive Queue is full or not enough resources to complete this operation");
        break;
    case EFAULT:
        HSB_LOG_ERROR("Invalid QueuePair");
        break;
    default:
        HSB_LOG_ERROR("Failed to post recv, {}", strerror(errno));
    }
    return err == 0;
}

void swap(QueuePair& lhs, QueuePair& rhs) noexcept
{
    using std::swap;
    swap(lhs.ptr_, rhs.ptr_);
}

MemoryRegion::MemoryRegion(::ibv_mr* mr)
    : ptr_(mr)
{
}

MemoryRegion::MemoryRegion(MemoryRegion&& other)
    : MemoryRegion()
{
    swap(*this, other);
}

MemoryRegion& MemoryRegion::operator=(MemoryRegion&& other)
{
    swap(*this, other);
    return *this;
}
MemoryRegion::~MemoryRegion()
{
    HSB_LOG_DEBUG("Deregistering memory");
    if (ptr_ && ibv_dereg_mr(ptr_))
        HSB_LOG_ERROR("Failed to deregister Memrory Region");
}

MemoryRegion::operator bool() const
{
    return static_cast<bool>(ptr_);
}

void swap(MemoryRegion& lhs, MemoryRegion& rhs) noexcept
{
    using std::swap;
    swap(lhs.ptr_, rhs.ptr_);
}

CompletionQueue::CompletionQueue(::ibv_cq* cq)
    : ptr_(cq)
{
}

CompletionQueue::CompletionQueue(CompletionQueue&& other)
    : CompletionQueue()
{
    swap(*this, other);
}

CompletionQueue& CompletionQueue::operator=(CompletionQueue&& other)
{
    swap(*this, other);
    return *this;
}
CompletionQueue::~CompletionQueue()
{
    if (ptr_ && ibv_destroy_cq(ptr_))
        HSB_LOG_ERROR("Failed to destroy CQ");
}

CompletionQueue::operator bool() const
{
    return static_cast<bool>(ptr_);
}

std::optional<::ibv_wc> CompletionQueue::poll()
{
    ::ibv_wc wc;
    auto poll_result = ibv_poll_cq(ptr_, 1, &wc);
    if (poll_result < 0)
        THROW_ERROR("Failed to poll Completion Queue");
    if (poll_result == 0)
        return std::optional<::ibv_wc>();
    return wc;
}

void swap(CompletionQueue& lhs, CompletionQueue& rhs) noexcept
{
    using std::swap;
    swap(lhs.ptr_, rhs.ptr_);
}

ProtectionDomain::ProtectionDomain(::ibv_pd* pd)
    : ptr_(pd)
{
}

ProtectionDomain::ProtectionDomain(ProtectionDomain&& other)
    : ProtectionDomain()
{
    swap(*this, other);
}

ProtectionDomain& ProtectionDomain::operator=(ProtectionDomain&& other)
{
    swap(*this, other);
    return *this;
}
ProtectionDomain::~ProtectionDomain()
{
    if (ptr_ && ibv_dealloc_pd(ptr_))
        HSB_LOG_ERROR("Failed to deallocate Protection Domain");
}

ProtectionDomain::operator bool() const
{
    return static_cast<bool>(ptr_);
}

MemoryRegion ProtectionDomain::register_memory_region(void* addr, size_t length, int access)
{
    HSB_LOG_DEBUG("Registering memory({}): {} bytes", addr, length);
    auto reg = ibv_reg_mr(ptr_, addr, length, access);
    if (!reg) {
        switch (errno) {
        case EINVAL:
            THROW_ERROR("Failed to register Memory Region, Invalid access value");
        case ENOMEM:
            THROW_ERROR("Failed to register Memory Region, Not enough resources "
                        "(either in operating system or in RDMA device) to complete "
                        "this operation");
        default:
            THROW_ERROR("Failed to register Memory Region, " << strerror(errno));
        }
    }
    return MemoryRegion(reg);
}

QueuePair ProtectionDomain::create_queue_pair(QueuePair::InitAttr& init_attr)
{
    auto qp = ibv_create_qp(ptr_, &init_attr);
    if (!qp)
        THROW_ERROR("Failed to create Queue Pair, " << strerrordesc_np(errno));
    return QueuePair(qp);
}

void swap(ProtectionDomain& lhs, ProtectionDomain& rhs) noexcept
{
    using std::swap;
    swap(lhs.ptr_, rhs.ptr_);
}

Context::Context()
    : Context(nullptr)
{
}

Context::Context(const std::string& dev_name)
    : Context([&dev_name] {
        HSB_LOG_INFO("Searching for IB devices in host");
        // Get device names in the system
        DeviceList device_list;
        auto device = device_list.find(dev_name);

        // if the device wasn't found in host
        if (!device)
            THROW_ERROR("Failed to find Device '" << dev_name << "'");

        // get device handle
        return device->get_context();
    }())
{
}

Context::Context(ibv_context* context)
    : ptr_(context)
{
}

Context::Context(Context&& other)
    : Context()
{
    swap(*this, other);
}

Context& Context::operator=(Context&& other)
{
    swap(*this, other);
    return *this;
}

Context::~Context()
{
    if (ptr_ && ibv_close_device(ptr_))
        HSB_LOG_ERROR("Failed to close Device");
}

Context::operator bool() const
{
    return static_cast<bool>(ptr_);
}

::ibv_device_attr Context::query_device()
{
    ::ibv_device_attr device_attr;
    if (ibv_query_device(ptr_, &device_attr))
        THROW_ERROR("Failed to get Device Attributes");
    return device_attr;
}

::ibv_port_attr Context::query_port(int port)
{
    ::ibv_port_attr port_attr;
    if (ibv_query_port(ptr_, port, &port_attr))
        THROW_ERROR("Failed to get Port Attributes");
    return port_attr;
}

ibv_gid Context::query_gid(int ib_port, int index)
{
    ibv_gid gid;
    if (ibv_query_gid(ptr_, ib_port, index, &gid))
        THROW_ERROR("Failed to query port GID");
    return gid;
}

int Context::query_gid_roce_v2_ip()
{
    constexpr size_t MAX_GID_ENTRIES = 20;

    ::ibv_gid_entry gid_entries[MAX_GID_ENTRIES];
    auto tot_entries = ibv_query_gid_table(ptr_, gid_entries, MAX_GID_ENTRIES, 0);

    // Finding RoCE V2 GID with IPv4 GID prefix
    int gid_index = 0;
    while (!((gid_entries[gid_index].gid_type == IBV_GID_TYPE_ROCE_V2) && (gid_entries[gid_index].gid.global.subnet_prefix == 0) && ((gid_entries[gid_index].gid.global.interface_id & 0xffffffff) == 0xffff0000)) && (gid_index < tot_entries)) {
        gid_index++;
    }

    if (gid_index == tot_entries || gid_entries[gid_index].gid_type != IBV_GID_TYPE_ROCE_V2)
        THROW_ERROR("Failed to find GID with IP based RoCE V2");
    return gid_entries[gid_index].gid_index;
}

ProtectionDomain Context::allocate_protection_domain()
{
    auto pd = ibv_alloc_pd(ptr_);
    if (!pd)
        THROW_ERROR("Failed to allocate Protection Domain");
    return ProtectionDomain(pd);
}

CompletionQueue Context::create_completion_queue(int size)
{
    auto cq = ibv_create_cq(ptr_, size, nullptr, nullptr, 0);
    if (!cq)
        THROW_ERROR("Failed to create Completion Queue with " << size << " entries");
    return CompletionQueue(cq);
}

void swap(Context& lhs, Context& rhs) noexcept
{
    using std::swap;
    swap(lhs.ptr_, rhs.ptr_);
}

Device::Device()
    : Device(nullptr)
{
}

Device::Device(::ibv_device* device)
    : ptr_(device)
{
}

Device::operator bool() const
{
    return static_cast<bool>(ptr_);
}

const char* Device::name() const
{
    return ibv_get_device_name(ptr_);
}

Context Device::get_context()
{
    auto ctx = ibv_open_device(ptr_);
    if (!ctx)
        THROW_ERROR("Failed to open Device");
    return Context(ctx);
}

DeviceList::DeviceList()
{
    ibv_devices_ = ibv_get_device_list(&num_devices_);
    if (!ibv_devices_)
        THROW_ERROR("Failed to get Device list");
}

DeviceList::~DeviceList()
{
    ibv_free_device_list(ibv_devices_);
}

size_t DeviceList::size() const
{
    return num_devices_;
}

Device DeviceList::get(size_t index) const
{
    if (index >= num_devices_)
        THROW_ERROR("Index out of bounds");
    return Device(ibv_devices_[index]);
}

std::optional<Device> DeviceList::find(const std::string& device_name) const
{
    for (int i = 0; i < num_devices_; ++i) {
        Device device(ibv_devices_[i]);
        auto local_device_name = device.name();
        HSB_LOG_DEBUG("Device name: {}", local_device_name);
        if (device_name.empty()) {
            HSB_LOG_WARN("No device name given, taken first device on list");
            return device;
        }
        if (device_name == local_device_name)
            return device;
    }
    return std::optional<Device>();
}

} // namespace hololink::operators::ibv
