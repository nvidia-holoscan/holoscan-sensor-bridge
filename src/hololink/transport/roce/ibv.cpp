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
#include "string_utils.hpp"

#include <filesystem>
#include <limits>
#include <thread>

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <net/if.h>
#include <poll.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <system_error>
#include <unistd.h>
#include <utility>

// size of the RoCEv2 header overhead: ip header (20) + udp header (8) + bth header (12) = 40
#define ROCEV2_HEADER_OVERHEAD 40u
#define IBV_VLAN_OVERHEAD 4u

namespace hololink::transport::roce::ibv {

namespace {

    template <typename... Parts>
    [[noreturn]] void throw_error(Parts&&... parts)
    {
        throw Error(detail::concatenate(std::forward<Parts>(parts)...));
    }

} // namespace

// Based on the assigned IP to the Hololink board, it's GID is computed.
::ibv_gid ipv4_to_gid(const std::string& ip)
{
    ::ibv_gid ipv4_gid {};

    // IPv4-mapped IPv6 in network byte order: 80 zero bits, 0xffff, IPv4.
    ipv4_gid.raw[10] = 0xff;
    ipv4_gid.raw[11] = 0xff;
    if (inet_pton(AF_INET, ip.c_str(), ipv4_gid.raw + 12) < 1)
        throw_error("Unable to convert IPv4 address (", ip, ") from text to binary form");

    return ipv4_gid;
}

std::string gid_to_ipv4(const ::ibv_gid& gid)
{
    char ip_str[INET_ADDRSTRLEN] {};
    ::inet_ntop(AF_INET, gid.raw + 12, ip_str, INET_ADDRSTRLEN);
    return std::string(ip_str);
}

bool is_roce_v2_ipv4_mapped(const ::ibv_gid_entry& entry)
{
    if (entry.gid_type != IBV_GID_TYPE_ROCE_V2)
        return false;
    const auto* bytes = entry.gid.raw;
    for (int i = 0; i < 10; ++i) {
        if (bytes[i] != 0)
            return false;
    }
    return bytes[10] == 0xff && bytes[11] == 0xff;
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

SendWorkRequest& SendWorkRequest::inline_data(bool enabled)
{
    const auto inline_flag
        = static_cast<decltype(send_flags)>(IBV_SEND_INLINE);
    if (enabled)
        send_flags |= inline_flag;
    else
        send_flags &= ~inline_flag;
    return *this;
}

WriteWithImmWorkRequest::WriteWithImmWorkRequest(
    uint64_t wr_id,
    ::ibv_sge* sge,
    uint64_t remote_addr,
    uint32_t rkey,
    uint32_t imm_be)
{
    memset(this, 0, sizeof(*this));

    this->wr_id = wr_id;
    next = nullptr;
    sg_list = sge;
    num_sge = 1;
    send_flags = IBV_SEND_SIGNALED;
    opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    imm_data = imm_be;
    wr.rdma.remote_addr = remote_addr;
    wr.rdma.rkey = rkey;
}

WriteWithImmWorkRequest& WriteWithImmWorkRequest::inline_data(bool enabled)
{
    const auto inline_flag
        = static_cast<decltype(send_flags)>(IBV_SEND_INLINE);
    if (enabled)
        send_flags |= inline_flag;
    else
        send_flags &= ~inline_flag;
    return *this;
}

ReceiveWorkRequest::ReceiveWorkRequest(uint64_t wr_id, ::ibv_sge* sge)
{
    memset(this, 0, sizeof(*this));

    this->wr_id = wr_id;
    next = nullptr;
    sg_list = sge;
    num_sge = sge != nullptr ? 1 : 0;
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

QueuePair::Capacity& QueuePair::Capacity::max_receive_scatter_gather_entry(uint32_t max_recv_sge)
{
    this->max_recv_sge = max_recv_sge;
    return *this;
}

QueuePair::Capacity& QueuePair::Capacity::inline_data_size(uint32_t bytes)
{
    this->max_inline_data = bytes;
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

QueuePair::QueuePair(QueuePair&& other) noexcept
    : QueuePair()
{
    swap(*this, other);
}

QueuePair& QueuePair::operator=(QueuePair&& other) noexcept
{
    swap(*this, other);
    return *this;
}
QueuePair::~QueuePair()
{
    if (ptr_)
        (void)ibv_destroy_qp(ptr_);
}

QueuePair::operator bool() const
{
    return static_cast<bool>(ptr_);
}

int QueuePair::reset_to_init(uint32_t device_port)
{
    if (device_port == 0
        || device_port > std::numeric_limits<std::uint8_t>::max())
        throw_error(
            "IB device port ", device_port, " does not fit a non-zero uint8_t");
    Attr attr;
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = static_cast<std::uint8_t>(device_port);
    // SEND needs no flags, but to be general
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    auto flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    return ibv_modify_qp(ptr_, &attr, flags);
}

namespace {

    int get_hif_mtu(const std::string& host_if_name)
    {
        struct ifreq ifr = { 0 };
        int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0)
            return -1;
        strncpy(ifr.ifr_name, host_if_name.c_str(), IFNAMSIZ - 1);
        if (ioctl(sockfd, SIOCGIFMTU, &ifr) < 0) {
            close(sockfd);
            return -2;
        }
        close(sockfd);
        return ifr.ifr_mtu;
    }

    // A failed host-MTU probe is an allowed fallback; invalid/incompatible
    // active MTU values remain fatal.
    void validate_mtu(int hif_mtu, enum ibv_mtu active_mtu)
    {
        int active_mtu_int = 0;
        switch (active_mtu) {
        case IBV_MTU_256:
            active_mtu_int = 256;
            break;
        case IBV_MTU_512:
            active_mtu_int = 512;
            break;
        case IBV_MTU_1024:
            active_mtu_int = 1024;
            break;
        case IBV_MTU_2048:
            active_mtu_int = 2048;
            break;
        case IBV_MTU_4096:
            active_mtu_int = 4096;
            break;
        default:
            throw_error(
                "Unknown active MTU: ", static_cast<int>(active_mtu));
        }
        // If the host MTU cannot be queried, use the verbs active MTU.
        if (hif_mtu >= 0) {
            // fatal error
            if (hif_mtu < (int)active_mtu_int + ROCEV2_HEADER_OVERHEAD + IBV_VLAN_OVERHEAD) {
                throw_error(
                    "Host Interface MTU, ", hif_mtu,
                    ", is less than the active MTU, ", active_mtu_int,
                    ", + Ethernet overhead required for RoCEv2, ",
                    ROCEV2_HEADER_OVERHEAD + IBV_VLAN_OVERHEAD,
                    " - try taking the interface down and up and verify with "
                    "ibv_devinfo -D <ibv device name>");
            }
        }
    }

    // this is a basic query of the ibv port active mtu and only relevant to P2P
    // connections to an HSB where we are setting the packet size from host
    // adding a switch will require a more sophisticated path_mtu query
    // will validate against the host's ethernet mtu
    enum ibv_mtu get_path_mtu(const std::string& host_if_name, ibv_port_attr& port_attr)
    {
        enum ibv_mtu active_mtu = port_attr.active_mtu;
        int hif_mtu = get_hif_mtu(host_if_name);
        validate_mtu(hif_mtu, active_mtu);
        return active_mtu;
    }

} // namespace

int QueuePair::init_to_rtr(
    uint32_t device_port,
    ibv_port_attr& port_attr,
    const std::string& host_if_name,
    const ::ibv_gid& remote_gid,
    uint32_t remote_qp_num,
    int gid_index,
    const ::ibv_mtu* path_mtu)
{
    // Throws and identifier rejections happen before ibv_modify_qp. Only the
    // retry loop below can leave the QP indeterminate.
    if (device_port == 0
        || device_port > std::numeric_limits<std::uint8_t>::max()
        || gid_index < 0
        || gid_index > std::numeric_limits<std::uint8_t>::max()
        || remote_qp_num > 0x00FFFFFF) {
        throw_error(
            "Invalid RTR identifiers: port=", device_port,
            " gid_index=", gid_index, " remote_qp=", detail::hex(remote_qp_num));
    }
    Attr attr;
    // More generic values, e.g., UD only needs qp_state
    attr.qp_state = IBV_QPS_RTR;
    // The default is the port active MTU, checked against the host interface.
    // A caller-supplied value skips that check: receive-only queue pairs
    // program 4096 so the FPGA can fill the interface MTU.
    attr.path_mtu = path_mtu != nullptr
        ? *path_mtu
        : get_path_mtu(host_if_name, port_attr);
    attr.dest_qp_num = remote_qp_num;
    attr.rq_psn = 0;

    attr.ah_attr.is_global = 1;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = static_cast<std::uint8_t>(device_port);
    attr.ah_attr.grh.dgid = remote_gid;
    attr.ah_attr.grh.hop_limit = 0xFF;
    attr.ah_attr.grh.sgid_index = static_cast<std::uint8_t>(gid_index);

    auto flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
    // We see occasional return code 110 (ETIMEDOUT); no idea
    // what causes this but it works on retry.
    for (int retry = 5; retry--;) {
        auto r = ibv_modify_qp(ptr_, &attr, flags);
        if (!r)
            break;
        if (r != ETIMEDOUT || !retry)
            return r;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return 0;
}

int QueuePair::rtr_to_rts()
{
    Attr attr;
    attr.qp_state = IBV_QPS_RTS;

    auto flags = IBV_QP_STATE | IBV_QP_SQ_PSN;

    return ibv_modify_qp(ptr_, &attr, flags);
}

bool QueuePair::move_to_error() noexcept
{
    Attr attr;
    attr.qp_state = IBV_QPS_ERR;

    const auto err = ibv_modify_qp(ptr_, &attr, IBV_QP_STATE);
    return err == 0;
}

int QueuePair::post_send(ibv_send_wr& send_wr)
{
    ibv_send_wr* bad_wr {};
    return ibv_post_send(ptr_, &send_wr, &bad_wr);
}

int QueuePair::post_receive(ibv_recv_wr& receive_wr)
{
    ibv_recv_wr* bad_wr {};
    return ibv_post_recv(ptr_, &receive_wr, &bad_wr);
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

MemoryRegion::MemoryRegion(MemoryRegion&& other) noexcept
    : MemoryRegion()
{
    swap(*this, other);
}

MemoryRegion& MemoryRegion::operator=(MemoryRegion&& other) noexcept
{
    swap(*this, other);
    return *this;
}
MemoryRegion::~MemoryRegion()
{
    if (ptr_)
        (void)ibv_dereg_mr(ptr_);
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

CompletionQueue::CompletionQueue(CompletionQueue&& other) noexcept
    : CompletionQueue()
{
    swap(*this, other);
}

CompletionQueue& CompletionQueue::operator=(CompletionQueue&& other) noexcept
{
    swap(*this, other);
    return *this;
}
CompletionQueue::~CompletionQueue()
{
    if (ptr_)
        (void)ibv_destroy_cq(ptr_);
}

CompletionQueue::operator bool() const
{
    return static_cast<bool>(ptr_);
}

int CompletionQueue::poll(::ibv_wc* completions, int max_entries)
{
    if (!ptr_ || completions == nullptr || max_entries <= 0)
        throw_error(
            "Completion Queue poll requires a live CQ, a non-null buffer, and "
            "a positive entry count");
    const auto poll_result = ibv_poll_cq(ptr_, max_entries, completions);
    if (poll_result < 0)
        throw_error(
            "Failed to poll Completion Queue, raw return=", poll_result);
    return poll_result;
}

std::optional<::ibv_wc> CompletionQueue::poll()
{
    ::ibv_wc wc;
    if (poll(&wc, 1) == 0)
        return std::nullopt;
    return wc;
}

void CompletionQueue::discard_pending() noexcept
{
    if (!ptr_)
        return;
    ::ibv_wc completions[16] {};
    while (true) {
        const int n = ibv_poll_cq(ptr_, 16, completions);
        if (n < 0)
            return;
        if (n == 0)
            return;
    }
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

ProtectionDomain::ProtectionDomain(ProtectionDomain&& other) noexcept
    : ProtectionDomain()
{
    swap(*this, other);
}

ProtectionDomain& ProtectionDomain::operator=(ProtectionDomain&& other) noexcept
{
    swap(*this, other);
    return *this;
}
ProtectionDomain::~ProtectionDomain()
{
    if (ptr_)
        (void)ibv_dealloc_pd(ptr_);
}

ProtectionDomain::operator bool() const
{
    return static_cast<bool>(ptr_);
}

namespace {

    // MR, CQ, and QP creation can fail with EPERM or ENOMEM when the
    // process cannot pin more memory. Include the current RLIMIT_MEMLOCK
    // as diagnostic context; these errno values can also have other
    // causes. Finite soft_bytes/hard_bytes values are bytes. Do not use
    // this helper for PD allocation: ibv_alloc_pd does not pin queue
    // buffers.
    [[noreturn]] void throw_verbs_resource_error(
        const char* operation, int resource_error)
    {
        if (resource_error == ENOMEM || resource_error == EPERM) {
            struct rlimit limit { };
            if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0) {
                const auto render_limit = [](rlim_t value) {
                    return value == RLIM_INFINITY
                        ? std::string("unlimited")
                        : std::to_string(value);
                };
                throw_error(
                    operation, ", ",
                    std::generic_category().message(resource_error),
                    "; current RLIMIT_MEMLOCK soft_bytes=",
                    render_limit(limit.rlim_cur), " hard_bytes=",
                    render_limit(limit.rlim_max),
                    " (possible pinned-memory constraint)");
            }
            const int limit_error = errno;
            const auto resource_message
                = std::generic_category().message(resource_error);
            throw_error(operation, ", ", resource_message,
                "; RLIMIT_MEMLOCK query failed: ",
                std::generic_category().message(limit_error));
        }
        if (resource_error == EINVAL)
            throw_error(operation, ", invalid arguments");
        throw_error(
            operation, ", ",
            std::generic_category().message(resource_error));
    }

} // namespace

MemoryRegion ProtectionDomain::register_memory_region(void* addr, size_t length, int access)
{
    auto reg = ibv_reg_mr(ptr_, addr, length, access);
    if (!reg) {
        const int err = errno;
        throw_verbs_resource_error("Failed to register Memory Region", err);
    }
    return MemoryRegion(reg);
}

MemoryRegion ProtectionDomain::register_memory_region_iova(
    void* addr, size_t length, uint64_t iova, int access)
{
    auto reg = ibv_reg_mr_iova(ptr_, addr, length, iova, access);
    if (!reg) {
        const int err = errno;
        throw_verbs_resource_error(
            "Failed to register Memory Region with iova", err);
    }
    return MemoryRegion(reg);
}

MemoryRegion ProtectionDomain::register_dmabuf_memory_region(
    uint64_t offset, size_t length, uint64_t iova, int fd, int access)
{
    auto reg = ibv_reg_dmabuf_mr(ptr_, offset, length, iova, fd, access);
    if (!reg) {
        const int err = errno;
        throw_verbs_resource_error(
            "Failed to register DMA-BUF Memory Region", err);
    }
    return MemoryRegion(reg);
}

QueuePair ProtectionDomain::try_create_queue_pair(
    QueuePair::InitAttr& init_attr) noexcept
{
    return QueuePair(ibv_create_qp(ptr_, &init_attr));
}

QueuePair ProtectionDomain::create_queue_pair(QueuePair::InitAttr& init_attr)
{
    // Call ibv_create_qp here, not via try_create_queue_pair: errno is only
    // reliable immediately after the failing verbs call.
    auto* qp = ibv_create_qp(ptr_, &init_attr);
    if (!qp) {
        const int error = errno;
        throw_verbs_resource_error("Failed to create Queue Pair", error);
    }
    return QueuePair(qp);
}

void swap(ProtectionDomain& lhs, ProtectionDomain& rhs) noexcept
{
    using std::swap;
    swap(lhs.ptr_, rhs.ptr_);
}

DeviceList::DeviceList()
{
    ibv_devices_ = ibv_get_device_list(&num_devices_);
    if (!ibv_devices_) {
        const int error = errno;
        throw_error(
            "Failed to get IB device list, ",
            std::generic_category().message(error));
    }
}

DeviceList::~DeviceList()
{
    if (ibv_devices_)
        ibv_free_device_list(ibv_devices_);
}

size_t DeviceList::size() const
{
    return num_devices_ > 0 ? static_cast<size_t>(num_devices_) : 0;
}

std::string DeviceList::name(size_t index) const
{
    if (index >= size())
        throw_error("Index out of bounds");
    const char* device_name = ibv_get_device_name(ibv_devices_[index]);
    if (device_name == nullptr)
        throw_error("Failed to get IB device name");
    return device_name;
}

std::optional<int> DeviceList::index_of(const std::string& device_name) const
{
    for (int index = 0; index < num_devices_; ++index) {
        const char* local_name = ibv_get_device_name(ibv_devices_[index]);
        if (local_name != nullptr && device_name == local_name)
            return index;
    }
    return std::nullopt;
}

std::optional<std::string> DeviceList::first_name() const
{
    if (size() == 0)
        return std::nullopt;
    return name(0);
}

Context DeviceList::open(const std::string& device_name) const
{
    if (device_name.empty())
        throw_error("IB device name must not be empty");
    const auto index = index_of(device_name);
    if (!index)
        throw_error("Failed to find IB device '", device_name, "'");
    ::ibv_context* context = ibv_open_device(ibv_devices_[*index]);
    if (!context) {
        const int error = errno;
        throw_error(
            "Failed to open IB device '", device_name, "', ",
            std::generic_category().message(error));
    }
    return Context(context);
}

Context::Context(const std::string& dev_name)
    : Context(DeviceList().open(dev_name))
{
}

Context::Context(ibv_context* context)
    : ptr_(context)
{
}

Context::Context(Context&& other) noexcept
    : Context()
{
    swap(*this, other);
}

Context& Context::operator=(Context&& other) noexcept
{
    swap(*this, other);
    return *this;
}

Context::~Context()
{
    if (ptr_)
        (void)ibv_close_device(ptr_);
}

Context::operator bool() const
{
    return static_cast<bool>(ptr_);
}

::ibv_port_attr Context::query_port(int port)
{
    if (port <= 0 || port > std::numeric_limits<std::uint8_t>::max())
        throw_error("IB device port ", port, " does not fit a non-zero uint8_t");
    ::ibv_port_attr port_attr;
    const int error
        = ibv_query_port(ptr_, static_cast<std::uint8_t>(port), &port_attr);
    if (error != 0)
        throw_error(
            "Failed to query attributes for IB port ", port,
            ", ibv_query_port returned ", error, " (",
            std::generic_category().message(error), ")");
    return port_attr;
}

::ibv_device_attr Context::query_device()
{
    ::ibv_device_attr device_attr;
    if (ibv_query_device(ptr_, &device_attr))
        throw_error("Failed to get Device Attributes");
    return device_attr;
}

std::string Context::get_host_if_name()
try {
    if (!ptr_ || !ptr_->device)
        throw_error("Device is null");
    const char* name = ibv_get_device_name(ptr_->device);

    // /sys/class/infiniband/<ibdev>/device/net/<ifname>
    const std::filesystem::path net_dir = std::filesystem::path(ptr_->device->ibdev_path) / "device" / "net";

    if (!std::filesystem::is_directory(net_dir))
        throw_error("No net sysfs directory for IB device '", name, "' at ", net_dir.string());

    for (const auto& entry : std::filesystem::directory_iterator(net_dir)) {
        if (entry.is_directory())
            return entry.path().filename().string();
    }

    throw_error("No ethernet interface found under ", net_dir.string(), " for IB device '", name, "'");
} catch (const std::filesystem::filesystem_error& error) {
    throw_error("Failed to inspect IB device network interfaces: ", error.what());
}

::ibv_gid Context::query_gid(int port, int index)
{
    ::ibv_gid gid;
    if (ibv_query_gid(ptr_, port, index, &gid))
        throw_error("Failed to query port GID");
    return gid;
}

int Context::query_gid_roce_v2_ip()
{
    constexpr int max_gid_entries = 20;

    ::ibv_gid_entry gid_entries[max_gid_entries] {};
    const int tot_entries = ibv_query_gid_table(ptr_, gid_entries, max_gid_entries, 0);
    if (tot_entries < 0) {
        // ibv_query_gid_table returns -errno and does not set errno.
        const int error = -tot_entries;
        throw_error(
            "Failed to query GID table, ibv_query_gid_table returned ",
            tot_entries, " (", std::generic_category().message(error), ")");
    }

    for (int gid_index = 0; gid_index < tot_entries && gid_index < max_gid_entries; ++gid_index) {
        if (is_roce_v2_ipv4_mapped(gid_entries[gid_index]))
            return gid_entries[gid_index].gid_index;
    }
    throw_error("Failed to find GID with IP based RoCE V2");
}

std::optional<::ibv_gid_entry> Context::query_gid_ex(int port, int index)
{
    if (port <= 0
        || port > std::numeric_limits<std::uint8_t>::max()
        || index < 0) {
        throw_error(
            "Invalid GID query identifiers: port=", port,
            " gid_index=", index);
    }
    ::ibv_gid_entry entry {};
    const int result = ibv_query_gid_ex(ptr_, port, index, &entry, 0);
    if (result != 0) {
        // The documented contract returns an errno value directly. Preserve
        // compatibility with providers that instead return -1 and set errno,
        // while retaining the raw verbs return in the diagnostic.
        const int captured_errno = result < 0 ? errno : 0;
        const int error = result > 0 ? result : captured_errno;
        if (error == ENODATA)
            return std::nullopt;
        if (result > 0) {
            throw_error(
                "Failed to query GID index ", index, " on port ", port,
                ", ibv_query_gid_ex returned ", result, " (",
                std::generic_category().message(result), ")");
        }
        throw_error(
            "Failed to query GID index ", index, " on port ", port,
            ", ibv_query_gid_ex returned ", result, " (errno ",
            captured_errno, ": ",
            std::generic_category().message(captured_errno), ")");
    }
    return entry;
}

ProtectionDomain Context::allocate_protection_domain()
{
    auto pd = ibv_alloc_pd(ptr_);
    if (!pd) {
        const int err = errno;
        throw_error(
            "Failed to allocate Protection Domain, ",
            std::generic_category().message(err));
    }
    return ProtectionDomain(pd);
}

CompletionQueue Context::create_completion_queue(int size)
{
    auto cq = ibv_create_cq(ptr_, size, nullptr, nullptr, 0);
    if (!cq) {
        const int err = errno;
        const auto operation = detail::concatenate(
            "Failed to create Completion Queue with ", size, " entries");
        throw_verbs_resource_error(operation.c_str(), err);
    }
    return CompletionQueue(cq);
}

void Context::enable_nonblocking_async_events()
{
    if (!ptr_)
        throw_error("Context is null");
    const int flags = fcntl(ptr_->async_fd, F_GETFL);
    if (flags < 0) {
        const int error = errno;
        throw_error(
            "Failed to read IB async fd flags, ",
            std::generic_category().message(error));
    }
    if ((flags & O_NONBLOCK) == 0) {
        if (fcntl(ptr_->async_fd, F_SETFL, flags | O_NONBLOCK) < 0) {
            const int error = errno;
            throw_error(
                "Failed to set IB async fd nonblocking, ",
                std::generic_category().message(error));
        }
    }
}

std::optional<Context::AsyncEvent> Context::poll_async_event()
{
    if (!ptr_)
        throw_error("Context is null");
    pollfd pfd {};
    pfd.fd = ptr_->async_fd;
    pfd.events = POLLIN;
    const int ready = ::poll(&pfd, 1, 0);
    if (ready < 0) {
        const int error = errno;
        if (error == EINTR)
            return std::nullopt;
        throw_error(
            "Failed to poll IB async fd, ",
            std::generic_category().message(error));
    }
    if (ready == 0)
        return std::nullopt;
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
        throw_error("IB async fd is not readable (revents=", pfd.revents, ")");
    // Unreachable with the current POLLIN-only request mask after ERR/HUP/NVAL
    // already threw. Kept so a later mask change (e.g. POLLPRI) cannot treat
    // an unrequested bit as a get.
    if ((pfd.revents & POLLIN) == 0)
        return std::nullopt;
    // Readiness alone does not make the get safe: ibv_get_async_event() reads
    // async_fd and permits simultaneous callers, only one of which gets the
    // event. Establish O_NONBLOCK here, on the rare ready path, so a racing
    // consumer leaves this one with EAGAIN instead of a blocked read.
    enable_nonblocking_async_events();
    ::ibv_async_event event {};
    const int result = ibv_get_async_event(ptr_, &event);
    if (result != 0) {
        const int error = errno;
        if (error == EAGAIN || error == EWOULDBLOCK || error == EINTR)
            return std::nullopt;
        throw_error(
            "Failed to get IB async event, ",
            std::generic_category().message(error));
    }
    AsyncEvent out;
    out.type = event.event_type;
    switch (event.event_type) {
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_GID_CHANGE:
        out.port_num = static_cast<std::uint32_t>(event.element.port_num);
        break;
    default:
        break;
    }
    // Acknowledge before returning so an allocation in the caller cannot leak
    // an event acknowledgement.
    ibv_ack_async_event(&event);
    return out;
}

void swap(Context& lhs, Context& rhs) noexcept
{
    using std::swap;
    swap(lhs.ptr_, rhs.ptr_);
}

} // namespace hololink::transport::roce::ibv
