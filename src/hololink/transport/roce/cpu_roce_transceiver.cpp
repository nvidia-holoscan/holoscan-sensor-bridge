/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "cpu_roce_transceiver_internal.hpp"
#include "host_trailer_key.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <net/if.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#include "ibv.hpp"
#include "string_utils.hpp"

namespace hololink::transport::roce {

namespace detail {

    LocalRoceV2Ipv4Gid query_local_roce_v2_ipv4_gid(
        ibv::Context& context, std::uint32_t port, const std::string& local_ip)
    {
        if (port > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw ibv::Error(
                "IB device port " + std::to_string(port)
                + " does not fit an int");
        }
        const int port_num = static_cast<int>(port);
        const auto port_attr = context.query_port(port_num);
        const auto requested_gid = local_ip.empty()
            ? std::optional<::ibv_gid> {}
            : std::optional<::ibv_gid> { ibv::ipv4_to_gid(local_ip) };
        bool found_unusable_netdev = false;

        for (int index = 0; index < port_attr.gid_tbl_len; ++index) {
            const auto entry = context.query_gid_ex(port_num, index);
            if (!entry.has_value()) {
                continue;
            }
            if (!ibv::is_roce_v2_ipv4_mapped(*entry)) {
                continue;
            }
            if (requested_gid
                && std::memcmp(
                       &entry->gid, &*requested_gid, sizeof(entry->gid))
                    != 0) {
                continue;
            }
            if (index > std::numeric_limits<std::uint8_t>::max()) {
                throw ibv::Error(
                    "RoCE v2 GID index " + std::to_string(index) + " on port "
                    + std::to_string(port) + " does not fit sgid_index");
            }

            char netdev[IF_NAMESIZE] {};
            if (entry->ndev_ifindex <= 0
                || if_indextoname(entry->ndev_ifindex, netdev) == nullptr) {
                found_unusable_netdev = true;
                continue;
            }
            return LocalRoceV2Ipv4Gid {
                entry->gid,
                index,
                static_cast<std::uint32_t>(entry->ndev_ifindex),
                netdev
            };
        }

        if (found_unusable_netdev) {
            if (local_ip.empty()) {
                throw ibv::Error(
                    "IPv4 RoCE v2 GIDs on port " + std::to_string(port)
                    + " have no usable network interface");
            }
            throw ibv::Error(
                "IPv4 RoCE v2 GIDs for local IP '" + local_ip + "' on port "
                + std::to_string(port) + " have no usable network interface");
        }
        if (local_ip.empty()) {
            throw ibv::Error(
                "Failed to find an IPv4 RoCE v2 GID on port "
                + std::to_string(port));
        }
        throw ibv::Error(
            "Failed to find an IPv4 RoCE v2 GID for local IP '" + local_ip
            + "' on port " + std::to_string(port));
    }

    bool selected_gid_matches(
        const LocalRoceV2Ipv4Gid& selected,
        const std::optional<::ibv_gid_entry>& current) noexcept
    {
        return current.has_value()
            && ibv::is_roce_v2_ipv4_mapped(*current)
            && current->ndev_ifindex > 0
            && current->ndev_ifindex == selected.ndev_ifindex
            && std::memcmp(
                   &current->gid, &selected.gid, sizeof(selected.gid))
            == 0;
    }

    AsyncEventDisposition classify_async_event(
        ::ibv_event_type type,
        std::uint32_t event_port,
        std::uint32_t device_port) noexcept
    {
        switch (type) {
        case IBV_EVENT_CQ_ERR:
        case IBV_EVENT_QP_FATAL:
        case IBV_EVENT_QP_REQ_ERR:
        case IBV_EVENT_QP_ACCESS_ERR:
        case IBV_EVENT_DEVICE_FATAL:
            return AsyncEventDisposition::Terminal;
        case IBV_EVENT_PORT_ERR:
            return event_port == device_port
                ? AsyncEventDisposition::Terminal
                : AsyncEventDisposition::Silent;
        case IBV_EVENT_GID_CHANGE:
            return event_port == device_port
                ? AsyncEventDisposition::ValidateSelectedGid
                : AsyncEventDisposition::Silent;
        case IBV_EVENT_COMM_EST:
        case IBV_EVENT_PORT_ACTIVE:
        case IBV_EVENT_LID_CHANGE:
        case IBV_EVENT_PKEY_CHANGE:
        case IBV_EVENT_SM_CHANGE:
        case IBV_EVENT_CLIENT_REREGISTER:
            return AsyncEventDisposition::Silent;
        case IBV_EVENT_SQ_DRAINED:
        case IBV_EVENT_PATH_MIG:
        case IBV_EVENT_PATH_MIG_ERR:
        case IBV_EVENT_SRQ_ERR:
        case IBV_EVENT_SRQ_LIMIT_REACHED:
        case IBV_EVENT_QP_LAST_WQE_REACHED:
            return AsyncEventDisposition::UnexpectedInvariant;
        default:
            // rdma-core v62 appended DEVICE_SPEED_CHANGE at this stable ABI
            // value. The intervening value is the older WQ_FATAL event, which
            // deliberately remains terminal through the default.
            constexpr int kIbvEventDeviceSpeedChange
                = static_cast<int>(IBV_EVENT_GID_CHANGE) + 2;
            if (static_cast<int>(type) == kIbvEventDeviceSpeedChange)
                return AsyncEventDisposition::Silent;
            return AsyncEventDisposition::Terminal;
        }
    }

} // namespace detail

namespace {

    using detail::host_page_size;

    // Every SGE we post covers at most one slot, so the slot stride and the TX
    // payload must both fit ibv_sge::length.
    constexpr std::size_t kMaxSgeLength
        = std::numeric_limits<decltype(ibv_sge::length)>::max();

    // HSB IP 0x2602 and newer packs the slot in the low 12 bits; PSN starts
    // at bit 12. The older 0x2510 layout uses 8 slot bits and is not supported.
    // Never derive this mask from the local ring size: an out-of-range wire
    // page must be rejected instead of aliased.
    constexpr std::uint32_t kModernHsbImmSlotMask = 0xFFF;
    constexpr std::uint32_t kMaxUcQpNumber = 0x00FFFFFF;

    std::size_t checked_ring_bytes(std::size_t stride, std::uint32_t slot_count)
    {
        if (stride == 0 || slot_count == 0) {
            throw std::invalid_argument(
                "RoCE ring stride and slot_count must be non-zero");
        }
        if (stride > std::numeric_limits<std::size_t>::max() / slot_count) {
            throw std::invalid_argument("RoCE ring byte size overflows size_t");
        }
        return stride * slot_count;
    }

    // Cyclic TX maps sequence through `slot_index()` with a mask. RX placement
    // is the CQE identity (wr_id or IMM page), never a wrap, so RX may be any
    // nonzero count up to the HSB IMM field. Returns slot_count so TxRing can
    // reject in the member-init list before RingStorage allocates.
    std::uint32_t checked_cyclic_tx_slot_count(std::uint32_t slot_count)
    {
        if (slot_count == 0 || (slot_count & (slot_count - 1)) != 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver TX slot_count must be a non-zero power of "
                "two (slot_count="
                + std::to_string(slot_count) + ")");
        }
        return slot_count;
    }

    // Caller transmit posts by slot index, not by a cyclic mask.
    std::uint32_t checked_caller_tx_slot_count(std::uint32_t slot_count)
    {
        constexpr std::uint32_t kMaxSlotCount = kModernHsbImmSlotMask + 1;
        if (slot_count == 0 || slot_count > kMaxSlotCount) {
            throw std::invalid_argument(
                "CpuRoceTransceiver caller transmit slot_count must be from "
                "1 to 4096 (slot_count="
                + std::to_string(slot_count) + ")");
        }
        return slot_count;
    }

    // libibverbs has reentrancy problems in processes that fork, which the
    // applications we run under may do. RoceTransmitterOp and RoceReceiver do
    // the same thing from their own start paths; this call cannot order itself
    // against registrations those make, so each verbs user opts in for itself.
    void initialize_fork_safety()
    {
        // Function-local static initialization serializes concurrent callers
        // and retains a failure, so a later setup() cannot proceed to
        // registration believing fork safety is on.
        static const int result = ibv_fork_init();
        if (result != 0) {
            throw ibv::Error(
                "ibv_fork_init failed: "
                + std::generic_category().message(result));
        }
    }

    void require_host_range_in_address_space(
        const HostMemory& host, std::size_t bytes)
    {
        const auto begin = reinterpret_cast<std::uintptr_t>(host.address);
        if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
            throw std::invalid_argument(
                "Host RoCE ring address range overflows uintptr_t");
        }
    }

    void require_borrowed_host_ring(const HostMemory& host, std::size_t bytes)
    {
        if (host.address == nullptr || host.size < bytes) {
            throw std::invalid_argument(
                "Host RoCE ring memory must be non-null and cover "
                "slot_count * stride");
        }
        require_host_range_in_address_space(host, bytes);
    }

    void require_borrowed_rx_page_alignment(const HostMemory& host)
    {
        // Zero-based RX registration requires a page-aligned host address.
        // TX uses a virtual-address IOVA and is not checked here.
        const auto page = host_page_size();
        const auto address = reinterpret_cast<std::uintptr_t>(host.address);
        if (address % page != 0) {
            throw std::invalid_argument(
                "Borrowed RX host memory must be OS-page aligned for "
                "zero-based RX registration");
        }
    }

    bool host_ranges_overlap(
        const HostMemory& left, std::size_t left_bytes,
        const HostMemory& right, std::size_t right_bytes)
    {
        if (left.address == nullptr || right.address == nullptr) {
            return false;
        }

        const auto left_begin = reinterpret_cast<std::uintptr_t>(left.address);
        const auto right_begin = reinterpret_cast<std::uintptr_t>(right.address);
        const auto left_end = left_begin + left_bytes;
        const auto right_end = right_begin + right_bytes;
        return left_begin < right_end && right_begin < left_end;
    }

    void require_dmabuf_ring(
        const DmaBufMemory& dmabuf, std::size_t bytes)
    {
        if (dmabuf.address == nullptr || dmabuf.fd < 0
            || dmabuf.size < bytes) {
            throw std::invalid_argument(
                "DMA-BUF RoCE ring memory must have a non-null application "
                "address, a valid fd, and cover slot_count * stride");
        }
        if (dmabuf.offset % host_page_size() != 0) {
            throw std::invalid_argument(
                "DMA-BUF RoCE ring offset must be OS-page aligned");
        }
        const auto address
            = reinterpret_cast<std::uintptr_t>(dmabuf.address);
        if (address > std::numeric_limits<std::uintptr_t>::max() - bytes) {
            throw std::invalid_argument(
                "DMA-BUF RoCE ring application address range overflows "
                "uintptr_t");
        }
        if (dmabuf.offset
            > std::numeric_limits<std::uint64_t>::max() - bytes) {
            throw std::invalid_argument(
                "DMA-BUF RoCE ring fd range overflows uint64_t");
        }
    }

    bool dmabuf_ranges_overlap(
        const DmaBufMemory& left, std::size_t left_bytes,
        const DmaBufMemory& right, std::size_t right_bytes)
    {
        // Same integer fd only. dup()ed fds and other aliases of one object
        // are the caller's duty, like two host virtual addresses over one map.
        if (left.fd < 0 || right.fd < 0 || left.fd != right.fd) {
            return false;
        }
        const auto left_end = left.offset + left_bytes;
        const auto right_end = right.offset + right_bytes;
        return left.offset < right_end && right.offset < left_end;
    }

    void require_gpu_pointer_ring(const GpuPointerMemory& gpu, std::size_t bytes)
    {
        if (gpu.address == nullptr || gpu.size < bytes) {
            throw std::invalid_argument(
                "GPU-pointer RoCE ring memory must be non-null and cover "
                "slot_count * stride");
        }
        const auto begin = reinterpret_cast<std::uintptr_t>(gpu.address);
        if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
            throw std::invalid_argument(
                "GPU-pointer RoCE ring address range overflows uintptr_t");
        }
    }

    void require_gpu_pointer_rx_page_alignment(const GpuPointerMemory& gpu)
    {
        // IOVA zero requires the GPU pointer's page offset to be zero.
        const auto page = host_page_size();
        const auto address = reinterpret_cast<std::uintptr_t>(gpu.address);
        if (address % page != 0) {
            throw std::invalid_argument(
                "GPU-pointer RX memory must be OS-page aligned for zero-based "
                "registration");
        }
    }

    bool gpu_pointer_ranges_overlap(
        const GpuPointerMemory& left, std::size_t left_bytes,
        const GpuPointerMemory& right, std::size_t right_bytes)
    {
        if (left.address == nullptr || right.address == nullptr) {
            return false;
        }
        const auto left_begin = reinterpret_cast<std::uintptr_t>(left.address);
        const auto right_begin
            = reinterpret_cast<std::uintptr_t>(right.address);
        const auto left_end = left_begin + left_bytes;
        const auto right_end = right_begin + right_bytes;
        return left_begin < right_end && right_begin < left_end;
    }

    bool is_ipv4_address(const std::string& address)
    {
        ::in_addr parsed {};
        return !address.empty()
            && inet_pton(AF_INET, address.c_str(), &parsed) == 1;
    }

    // Every mode switch outside blocking_monitor() ends here, so an added
    // enumerator fails loudly at setup instead of inheriting a default.
    [[noreturn]] void throw_invalid_mode(CpuRoceTransceiver::Mode mode)
    {
        throw std::invalid_argument(
            "CpuRoceTransceiver mode is invalid ("
            + std::to_string(static_cast<int>(mode)) + ")");
    }

    struct ModeLayout {
        bool rx = false;
        bool tx = false;
        enum class Outbound { None,
            Tx,
            Rx };
        Outbound outbound = Outbound::None;
    };

    // One table per mode. `rx`/`tx` say which payload rings setup() builds and
    // validate_config() checks; `outbound` says which ring the wire reads from.
    // Those are different questions: Forward allocates only RX and sends from
    // RX storage. Do not use `outbound` to decide what to allocate, or
    // `rx`/`tx` to decide what to transmit. A new mode answers all three here.
    [[nodiscard]] ModeLayout mode_layout(CpuRoceTransceiver::Mode mode)
    {
        switch (mode) {
        case CpuRoceTransceiver::Mode::Rx:
            return { true, false, ModeLayout::Outbound::None };
        case CpuRoceTransceiver::Mode::Forward:
            return { true, false, ModeLayout::Outbound::Rx };
        case CpuRoceTransceiver::Mode::Tx:
            return { false, true, ModeLayout::Outbound::Tx };
        case CpuRoceTransceiver::Mode::Duplex:
        case CpuRoceTransceiver::Mode::Unified:
        case CpuRoceTransceiver::Mode::CallerDriven:
            return { true, true, ModeLayout::Outbound::Tx };
        default:
            throw_invalid_mode(mode);
        }
    }

    // Geometry blocking_monitor() actually transmits, for peer validation
    // and inline negotiation.
    [[nodiscard]] const RingConfig* active_outbound_ring(
        const CpuRoceTransceiver::Config& config)
    {
        switch (mode_layout(config.mode).outbound) {
        case ModeLayout::Outbound::None:
            return nullptr;
        case ModeLayout::Outbound::Tx:
            return &config.tx;
        case ModeLayout::Outbound::Rx:
            return &config.rx;
        }
        throw_invalid_mode(config.mode);
    }

    void require_write_with_imm_remote_range(
        const CpuRoceTransceiver::Peer& peer,
        std::uint32_t slot_count,
        std::size_t payload_size)
    {
        const auto max_u64 = std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t last_index = slot_count - 1;
        if (peer.rx_stride != 0 && last_index > max_u64 / peer.rx_stride) {
            throw std::invalid_argument(
                "CpuRoceTransceiver WriteWithImm remote slot span overflows "
                "uint64_t (slot_count="
                + std::to_string(slot_count) + ", rx_stride="
                + std::to_string(peer.rx_stride) + ")");
        }
        const std::uint64_t slot_span = last_index * peer.rx_stride;
        if (peer.rx_base > max_u64 - slot_span) {
            throw std::invalid_argument(
                "CpuRoceTransceiver WriteWithImm remote address overflows "
                "uint64_t (rx_base="
                + std::to_string(peer.rx_base) + ", slot_span="
                + std::to_string(slot_span) + ")");
        }
        const std::uint64_t last_slot_start = peer.rx_base + slot_span;
        if (payload_size == 0 || last_slot_start > max_u64 - (payload_size - 1)) {
            throw std::invalid_argument(
                "CpuRoceTransceiver WriteWithImm last addressed byte overflows "
                "uint64_t (last_slot_start="
                + std::to_string(last_slot_start) + ", payload_size="
                + std::to_string(payload_size) + ")");
        }
    }

    void validate_ring_config(const RingConfig& ring)
    {
        if (ring.stride == 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver stride must be non-zero");
        }
        if (ring.stride > kMaxSgeLength) {
            throw std::invalid_argument(
                "CpuRoceTransceiver stride must fit in an ibv_sge length "
                "(stride="
                + std::to_string(ring.stride) + ", max="
                + std::to_string(kMaxSgeLength) + ")");
        }
        if (ring.frame_layout.payload_size == 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver payload_size must be non-zero");
        }
        if (ring.frame_layout.payload_size > ring.stride) {
            throw std::invalid_argument(
                "CpuRoceTransceiver payload_size must fit in its slot "
                "(payload_size="
                + std::to_string(ring.frame_layout.payload_size) + ", stride="
                + std::to_string(ring.stride) + ")");
        }
        if (ring.slot_count == 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver slot_count must be non-zero");
        }
        constexpr std::uint32_t kMaxSlotCount = kModernHsbImmSlotMask + 1;
        if (ring.slot_count > kMaxSlotCount) {
            throw std::invalid_argument(
                "CpuRoceTransceiver slot_count must not exceed the 4096 slots "
                "addressable by the modern HSB 12-bit IMM page field "
                "(slot_count="
                + std::to_string(ring.slot_count)
                + ", max=" + std::to_string(kMaxSlotCount) + ")");
        }
    }

    void require_rx_metadata_geometry(const RingConfig& rx)
    {
        if (rx.frame_layout.metadata_size == 0) {
            return;
        }
        if (rx.frame_layout.metadata_offset > rx.stride
            || rx.frame_layout.metadata_offset
                < rx.frame_layout.payload_size
            || rx.frame_layout.metadata_size
                > rx.stride - rx.frame_layout.metadata_offset) {
            throw std::invalid_argument(
                "CpuRoceTransceiver RX metadata must follow the payload and "
                "fit in its slot "
                "(metadata_offset="
                + std::to_string(rx.frame_layout.metadata_offset)
                + ", metadata_size="
                + std::to_string(rx.frame_layout.metadata_size)
                + ", stride=" + std::to_string(rx.stride) + ")");
        }
    }

    // The RX ring is an ibv_reg_mr target for the peer, so a borrowed span
    // must also be page aligned. Returns the ring's byte size.
    std::size_t validate_rx_ring(const RingConfig& rx)
    {
        validate_ring_config(rx);
        require_rx_metadata_geometry(rx);
        const auto bytes = checked_ring_bytes(rx.stride, rx.slot_count);
        if (const auto* host = std::get_if<HostMemory>(&rx.payload_memory)) {
            require_borrowed_host_ring(*host, bytes);
            require_borrowed_rx_page_alignment(*host);
        } else if (
            const auto* dmabuf
            = std::get_if<DmaBufMemory>(&rx.payload_memory)) {
            require_dmabuf_ring(*dmabuf, bytes);
        } else if (
            const auto* gpu
            = std::get_if<GpuPointerMemory>(&rx.payload_memory)) {
            require_gpu_pointer_ring(*gpu, bytes);
            require_gpu_pointer_rx_page_alignment(*gpu);
        }
        return bytes;
    }

    // Returns the ring's byte size. cyclic_tx is the flag path. Caller
    // transmit passes false and may use a depth that is not a power of two.
    std::size_t validate_tx_ring(const RingConfig& tx, bool cyclic_tx)
    {
        if (tx.frame_layout.metadata_size != 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver does not support TX frame metadata");
        }
        validate_ring_config(tx);
        if (cyclic_tx) {
            checked_cyclic_tx_slot_count(tx.slot_count);
        } else {
            checked_caller_tx_slot_count(tx.slot_count);
        }
        const auto bytes = checked_ring_bytes(tx.stride, tx.slot_count);
        if (const auto* host = std::get_if<HostMemory>(&tx.payload_memory)) {
            require_borrowed_host_ring(*host, bytes);
        } else if (
            const auto* dmabuf
            = std::get_if<DmaBufMemory>(&tx.payload_memory)) {
            require_dmabuf_ring(*dmabuf, bytes);
        } else if (
            const auto* gpu
            = std::get_if<GpuPointerMemory>(&tx.payload_memory)) {
            require_gpu_pointer_ring(*gpu, bytes);
        }
        return bytes;
    }

    std::uint32_t mtu_payload_bytes(::ibv_mtu mtu)
    {
        switch (mtu) {
        case IBV_MTU_256:
            return 256;
        case IBV_MTU_512:
            return 512;
        case IBV_MTU_1024:
            return 1024;
        case IBV_MTU_2048:
            return 2048;
        case IBV_MTU_4096:
            return 4096;
        default:
            return 0;
        }
    }

    int interface_mtu(const std::string& name)
    {
        if (name.empty() || name.size() >= IFNAMSIZ) {
            return -1;
        }
        const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd < 0) {
            return -1;
        }
        ifreq request {};
        std::strncpy(request.ifr_name, name.c_str(), IFNAMSIZ - 1);
        const int rc = ::ioctl(fd, SIOCGIFMTU, &request);
        ::close(fd);
        return rc < 0 ? -1 : request.ifr_mtu;
    }

    // Largest UC path MTU that fits the port and the netdev MTU minus the
    // RoCEv2 and VLAN headers (44 bytes). A failed netdev query keeps the
    // port active MTU.
    ::ibv_mtu clamped_path_mtu(const std::string& netdev, ::ibv_mtu active)
    {
        const std::uint32_t active_bytes = mtu_payload_bytes(active);
        if (active_bytes == 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver caller transmit path MTU is unknown");
        }
        std::uint32_t room = active_bytes;
        const int host_mtu = interface_mtu(netdev);
        constexpr int kRoceOverhead = 44;
        if (host_mtu >= kRoceOverhead) {
            const auto host_room
                = static_cast<std::uint32_t>(host_mtu - kRoceOverhead);
            if (host_room < room) {
                room = host_room;
            }
        }
        static constexpr std::uint32_t kSteps[]
            = { 4096, 2048, 1024, 512, 256 };
        static constexpr ::ibv_mtu kEnums[] = {
            IBV_MTU_4096, IBV_MTU_2048, IBV_MTU_1024, IBV_MTU_512, IBV_MTU_256
        };
        for (std::size_t i = 0; i < 5; ++i) {
            if (room >= kSteps[i]) {
                return kEnums[i];
            }
        }
        throw std::invalid_argument(
            "CpuRoceTransceiver caller transmit path MTU is below 256 bytes");
    }

    void validate_config_impl(
        const CpuRoceTransceiver::Config& config, bool any_tx_depth)
    {
        switch (config.tx_operation) {
        case CpuRoceTransceiver::TxOperation::Send:
        case CpuRoceTransceiver::TxOperation::WriteWithImm:
            break;
        default:
            throw std::invalid_argument(
                "CpuRoceTransceiver tx_operation is invalid");
        }
        if (config.ibv_name.empty()) {
            throw std::invalid_argument(
                "CpuRoceTransceiver ibv_name must not be empty");
        }
        if (config.ibv_port == 0
            || config.ibv_port > std::numeric_limits<std::uint8_t>::max()) {
            throw std::invalid_argument(
                "CpuRoceTransceiver ibv_port must fit a non-zero uint8_t");
        }
        if (!config.local_ip.empty() && !is_ipv4_address(config.local_ip)) {
            throw std::invalid_argument(
                "CpuRoceTransceiver local_ip must be a valid IPv4 address");
        }
        if (config.rx_cpu.has_value() != config.tx_cpu.has_value()) {
            throw std::invalid_argument(
                "CpuRoceTransceiver Duplex requires both rx_cpu and tx_cpu, "
                "or neither");
        }
        if (config.mode == CpuRoceTransceiver::Mode::Duplex
            && config.poll_cpu.has_value()) {
            throw std::invalid_argument(
                "CpuRoceTransceiver poll_cpu is not valid in Duplex mode; "
                "use rx_cpu and tx_cpu");
        }
        if (config.mode != CpuRoceTransceiver::Mode::Duplex
            && config.rx_cpu.has_value()) {
            throw std::invalid_argument(
                "CpuRoceTransceiver rx_cpu and tx_cpu are Duplex-only");
        }
        if (config.mode == CpuRoceTransceiver::Mode::CallerDriven
            && config.poll_cpu.has_value()) {
            throw std::invalid_argument(
                "CpuRoceTransceiver poll_cpu does not apply to CallerDriven; "
                "the caller thread is the poller");
        }
        if (config.poll_cpu && !config.poll_cpu_best_effort
            && (*config.poll_cpu < 0 || *config.poll_cpu >= CPU_SETSIZE)) {
            throw std::invalid_argument(
                "CpuRoceTransceiver poll_cpu is out of range");
        }
        if (config.rx_cpu
            && (*config.rx_cpu < 0 || *config.rx_cpu >= CPU_SETSIZE
                || *config.tx_cpu < 0 || *config.tx_cpu >= CPU_SETSIZE)) {
            throw std::invalid_argument(
                "CpuRoceTransceiver Duplex CPU affinity is out of range");
        }
        if (config.rx_cpu && *config.rx_cpu == *config.tx_cpu) {
            throw std::invalid_argument(
                "CpuRoceTransceiver Duplex RX and TX workers require "
                "different CPUs");
        }
        const auto layout = mode_layout(config.mode);
        std::size_t rx_bytes = 0;
        std::size_t tx_bytes = 0;
        if (layout.rx) {
            rx_bytes = validate_rx_ring(config.rx);
        }
        if (layout.tx) {
            tx_bytes = validate_tx_ring(config.tx, !any_tx_depth);
        }
        if (layout.rx && layout.tx) {
            const auto* rx_host
                = std::get_if<HostMemory>(&config.rx.payload_memory);
            const auto* tx_host
                = std::get_if<HostMemory>(&config.tx.payload_memory);
            if (rx_host != nullptr && tx_host != nullptr
                && host_ranges_overlap(
                    *rx_host, rx_bytes, *tx_host, tx_bytes)) {
                throw std::invalid_argument(
                    "CpuRoceTransceiver modes that allocate both rings "
                    "require disjoint host RX and TX rings");
            }
            const auto* rx_dmabuf
                = std::get_if<DmaBufMemory>(&config.rx.payload_memory);
            const auto* tx_dmabuf
                = std::get_if<DmaBufMemory>(&config.tx.payload_memory);
            if (rx_dmabuf != nullptr && tx_dmabuf != nullptr
                && dmabuf_ranges_overlap(
                    *rx_dmabuf, rx_bytes, *tx_dmabuf, tx_bytes)) {
                throw std::invalid_argument(
                    "CpuRoceTransceiver modes that allocate both rings "
                    "require disjoint DMA-BUF RX and TX rings");
            }
            const auto* rx_gpu
                = std::get_if<GpuPointerMemory>(&config.rx.payload_memory);
            const auto* tx_gpu
                = std::get_if<GpuPointerMemory>(&config.tx.payload_memory);
            if (rx_gpu != nullptr && tx_gpu != nullptr
                && gpu_pointer_ranges_overlap(
                    *rx_gpu, rx_bytes, *tx_gpu, tx_bytes)) {
                throw std::invalid_argument(
                    "CpuRoceTransceiver Duplex and Unified require disjoint "
                    "GPU-pointer RX and TX rings");
            }
        }
        const bool pairs_slots
            = config.mode == CpuRoceTransceiver::Mode::Unified
            || config.mode == CpuRoceTransceiver::Mode::CallerDriven;
        if (pairs_slots
            && (std::holds_alternative<DmaBufMemory>(
                    config.rx.payload_memory)
                || std::holds_alternative<DmaBufMemory>(
                    config.tx.payload_memory)
                || std::holds_alternative<GpuPointerMemory>(
                    config.rx.payload_memory)
                || std::holds_alternative<GpuPointerMemory>(
                    config.tx.payload_memory))) {
            throw std::invalid_argument(
                "CpuRoceTransceiver Unified and CallerDriven require "
                "CPU-addressable payload memory");
        }
        if (pairs_slots && config.rx.slot_count != config.tx.slot_count) {
            throw std::invalid_argument(
                "CpuRoceTransceiver Unified and CallerDriven require equal RX "
                "and TX slot_count values (rx="
                + std::to_string(config.rx.slot_count) + ", tx="
                + std::to_string(config.tx.slot_count) + ")");
        }
        if (config.rx_path_mtu_4096 && config.mode != CpuRoceTransceiver::Mode::Rx) {
            throw std::invalid_argument(
                "CpuRoceTransceiver rx_path_mtu_4096 is valid only in Rx mode");
        }
    }

    void validate_outbound_against_peer(
        const CpuRoceTransceiver::Config& config,
        const CpuRoceTransceiver::Peer& peer,
        const RingConfig& outbound)
    {
        const auto payload = outbound.frame_layout.payload_size;
        const auto local_slots = outbound.slot_count;
        if (peer.rx_stride < payload) {
            throw std::invalid_argument(
                "CpuRoceTransceiver outbound payload_size exceeds peer RX "
                "stride (payload_size="
                + std::to_string(payload) + ", rx_stride="
                + std::to_string(peer.rx_stride) + ")");
        }
        if (config.tx_operation
            != CpuRoceTransceiver::TxOperation::WriteWithImm) {
            return;
        }
        if (peer.rx_rkey == 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver WriteWithImm requires a non-zero peer RX "
                "rkey");
        }
        if (peer.rx_slot_count == 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver WriteWithImm requires a non-zero peer "
                "RX slot_count");
        }
        if (local_slots > peer.rx_slot_count) {
            throw std::invalid_argument(
                "CpuRoceTransceiver WriteWithImm outbound slot_count exceeds "
                "peer RX slot_count (outbound="
                + std::to_string(local_slots) + ", peer="
                + std::to_string(peer.rx_slot_count) + ")");
        }
        require_write_with_imm_remote_range(peer, local_slots, payload);
    }

    void validate_peer_impl(
        const CpuRoceTransceiver::Config& config,
        const CpuRoceTransceiver::Peer& peer)
    {
        if (!is_ipv4_address(peer.ip)) {
            throw std::invalid_argument(
                "CpuRoceTransceiver peer IP must be a valid IPv4 address");
        }
        if (peer.qp > kMaxUcQpNumber) {
            throw std::invalid_argument(
                "CpuRoceTransceiver peer QP must fit the 24-bit QPN field");
        }
        const auto layout = mode_layout(config.mode);
        const RingConfig* outbound = active_outbound_ring(config);
        // A TX ring or a wire outbound both need a peer QP. The OR covers a
        // layout that allocates TX storage but forgets Outbound.
        if ((layout.tx || outbound != nullptr) && peer.qp == 0) {
            throw std::invalid_argument(
                "CpuRoceTransceiver transmit modes require a non-zero peer QP");
        }
        if (outbound != nullptr) {
            validate_outbound_against_peer(config, peer, *outbound);
        }
    }

} // namespace

namespace detail {

    void validate_config(
        const CpuRoceTransceiver::Config& config, bool any_tx_depth)
    {
        validate_config_impl(config, any_tx_depth);
    }

    void validate_peer(
        const CpuRoceTransceiver::Config& config,
        const CpuRoceTransceiver::Peer& peer)
    {
        validate_peer_impl(config, peer);
    }

    HostAllocation::HostAllocation(std::size_t size, std::size_t alignment)
    {
        if (alignment < alignof(void*) || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("Host allocation alignment must be a power of two");
        }
        // Transfer ownership only after allocation succeeds, so a throwing
        // constructor never exposes memory for the destructor to free.
        void* allocated = nullptr;
        if (posix_memalign(&allocated, alignment, size) != 0) {
            throw std::runtime_error("Unable to allocate a host RoCE ring");
        }
        address_ = allocated;
        std::memset(address_, 0, size);
    }

    HostAllocation::~HostAllocation()
    {
        reset();
    }

    HostAllocation::HostAllocation(HostAllocation&& other) noexcept
    {
        swap(other);
    }

    HostAllocation& HostAllocation::operator=(HostAllocation&& other) noexcept
    {
        HostAllocation temporary(std::move(other));
        swap(temporary);
        return *this;
    }

    void* HostAllocation::get() const { return address_; }

    void HostAllocation::reset() noexcept
    {
        if (address_ == nullptr) {
            return;
        }
        std::free(address_);
        address_ = nullptr;
    }

    void HostAllocation::swap(HostAllocation& other) noexcept
    {
        std::swap(address_, other.address_);
    }

    RingStorage::RingStorage(
        const PayloadMemory& memory, std::size_t stride, std::uint32_t slot_count,
        bool gpu_pointer_iova_zero)
    {
        const std::size_t data_bytes = checked_ring_bytes(stride, slot_count);
        const std::size_t flags_bytes
            = checked_ring_bytes(sizeof(std::uint64_t), slot_count);

        stride_ = stride;
        slot_count_ = slot_count;

        if (std::holds_alternative<TransceiverAllocation>(memory)) {
            // Align to an OS memory page (ibv_reg_mr), not to a ring slot.
            owned_data_ = HostAllocation(data_bytes, host_page_size());
            data_ = static_cast<std::uint8_t*>(owned_data_.get());
            sge_base_ = reinterpret_cast<std::uint64_t>(data_);
        } else if (const auto* host = std::get_if<HostMemory>(&memory)) {
            require_borrowed_host_ring(*host, data_bytes);
            data_ = static_cast<std::uint8_t*>(host->address);
            sge_base_ = reinterpret_cast<std::uint64_t>(data_);
        } else if (const auto* gpu = std::get_if<GpuPointerMemory>(&memory)) {
            require_gpu_pointer_ring(*gpu, data_bytes);
            data_ = static_cast<std::uint8_t*>(gpu->address);
            gpu_pointer_ = true;
            if (gpu_pointer_iova_zero) {
                require_gpu_pointer_rx_page_alignment(*gpu);
                sge_base_ = 0;
            } else {
                sge_base_ = reinterpret_cast<std::uint64_t>(data_);
            }
        } else {
            const auto& dmabuf = std::get<DmaBufMemory>(memory);
            require_dmabuf_ring(dmabuf, data_bytes);
            data_ = static_cast<std::uint8_t*>(dmabuf.address);
            sge_base_ = 0;
            dmabuf_ = dmabuf;
        }

        // Flags are CPU-only; 64-byte alignment is a cache line.
        owned_flags_ = HostAllocation(flags_bytes, 64);
        flags_ = static_cast<std::uint64_t*>(owned_flags_.get());
    }

    std::uint64_t RingStorage::slot_address(std::uint32_t slot) const
    {
        if (slot >= slot_count_) {
            throw std::out_of_range("RoCE ring slot is out of range");
        }
        return reinterpret_cast<std::uint64_t>(data_) + slot * stride_;
    }

    std::uint64_t RingStorage::slot_sge_address(std::uint32_t slot) const
    {
        if (slot >= slot_count_) {
            throw std::out_of_range("RoCE ring slot is out of range");
        }
        return sge_base_ + slot * stride_;
    }

    RxRing::RxRing(
        const PayloadMemory& memory, std::size_t stride, std::uint32_t slot_count)
        : storage_(memory, stride, slot_count, true)
        , receive_policy_(
              storage_.dmabuf() || storage_.gpu_pointer()
                  ? ReceivePolicy::WriteWithImmOnly
                  : ReceivePolicy::SendAndWriteWithImm)
    {
    }

    RxPublishResult RxRing::publish_rx(std::uint32_t slot)
    {
        if (slot >= storage_.slot_count()) {
            return RxPublishResult::InvalidSlot;
        }

        if (CpuRoceTransceiver::load_flag_acquire(&storage_.flags()[slot]) != 0) {
            return RxPublishResult::Collision;
        }

        // One monitor owns RX publication. The consumer releases by storing
        // zero; that is the only RX occupancy signal.
        CpuRoceTransceiver::store_flag_release(
            &storage_.flags()[slot], storage_.slot_address(slot));
        return RxPublishResult::Published;
    }

    TxRing::TxRing(
        const PayloadMemory& memory, std::size_t stride, std::uint32_t slot_count,
        bool cyclic)
        : storage_(memory, stride,
            cyclic ? checked_cyclic_tx_slot_count(slot_count)
                   : checked_caller_tx_slot_count(slot_count))
    {
    }

    std::uint32_t TxRing::slot_index(std::uint64_t sequence) const
    {
        // Mask wrap; TX slot_count is a power of two.
        return static_cast<std::uint32_t>(sequence & (storage_.slot_count() - 1));
    }

    TxClaimResult TxRing::try_claim_tx(std::uint32_t slot)
    {
        if (slot >= storage_.slot_count()) {
            throw std::out_of_range("TX slot is out of range");
        }
        // An unpublished slot is read-only on this path. Do not fold the load
        // and the clear into an unconditional exchange: the TX poller spins here,
        // and an RMW per idle poll dirties the line the producer is writing.
        volatile std::uint64_t* const flag = &storage_.flags()[slot];
        std::uint64_t published_address
            = CpuRoceTransceiver::load_flag_acquire(flag);
        if (published_address == 0) {
            return TxClaimResult::NotReady;
        }
        if (published_address != storage_.slot_address(slot)) {
            throw std::runtime_error(
                "Published TX address does not match the expected ring slot");
        }
        // An out-of-contract same-address republication is indistinguishable
        // and may coalesce. Replacing this load/store with a claim CAS cannot
        // detect or prevent that ABA case.
        CpuRoceTransceiver::store_flag_release(flag, 0);

        return TxClaimResult::Claimed;
    }

    RxCqeDecode decode_rx_cqe(
        const ::ibv_wc& wc,
        std::uint32_t slot_count,
        bool write_with_imm_only)
    {
        RxCqeDecode decoded;
        decoded.recv_wr_id = wc.wr_id;
        if (wc.status != IBV_WC_SUCCESS) {
            decoded.status = RxCqeDecode::Status::FailedStatus;
            return decoded;
        }

        const bool wr_id_in_range
            = wc.wr_id < static_cast<std::uint64_t>(slot_count);

        switch (wc.opcode) {
        case IBV_WC_RECV:
            if (write_with_imm_only) {
                decoded.status = RxCqeDecode::Status::UnexpectedSendForCreditOnly;
                return decoded;
            }
            if (!wr_id_in_range) {
                decoded.status = RxCqeDecode::Status::InvalidWrId;
                return decoded;
            }
            decoded.payload_slot = static_cast<std::uint32_t>(wc.wr_id);
            decoded.status = RxCqeDecode::Status::Ok;
            return decoded;
        case IBV_WC_RECV_RDMA_WITH_IMM: {
            if ((wc.wc_flags & IBV_WC_WITH_IMM) == 0) {
                decoded.status = RxCqeDecode::Status::MissingImm;
                return decoded;
            }
            if (!wr_id_in_range) {
                decoded.status = RxCqeDecode::Status::InvalidWrId;
                return decoded;
            }
            const std::uint32_t imm = ntohl(wc.imm_data);
            const std::uint32_t page = imm & kModernHsbImmSlotMask;
            if (page >= slot_count) {
                decoded.status = RxCqeDecode::Status::InvalidImmSlot;
                return decoded;
            }
            decoded.payload_slot = page;
            decoded.status = RxCqeDecode::Status::Ok;
            return decoded;
        }
        default:
            decoded.status = RxCqeDecode::Status::UnsupportedOpcode;
            return decoded;
        }
    }

    RxCqeDecode decode_handled_rx_cqe(
        const ::ibv_wc& wc,
        std::uint32_t credit_count)
    {
        RxCqeDecode decoded;
        decoded.recv_wr_id = wc.wr_id;
        if (wc.status != IBV_WC_SUCCESS) {
            decoded.status = RxCqeDecode::Status::FailedStatus;
            return decoded;
        }

        const bool wr_id_in_range
            = wc.wr_id < static_cast<std::uint64_t>(credit_count);
        switch (wc.opcode) {
        case IBV_WC_RECV:
            // These credits have no buffer. A Send with payload completes as
            // a local length error. A zero-length Send-with-Immediate would
            // otherwise publish a slot whose bytes were never written.
            decoded.status = RxCqeDecode::Status::UnexpectedSend;
            return decoded;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            if ((wc.wc_flags & IBV_WC_WITH_IMM) == 0) {
                decoded.status = RxCqeDecode::Status::MissingImm;
                return decoded;
            }
            if (!wr_id_in_range) {
                decoded.status = RxCqeDecode::Status::InvalidWrId;
                return decoded;
            }
            decoded.status = RxCqeDecode::Status::Ok;
            return decoded;
        default:
            decoded.status = RxCqeDecode::Status::UnsupportedOpcode;
            return decoded;
        }
    }

    RxCompletion make_rx_completion(
        const ::ibv_wc& wc, std::uint32_t credit_id)
    {
        // Presence follows IBV_WC_WITH_IMM: Send-with-Imm also carries IMM.
        std::optional<std::uint32_t> immediate_data;
        if ((wc.wc_flags & IBV_WC_WITH_IMM) != 0) {
            immediate_data = ntohl(wc.imm_data);
        }
        return { credit_id, immediate_data };
    }

    TxCqeDecode decode_tx_cqe(
        const ::ibv_wc& wc,
        std::uint64_t expected_sequence,
        ibv_wc_opcode expected_opcode) noexcept
    {
        TxCqeDecode decoded;
        decoded.sequence = wc.wr_id;
        if (wc.status != IBV_WC_SUCCESS) {
            decoded.status = TxCqeDecode::Status::FailedStatus;
        } else if (wc.opcode != expected_opcode) {
            decoded.status = TxCqeDecode::Status::UnsupportedOpcode;
        } else if (wc.wr_id != expected_sequence) {
            decoded.status = TxCqeDecode::Status::OutOfSequence;
        } else {
            decoded.status = TxCqeDecode::Status::Ok;
        }
        return decoded;
    }

} // namespace detail

namespace {

    constexpr int kRxPollBatch = 16;
    constexpr int kTxPollBatch = 16;
    constexpr int kMaxAsyncEventsPerCheck = 8;
    // Async events are fatal notifications, not the CQE path. A nonblocking
    // drain every spin is several times an empty ibv_poll_cq; bound it to
    // every 1024 iterations.
    constexpr unsigned kAsyncCheckInterval = 1024;
    static_assert((kAsyncCheckInterval & (kAsyncCheckInterval - 1)) == 0,
        "async check interval must be a power of two");
    // CallerDriven counts every rx_poll() and resets the count on a drain. An
    // empty call drains once this many calls have passed since the last
    // drain. Twice that, with no drain, drains before the claim so a
    // selected-GID change is seen while requests keep arriving.
    constexpr unsigned kCallerDrivenAsyncCheckInterval = 65536;
    static_assert(
        (kCallerDrivenAsyncCheckInterval
            & (kCallerDrivenAsyncCheckInterval - 1))
            == 0,
        "CallerDriven async check interval must be a power of two");

    // Provisional mlx5 policy: keep an inline WQE within four 64-byte WQEBBs.
    // WriteWithImm spends one additional 16-byte segment on remote addressing.
    // Benchmark these cutoffs on target hardware before raising them.
    constexpr std::uint32_t kMaxInlineSendPayload = 236;
    constexpr std::uint32_t kMaxInlineWriteWithImmPayload = 220;

    using detail::cpu_relax;

    constexpr std::uint32_t max_inline_payload(
        CpuRoceTransceiver::TxOperation operation) noexcept
    {
        return operation == CpuRoceTransceiver::TxOperation::Send
            ? kMaxInlineSendPayload
            : kMaxInlineWriteWithImmPayload;
    }

    std::string async_event_name(::ibv_event_type type)
    {
        const char* const name = ibv_event_type_str(type);
        return name != nullptr ? std::string(name)
                               : std::string("unknown event ")
                + std::to_string(static_cast<int>(type));
    }

    // Consume and classify up to kMaxAsyncEventsPerCheck pending events.
    // Benign events are acknowledged; returns a description of the first
    // condition that makes this transceiver session terminal.
    std::optional<std::string> drain_async_events(
        ibv::Context& context,
        std::uint32_t device_port,
        const detail::LocalRoceV2Ipv4Gid& selected_gid)
    {
        for (int i = 0; i < kMaxAsyncEventsPerCheck; ++i) {
            const auto event = context.poll_async_event();
            if (!event) {
                return std::nullopt;
            }
            switch (detail::classify_async_event(
                event->type, event->port_num, device_port)) {
            case detail::AsyncEventDisposition::Terminal:
                if (event->type == IBV_EVENT_PORT_ERR) {
                    // Session policy, not a verbs guarantee that every QP on
                    // the port enters Error. This UC path has no recovery,
                    // retry, or resync, so a down configured port is terminal.
                    return async_event_name(event->type) + " on port "
                        + std::to_string(event->port_num);
                }
                return async_event_name(event->type);
            case detail::AsyncEventDisposition::Silent:
                break;
            case detail::AsyncEventDisposition::ValidateSelectedGid:
                // The event is port-wide and does not name the changed slot.
                // Revalidate only the entry selected during setup; unrelated
                // table churn does not terminate this session.
                if (!detail::selected_gid_matches(
                        selected_gid,
                        context.query_gid_ex(device_port, selected_gid.index))) {
                    return async_event_name(event->type)
                        + " invalidated selected GID index "
                        + std::to_string(selected_gid.index) + " on port "
                        + std::to_string(device_port);
                }
                break;
            case detail::AsyncEventDisposition::UnexpectedInvariant:
                return async_event_name(event->type)
                    + " is unexpected for CpuRoceTransceiver";
            }
        }
        return std::nullopt;
    }

    ibv::MemoryRegion register_gpu_pointer_region(
        ibv::ProtectionDomain& protection_domain,
        void* address,
        std::size_t bytes,
        std::uint64_t iova,
        int access)
    {
        try {
            return protection_domain.register_memory_region_iova(
                address, bytes, iova, access);
        } catch (const ibv::Error& error) {
            throw ibv::Error(
                "Failed to register a GPU pointer. This path needs "
                "nvidia-peermem; DMA-BUF registration does not. "
                + std::string(error.what()));
        }
    }

    struct VerbsResources {
        VerbsResources(
            const CpuRoceTransceiver::Config& config,
            detail::RxRing* rx_ring,
            detail::TxRing* tx_ring,
            std::uint32_t receive_depth,
            bool register_local_rx_alias,
            bool host_metadata,
            std::uint32_t max_send_sge,
            void* extra_host,
            std::uint32_t extra_host_bytes,
            bool allow_inline)
            : rx_ring(rx_ring)
            , tx_ring(tx_ring)
        {
            context = ibv::Context(config.ibv_name);
            port = context.query_port(config.ibv_port);
            local_gid = detail::query_local_roce_v2_ipv4_gid(
                context, config.ibv_port, config.local_ip);
            protection_domain = context.allocate_protection_domain();
            // RX needs one CQE per posted receive. Tx/Duplex/Unified use TX
            // ring depth until completions are coalesced. Use depth one as
            // the portable minimum for an inactive direction. Forward has no
            // dedicated TX ring but can have one outstanding send.
            const std::uint32_t send_depth
                = tx_ring != nullptr ? tx_ring->slot_count() : 1;
            send_completion_queue
                = context.create_completion_queue(send_depth);
            receive_completion_queue
                = context.create_completion_queue(receive_depth);

            // Host RX storage has two registrations with distinct contracts:
            //
            // - peer_visible_rx_memory is remotely writable and has IOVA
            //   base zero. Its rkey is exported to HSB, whose 32-bit page
            //   address cannot represent a host virtual address.
            // - local_rx_memory uses the host virtual address. Its lkey
            //   is used for receive and local-send SGEs because providers may
            //   copy small incoming Sends from a CQE in software and therefore
            //   dereference the SGE address as a process pointer.
            //
            // Both host registrations are required when a receive WQE carries
            // an SGE. Continuing with only the zero-based registration could
            // turn a small Send into a write near address zero. Handled
            // receive posts no SGE, so it skips the alias. DMA-BUF and a
            // GPU pointer have one zero-based registration and no alias:
            // their receive WQEs carry no SGE. A GPU-pointer TX MR is
            // separate and uses the pointer as its IOVA.
            if (rx_ring != nullptr) {
                const auto rx_bytes = checked_ring_bytes(
                    rx_ring->stride(), rx_ring->slot_count());
                if (const auto& dmabuf = rx_ring->dmabuf()) {
                    peer_visible_rx_memory
                        = protection_domain.register_dmabuf_memory_region(
                            dmabuf->offset, rx_bytes, 0,
                            dmabuf->fd,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
                } else if (rx_ring->gpu_pointer()) {
                    peer_visible_rx_memory = register_gpu_pointer_region(
                        protection_domain, rx_ring->data(), rx_bytes, 0,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
                } else {
                    peer_visible_rx_memory
                        = protection_domain.register_memory_region_iova(
                            rx_ring->data(), rx_bytes, 0,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
                    if (register_local_rx_alias) {
                        try {
                            local_rx_memory
                                = protection_domain.register_memory_region(
                                    rx_ring->data(), rx_bytes,
                                    IBV_ACCESS_LOCAL_WRITE);
                        } catch (const ibv::Error& error) {
                            throw ibv::Error(
                                std::string(
                                    "Failed to register required virtual-address "
                                    "RX memory alias: ")
                                + error.what());
                        }
                    }
                }
            }
            // Host TX flags publish host virtual addresses and use that same
            // IOVA. DMA-BUF flags publish the opaque application-visible
            // address, while SGEs use the fixed zero-based IOVA. A GPU
            // pointer is both the IOVA and the SGE base.
            if (tx_ring != nullptr) {
                const auto tx_bytes = checked_ring_bytes(
                    tx_ring->stride(), tx_ring->slot_count());
                if (const auto& dmabuf = tx_ring->dmabuf()) {
                    try {
                        send_memory
                            = protection_domain.register_dmabuf_memory_region(
                                dmabuf->offset, tx_bytes, 0,
                                dmabuf->fd, IBV_ACCESS_LOCAL_WRITE);
                    } catch (const ibv::Error& error) {
                        throw RegistrationError(error.what());
                    }
                } else if (tx_ring->gpu_pointer()) {
                    send_memory = register_gpu_pointer_region(
                        protection_domain, tx_ring->data(), tx_bytes,
                        tx_ring->slot_sge_address(0),
                        IBV_ACCESS_LOCAL_WRITE);
                } else {
                    send_memory = protection_domain.register_memory_region(
                        tx_ring->data(), tx_bytes, IBV_ACCESS_LOCAL_WRITE);
                }
            }
            if (extra_host_bytes != 0) {
                if (extra_host == nullptr || tx_ring == nullptr) {
                    throw std::invalid_argument(
                        "CpuRoceTransceiver caller transmit extra host bytes "
                        "require a caller-owned buffer and a TX ring");
                }
                extra_host_base = extra_host;
                this->extra_host_bytes = extra_host_bytes;
                extra_memory = protection_domain.register_memory_region(
                    extra_host, extra_host_bytes, IBV_ACCESS_LOCAL_WRITE);
            }

            auto capacity = ibv::QueuePair::Capacity()
                                .max_send_write_read(send_depth)
                                .max_receive_write_read(receive_depth)
                                .max_send_scatter_gather_entry(max_send_sge)
                                .max_receive_scatter_gather_entry(1);
            // Inline is all-or-nothing. The policy threshold decides whether
            // to try; the QP request remains the full transmitted payload.
            const RingConfig* outbound = active_outbound_ring(config);
            const std::uint32_t payload_size = outbound != nullptr
                ? static_cast<std::uint32_t>(
                    outbound->frame_layout.payload_size)
                : 0;
            const bool attempt_inline
                = allow_inline
                && outbound != nullptr
                && !std::holds_alternative<DmaBufMemory>(
                    outbound->payload_memory)
                && !std::holds_alternative<GpuPointerMemory>(
                    outbound->payload_memory)
                && payload_size <= max_inline_payload(config.tx_operation);
            if (attempt_inline) {
                capacity.inline_data_size(payload_size);
            }
            auto init_attr = ibv::QueuePair::InitAttr()
                                 .query_pair_type(IBV_QPT_UC)
                                 .send_completion_queue(send_completion_queue.ptr_)
                                 .receive_completion_queue(
                                     receive_completion_queue.ptr_)
                                 .capacity(capacity);
            // HSB speaks UC. Lost UC messages are not recovered, so the Stall
            // policy is unbounded with respect to the peer.
            if (attempt_inline) {
                // A rejected inline request usually means this provider
                // cannot carry the payload in the WQE: retry without inline.
                // On success, ibv_create_qp grants at least the requested
                // max_inline_data. If the non-inline create also fails,
                // create_queue_pair throws with the memlock annotation.
                auto inline_probe
                    = protection_domain.try_create_queue_pair(init_attr);
                if (inline_probe) {
                    queue_pair = std::move(inline_probe);
                    tx_inline = true;
                } else {
                    capacity.inline_data_size(0);
                    init_attr.capacity(capacity);
                    queue_pair = protection_domain.create_queue_pair(init_attr);
                }
            } else {
                queue_pair = protection_domain.create_queue_pair(init_attr);
            }
            if (const int error = queue_pair.reset_to_init(config.ibv_port);
                error != 0) {
                throw ibv::Error(detail::concatenate(
                    "Failed to move CpuRoceTransceiver QP to INIT, "
                    "ibv_modify_qp returned ",
                    error, " (",
                    std::generic_category().message(error), ")"));
            }
            configure_device_metadata_umr(config, host_metadata);
        }

        void configure_device_metadata_umr(
            const CpuRoceTransceiver::Config& config, bool host_metadata);

        ibv::ScatterGatherElement make_local_rx_sge(
            std::uint32_t slot, std::uint32_t length) const
        {
            assert(rx_ring != nullptr);
            const auto* memory = local_rx_memory
                ? local_rx_memory.ptr_
                : peer_visible_rx_memory.ptr_;
            assert(memory != nullptr);
            return detail::make_local_rx_sge(
                *rx_ring, slot, length, memory->lkey);
        }

        ibv::ScatterGatherElement make_full_slot_rx_sge(std::uint32_t slot) const
        {
            assert(rx_ring != nullptr);
            return make_local_rx_sge(
                slot, static_cast<std::uint32_t>(rx_ring->stride()));
        }

        ibv::ScatterGatherElement make_tx_sge(
            std::uint32_t slot, std::uint32_t length) const
        {
            assert(tx_ring != nullptr && send_memory.ptr_ != nullptr);
            return detail::make_local_tx_sge(
                *tx_ring, slot, length, send_memory.ptr_->lkey);
        }

        ~VerbsResources()
        {
            quiesce();
        }

        VerbsResources(const VerbsResources&) = delete;
        VerbsResources& operator=(const VerbsResources&) = delete;

        void quiesce() noexcept
        {
            if (queue_pair) {
                queue_pair.move_to_error();
                receive_completion_queue.discard_pending();
                send_completion_queue.discard_pending();
                queue_pair = {};
            }

            // Drop the extra-host registration before the payload MR. The
            // bytes belong to the caller and outlive this object.
            extra_memory = {};
            send_memory = {};
            local_rx_memory = {};
            // The indirect key's payload entries name the peer-visible region.
            host_trailer_key.reset();
            peer_visible_rx_memory = {};
            extra_host_base = nullptr;
            extra_host_bytes = 0;
            send_completion_queue = {};
            receive_completion_queue = {};
            protection_domain = {};
            context = {};
            rx_ring = nullptr;
            tx_ring = nullptr;
            connected = false;
            connect_attempted = false;
        }

        // Declaration order is ownership order; quiesce() explicitly releases
        // in the reverse dependency order before ring storage is freed.
        ibv::Context context;
        ::ibv_port_attr port {};
        detail::LocalRoceV2Ipv4Gid local_gid;
        ibv::ProtectionDomain protection_domain;
        ibv::CompletionQueue receive_completion_queue;
        ibv::CompletionQueue send_completion_queue;
        ibv::MemoryRegion peer_visible_rx_memory;
        ibv::MemoryRegion local_rx_memory;
        // Empty unless handled receive requested host trailers and the NIC
        // accepted the indirect key. Destroyed before the payload region.
        std::unique_ptr<detail::HostTrailerKey> host_trailer_key;
        ibv::MemoryRegion send_memory;
        // Destroyed before send_memory. Empty unless caller transmit
        // registered a second scatter/gather region. The bytes are not owned.
        ibv::MemoryRegion extra_memory;
        ibv::QueuePair queue_pair;
        // Non-owning. Host Recv SGEs use the local virtual-address alias;
        // DMA-BUF receive WQEs are credit-only and carry no SGE.
        // Impl keeps the ring alive until after verbs are destroyed.
        detail::RxRing* rx_ring = nullptr;
        // Non-owning. Send SGEs use the ring's registered IOVA and
        // send_memory's lkey. try_claim_tx validates the slot before each post
        // in the Tx/Duplex path; Unified uses decoded.payload_slot and does
        // not claim.
        detail::TxRing* tx_ring = nullptr;
        bool tx_inline = false;
        // Non-owning. The caller keeps these bytes alive past verbs.reset().
        void* extra_host_base = nullptr;
        std::uint32_t extra_host_bytes = 0;
        bool connected = false;
        bool connect_attempted = false;
        // Payload bytes programmed at RTR. Published only after RTS.
        std::uint32_t path_mtu_payload = 0;
    };

    void VerbsResources::configure_device_metadata_umr(
        const CpuRoceTransceiver::Config& config, bool host_metadata)
    {
        const auto& layout = config.rx.frame_layout;
        // The trailer must end the slot, so the two KLM entries per slot
        // rebuild the stride the sender writes. validate_config() bounds the
        // port, the stride and the slot count.
        if (!host_metadata || rx_ring == nullptr
            || (!rx_ring->dmabuf() && !rx_ring->gpu_pointer())
            || layout.metadata_size == 0
            || layout.metadata_offset + layout.metadata_size != rx_ring->stride()) {
            return;
        }
        host_trailer_key = detail::try_host_trailer_key(context,
            protection_domain, static_cast<std::uint8_t>(config.ibv_port),
            port.active_mtu, local_gid, peer_visible_rx_memory, rx_ring->stride(),
            static_cast<std::uint32_t>(layout.metadata_size),
            rx_ring->slot_count());
    }

} // namespace

struct CpuRoceTransceiver::Impl {
    explicit Impl(CpuRoceTransceiver::Config configuration)
        : config(std::move(configuration))
    {
    }

    CpuRoceTransceiver::Config config;
    std::unique_ptr<detail::RxRing> rx_ring;
    std::unique_ptr<detail::TxRing> tx_ring;
    std::unique_ptr<VerbsResources> verbs;
    std::mutex teardown;
    std::condition_variable monitor_state_changed;
    bool monitor_started = false;
    bool monitor_active = false;
    // Caller of blocking_monitor() while monitor_active; lets close() reject a
    // self-call instead of waiting for a thread that is waiting on it. Duplex
    // workers are not recorded: their bodies have no user code today, so they
    // cannot call close(). A worker close() would deadlock — the coordinator
    // is blocked joining that worker, and this id would not match. If Duplex
    // ever gains a per-slot callback, record worker ids here too. Unified is
    // single-poller: UnifiedDispatchFn runs on this thread, so the guard
    // covers it without extra ids.
    std::thread::id monitor_thread;
    std::atomic<bool> exit_flag { false };
    std::atomic<bool> closed { false };
    std::atomic<bool> terminal_failure { false };
    std::atomic<std::uint64_t> rx_collision_count { 0 };
    UnifiedDispatchFn unified_dispatch = nullptr;
    void* unified_context = nullptr;

    // CallerDriven queue and per-slot send occupancy. Hook calls must not
    // overlap; successive calls may change threads. The caller orders them,
    // for example by joining the previous thread or holding a mutex. Stop
    // those calls before calling close().
    struct CallerDriven {
        struct Held {
            std::uint32_t payload_slot = 0;
            std::uint32_t recv_wr_id = 0;
        };

        std::vector<Held> held;
        std::vector<std::uint8_t> tx_in_flight;
        unsigned sends_in_flight = 0;
        // Starts at the interval so the first empty rx_poll() drains once.
        unsigned calls_since_async = kCallerDrivenAsyncCheckInterval;
        bool gid_checked = false;
        // One CQ poll buffer, allocated with the queue and kept off the
        // per-call fields above. Not value-initialized on the poll path;
        // ibv_poll_cq writes the entries it returns.
        std::array<::ibv_wc, kRxPollBatch> completions {};
    };

    std::unique_ptr<CallerDriven> caller_driven;
    // Published with release at the end of connect(), cleared on close() and
    // on a fatal CallerDriven error. Hooks load it with acquire and return
    // false while it is clear, so a dispatcher spinning before connect() does
    // not throw and does not race the plain verbs->connected flag.
    std::atomic<bool> caller_driven_ready { false };

    // Set once move_to_error has succeeded, so a second terminalize does not
    // destroy the queue pair and discard flush completions.
    bool queue_pair_errored = false;

    detail::CpuRoceRxCompletionHandler::Fn rx_completion_handler = nullptr;
    void* rx_completion_context = nullptr;
    std::uint32_t receive_credit_count = 0;
    bool host_metadata_requested = false;

    // Set by CpuRoceCallerTx::configure() before setup(). Not a mode.
    struct CallerPost {
        bool configured = false;
        void* extra_host = nullptr;
        std::uint32_t extra_host_bytes = 0;
        bool clamp_path_mtu = false;
        // Next send-queue PSN. SQ starts at 0. Advanced only after an
        // accepted caller post. The owner thread is the only writer.
        std::uint32_t next_psn = 0;
    };
    CallerPost caller_post;

    void terminalize_queue_pair();
    void note_caller_driven_gid();
    void drain_caller_driven_async();
    void reap_caller_driven_sends(bool one_batch = false);
    bool pull_caller_driven_receives();
    bool claim_caller_driven_slot(std::uint32_t* out_slot, bool& reap_required);
    bool finish_caller_driven_claim(std::uint32_t* out_slot);
};

namespace {

    void post_receive_slot(VerbsResources& verbs, std::uint32_t slot)
    {
        assert(verbs.rx_ring != nullptr);
        // DMA-BUF and GPU-pointer RX are used with RDMA Write-with-Immediate.
        // The receive WQE supplies only notification credit: remote_addr
        // selects the payload destination. A zero-SGE WQE also prevents a
        // provider from trying to satisfy an ordinary Send by CPU-copying
        // into that memory.
        std::optional<ibv::ScatterGatherElement> sge;
        if (verbs.rx_ring->posts_receive_sge()) {
            sge.emplace(verbs.make_full_slot_rx_sge(slot));
        }
        ibv::ReceiveWorkRequest work_request(
            slot, sge ? &*sge : nullptr);
        if (const int error = verbs.queue_pair.post_receive(work_request);
            error != 0) {
            throw ibv::Error(detail::concatenate(
                "Failed to post CpuRoceTransceiver receive WQE for slot ", slot,
                ", ibv_post_recv returned ", error, " (",
                std::generic_category().message(error), ")"));
        }
    }

    void post_receive_credit(VerbsResources& verbs, std::uint32_t credit_id)
    {
        // Handled receive WQEs carry no SGE. Write-with-Immediate places the
        // payload by remote address; the WQE is only the credit that produces
        // the completion. The handler reads the slot from the immediate.
        ibv::ReceiveWorkRequest work_request(credit_id, nullptr);
        if (const int error = verbs.queue_pair.post_receive(work_request);
            error != 0) {
            throw ibv::Error(detail::concatenate(
                "Failed to post CpuRoceTransceiver receive credit ", credit_id,
                ", ibv_post_recv returned ", error, " (",
                std::generic_category().message(error), ")"));
        }
    }

    void fill_handled_receive_credits(
        VerbsResources& verbs, std::uint32_t credit_count)
    {
        for (std::uint32_t credit = 0; credit < credit_count; ++credit) {
            post_receive_credit(verbs, credit);
        }
    }

    void fill_receive_queue(VerbsResources& verbs)
    {
        assert(verbs.rx_ring != nullptr);
        for (std::uint32_t slot = 0; slot < verbs.rx_ring->slot_count();
             ++slot) {
            post_receive_slot(verbs, slot);
        }
    }

    std::string rx_cqe_summary(const ::ibv_wc& wc)
    {
        const char* const status_name = ibv_wc_status_str(wc.status);
        if (wc.status != IBV_WC_SUCCESS) {
            // libibverbs guarantees only wr_id, status, qp_num, and vendor_err
            // for failed completions. Other fields may contain stale data.
            return detail::concatenate(
                "status=", static_cast<int>(wc.status), " (",
                status_name != nullptr ? status_name : "unknown",
                ") vendor_err=", detail::hex(wc.vendor_err),
                " wr_id=", detail::hex(wc.wr_id),
                " qp_num=", detail::hex(wc.qp_num));
        }

        std::string summary = detail::concatenate(
            "status=", static_cast<int>(wc.status), " (",
            status_name != nullptr ? status_name : "unknown",
            ") vendor_err=", detail::hex(wc.vendor_err),
            " opcode=", static_cast<int>(wc.opcode),
            " wc_flags=", detail::hex(static_cast<std::uint32_t>(wc.wc_flags)),
            " byte_len=", wc.byte_len,
            " wr_id=", detail::hex(wc.wr_id),
            " qp_num=", detail::hex(wc.qp_num));
        if ((wc.wc_flags & IBV_WC_WITH_IMM) != 0) {
            summary += detail::concatenate(
                " imm_host=", detail::hex(ntohl(wc.imm_data)));
        }
        return summary;
    }

    [[noreturn]] void throw_rx_cqe_error(
        const detail::RxCqeDecode& decoded, const ::ibv_wc& wc)
    {
        const std::string summary = rx_cqe_summary(wc);
        switch (decoded.status) {
        case detail::RxCqeDecode::Status::FailedStatus:
            // This UC RX CQ path cannot safely continue or restart the
            // consumed batch while keeping RQ credit and slot publication
            // consistent. Not a claim that every RC/UC WR error moves every
            // QP to the same state.
            throw std::runtime_error(
                "CpuRoceTransceiver recv CQE failed: " + summary);
        case detail::RxCqeDecode::Status::UnsupportedOpcode:
            throw std::runtime_error(
                "CpuRoceTransceiver recv CQE has an unsupported opcode: "
                + summary);
        case detail::RxCqeDecode::Status::UnexpectedSendForCreditOnly:
            throw std::runtime_error(
                "CpuRoceTransceiver DMA-BUF or GPU-pointer RX requires "
                "RDMA Write-with-Immediate: "
                + summary);
        case detail::RxCqeDecode::Status::UnexpectedSend:
            throw std::runtime_error(
                "CpuRoceTransceiver handled receive requires "
                "RDMA Write-with-Immediate: "
                + summary);
        case detail::RxCqeDecode::Status::MissingImm:
            throw std::runtime_error(
                "CpuRoceTransceiver Write-with-Imm CQE is missing immediate "
                "data: "
                + summary);
        case detail::RxCqeDecode::Status::InvalidWrId:
            throw std::runtime_error(
                "CpuRoceTransceiver recv CQE wr_id is out of range: "
                + summary);
        case detail::RxCqeDecode::Status::InvalidImmSlot:
            // Recurring protocol / slot_count / ring-geometry mismatch.
            // Continuing would keep rejecting frames while looking alive.
            throw std::runtime_error(
                "CpuRoceTransceiver recv CQE IMM slot is out of range: "
                + summary);
        case detail::RxCqeDecode::Status::Ok:
            break;
        }
        throw std::runtime_error("CpuRoceTransceiver recv CQE is invalid");
    }

    void check_fatal_async_events(
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config);

    bool handle_rx_completion(
        detail::RxRing& rx_ring,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        const ::ibv_wc& completion,
        bool write_with_imm_only,
        const std::atomic<bool>& exit_flag,
        std::atomic<std::uint64_t>& collision_count,
        unsigned& spin)
    {
        const auto decoded = detail::decode_rx_cqe(
            completion, config.rx.slot_count, write_with_imm_only);
        if (decoded.status != detail::RxCqeDecode::Status::Ok) {
            throw_rx_cqe_error(decoded, completion);
        }

        auto try_publish_and_rearm = [&]() {
            switch (rx_ring.publish_rx(decoded.payload_slot)) {
            case detail::RxPublishResult::Published:
                // Publish before restoring RQ credit, matching CUDA-Q's
                // externally visible handoff. Send and Write-with-Imm both
                // consume a receive WQE. Delaying this repost reduces
                // available RQ credit and can bound future notified
                // operations once the remaining credit is exhausted. It
                // cannot undo payload DMA completed before this CQE.
                post_receive_slot(
                    verbs, static_cast<std::uint32_t>(decoded.recv_wr_id));
                return true;
            case detail::RxPublishResult::Collision:
                return false;
            case detail::RxPublishResult::InvalidSlot:
                throw std::runtime_error(
                    "CpuRoceTransceiver RX publication used an invalid slot "
                    + std::to_string(decoded.payload_slot));
            default:
                throw std::logic_error(
                    "CpuRoceTransceiver RX publication returned an invalid result");
            }
        };

        if (try_publish_and_rearm()) {
            return true;
        }

        // Occupied flag: wait for the consumer's release-store, then retry the
        // publication. decode_rx_cqe already range-checked this slot, and
        // publish_rx re-checks occupancy, so observing zero here is a hint to
        // retry rather than a decision to overwrite.
        collision_count.fetch_add(1, std::memory_order_relaxed);
        volatile std::uint64_t* const flag
            = &rx_ring.flags()[decoded.payload_slot];
        for (;;) {
            if (exit_flag.load(std::memory_order_acquire)) {
                return false;
            }
            if ((spin & (kAsyncCheckInterval - 1)) == 0) {
                check_fatal_async_events(verbs, config);
            }
            ++spin;
            if (CpuRoceTransceiver::load_flag_acquire(flag) != 0) {
                cpu_relax();
                continue;
            }
            if (try_publish_and_rearm()) {
                return true;
            }
            cpu_relax();
        }
    }

    std::string tx_cqe_summary(const ::ibv_wc& wc)
    {
        const char* const status_name = ibv_wc_status_str(wc.status);
        if (wc.status != IBV_WC_SUCCESS) {
            return detail::concatenate(
                "status=", static_cast<int>(wc.status), " (",
                status_name != nullptr ? status_name : "unknown",
                ") vendor_err=", detail::hex(wc.vendor_err),
                " wr_id=", detail::hex(wc.wr_id),
                " qp_num=", detail::hex(wc.qp_num));
        }
        return detail::concatenate(
            "status=", static_cast<int>(wc.status), " (",
            status_name != nullptr ? status_name : "unknown",
            ") opcode=", static_cast<int>(wc.opcode),
            " wr_id=", detail::hex(wc.wr_id),
            " qp_num=", detail::hex(wc.qp_num));
    }

    void validate_tx_completion(
        const ::ibv_wc& completion,
        std::uint64_t expected_sequence,
        ibv_wc_opcode expected_opcode)
    {
        const auto decoded
            = detail::decode_tx_cqe(
                completion, expected_sequence, expected_opcode);
        switch (decoded.status) {
        case detail::TxCqeDecode::Status::FailedStatus:
            throw std::runtime_error(
                "CpuRoceTransceiver send CQE failed: "
                + tx_cqe_summary(completion));
        case detail::TxCqeDecode::Status::UnsupportedOpcode:
            throw std::runtime_error(
                "CpuRoceTransceiver send CQE has an unsupported opcode: "
                + tx_cqe_summary(completion));
        case detail::TxCqeDecode::Status::OutOfSequence:
            throw std::runtime_error(
                "CpuRoceTransceiver send CQE is out of sequence: expected "
                + std::to_string(expected_sequence) + ", "
                + tx_cqe_summary(completion));
        case detail::TxCqeDecode::Status::Ok:
            break;
        }
    }

    void handle_tx_completion(
        const ::ibv_wc& completion,
        std::uint64_t& next_completion_sequence,
        ibv_wc_opcode expected_opcode)
    {
        validate_tx_completion(
            completion, next_completion_sequence, expected_opcode);
        ++next_completion_sequence;
    }

    void check_fatal_async_events(
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config)
    {
        const auto terminal_failure = drain_async_events(
            verbs.context, config.ibv_port, verbs.local_gid);
        if (terminal_failure) {
            throw std::runtime_error(
                "CpuRoceTransceiver fatal IB async event: "
                + *terminal_failure);
        }
    }

    void run_rx_loop(
        detail::RxRing& rx_ring,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        const std::atomic<bool>& exit_flag,
        std::atomic<std::uint64_t>& collision_count)
    {
        const bool write_with_imm_only = !rx_ring.accepts_send();
        std::array<::ibv_wc, kRxPollBatch> completions {};
        unsigned spin = 0;
        while (!exit_flag.load(std::memory_order_acquire)) {
            if ((spin & (kAsyncCheckInterval - 1)) == 0) {
                check_fatal_async_events(verbs, config);
            }
            ++spin;
            const int n = verbs.receive_completion_queue.poll(
                completions.data(), kRxPollBatch);
            if (n == 0) {
                cpu_relax();
                continue;
            }
            for (int i = 0; i < n; ++i) {
                if (!handle_rx_completion(
                        rx_ring, verbs, config,
                        completions[static_cast<std::size_t>(i)],
                        write_with_imm_only, exit_flag,
                        collision_count, spin)) {
                    return;
                }
            }
        }
    }

    void run_handled_rx_loop(
        const std::atomic<bool>& exit_flag,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        detail::CpuRoceRxCompletionHandler::Fn handler,
        void* handler_context,
        std::uint32_t credit_count)
    {
        assert(verbs.rx_ring != nullptr);
        // Camera-rate handled RX normally has one CQE. Asking mlx5 for one
        // avoids probing the next CQE owner byte on the completion path.
        std::array<::ibv_wc, 1> completions {};
        // Every pass counts, including passes that return a completion.
        // A selected-GID change can arrive while frames keep coming.
        unsigned spins = 0;
        for (;;) {
            if (exit_flag.load(std::memory_order_acquire)) {
                return;
            }
            if ((spins & 0xFFFFu) == 0u) {
                check_fatal_async_events(verbs, config);
            }
            ++spins;
            const int count = verbs.receive_completion_queue.poll(
                completions.data(), 1);
            if (count == 0) {
                cpu_relax();
                continue;
            }
            for (int i = 0; i < count; ++i) {
                const auto& completion
                    = completions[static_cast<std::size_t>(i)];
                const auto decoded = detail::decode_handled_rx_cqe(
                    completion, credit_count);
                if (decoded.status != detail::RxCqeDecode::Status::Ok) {
                    throw_rx_cqe_error(decoded, completion);
                }
                const auto credit_id
                    = static_cast<std::uint32_t>(decoded.recv_wr_id);
                handler(
                    handler_context,
                    detail::make_rx_completion(completion, credit_id));
                // A return reposts this credit. A throw skips the repost and
                // terminalizes the monitor. The immediate's slot is the
                // handler's to accept or drop.
                post_receive_credit(verbs, credit_id);
            }
        }
    }

    template <CpuRoceTransceiver::TxOperation Operation>
    void run_forward_loop(
        detail::RxRing& rx_ring,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        const std::atomic<bool>& exit_flag)
    {
        constexpr ibv_wc_opcode expected_completion_opcode
            = Operation == CpuRoceTransceiver::TxOperation::Send
            ? IBV_WC_SEND
            : IBV_WC_RDMA_WRITE;
        constexpr const char* tx_operation_name
            = Operation == CpuRoceTransceiver::TxOperation::Send
            ? "SEND"
            : "WRITE_WITH_IMM";

        // Full-frame echo, matching CUDA-Q forward_loop and the GPU forward
        // kernel: always TX rx.frame_layout.payload_size from RX storage.
        // Recv WQEs cover stride; CQE byte_len is not used. A short inbound
        // SEND leaves stale slot tail in the echoed payload. Write-with-Imm
        // CQE byte_len is not the frame size (HSB may terminate with a small
        // write). Do not fail-stop on a short SEND; that would be stricter
        // than CUDA-Q Forward.
        const auto payload_size = static_cast<std::uint32_t>(
            config.rx.frame_layout.payload_size);
        auto sge = verbs.make_local_rx_sge(0, payload_size);
        // One work request for the loop. Each post patches wr_id and, for
        // WRITE_WITH_IMM, the remote address and immediate.
        auto work_request = [&]() {
            if constexpr (Operation == CpuRoceTransceiver::TxOperation::Send) {
                auto request = ibv::SendWorkRequest(0, &sge);
                request.inline_data(verbs.tx_inline);
                return request;
            } else {
                auto request = ibv::WriteWithImmWorkRequest(
                    0, &sge, config.peer.rx_base, config.peer.rx_rkey, htonl(0));
                request.inline_data(verbs.tx_inline);
                return request;
            }
        }();

        if (!detail::selected_gid_matches(
                verbs.local_gid,
                verbs.context.query_gid_ex(
                    config.ibv_port, verbs.local_gid.index))) {
            throw std::runtime_error(
                "CpuRoceTransceiver selected GID changed before Forward start");
        }

        const bool write_with_imm_only = !rx_ring.accepts_send();
        std::array<::ibv_wc, kRxPollBatch> receive_completions {};
        std::uint64_t send_sequence = 0;
        unsigned spin = 0;
        while (!exit_flag.load(std::memory_order_acquire)) {
            if ((spin & (kAsyncCheckInterval - 1)) == 0) {
                check_fatal_async_events(verbs, config);
            }
            ++spin;
            const int count = verbs.receive_completion_queue.poll(
                receive_completions.data(), kRxPollBatch);
            if (count == 0) {
                cpu_relax();
                continue;
            }

            for (int i = 0; i < count; ++i) {
                const auto& completion
                    = receive_completions[static_cast<std::size_t>(i)];
                const auto decoded
                    = detail::decode_rx_cqe(
                        completion, config.rx.slot_count, write_with_imm_only);
                if (decoded.status != detail::RxCqeDecode::Status::Ok) {
                    throw_rx_cqe_error(decoded, completion);
                }

                const auto slot = decoded.payload_slot;
                // SGE uses the ring's registered IOVA: host VA for host
                // storage, zero-based IOVA for DMA-BUF.
                sge.addr = rx_ring.slot_sge_address(slot);
                work_request.wr_id = send_sequence;
                if constexpr (Operation
                    == CpuRoceTransceiver::TxOperation::WriteWithImm) {
                    work_request.wr.rdma.remote_addr = config.peer.rx_base
                        + static_cast<std::uint64_t>(slot) * config.peer.rx_stride;
                    work_request.imm_data = htonl(slot);
                }
                if (const int error = verbs.queue_pair.post_send(work_request);
                    error != 0) {
                    throw ibv::Error(detail::concatenate(
                        "Failed to post CpuRoceTransceiver Forward ",
                        tx_operation_name, " WQE for sequence ", send_sequence,
                        ", slot ", slot, ", ibv_post_send returned ", error,
                        " (", std::generic_category().message(error), ")"));
                }

                ::ibv_wc send_completion {};
                while (!exit_flag.load(std::memory_order_acquire)) {
                    if ((spin & (kAsyncCheckInterval - 1)) == 0) {
                        check_fatal_async_events(verbs, config);
                    }
                    ++spin;
                    if (verbs.send_completion_queue.poll(&send_completion, 1)
                        == 0) {
                        cpu_relax();
                        continue;
                    }
                    validate_tx_completion(
                        send_completion, send_sequence,
                        expected_completion_opcode);
                    break;
                }
                if (exit_flag.load(std::memory_order_acquire)) {
                    return;
                }

                ++send_sequence;
                // The same RX storage is the send source. Re-arm only after
                // the send CQE releases it back to the NIC.
                post_receive_slot(
                    verbs, static_cast<std::uint32_t>(decoded.recv_wr_id));
            }
        }
    }

    template <CpuRoceTransceiver::TxOperation Operation>
    void run_tx_loop(
        detail::TxRing& tx_ring,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        const std::atomic<bool>& exit_flag,
        bool owns_async_drain)
    {
        static_assert(
            Operation == CpuRoceTransceiver::TxOperation::Send
                || Operation
                    == CpuRoceTransceiver::TxOperation::WriteWithImm,
            "Unsupported TX operation");
        constexpr ibv_wc_opcode expected_completion_opcode
            = Operation == CpuRoceTransceiver::TxOperation::Send
            ? IBV_WC_SEND
            : IBV_WC_RDMA_WRITE;
        constexpr const char* tx_operation_name
            = Operation == CpuRoceTransceiver::TxOperation::Send
            ? "SEND"
            : "WRITE_WITH_IMM";

        std::array<::ibv_wc, kTxPollBatch> completions {};
        std::uint64_t next_post_sequence = 0;
        std::uint64_t next_completion_sequence = 0;
        unsigned spin = 0;
        const auto payload_size = static_cast<std::uint32_t>(
            config.tx.frame_layout.payload_size);
        auto sge = verbs.make_tx_sge(0, payload_size);
        // One work request for the loop. Each post patches wr_id and, for
        // WRITE_WITH_IMM, the remote address and immediate.
        auto work_request = [&]() {
            if constexpr (Operation == CpuRoceTransceiver::TxOperation::Send) {
                auto request = ibv::SendWorkRequest(0, &sge);
                request.inline_data(verbs.tx_inline);
                return request;
            } else {
                auto request = ibv::WriteWithImmWorkRequest(
                    0, &sge, config.peer.rx_base, config.peer.rx_rkey, htonl(0));
                request.inline_data(verbs.tx_inline);
                return request;
            }
        }();

        if (!detail::selected_gid_matches(
                verbs.local_gid,
                verbs.context.query_gid_ex(
                    config.ibv_port, verbs.local_gid.index))) {
            throw std::runtime_error(
                "CpuRoceTransceiver selected GID changed before TX start");
        }

        while (!exit_flag.load(std::memory_order_acquire)) {
            if (owns_async_drain
                && (spin & (kAsyncCheckInterval - 1)) == 0) {
                check_fatal_async_events(verbs, config);
            }
            ++spin;

            bool progressed = false;
            if (next_completion_sequence != next_post_sequence) {
                const int n = verbs.send_completion_queue.poll(
                    completions.data(), kTxPollBatch);
                progressed = n != 0;
                for (int i = 0; i < n; ++i) {
                    handle_tx_completion(
                        completions[static_cast<std::size_t>(i)],
                        next_completion_sequence,
                        expected_completion_opcode);
                }
            }

            // Cancellation is checked after each post, not before each claim:
            // an unpublished slot leaves this loop immediately and the outer
            // loop reloads exit_flag. Either way at most one post can race a
            // concurrent shutdown.
            for (;;) {
                // Bound the unpolled inner burst to the ring depth before
                // consuming the next doorbell. The requested SQ depth matches
                // that window; the provider may grant a larger SQ. Cyclic
                // posting then proves a slot cannot have two outstanding WQEs.
                if (!detail::cyclic_tx_window_has_capacity(
                        next_post_sequence, next_completion_sequence,
                        tx_ring.slot_count())) {
                    break;
                }
                const std::uint32_t slot
                    = tx_ring.slot_index(next_post_sequence);
                if (tx_ring.try_claim_tx(slot)
                    == detail::TxClaimResult::NotReady) {
                    break;
                }

                sge.addr = tx_ring.slot_sge_address(slot);
                // Every TX work request stays signaled. Its CQE advances the
                // cyclic post/completion window; the public doorbell was
                // consumed at claim and is never a completion indication.
                // WRITE_WITH_IMM names this slot in the immediate. The slot
                // fits the 12-bit IMM page field because validate_ring_config
                // caps tx.slot_count at 4096.
                work_request.wr_id = next_post_sequence;
                if constexpr (Operation
                    == CpuRoceTransceiver::TxOperation::WriteWithImm) {
                    work_request.wr.rdma.remote_addr = config.peer.rx_base
                        + static_cast<std::uint64_t>(slot) * config.peer.rx_stride;
                    work_request.imm_data = htonl(slot);
                }
                if (const int error = verbs.queue_pair.post_send(work_request);
                    error != 0) {
                    // The occupancy window makes expected SQ-full backpressure
                    // unreachable. Do not retry an unclassified post failure
                    // automatically. The consumed publication is not restored;
                    // terminalize this one-shot instance.
                    throw ibv::Error(detail::concatenate(
                        "Failed to post CpuRoceTransceiver ", tx_operation_name,
                        " WQE for sequence ", next_post_sequence, ", slot ", slot,
                        ", ibv_post_send returned ", error, " (",
                        std::generic_category().message(error), ")"));
                }
                ++next_post_sequence;
                progressed = true;

                if (exit_flag.load(std::memory_order_acquire)) {
                    return;
                }
            }

            if (!progressed) {
                cpu_relax();
            }
        }
    }

    template <CpuRoceTransceiver::TxOperation Operation>
    void run_unified_loop(
        detail::RxRing& rx_ring,
        detail::TxRing& tx_ring,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        CpuRoceTransceiver::UnifiedDispatchFn dispatch,
        void* dispatch_context,
        const std::atomic<bool>& exit_flag)
    {
        constexpr ibv_wc_opcode expected_completion_opcode
            = Operation == CpuRoceTransceiver::TxOperation::Send
            ? IBV_WC_SEND
            : IBV_WC_RDMA_WRITE;
        constexpr const char* tx_operation_name
            = Operation == CpuRoceTransceiver::TxOperation::Send
            ? "SEND"
            : "WRITE_WITH_IMM";

        if (!detail::selected_gid_matches(
                verbs.local_gid,
                verbs.context.query_gid_ex(
                    config.ibv_port, verbs.local_gid.index))) {
            throw std::runtime_error(
                "CpuRoceTransceiver selected GID changed before Unified start");
        }

        const auto rx_payload_size = config.rx.frame_layout.payload_size;
        const auto tx_capacity = config.tx.frame_layout.payload_size;
        auto sge = verbs.make_tx_sge(0, 0);
        // One work request for the loop. Each post patches wr_id, the SGE
        // length, and, for WRITE_WITH_IMM, the remote address and immediate.
        auto work_request = [&]() {
            if constexpr (Operation == CpuRoceTransceiver::TxOperation::Send) {
                auto request = ibv::SendWorkRequest(0, &sge);
                request.inline_data(verbs.tx_inline);
                return request;
            } else {
                auto request = ibv::WriteWithImmWorkRequest(
                    0, &sge, config.peer.rx_base, config.peer.rx_rkey, htonl(0));
                request.inline_data(verbs.tx_inline);
                return request;
            }
        }();

        // At most one signaled send per TX slot. wr_id is that slot; SQ
        // completions stay in post order, so waiting for slot N drains older
        // CQEs first. Do not reuse Duplex's cyclic send sequence: Drop skips
        // the SQ, so post order is not slot order.
        std::vector<std::uint8_t> tx_in_flight(config.tx.slot_count, 0);
        unsigned spin = 0;
        const bool write_with_imm_only = !rx_ring.accepts_send();

        const auto poll_one_tx_completion = [&]() {
            ::ibv_wc completion {};
            if (verbs.send_completion_queue.poll(&completion, 1) == 0) {
                return false;
            }
            if (completion.status != IBV_WC_SUCCESS) {
                throw std::runtime_error(
                    "CpuRoceTransceiver Unified send CQE failed: "
                    + tx_cqe_summary(completion));
            }
            if (completion.opcode != expected_completion_opcode) {
                throw std::runtime_error(
                    "CpuRoceTransceiver Unified send CQE has an unsupported "
                    "opcode: "
                    + tx_cqe_summary(completion));
            }
            if (completion.wr_id >= config.tx.slot_count
                || tx_in_flight[static_cast<std::uint32_t>(completion.wr_id)]
                    == 0) {
                throw std::runtime_error(
                    "CpuRoceTransceiver Unified TX completion does not "
                    "identify an in-flight slot: "
                    + tx_cqe_summary(completion));
            }
            tx_in_flight[static_cast<std::uint32_t>(completion.wr_id)] = 0;
            return true;
        };

        std::array<::ibv_wc, kRxPollBatch> receive_completions {};
        while (!exit_flag.load(std::memory_order_acquire)) {
            if ((spin & (kAsyncCheckInterval - 1)) == 0) {
                check_fatal_async_events(verbs, config);
            }
            ++spin;
            const int count = verbs.receive_completion_queue.poll(
                receive_completions.data(), kRxPollBatch);
            if (count == 0) {
                cpu_relax();
                continue;
            }

            for (int i = 0; i < count; ++i) {
                const auto& completion
                    = receive_completions[static_cast<std::size_t>(i)];
                const auto decoded
                    = detail::decode_rx_cqe(
                        completion, config.rx.slot_count, write_with_imm_only);
                if (decoded.status != detail::RxCqeDecode::Status::Ok) {
                    throw_rx_cqe_error(decoded, completion);
                }
                const auto slot = decoded.payload_slot;

                // The callback may overwrite TX slot N only after the NIC has
                // released the previous response from that same slot. Polling
                // older completions is necessary because the SQ CQ is ordered.
                while (tx_in_flight[slot] != 0) {
                    if (exit_flag.load(std::memory_order_acquire)) {
                        return;
                    }
                    if ((spin & (kAsyncCheckInterval - 1)) == 0) {
                        check_fatal_async_events(verbs, config);
                    }
                    ++spin;
                    if (!poll_one_tx_completion()) {
                        cpu_relax();
                    }
                }
                if (exit_flag.load(std::memory_order_acquire)) {
                    return;
                }

                const auto result = dispatch(
                    dispatch_context,
                    rx_ring.data()
                        + static_cast<std::size_t>(slot) * config.rx.stride,
                    rx_payload_size,
                    tx_ring.data()
                        + static_cast<std::size_t>(slot) * config.tx.stride,
                    tx_capacity);
                if (exit_flag.load(std::memory_order_acquire)) {
                    return;
                }

                bool send_response = false;
                switch (result.disposition) {
                case CpuRoceTransceiver::UnifiedDispatchDisposition::Send:
                    if (result.bytes > tx_capacity) {
                        throw std::runtime_error(
                            "CpuRoceTransceiver Unified callback response "
                            "exceeds TX capacity (bytes="
                            + std::to_string(result.bytes) + ", capacity="
                            + std::to_string(tx_capacity) + ")");
                    }
                    // Wire-identical to Drop. Nonzero Drop is rejected so
                    // the two dispositions stay distinguishable.
                    send_response = result.bytes != 0;
                    break;
                case CpuRoceTransceiver::UnifiedDispatchDisposition::Drop:
                    if (result.bytes != 0) {
                        throw std::runtime_error(
                            "CpuRoceTransceiver Unified Drop result must have "
                            "bytes == 0");
                    }
                    break;
                case CpuRoceTransceiver::UnifiedDispatchDisposition::Fatal:
                    throw std::runtime_error(
                        "CpuRoceTransceiver Unified callback reported Fatal");
                default:
                    throw std::runtime_error(
                        "CpuRoceTransceiver Unified callback returned an "
                        "invalid disposition");
                }

                if (send_response) {
                    // Inline was negotiated for tx_capacity at QP create;
                    // this SGE may be shorter.
                    sge.addr = tx_ring.slot_sge_address(slot);
                    sge.length = static_cast<std::uint32_t>(result.bytes);
                    work_request.wr_id = slot;
                    if constexpr (Operation
                        == CpuRoceTransceiver::TxOperation::WriteWithImm) {
                        work_request.wr.rdma.remote_addr = config.peer.rx_base
                            + static_cast<std::uint64_t>(slot)
                                * config.peer.rx_stride;
                        work_request.imm_data = htonl(slot);
                    }
                    if (const int error = verbs.queue_pair.post_send(work_request);
                        error != 0) {
                        throw ibv::Error(detail::concatenate(
                            "Failed to post CpuRoceTransceiver Unified ",
                            tx_operation_name, " WQE for slot ", slot,
                            ", ibv_post_send returned ", error, " (",
                            std::generic_category().message(error), ")"));
                    }
                    tx_in_flight[slot] = 1;
                }

                // Drop and zero-byte Send skip our Send WQE, matching CUDA-Q.
                // They do not restore a peer RQ WQE that was waiting for a
                // response. Re-arm restores the local recv consumed by this
                // CQE.
                post_receive_slot(
                    verbs, static_cast<std::uint32_t>(decoded.recv_wr_id));
            }
        }
    }

    // The TX verb is a template argument so each loop compiles with a fixed
    // opcode, WQE type, and IMM handling. These two wrappers resolve it once
    // per monitor entry; do not move the choice inside a poll loop.
    void run_configured_tx(
        detail::TxRing& tx_ring,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        const std::atomic<bool>& exit_flag,
        bool owns_async_drain)
    {
        if (config.tx_operation
            == CpuRoceTransceiver::TxOperation::WriteWithImm) {
            run_tx_loop<CpuRoceTransceiver::TxOperation::WriteWithImm>(
                tx_ring, verbs, config, exit_flag, owns_async_drain);
        } else {
            run_tx_loop<CpuRoceTransceiver::TxOperation::Send>(
                tx_ring, verbs, config, exit_flag, owns_async_drain);
        }
    }

    void run_configured_forward(
        detail::RxRing& rx_ring,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        const std::atomic<bool>& exit_flag)
    {
        if (config.tx_operation
            == CpuRoceTransceiver::TxOperation::WriteWithImm) {
            run_forward_loop<CpuRoceTransceiver::TxOperation::WriteWithImm>(
                rx_ring, verbs, config, exit_flag);
        } else {
            run_forward_loop<CpuRoceTransceiver::TxOperation::Send>(
                rx_ring, verbs, config, exit_flag);
        }
    }

    void run_configured_unified(
        detail::RxRing& rx_ring,
        detail::TxRing& tx_ring,
        VerbsResources& verbs,
        const CpuRoceTransceiver::Config& config,
        CpuRoceTransceiver::UnifiedDispatchFn dispatch,
        void* dispatch_context,
        const std::atomic<bool>& exit_flag)
    {
        if (config.tx_operation
            == CpuRoceTransceiver::TxOperation::WriteWithImm) {
            run_unified_loop<CpuRoceTransceiver::TxOperation::WriteWithImm>(
                rx_ring, tx_ring, verbs, config, dispatch, dispatch_context,
                exit_flag);
        } else {
            run_unified_loop<CpuRoceTransceiver::TxOperation::Send>(
                rx_ring, tx_ring, verbs, config, dispatch, dispatch_context,
                exit_flag);
        }
    }

} // namespace

void CpuRoceTransceiver::Impl::terminalize_queue_pair()
{
    exit_flag.store(true, std::memory_order_release);
    caller_driven_ready.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(teardown);
    if (verbs && verbs->queue_pair && !queue_pair_errored) {
        if (!verbs->queue_pair.move_to_error()) {
            // The error transition itself failed, so destruction is the
            // retirement proof. A later successful ERR state is not claimed.
            verbs.reset();
        } else {
            verbs->connected = false;
            queue_pair_errored = true;
        }
    }
    terminal_failure.store(true, std::memory_order_release);
}

void CpuRoceTransceiver::Impl::note_caller_driven_gid()
{
    if (caller_driven->gid_checked) {
        return;
    }
    if (!detail::selected_gid_matches(
            verbs->local_gid,
            verbs->context.query_gid_ex(
                config.ibv_port, verbs->local_gid.index))) {
        throw std::runtime_error(
            "CpuRoceTransceiver selected GID changed before CallerDriven "
            "start");
    }
    caller_driven->gid_checked = true;
}

// Reads the async fd and restarts the rx_poll() counter. A selected-GID
// change is terminal. tx_publish() does not call this.
void CpuRoceTransceiver::Impl::drain_caller_driven_async()
{
    caller_driven->calls_since_async = 0;
    check_fatal_async_events(*verbs, config);
}

// one_batch stops after a single poll. The empty rx_poll() path uses that so
// a spinning dispatcher retires finished sends between requests. Slot reuse
// and tx_publish() pass false and drain the queue.
void CpuRoceTransceiver::Impl::reap_caller_driven_sends(bool one_batch)
{
    if (caller_driven->sends_in_flight == 0) {
        return;
    }
    const auto expected_opcode
        = config.tx_operation == CpuRoceTransceiver::TxOperation::Send
        ? IBV_WC_SEND
        : IBV_WC_RDMA_WRITE;
    for (;;) {
        const int count = verbs->send_completion_queue.poll(
            caller_driven->completions.data(), kRxPollBatch);
        if (count <= 0) {
            return;
        }
        for (int i = 0; i < count; ++i) {
            const auto& completion
                = caller_driven->completions[static_cast<std::size_t>(i)];
            if (completion.status != IBV_WC_SUCCESS) {
                throw std::runtime_error(
                    "CpuRoceTransceiver CallerDriven send CQE failed: "
                    + tx_cqe_summary(completion));
            }
            if (completion.opcode != expected_opcode) {
                throw std::runtime_error(
                    "CpuRoceTransceiver CallerDriven send CQE has an "
                    "unsupported opcode: "
                    + tx_cqe_summary(completion));
            }
            if (completion.wr_id >= caller_driven->tx_in_flight.size()
                || caller_driven->tx_in_flight[static_cast<std::uint32_t>(
                       completion.wr_id)]
                    == 0) {
                throw std::runtime_error(
                    "CpuRoceTransceiver CallerDriven TX completion does not "
                    "identify an in-flight slot: "
                    + tx_cqe_summary(completion));
            }
            caller_driven->tx_in_flight[static_cast<std::uint32_t>(
                completion.wr_id)]
                = 0;
            --caller_driven->sends_in_flight;
        }
        if (one_batch || count < kRxPollBatch) {
            return;
        }
    }
}

bool CpuRoceTransceiver::Impl::pull_caller_driven_receives()
{
    const int count = verbs->receive_completion_queue.poll(
        caller_driven->completions.data(), kRxPollBatch);
    if (count <= 0) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        const auto& completion
            = caller_driven->completions[static_cast<std::size_t>(i)];
        // CallerDriven rejects DMA-BUF, so every receive carries a local SGE
        // and wr_id is the posted slot.
        const auto decoded = detail::decode_rx_cqe(
            completion, config.rx.slot_count, false);
        if (decoded.status != detail::RxCqeDecode::Status::Ok) {
            throw_rx_cqe_error(decoded, completion);
        }
        caller_driven->held.push_back(
            CallerDriven::Held {
                decoded.payload_slot,
                static_cast<std::uint32_t>(decoded.recv_wr_id),
            });
    }
    return true;
}

// Returns true when a slot was handed to the caller. Sets reap_required
// when a held slot still has a send outstanding; the caller reaps once
// and retries. Slots with a clear send bit are eligible without a CQ poll.
bool CpuRoceTransceiver::Impl::claim_caller_driven_slot(
    std::uint32_t* out_slot, bool& reap_required)
{
    reap_required = false;
    for (auto it = caller_driven->held.begin(); it != caller_driven->held.end();
         ++it) {
        if (it->payload_slot >= caller_driven->tx_in_flight.size()) {
            throw std::runtime_error(
                "CpuRoceTransceiver CallerDriven payload slot is outside the "
                "TX ring");
        }
        if (caller_driven->tx_in_flight[it->payload_slot] != 0) {
            reap_required = true;
            continue;
        }
        if (CpuRoceTransceiver::load_flag_acquire(
                &tx_ring->flags()[it->payload_slot])
            != 0) {
            continue;
        }
        const auto payload_slot = it->payload_slot;
        const auto recv_wr_id = it->recv_wr_id;
        caller_driven->held.erase(it);
        post_receive_slot(*verbs, recv_wr_id);
        *out_slot = payload_slot;
        return true;
    }
    return false;
}

bool CpuRoceTransceiver::Impl::finish_caller_driven_claim(
    std::uint32_t* out_slot)
{
    bool reap_required = false;
    if (claim_caller_driven_slot(out_slot, reap_required)) {
        return true;
    }
    if (!reap_required) {
        return false;
    }
    reap_caller_driven_sends();
    return claim_caller_driven_slot(out_slot, reap_required);
}

detail::ThreadAffinityScope::ThreadAffinityScope(
    std::optional<int> cpu, bool best_effort)
{
    if (!cpu) {
        return;
    }

    if (sched_getaffinity(0, sizeof(original_), &original_) != 0) {
        const int error = errno;
        if (best_effort) {
            return;
        }
        throw std::runtime_error(
            "CpuRoceTransceiver failed to read thread affinity: "
            + std::generic_category().message(error));
    }
    if (const int error = try_pin_current_thread(*cpu); error != 0) {
        if (best_effort) {
            return;
        }
        throw std::runtime_error(
            "CpuRoceTransceiver failed to set thread affinity to CPU "
            + std::to_string(*cpu) + ": "
            + std::generic_category().message(error));
    }
    active_ = true;
}

detail::ThreadAffinityScope::~ThreadAffinityScope() noexcept
{
    if (active_) {
        (void)sched_setaffinity(0, sizeof(original_), &original_);
    }
}

void detail::ThreadAffinityScope::restore()
{
    if (!active_) {
        return;
    }
    if (sched_setaffinity(0, sizeof(original_), &original_) != 0) {
        const int error = errno;
        throw detail::ThreadAffinityRestoreError(
            "CpuRoceTransceiver failed to restore thread affinity: "
            + std::generic_category().message(error));
    }
    active_ = false;
}

std::thread detail::spawn_duplex_worker(detail::DuplexWorkerBody body)
{
    return std::thread(std::move(body));
}

void detail::run_duplex_workers(
    std::atomic<bool>& exit_flag,
    const detail::DuplexWorkerBody& rx_body,
    const detail::DuplexWorkerBody& tx_body,
    const detail::DuplexWorkerSpawn& spawn)
{
    std::mutex failure_mutex;
    std::exception_ptr first_failure;

    const auto capture = [&](std::exception_ptr failure) {
        {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (first_failure == nullptr) {
                first_failure = std::move(failure);
            }
        }
        // Release the peer worker from its poll loop before any join.
        exit_flag.store(true, std::memory_order_release);
    };

    std::thread rx_worker;
    std::thread tx_worker;
    try {
        rx_worker = spawn([&rx_body, &capture] {
            try {
                rx_body();
            } catch (...) {
                capture(std::current_exception());
            }
        });
        tx_worker = spawn([&tx_body, &capture] {
            try {
                tx_body();
            } catch (...) {
                capture(std::current_exception());
            }
        });
    } catch (...) {
        // A failed second spawn still has a running first worker to stop.
        capture(std::current_exception());
    }

    // Unconditional: the bodies reference locals of this frame, and a
    // joinable std::thread destructor would terminate. Do not catch join():
    // a throwing join may leave the thread joinable, and then ~thread
    // terminates instead of propagating.
    if (rx_worker.joinable()) {
        rx_worker.join();
    }
    if (tx_worker.joinable()) {
        tx_worker.join();
    }

    if (first_failure != nullptr) {
        std::rethrow_exception(first_failure);
    }
}

CpuRoceTransceiver::CpuRoceTransceiver(Config config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

CpuRoceTransceiver::~CpuRoceTransceiver() noexcept
{
    // Destruction before the monitor has stopped cannot safely release the
    // rings and verbs objects. close() therefore intentionally terminates via
    // this noexcept boundary if its threading precondition is violated.
    close();
}

void CpuRoceTransceiver::setup()
{
    if (impl_->closed.load(std::memory_order_acquire)) {
        throw std::logic_error("CpuRoceTransceiver cannot be set up after close()");
    }
    if (impl_->terminal_failure.load(std::memory_order_acquire)) {
        throw std::logic_error(
            "CpuRoceTransceiver cannot be reused after a monitor failure");
    }
    detail::validate_config(impl_->config, impl_->caller_post.configured);
    if (impl_->config.mode == Mode::Unified
        && impl_->unified_dispatch == nullptr) {
        throw std::invalid_argument(
            "CpuRoceTransceiver Unified requires a dispatch callback");
    }
    if (impl_->verbs != nullptr) {
        return;
    }

    initialize_fork_safety();

    // Build only the rings this mode uses. A ring left null means no MR and
    // no TX flags. Use depth one as the portable minimum for an inactive
    // direction. Forward has no dedicated TX ring but can have one
    // outstanding send, so it never builds a TxRing over RX storage.
    const Config& config = impl_->config;
    const auto layout = mode_layout(config.mode);
    std::unique_ptr<detail::RxRing> rx_ring;
    std::unique_ptr<detail::TxRing> tx_ring;
    if (layout.rx) {
        rx_ring = std::make_unique<detail::RxRing>(
            config.rx.payload_memory, config.rx.stride, config.rx.slot_count);
    }
    if (layout.tx) {
        tx_ring = std::make_unique<detail::TxRing>(
            config.tx.payload_memory, config.tx.stride, config.tx.slot_count,
            !impl_->caller_post.configured);
    }

    const bool handled_rx = rx_ring != nullptr
        && impl_->rx_completion_handler != nullptr;
    std::uint32_t receive_depth = 1;
    if (rx_ring != nullptr) {
        receive_depth = handled_rx ? impl_->receive_credit_count
                                   : rx_ring->slot_count();
    }
    const bool caller_post = impl_->caller_post.configured;
    const std::uint32_t send_sge = caller_post
        ? (impl_->caller_post.extra_host_bytes != 0 ? 2u : 1u)
        : 1u;
    auto verbs = std::make_unique<VerbsResources>(
        config, rx_ring.get(), tx_ring.get(), receive_depth, !handled_rx,
        handled_rx && impl_->host_metadata_requested, send_sge,
        caller_post ? impl_->caller_post.extra_host : nullptr,
        caller_post ? impl_->caller_post.extra_host_bytes : 0u,
        !caller_post);
    // Recv WQEs are valid in INIT. Post them before publishing the QP so a
    // failed fill does not leave get_qp_number() valid with a half-armed RQ,
    // and so the RQ has credit as soon as connect() reaches RTR.
    if (rx_ring != nullptr) {
        if (handled_rx) {
            fill_handled_receive_credits(*verbs, impl_->receive_credit_count);
        } else {
            fill_receive_queue(*verbs);
        }
    }

    // Allocate CallerDriven state before committing the QP. A throw here
    // destroys the local verbs object and leaves the instance unset, so a
    // later setup() retries instead of returning success with hooks that
    // cannot run.
    std::unique_ptr<Impl::CallerDriven> caller_driven;
    if (config.mode == Mode::CallerDriven) {
        caller_driven = std::make_unique<Impl::CallerDriven>();
        caller_driven->tx_in_flight.assign(config.tx.slot_count, 0);
        // One held completion per posted recv. Reserving here keeps rx_poll()
        // from allocating on the dispatcher thread.
        caller_driven->held.reserve(config.rx.slot_count);
    }

    impl_->rx_ring = std::move(rx_ring);
    impl_->tx_ring = std::move(tx_ring);
    impl_->verbs = std::move(verbs);
    impl_->caller_driven = std::move(caller_driven);
}

void CpuRoceTransceiver::connect(Peer peer)
{
    if (impl_->closed.load(std::memory_order_acquire)) {
        throw std::logic_error("CpuRoceTransceiver cannot connect after close()");
    }
    if (impl_->terminal_failure.load(std::memory_order_acquire)) {
        throw std::logic_error(
            "CpuRoceTransceiver cannot connect after a monitor failure");
    }
    if (impl_->verbs == nullptr) {
        throw std::logic_error(
            "CpuRoceTransceiver::setup must succeed before connect()");
    }
    if (impl_->verbs->connected) {
        throw std::logic_error("CpuRoceTransceiver is already connected");
    }
    // Transient ETIMEDOUT on INIT→RTR is retried inside init_to_rtr()
    // (five attempts, 200 ms apart). A returned error still forbids a later
    // connect(): a failed ibv_modify_qp can leave the QP indeterminate.
    // A throw from init_to_rtr is pre-modify and does not set this flag.
    if (impl_->verbs->connect_attempted) {
        throw std::logic_error(
            "CpuRoceTransceiver cannot retry a failed QP transition");
    }
    detail::validate_peer(impl_->config, peer);

    const auto remote_gid = ibv::ipv4_to_gid(peer.ip);
    const ::ibv_mtu forced_mtu = IBV_MTU_4096;
    ::ibv_mtu handled_mtu = IBV_MTU_256;
    const ::ibv_mtu* path_mtu = nullptr;
    if (impl_->config.rx_path_mtu_4096) {
        path_mtu = &forced_mtu;
    } else if (impl_->caller_post.configured && impl_->caller_post.clamp_path_mtu) {
        handled_mtu = clamped_path_mtu(
            impl_->verbs->local_gid.netdev, impl_->verbs->port.active_mtu);
        path_mtu = &handled_mtu;
    }
    // Throws from init_to_rtr are pre-modify; the QP stays INIT and a later
    // connect() is still allowed. A return means modify was attempted.
    const int rtr_error = impl_->verbs->queue_pair.init_to_rtr(
        impl_->config.ibv_port, impl_->verbs->port,
        impl_->verbs->local_gid.netdev, remote_gid, peer.qp,
        impl_->verbs->local_gid.index, path_mtu);
    impl_->verbs->connect_attempted = true;
    if (rtr_error != 0) {
        throw ibv::Error(detail::concatenate(
            "Failed to move CpuRoceTransceiver QP from INIT to RTR, "
            "ibv_modify_qp returned ",
            rtr_error, " (",
            std::generic_category().message(rtr_error), ")"));
    }
    if (const int rts_error = impl_->verbs->queue_pair.rtr_to_rts();
        rts_error != 0) {
        throw ibv::Error(detail::concatenate(
            "Failed to move CpuRoceTransceiver QP from RTR to RTS, "
            "ibv_modify_qp returned ",
            rts_error, " (",
            std::generic_category().message(rts_error), ")"));
    }

    impl_->config.peer = std::move(peer);
    impl_->verbs->path_mtu_payload = path_mtu != nullptr
        ? mtu_payload_bytes(*path_mtu)
        : mtu_payload_bytes(impl_->verbs->port.active_mtu);
    impl_->verbs->connected = true;
    if (impl_->config.mode == Mode::CallerDriven) {
        // Release-publish everything connect() wrote, including peer and
        // connected. Hooks acquire-load this flag instead of the plain bool.
        impl_->caller_driven_ready.store(true, std::memory_order_release);
    }
}

void CpuRoceTransceiver::start()
{
    setup();
    connect(impl_->config.peer);
}

void CpuRoceTransceiver::blocking_monitor()
{
    std::optional<detail::ThreadAffinityScope> affinity;
    {
        std::lock_guard<std::mutex> lock(impl_->teardown);
        // CallerDriven has no monitor thread. Return without consuming the
        // one-shot latch so the caller can drive rx_poll()/tx_publish(),
        // including after a second blocking_monitor().
        if (impl_->config.mode == Mode::CallerDriven) {
            if (impl_->terminal_failure.load(std::memory_order_acquire)) {
                throw std::logic_error(
                    "CpuRoceTransceiver cannot monitor after a datapath "
                    "failure");
            }
            if (impl_->closed.load(std::memory_order_relaxed)
                || impl_->exit_flag.load(std::memory_order_acquire)) {
                throw std::logic_error(
                    "CpuRoceTransceiver cannot monitor after close()");
            }
            if (!impl_->caller_driven_ready.load(std::memory_order_acquire)) {
                throw std::logic_error(
                    "CpuRoceTransceiver must be connected before "
                    "blocking_monitor()");
            }
            return;
        }
        if (impl_->caller_post.configured) {
            throw std::logic_error(
                "CpuRoceTransceiver caller transmit has no monitor thread");
        }
        if (impl_->closed.load(std::memory_order_relaxed)) {
            throw std::logic_error(
                "CpuRoceTransceiver cannot monitor after close()");
        }
        // Validate the mode before the one-shot latch and outside the try, so
        // a precondition error neither consumes the monitor nor terminalizes
        // the instance. The datapath switch below covers the same set; its
        // default is then unreachable.
        switch (impl_->config.mode) {
        case Mode::Rx:
        case Mode::Tx:
        case Mode::Duplex:
        case Mode::Forward:
        case Mode::Unified:
            break;
        default:
            throw std::logic_error(
                "CpuRoceTransceiver monitor mode is unhandled ("
                + std::to_string(static_cast<int>(impl_->config.mode))
                + ")");
        }
        if (impl_->monitor_started) {
            throw std::logic_error(
                "CpuRoceTransceiver::blocking_monitor is one-shot; close and "
                "recreate the transceiver");
        }
        if (impl_->verbs == nullptr || !impl_->verbs->connected) {
            throw std::logic_error(
                "CpuRoceTransceiver must be connected before blocking_monitor()");
        }
        // The connected check above implies setup() ran, which validated
        // Config, so Duplex has an empty poll_cpu. Pin before the latch so a
        // placement failure is retryable. close() may wait on this mutex for
        // the two startup syscalls.
        affinity.emplace(
            impl_->config.poll_cpu, impl_->config.poll_cpu_best_effort);
        impl_->monitor_started = true;
        impl_->monitor_active = true;
        impl_->monitor_thread = std::this_thread::get_id();
        impl_->monitor_state_changed.notify_all();
    }

    struct MonitorExit {
        Impl& impl;
        ~MonitorExit()
        {
            std::lock_guard<std::mutex> lock(impl.teardown);
            impl.monitor_active = false;
            impl.monitor_state_changed.notify_all();
        }
    } on_exit { *impl_ };

    try {
        if (impl_->config.mode == Mode::Duplex) {
            auto* const impl = impl_.get();
            const auto pin_or_throw = [](std::optional<int> cpu) {
                if (!cpu) {
                    return;
                }
                if (const int error = detail::try_pin_current_thread(*cpu);
                    error != 0) {
                    // Both workers can reach this concurrently, so format with
                    // the thread-safe category rather than std::strerror.
                    throw std::runtime_error(
                        "CpuRoceTransceiver failed to set worker affinity to CPU "
                        + std::to_string(*cpu) + ": "
                        + std::generic_category().message(error));
                }
            };
            detail::run_duplex_workers(
                impl->exit_flag,
                [impl, pin_or_throw] {
                    pin_or_throw(impl->config.rx_cpu);
                    run_rx_loop(
                        *impl->rx_ring, *impl->verbs, impl->config,
                        impl->exit_flag, impl->rx_collision_count);
                },
                [impl, pin_or_throw] {
                    pin_or_throw(impl->config.tx_cpu);
                    // RX is the sole async-event drainer in Duplex.
                    run_configured_tx(
                        *impl->tx_ring, *impl->verbs, impl->config,
                        impl->exit_flag, false);
                });
        } else {
            switch (impl_->config.mode) {
            case Mode::Forward:
                run_configured_forward(
                    *impl_->rx_ring, *impl_->verbs, impl_->config,
                    impl_->exit_flag);
                break;
            case Mode::Tx:
                // Tx-only is its own async-event drainer.
                run_configured_tx(
                    *impl_->tx_ring, *impl_->verbs, impl_->config,
                    impl_->exit_flag, true);
                break;
            case Mode::Rx:
                // Select the publication protocol once, outside both loops.
                if (impl_->rx_completion_handler != nullptr) {
                    run_handled_rx_loop(
                        impl_->exit_flag, *impl_->verbs, impl_->config,
                        impl_->rx_completion_handler,
                        impl_->rx_completion_context,
                        impl_->receive_credit_count);
                } else {
                    run_rx_loop(
                        *impl_->rx_ring, *impl_->verbs, impl_->config,
                        impl_->exit_flag, impl_->rx_collision_count);
                }
                break;
            case Mode::Unified:
                run_configured_unified(
                    *impl_->rx_ring, *impl_->tx_ring, *impl_->verbs,
                    impl_->config, impl_->unified_dispatch,
                    impl_->unified_context, impl_->exit_flag);
                break;
            default:
                // Unreachable: the prologue admits only the five modes above,
                // and Duplex took the branch before this switch.
                throw std::logic_error(
                    "CpuRoceTransceiver monitor mode is unhandled ("
                    + std::to_string(static_cast<int>(impl_->config.mode))
                    + ")");
            }
        }
    } catch (...) {
        // A failed completion, async event, or violated ring protocol makes
        // this one-shot datapath terminal. Never leave an RTS QP active
        // without its monitor, and never permit a restart on this instance.
        // This runs while monitor_active is still set. ~MonitorExit clears
        // that flag only after this call returns, so close() cannot destroy
        // the queue pair concurrently with the error transition.
        impl_->terminalize_queue_pair();
        throw;
    }
    // Outside the datapath catch: an OS restoration failure must not poison
    // the QP or be classified as a completion/async-event failure.
    affinity.value().restore();
}

void CpuRoceTransceiver::request_exit() noexcept
{
    // CallerDriven has no monitor thread. Setting the flag would make a later
    // blocking_monitor() throw as if close() had already run.
    if (impl_->config.mode == Mode::CallerDriven) {
        return;
    }
    impl_->exit_flag.store(true, std::memory_order_release);
}

void CpuRoceTransceiver::close()
{
    impl_->exit_flag.store(true, std::memory_order_release);
    impl_->caller_driven_ready.store(false, std::memory_order_release);
    std::unique_lock<std::mutex> lock(impl_->teardown);
    // The exit flag is already set, so an erroneous self-call still requests
    // shutdown; only the wait is refused. Waiting here would be permanent:
    // monitor_active is cleared by this very thread on its way out of
    // blocking_monitor().
    if (impl_->monitor_active
        && impl_->monitor_thread == std::this_thread::get_id()) {
        throw std::logic_error(
            "CpuRoceTransceiver::close must not be called from the thread "
            "running blocking_monitor(); return from it instead");
    }
    impl_->monitor_state_changed.wait(lock, [&] {
        return !impl_->monitor_active;
    });
    if (impl_->closed.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    // Join the CQ spinner (if any) before releasing the QP: close() vs
    // blocking_monitor() is the one supported concurrent pair.
    impl_->verbs.reset();
    impl_->tx_ring.reset();
    impl_->rx_ring.reset();
}

bool CpuRoceTransceiver::rx_poll(std::uint32_t* out_slot)
{
    if (out_slot == nullptr) {
        throw std::invalid_argument(
            "CpuRoceTransceiver rx_poll requires an output slot");
    }
    if (impl_->config.mode != Mode::CallerDriven) {
        throw std::logic_error(
            "CpuRoceTransceiver rx_poll is CallerDriven-only");
    }
    if (!impl_->caller_driven_ready.load(std::memory_order_acquire)) {
        return false;
    }
    try {
        impl_->note_caller_driven_gid();
        // Count this call before claiming, so a fatal async event is not
        // reported after the recv WQE has already been re-armed.
        if (++impl_->caller_driven->calls_since_async
            >= 2 * kCallerDrivenAsyncCheckInterval) {
            impl_->drain_caller_driven_async();
        }
        if (impl_->finish_caller_driven_claim(out_slot)) {
            return true;
        }
        if (!impl_->pull_caller_driven_receives()) {
            impl_->reap_caller_driven_sends(/*one_batch=*/true);
            if (impl_->caller_driven->calls_since_async
                >= kCallerDrivenAsyncCheckInterval) {
                impl_->drain_caller_driven_async();
            }
            return false;
        }
        return impl_->finish_caller_driven_claim(out_slot);
    } catch (const std::logic_error&) {
        throw;
    } catch (...) {
        impl_->terminalize_queue_pair();
        throw;
    }
}

bool CpuRoceTransceiver::tx_publish(std::uint32_t slot)
{
    if (impl_->config.mode != Mode::CallerDriven) {
        throw std::logic_error(
            "CpuRoceTransceiver tx_publish is CallerDriven-only");
    }
    if (!impl_->caller_driven_ready.load(std::memory_order_acquire)) {
        return false;
    }
    try {
        auto& caller_driven = *impl_->caller_driven;
        auto& verbs = *impl_->verbs;
        const auto& config = impl_->config;
        impl_->note_caller_driven_gid();
        if (slot >= caller_driven.tx_in_flight.size()) {
            throw std::out_of_range(
                "CpuRoceTransceiver tx_publish slot is out of range");
        }
        if (caller_driven.tx_in_flight[slot] != 0) {
            impl_->reap_caller_driven_sends();
            if (caller_driven.tx_in_flight[slot] != 0) {
                throw std::logic_error(
                    "CpuRoceTransceiver tx_publish slot still has a send in "
                    "flight");
            }
        }

        const auto length = static_cast<std::uint32_t>(
            config.tx.frame_layout.payload_size);
        auto sge = verbs.make_tx_sge(slot, length);
        int error = 0;
        if (config.tx_operation
            == CpuRoceTransceiver::TxOperation::WriteWithImm) {
            auto request = ibv::WriteWithImmWorkRequest(
                slot,
                &sge,
                config.peer.rx_base
                    + static_cast<std::uint64_t>(slot) * config.peer.rx_stride,
                config.peer.rx_rkey,
                htonl(slot));
            request.inline_data(verbs.tx_inline);
            error = verbs.queue_pair.post_send(request);
        } else {
            auto request = ibv::SendWorkRequest(slot, &sge);
            request.inline_data(verbs.tx_inline);
            error = verbs.queue_pair.post_send(request);
        }
        if (error != 0) {
            throw ibv::Error(detail::concatenate(
                "Failed to post CpuRoceTransceiver CallerDriven send for slot ",
                slot, ", ibv_post_send returned ", error, " (",
                std::generic_category().message(error), ")"));
        }
        caller_driven.tx_in_flight[slot] = 1;
        ++caller_driven.sends_in_flight;
        return true;
    } catch (const std::logic_error&) {
        // Includes std::out_of_range: a bad slot is the caller's error and
        // does not poison the QP.
        throw;
    } catch (...) {
        impl_->terminalize_queue_pair();
        throw;
    }
}

void detail::CpuRoceRxCompletionHandler::enable_handled_receive(
    Fn fn, void* context, std::uint32_t receive_credit_count)
{
    auto& impl = *transceiver_.impl_;
    if (impl.config.mode != CpuRoceTransceiver::Mode::Rx) {
        throw std::logic_error(
            "CpuRoceTransceiver handled receive requires Mode::Rx");
    }
    if (fn == nullptr) {
        throw std::invalid_argument(
            "CpuRoceTransceiver handled receive callback must not be null");
    }
    if (receive_credit_count == 0
        || receive_credit_count > kMaxHandledReceiveCredits) {
        throw std::invalid_argument(
            "CpuRoceTransceiver receive credit count must be from 1 to "
            + std::to_string(kMaxHandledReceiveCredits));
    }
    std::lock_guard<std::mutex> lock(impl.teardown);
    if (impl.verbs != nullptr || impl.monitor_started
        || impl.closed.load(std::memory_order_relaxed)
        || impl.terminal_failure.load(std::memory_order_relaxed)) {
        throw std::logic_error(
            "CpuRoceTransceiver handled receive must be enabled before setup");
    }
    if (impl.rx_completion_handler != nullptr) {
        throw std::logic_error(
            "CpuRoceTransceiver handled receive is already enabled");
    }
    impl.rx_completion_handler = fn;
    impl.rx_completion_context = context;
    impl.receive_credit_count = receive_credit_count;
}

void detail::CpuRoceRxCompletionHandler::request_host_metadata()
{
    auto& impl = *transceiver_.impl_;
    std::lock_guard<std::mutex> lock(impl.teardown);
    if (impl.verbs != nullptr || impl.monitor_started
        || impl.closed.load(std::memory_order_relaxed)
        || impl.terminal_failure.load(std::memory_order_relaxed)) {
        throw std::logic_error(
            "CpuRoceTransceiver host metadata must be requested before setup");
    }
    impl.host_metadata_requested = true;
}

const std::uint8_t* detail::CpuRoceRxCompletionHandler::host_metadata() const
{
    auto& impl = *transceiver_.impl_;
    std::lock_guard<std::mutex> lock(impl.teardown);
    return impl.verbs != nullptr
            && impl.verbs->host_trailer_key != nullptr
        ? static_cast<const std::uint8_t*>(impl.verbs->host_trailer_key->trailers.get())
        : nullptr;
}

bool detail::CpuRoceMonitorTestAccess::wait_until_active(
    CpuRoceTransceiver& transceiver, int timeout_ms)
{
    if (transceiver.impl_ == nullptr) {
        return false;
    }
    std::unique_lock<std::mutex> lock(transceiver.impl_->teardown);
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(timeout_ms);
    return transceiver.impl_->monitor_state_changed.wait_until(
        lock, deadline, [&] { return transceiver.impl_->monitor_active; });
}

void CpuRoceTransceiver::set_unified_dispatch(
    UnifiedDispatchFn fn, void* context)
{
    if (impl_->closed.load(std::memory_order_acquire)) {
        throw std::logic_error(
            "CpuRoceTransceiver cannot set dispatch after close()");
    }
    if (impl_->terminal_failure.load(std::memory_order_acquire)) {
        throw std::logic_error(
            "CpuRoceTransceiver cannot set dispatch after a monitor failure");
    }
    if (impl_->verbs != nullptr) {
        throw std::logic_error(
            "CpuRoceTransceiver dispatch must be set before setup()");
    }
    if (impl_->config.mode != Mode::Unified) {
        throw std::logic_error(
            "CpuRoceTransceiver dispatch callback is Unified-only");
    }
    if (fn == nullptr) {
        throw std::invalid_argument(
            "CpuRoceTransceiver Unified dispatch callback must not be null");
    }
    impl_->unified_dispatch = fn;
    impl_->unified_context = context;
}

void CpuRoceTransceiver::set_local_ip(std::string local_ip)
{
    if (impl_->closed.load(std::memory_order_acquire)) {
        throw std::logic_error(
            "CpuRoceTransceiver cannot set local_ip after close()");
    }
    if (impl_->terminal_failure.load(std::memory_order_acquire)) {
        throw std::logic_error(
            "CpuRoceTransceiver cannot set local_ip after a monitor failure");
    }
    if (impl_->verbs != nullptr) {
        throw std::logic_error(
            "CpuRoceTransceiver local_ip must be set before setup()");
    }
    if (!local_ip.empty() && !is_ipv4_address(local_ip)) {
        throw std::invalid_argument(
            "CpuRoceTransceiver local_ip must be a valid IPv4 address");
    }
    impl_->config.local_ip = std::move(local_ip);
}

// Resource accessors below lock teardown because monitor failure and close()
// replace their pointers. The collision counter is an independent atomic
// diagnostic and deliberately does not take that lock.
std::uint32_t CpuRoceTransceiver::get_qp_number() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->verbs && impl_->verbs->queue_pair
        ? impl_->verbs->queue_pair.ptr_->qp_num
        : 0;
}

std::uint32_t CpuRoceTransceiver::get_path_mtu() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    // Recorded only after RTS, so a failed connect still reports zero.
    return impl_->verbs && impl_->verbs->connected
        ? impl_->verbs->path_mtu_payload
        : 0;
}

std::uint32_t CpuRoceTransceiver::get_rkey() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    if (impl_->verbs == nullptr) {
        return 0;
    }
    if (impl_->verbs->host_trailer_key != nullptr) {
        return impl_->verbs->host_trailer_key->rkey;
    }
    return impl_->verbs->peer_visible_rx_memory
        ? impl_->verbs->peer_visible_rx_memory.ptr_->rkey
        : 0;
}

std::uint64_t CpuRoceTransceiver::external_frame_memory() const
{
    return 0;
}

std::size_t CpuRoceTransceiver::get_rx_ring_stride() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->rx_ring ? impl_->rx_ring->stride() : 0;
}

std::uint32_t CpuRoceTransceiver::get_rx_ring_slot_count() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->rx_ring ? impl_->rx_ring->slot_count() : 0;
}

std::size_t CpuRoceTransceiver::get_tx_ring_stride() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->tx_ring ? impl_->tx_ring->stride() : 0;
}

std::uint32_t CpuRoceTransceiver::get_tx_ring_slot_count() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->tx_ring ? impl_->tx_ring->slot_count() : 0;
}

std::uint64_t CpuRoceTransceiver::get_rx_collision_count() const noexcept
{
    return impl_->rx_collision_count.load(std::memory_order_relaxed);
}

std::uint8_t* CpuRoceTransceiver::get_rx_ring_data_addr() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->rx_ring ? impl_->rx_ring->data() : nullptr;
}

std::uint64_t* CpuRoceTransceiver::get_rx_ring_flag_addr() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->rx_ring ? impl_->rx_ring->flags() : nullptr;
}

std::uint8_t* CpuRoceTransceiver::get_tx_ring_data_addr() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->tx_ring ? impl_->tx_ring->data() : nullptr;
}

std::uint64_t* CpuRoceTransceiver::get_tx_ring_flag_addr() const
{
    std::lock_guard<std::mutex> lock(impl_->teardown);
    return impl_->tx_ring ? impl_->tx_ring->flags() : nullptr;
}

void detail::CpuRoceCallerTx::configure(Config config)
{
    auto& impl = *transceiver_.impl_;
    std::lock_guard<std::mutex> lock(impl.teardown);
    if (impl.closed.load(std::memory_order_relaxed)
        || impl.terminal_failure.load(std::memory_order_relaxed)) {
        throw std::logic_error(
            "CpuRoceCallerTx::configure cannot run after close or a "
            "terminal failure");
    }
    if (impl.verbs != nullptr) {
        throw std::logic_error(
            "CpuRoceCallerTx::configure must run before setup()");
    }
    if (impl.caller_post.configured) {
        throw std::logic_error("CpuRoceCallerTx is already configured");
    }
    if (impl.config.mode != CpuRoceTransceiver::Mode::Tx) {
        throw std::invalid_argument("CpuRoceCallerTx requires Tx mode");
    }
    if (impl.config.tx_operation != CpuRoceTransceiver::TxOperation::Send) {
        throw std::invalid_argument("CpuRoceCallerTx requires Send");
    }
    if (config.extra_host_bytes != 0 && config.extra_host == nullptr) {
        throw std::invalid_argument(
            "CpuRoceCallerTx extra host bytes need a buffer");
    }
    impl.caller_post.configured = true;
    impl.caller_post.extra_host = config.extra_host;
    impl.caller_post.extra_host_bytes = config.extra_host_bytes;
    impl.caller_post.clamp_path_mtu = config.clamp_path_mtu;
}

bool detail::CpuRoceCallerTx::configured() const
{
    return transceiver_.impl_->caller_post.configured;
}

std::uint32_t detail::CpuRoceCallerTx::post(
    std::uint32_t slot, std::uint32_t length, std::uint64_t wr_id,
    const std::uint8_t* extra, std::uint32_t extra_length)
{
    auto& impl = *transceiver_.impl_;
    if (!impl.caller_post.configured) {
        throw std::logic_error("CpuRoceCallerTx is not configured");
    }
    // The owner thread is the only caller. close() runs after it has
    // stopped, so this load is the verbs object for the rest of the call.
    auto* verbs = impl.verbs.get();
    auto* tx_ring = impl.tx_ring.get();
    if (verbs == nullptr || !verbs->queue_pair || !verbs->connected
        || tx_ring == nullptr) {
        throw std::logic_error("CpuRoceCallerTx::post is not connected");
    }
    if (slot >= tx_ring->slot_count()) {
        throw std::out_of_range("CpuRoceCallerTx::post slot is out of range");
    }
    const auto capacity = impl.config.tx.frame_layout.payload_size;
    if (length == 0 || static_cast<std::size_t>(length) > capacity) {
        throw std::invalid_argument(
            "CpuRoceCallerTx::post length is outside the slot");
    }
    ibv::ScatterGatherElement sges[2] = {
        verbs->make_tx_sge(slot, length),
        ibv::ScatterGatherElement(0, 0, 0),
    };
    int num_sge = 1;
    if (extra != nullptr || extra_length != 0) {
        if (extra == nullptr || extra_length == 0
            || verbs->extra_host_base == nullptr
            || verbs->extra_host_bytes == 0
            || verbs->extra_memory.ptr_ == nullptr) {
            throw std::invalid_argument(
                "CpuRoceCallerTx::post has no extra host region");
        }
        const auto address = reinterpret_cast<std::uintptr_t>(extra);
        const auto base = reinterpret_cast<std::uintptr_t>(
            verbs->extra_host_base);
        const auto bytes = static_cast<std::uintptr_t>(
            verbs->extra_host_bytes);
        if (address < base
            || static_cast<std::uintptr_t>(extra_length) > bytes
            || address - base > bytes - extra_length) {
            throw std::invalid_argument(
                "CpuRoceCallerTx::post extra entry is outside the "
                "registered host region");
        }
        sges[1] = ibv::ScatterGatherElement(
            address, extra_length, verbs->extra_memory.ptr_->lkey);
        num_sge = 2;
    }
    const std::uint64_t message_bytes = static_cast<std::uint64_t>(length)
        + (num_sge == 2 ? sges[1].length : 0);
    if (message_bytes > impl.config.peer.rx_stride) {
        throw std::invalid_argument(
            "CpuRoceCallerTx::post message exceeds the peer RX stride");
    }
    // Signaled SEND, inline off. This work request is not the one the
    // flag, forward, or unified loops reuse.
    auto request = ibv::SendWorkRequest(wr_id, sges);
    request.num_sge = num_sge;
    const int post_error = verbs->queue_pair.post_send(request);
    if (post_error != 0) {
        impl.terminalize_queue_pair();
        throw ibv::Error(detail::concatenate(
            "Failed to post CpuRoceCallerTx send for slot ", slot,
            ", ibv_post_send returned ", post_error, " (",
            std::generic_category().message(post_error), ")"));
    }
    const std::uint32_t mtu = verbs->path_mtu_payload;
    const std::uint64_t packets = mtu == 0
        ? 1
        : (message_bytes + static_cast<std::uint64_t>(mtu) - 1u) / mtu;
    constexpr std::uint32_t kPsnMask = 0x00FFFFFFu;
    const std::uint32_t start = impl.caller_post.next_psn;
    const std::uint32_t last = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(start) + packets - 1u) & kPsnMask);
    impl.caller_post.next_psn = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(start) + packets) & kPsnMask);
    return last;
}

int detail::CpuRoceCallerTx::poll(Completion* out, int max_entries)
{
    if (out == nullptr || max_entries <= 0) {
        throw std::invalid_argument("CpuRoceCallerTx::poll needs a buffer");
    }
    auto& impl = *transceiver_.impl_;
    if (!impl.caller_post.configured) {
        throw std::logic_error("CpuRoceCallerTx is not configured");
    }
    // Re-read verbs. An errored queue pair is still here, so flush
    // completions remain readable. A destroyed one has nothing left.
    auto* verbs = impl.verbs.get();
    if (verbs == nullptr || !verbs->queue_pair) {
        return 0;
    }
    std::array<::ibv_wc, 16> completions;
    int count = 0;
    const int cap = max_entries < 16 ? max_entries : 16;
    try {
        count = verbs->send_completion_queue.poll(completions.data(), cap);
    } catch (...) {
        impl.terminalize_queue_pair();
        throw;
    }
    bool terminal = false;
    for (int i = 0; i < count; ++i) {
        const auto& completion = completions[static_cast<std::size_t>(i)];
        Status status = Status::Error;
        if (completion.status == IBV_WC_SUCCESS) {
            status = Status::Ok;
        } else if (completion.status == IBV_WC_WR_FLUSH_ERR) {
            status = Status::Flush;
        } else {
            terminal = true;
        }
        out[i] = Completion {
            completion.wr_id,
            status,
            status == Status::Ok && completion.opcode == IBV_WC_SEND,
        };
    }
    if (terminal) {
        impl.terminalize_queue_pair();
    }
    return count;
}

void detail::CpuRoceCallerTx::drain_async()
{
    auto& impl = *transceiver_.impl_;
    if (!impl.caller_post.configured) {
        throw std::logic_error("CpuRoceCallerTx is not configured");
    }
    std::exception_ptr fatal;
    {
        std::lock_guard<std::mutex> lock(impl.teardown);
        if (impl.verbs == nullptr) {
            return;
        }
        try {
            check_fatal_async_events(*impl.verbs, impl.config);
        } catch (...) {
            fatal = std::current_exception();
        }
    }
    if (fatal) {
        impl.terminalize_queue_pair();
        std::rethrow_exception(fatal);
    }
}

bool detail::CpuRoceCallerTx::queue_pair_destroyed() const
{
    auto& impl = *transceiver_.impl_;
    std::lock_guard<std::mutex> lock(impl.teardown);
    if (!impl.caller_post.configured) {
        return false;
    }
    return impl.verbs == nullptr || !impl.verbs->queue_pair;
}

void detail::CpuRoceCallerTx::fail()
{
    auto& impl = *transceiver_.impl_;
    if (!impl.caller_post.configured
        || impl.config.mode != CpuRoceTransceiver::Mode::Tx
        || impl.config.tx_operation
            != CpuRoceTransceiver::TxOperation::Send) {
        throw std::logic_error(
            "CpuRoceCallerTx::fail requires a configured caller-post "
            "transmit path");
    }
    impl.terminalize_queue_pair();
}

} // namespace hololink::transport::roce
