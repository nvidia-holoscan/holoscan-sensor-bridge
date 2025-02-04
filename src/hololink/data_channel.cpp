/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "data_channel.hpp"

#include <arpa/inet.h>

#include <hololink/logging.hpp>

namespace hololink {

namespace {

    // This memory map used by the Enumerator is only supported on CPNX FPGAs that are this
    // version or newer.
    constexpr int64_t MINIMUM_CPNX_VERSION = 0x2410;

    // Distance between sensor configuration blocks
    constexpr uint32_t SENSOR_CONFIGURATION_SIZE = 0x40;

    /** Hololink-lite data plane configuration is implied by the value
     * passed in the bootp transaction_id field, which is coopted
     * by FPGA to imply which port is publishing the request.  We use
     * that port ID to figure out what the address of the port's
     * configuration data is; which is the value listed here.
     */
    struct HololinkChannelConfiguration {
        uint32_t configuration_address;
        uint32_t vip_mask;
    };
    static const std::map<int, HololinkChannelConfiguration> BOOTP_TRANSACTION_ID_MAP {
        { 0, HololinkChannelConfiguration { 0x02000000, 0x1 } },
        { 1, HololinkChannelConfiguration { 0x02010000, 0x2 } },
    };

} // anonymous namespace

DataChannel::DataChannel(const Metadata& metadata, const std::function<std::shared_ptr<Hololink>(const Metadata& metadata)>& create_hololink)
{
    auto cpnx_version = metadata.get<int64_t>("cpnx_version"); // or None
    if (!cpnx_version) {
        throw UnsupportedVersion("No 'cpnx_version' field found.");
    }
    if (cpnx_version.value() < MINIMUM_CPNX_VERSION) {
        throw UnsupportedVersion(fmt::format("cpnx_version={:#X}; minimum supported version={:#X}.",
            cpnx_version.value(), MINIMUM_CPNX_VERSION));
    }
    hololink_ = create_hololink(metadata);
    network_configuration_address_ = metadata.get<int64_t>("configuration_address").value();
    peer_ip_ = metadata.get<std::string>("peer_ip").value();
    vip_mask_ = metadata.get<int64_t>("vip_mask").value();
    data_plane_ = metadata.get<int64_t>("data_plane").value();
    sensor_ = metadata.get<int64_t>("sensor").value();
    sensor_configuration_address_ = SENSOR_CONFIGURATION_SIZE * sensor_;
    qp_number_ = 0;
    rkey_ = 0;
    multicast_ = "";
    multicast_port_ = 0;
    broadcast_port_ = 0;
    auto multicast = metadata.get<std::string>("multicast");
    auto multicast_port = metadata.get<int64_t>("multicast_port");
    auto broadcast_port = metadata.get<int64_t>("broadcast_port");
    if (broadcast_port) {
        broadcast_port_ = static_cast<uint16_t>(broadcast_port.value());
        HSB_LOG_INFO(fmt::format("DataChannel broadcast port={}.", broadcast_port_));
    } else if (multicast && multicast_port) {
        multicast_ = multicast.value();
        multicast_port_ = static_cast<uint16_t>(multicast_port.value());
        HSB_LOG_INFO(fmt::format("DataChannel multicast address={} port={}.", multicast_, multicast_port_));
    }
}

/*static*/ bool DataChannel::enumerated(const Metadata& metadata)
{
    if (!metadata.get<int64_t>("configuration_address")) {
        return false;
    }
    if (!metadata.get<std::string>("peer_ip")) {
        return false;
    }
    if (!metadata.get<int64_t>("data_plane")) {
        return false;
    }
    return Hololink::enumerated(metadata);
}

/* static */ void DataChannel::use_multicast(Metadata& metadata, std::string address, uint16_t port)
{
    metadata["multicast"] = address;
    metadata["multicast_port"] = static_cast<int64_t>(port);
}

/* static */ void DataChannel::use_broadcast(Metadata& metadata, uint16_t port)
{
    metadata["broadcast_port"] = static_cast<int64_t>(port);
}

std::shared_ptr<Hololink> DataChannel::hololink() const { return hololink_; }

const std::string& DataChannel::peer_ip() const { return peer_ip_; }

void DataChannel::authenticate(uint32_t qp_number, uint32_t rkey)
{
    qp_number_ = qp_number;
    rkey_ = rkey;
}

static uint32_t compute_payload_size(uint32_t frame_size)
{
    const uint32_t mtu = 1472; // TCP/IP illustrated vol 1 (1994), section 11.6, page 151
    const uint32_t header_size = 78;
    const uint32_t page_size = hololink::native::PAGE_SIZE;
    const uint32_t payload_size = ((mtu - header_size + page_size - 1) / page_size) * page_size;
    const uint64_t packets = (frame_size + payload_size - 1) / payload_size; // round up
    HSB_LOG_INFO(
        "header_size={} payload_size={} packets={}", header_size, payload_size, packets);
    return payload_size;
}

void DataChannel::configure(uint64_t frame_memory, size_t frame_size, size_t page_size, unsigned pages, uint32_t local_data_port)
{
    // Contract enforcement
    if (frame_memory & (hololink::native::PAGE_SIZE - 1)) {
        throw std::runtime_error(fmt::format("frame_memory={:#x} must be {}-byte aligned.", frame_memory, hololink::native::PAGE_SIZE));
    }
    if (page_size & (hololink::native::PAGE_SIZE - 1)) {
        throw std::runtime_error(fmt::format("page_size={:#x} must be {}-byte aligned.", page_size, hololink::native::PAGE_SIZE));
    }
    uint32_t aligned_frame_size = hololink::native::round_up(frame_size, hololink::native::PAGE_SIZE);
    uint32_t metadata_size = hololink::native::PAGE_SIZE;
    uint32_t aligned_frame_with_metadata = aligned_frame_size + metadata_size;
    if (page_size < aligned_frame_with_metadata) {
        throw std::runtime_error(fmt::format("page_size={:#x} must be at least {:#x} bytes.", page_size, aligned_frame_with_metadata));
    }
    if (pages > 4) {
        throw std::runtime_error(fmt::format("pages={} can be at most 4.", pages));
    }
    if (pages < 1) {
        throw std::runtime_error(fmt::format("pages={} must be at least 1.", pages));
    }

    // Ok, we're good.
    uint32_t payload_size = compute_payload_size(frame_size);

    const std::string& peer_ip = this->peer_ip();
    auto [local_ip, local_device, local_mac] = native::local_ip_and_mac(peer_ip);

    // Data plane destination addresses
    uint32_t mac_high = (local_mac[0] << 8) | (local_mac[1] << 0);
    uint32_t mac_low
        = ((local_mac[2] << 24) | (local_mac[3] << 16) | (local_mac[4] << 8) | (local_mac[5] << 0));
    in_addr_t ip = inet_network(local_ip.c_str());
    uint32_t udp_port = local_data_port;

    // Override those if we're using multicast or broadcast.
    if (broadcast_port_) {
        ip = INADDR_BROADCAST;
        uint8_t broadcast_mac[] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
        mac_high = (broadcast_mac[0] << 8) | (broadcast_mac[1] << 0);
        mac_low = ((broadcast_mac[2] << 24) | (broadcast_mac[3] << 16) | (broadcast_mac[4] << 8) | (broadcast_mac[5] << 0));
        udp_port = static_cast<uint32_t>(broadcast_port_);
    } else if (!multicast_.empty()) {
        ip = inet_network(multicast_.c_str());
        uint8_t broadcast_mac[] = { 0x01, 0x00, 0x5E, static_cast<uint8_t>((ip >> 16) & 0x7F), static_cast<uint8_t>((ip >> 8) & 0xFF), static_cast<uint8_t>((ip >> 0) & 0xFF) };
        mac_high = (broadcast_mac[0] << 8) | (broadcast_mac[1] << 0);
        mac_low = ((broadcast_mac[2] << 24) | (broadcast_mac[3] << 16) | (broadcast_mac[4] << 8) | (broadcast_mac[5] << 0));
        udp_port = static_cast<uint32_t>(multicast_port_);
    }

    // Clearing DP_VIP_MASK should be unnecessary-- we should only
    // be here following a reset, but be defensive and make
    // sure we're not transmitting anything while we update.
#define PAGES(x) ((x) >> 7)
    hololink_->and_uint32(network_configuration_address_ + DP_VIP_MASK, ~vip_mask_);
    hololink_->write_uint32(network_configuration_address_ + DP_PACKET_SIZE, PAGES(payload_size));
    // Write the addresses for the pages we'll receive into.
    hololink_->write_uint32(sensor_configuration_address_ + DP_QP, qp_number_);
    hololink_->write_uint32(sensor_configuration_address_ + DP_RKEY, rkey_);
    hololink_->write_uint32(sensor_configuration_address_ + DP_ADDRESS_0, (pages > 0) ? PAGES(frame_memory) : 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_ADDRESS_1, (pages > 1) ? PAGES(frame_memory + page_size) : 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_ADDRESS_2, (pages > 2) ? PAGES(frame_memory + (page_size * 2)) : 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_ADDRESS_3, (pages > 3) ? PAGES(frame_memory + (page_size * 3)) : 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_BUFFER_LENGTH, frame_size);
    hololink_->write_uint32(sensor_configuration_address_ + DP_BUFFER_MASK, (1 << pages) - 1);
    hololink_->write_uint32(sensor_configuration_address_ + DP_HOST_MAC_LOW, mac_low);
    hololink_->write_uint32(sensor_configuration_address_ + DP_HOST_MAC_HIGH, mac_high);
    hololink_->write_uint32(sensor_configuration_address_ + DP_HOST_IP, ip);
    hololink_->write_uint32(sensor_configuration_address_ + DP_HOST_UDP_PORT, udp_port);
    // 0x1 meaning to connect sensor 1 to the current ethernet port
    hololink_->or_uint32(network_configuration_address_ + DP_VIP_MASK, vip_mask_);
}

void DataChannel::unconfigure()
{
    // This stops transmission.
    hololink_->and_uint32(network_configuration_address_ + DP_VIP_MASK, ~vip_mask_);
    //
    hololink_->write_uint32(sensor_configuration_address_ + DP_BUFFER_MASK, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_BUFFER_LENGTH, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_QP, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_RKEY, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_ADDRESS_0, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_ADDRESS_1, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_ADDRESS_2, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_ADDRESS_3, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_HOST_MAC_LOW, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_HOST_MAC_HIGH, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_HOST_IP, 0);
    hololink_->write_uint32(sensor_configuration_address_ + DP_HOST_UDP_PORT, 0);
    //
}

void DataChannel::configure_socket(int socket_fd)
{
    const std::string& peer_ip = this->peer_ip();
    auto [local_ip, local_device, local_mac] = native::local_ip_and_mac(peer_ip);

    if (broadcast_port_) {
        //
        HSB_LOG_INFO(fmt::format("DataChannel configure_socket socket_fd={} broadcast_port={} local_ip={} local_device={}.", socket_fd, broadcast_port_, local_ip, local_device));
        int on = 1;
        // Allow us to receive broadcast.
        int r = setsockopt(socket_fd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
        if (r != 0) {
            throw std::runtime_error(fmt::format("SO_BROADCAST failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        // Allow other programs to receive these broadcast packets.
        r = setsockopt(socket_fd, SOL_SOCKET, SO_REUSEPORT, &on, sizeof(on));
        if (r != 0) {
            throw std::runtime_error(fmt::format("SO_REUSEPORT failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        // We use INADDR_ANY to receive but listen only to the specific receiver interface.
        r = setsockopt(socket_fd, SOL_SOCKET, SO_BINDTODEVICE, local_device.data(), local_device.size());
        if (r != 0) {
            throw std::runtime_error(fmt::format("SO_BINDTODEVICE failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        // Configure what we listen to.
        sockaddr_in address {};
        address.sin_family = AF_INET;
        address.sin_port = htons(broadcast_port_);
        address.sin_addr.s_addr = INADDR_ANY;
        if (bind(socket_fd, (sockaddr*)&address, sizeof(address)) < 0) {
            throw std::runtime_error(fmt::format("bind failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        HSB_LOG_INFO(fmt::format("(done) DataChannel configure_socket socket_fd={} multicast_address={}.", socket_fd, multicast_));
    } else if (!multicast_.empty()) {
        //
        HSB_LOG_INFO(fmt::format("DataChannel configure_socket socket_fd={} multicast_address={} multicast_port={} local_ip={} local_device={}.", socket_fd, multicast_, multicast_port_, local_ip, local_device));
        struct ip_mreq mreq = {}; // fill with zeros.
        int r = inet_aton(multicast_.c_str(), &mreq.imr_multiaddr);
        if (r) {
            r = inet_aton(local_ip.c_str(), &mreq.imr_interface);
        }
        if (!r) {
            throw std::runtime_error(fmt::format("DataChannel failed socket configuration for multicast={}.", multicast_));
        }
        r = setsockopt(socket_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));
        if (r) {
            throw std::runtime_error(fmt::format("DataChannel failed to set socket configuration for multicast={}, errno={}.", multicast_, errno));
        }
        sockaddr_in address {};
        address.sin_family = AF_INET;
        address.sin_port = htons(multicast_port_);
        if (inet_pton(AF_INET, multicast_.c_str(), &address.sin_addr) != 1) {
            throw std::runtime_error(
                fmt::format("Failed to convert address {}", multicast_));
        }

        if (bind(socket_fd, (sockaddr*)&address, sizeof(address)) < 0) {
            throw std::runtime_error(fmt::format("bind failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        HSB_LOG_INFO(fmt::format("(done) DataChannel configure_socket socket_fd={} multicast_address={}.", socket_fd, multicast_));
    } else {
        // Not multicast; use bind(local_ip,0) so that the kernel assigns us a UDP port.
        sockaddr_in address {};
        address.sin_family = AF_INET;
        address.sin_port = htons(0);
        if (inet_pton(AF_INET, local_ip.c_str(), &address.sin_addr) != 1) {
            throw std::runtime_error(
                fmt::format("Failed to convert address {}", local_ip));
        }

        if (bind(socket_fd, (sockaddr*)&address, sizeof(address)) < 0) {
            throw std::runtime_error(fmt::format("bind failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        HSB_LOG_INFO(fmt::format("(done) DataChannel configure_socket socket_fd={} local_ip={}.", socket_fd, local_ip));
    }
}

/* static */ void DataChannel::use_data_plane_configuration(Metadata& metadata, int64_t data_plane)
{
    auto channel_configuration = BOOTP_TRANSACTION_ID_MAP.find(data_plane);
    if (channel_configuration == BOOTP_TRANSACTION_ID_MAP.cend()) {
        throw std::runtime_error(fmt::format("use_data_plane failed, data_plane={} is out-of-range.", data_plane));
    }
    HSB_LOG_TRACE(fmt::format("data_plane={}", data_plane));
    metadata["configuration_address"] = channel_configuration->second.configuration_address;
}

/* static */ void DataChannel::use_sensor(Metadata& metadata, int64_t sensor_number)
{
    auto channel_configuration = BOOTP_TRANSACTION_ID_MAP.find(sensor_number);
    if (channel_configuration == BOOTP_TRANSACTION_ID_MAP.cend()) {
        throw std::runtime_error(fmt::format("use_sensor failed, sensor_number={} is out-of-range.", sensor_number));
    }
    HSB_LOG_TRACE(fmt::format("sensor_number={}", sensor_number));
    metadata["sensor"] = sensor_number;
    metadata["vip_mask"] = channel_configuration->second.vip_mask;
}

} // namespace hololink
