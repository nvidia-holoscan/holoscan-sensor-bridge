/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <thread>

#include "logging_internal.hpp"

namespace hololink {

namespace {

    // This memory map used by DataChannel is only supported on FPGAs that are this
    // version or newer.
    constexpr int64_t MINIMUM_HSB_IP_VERSION = 0x2501;

} // anonymous namespace

DataChannel::DataChannel(const Metadata& metadata, const std::function<std::shared_ptr<Hololink>(const Metadata& metadata)>& create_hololink)
{
    auto hsb_ip_version = metadata.get<int64_t>("hsb_ip_version"); // or None
    if (!hsb_ip_version) {
        throw UnsupportedVersion("No 'hsb_ip_version' field found.");
    }
    if (hsb_ip_version.value() < MINIMUM_HSB_IP_VERSION) {
        throw UnsupportedVersion(fmt::format("hsb_ip_version={:#X}; minimum supported version={:#X}.",
            hsb_ip_version.value(), MINIMUM_HSB_IP_VERSION));
    }
    auto hololink = create_hololink(metadata);
    initialize(metadata, hololink);
}

// This protected constructor is only valid for programming tools which
// circumvent version checking done above.
DataChannel::DataChannel(const Metadata& metadata, std::shared_ptr<Hololink> hololink)
{
    initialize(metadata, hololink);
}

void DataChannel::initialize(const Metadata& metadata, std::shared_ptr<Hololink> hololink)
{
    enumeration_metadata_ = metadata;
    hololink_ = hololink;
    peer_ip_ = metadata.get<std::string>("peer_ip").value();
    vp_mask_ = metadata.get<int64_t>("vp_mask").value();
    data_plane_ = metadata.get<int64_t>("data_plane").value();
    sensor_ = metadata.get<int64_t>("sensor").value();
    sif_address_ = metadata.get<int64_t>("sif_address").value();
    vp_address_ = metadata.get<int64_t>("vp_address").value();
    hif_address_ = metadata.get<int64_t>("hif_address").value();
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
    frame_end_event_valid_ = false;
    frame_end_event_ = static_cast<Hololink::Event>(0); // not used when frame_end_event_valid_ is false
    auto frame_end_event = metadata.get<int64_t>("frame_end_event");
    if (frame_end_event) {
        frame_end_event_valid_ = true;
        frame_end_event_ = static_cast<Hololink::Event>(frame_end_event.value());
    }
    fpga_uuid_ = metadata.get<std::string>("fpga_uuid").value();
}

/*static*/ bool DataChannel::enumerated(const Metadata& metadata)
{
    if (!metadata.get<int64_t>("hif_address")) {
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

static uint32_t compute_payload_size(uint32_t frame_size, uint32_t header_size)
{
    const uint32_t mtu = 1472; // TCP/IP illustrated vol 1 (1994), section 11.6, page 151
    const uint32_t page_size = hololink::core::PAGE_SIZE;
    uint32_t payload_size = ((mtu - header_size + page_size - 1) / page_size) * page_size;
    if (payload_size > mtu) {
        /* Max payload size is 1408 bytes. */
        payload_size = 1408;
    }
    const uint64_t packets = (frame_size + payload_size - 1) / payload_size; // round up
    HSB_LOG_INFO(
        "header_size={} payload_size={} packets={}", header_size, payload_size, packets);
    return payload_size;
}

void DataChannel::configure_common(uint32_t frame_size, uint32_t header_size, uint32_t local_data_port)
{
    uint32_t payload_size = compute_payload_size(frame_size, header_size);

    const std::string& peer_ip = this->peer_ip();
    auto [local_ip, local_device, local_mac] = core::local_ip_and_mac(peer_ip);

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

#define PAGES(x) ((x) >> 7)
    hololink_->write_uint32(hif_address_ + DP_PACKET_SIZE, PAGES(payload_size));
    hololink_->write_uint32(hif_address_ + DP_PACKET_UDP_PORT, DATA_SOURCE_UDP_PORT);
    hololink_->write_uint32(vp_address_ + DP_BUFFER_LENGTH, frame_size);
    hololink_->write_uint32(vp_address_ + DP_HOST_MAC_LOW, mac_low);
    hololink_->write_uint32(vp_address_ + DP_HOST_MAC_HIGH, mac_high);
    hololink_->write_uint32(vp_address_ + DP_HOST_IP, ip);
    hololink_->write_uint32(vp_address_ + DP_HOST_UDP_PORT, udp_port);
}

void DataChannel::configure_roce(uint64_t frame_memory, size_t frame_size, size_t page_size, unsigned pages, uint32_t local_data_port)
{
    // Contract enforcement
    if (frame_memory & (hololink::core::PAGE_SIZE - 1)) {
        throw std::runtime_error(fmt::format("frame_memory={:#x} must be {}-byte aligned.", frame_memory, hololink::core::PAGE_SIZE));
    }
    if (page_size & (hololink::core::PAGE_SIZE - 1)) {
        throw std::runtime_error(fmt::format("page_size={:#x} must be {}-byte aligned.", page_size, hololink::core::PAGE_SIZE));
    }
    size_t aligned_frame_size = hololink::core::round_up(frame_size, hololink::core::PAGE_SIZE);
    size_t metadata_size = hololink::core::PAGE_SIZE;
    size_t aligned_frame_with_metadata = aligned_frame_size + metadata_size;
    if (page_size < aligned_frame_with_metadata) {
        throw std::runtime_error(fmt::format("page_size={:#x} must be at least {:#x} bytes.", page_size, aligned_frame_with_metadata));
    }
    if (pages > 4) {
        throw std::runtime_error(fmt::format("pages={} can be at most 4.", pages));
    }
    if (pages < 1) {
        throw std::runtime_error(fmt::format("pages={} must be at least 1.", pages));
    }
    size_t highest_address = frame_memory + page_size * pages;
    if (PAGES(highest_address) > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(fmt::format("highest_address={:#x}; pages={:#x} cannot fit in 32-bits.", highest_address, PAGES(highest_address)));
    }

    // Clearing DP_VP_MASK should be unnecessary-- we should only
    // be here following a reset, but be defensive and make
    // sure we're not transmitting anything while we update.
    hololink_->and_uint32(hif_address_ + DP_VP_MASK, ~vp_mask_);

    const uint32_t header_size = 78;
    configure_common(frame_size, header_size, local_data_port);
    hololink_->write_uint32(vp_address_ + DP_QP, qp_number_);
    hololink_->write_uint32(vp_address_ + DP_RKEY, rkey_);
    hololink_->write_uint32(vp_address_ + DP_ADDRESS_0, (pages > 0) ? PAGES(frame_memory) : 0);
    hololink_->write_uint32(vp_address_ + DP_ADDRESS_1, (pages > 1) ? PAGES(frame_memory + page_size) : 0);
    hololink_->write_uint32(vp_address_ + DP_ADDRESS_2, (pages > 2) ? PAGES(frame_memory + (page_size * 2)) : 0);
    hololink_->write_uint32(vp_address_ + DP_ADDRESS_3, (pages > 3) ? PAGES(frame_memory + (page_size * 3)) : 0);
    hololink_->write_uint32(vp_address_ + DP_BUFFER_MASK, (1 << pages) - 1);

    // Restore the DP_VP_MASK to re-enable the sensor.
    hololink_->or_uint32(hif_address_ + DP_VP_MASK, vp_mask_);
}

void DataChannel::configure_coe(uint8_t channel, size_t frame_size, uint32_t pixel_width, bool vlan_enabled)
{
    HSB_LOG_INFO("Enabling 1722 COE: channel={}, frame_size={}, pixel_width={}", channel, frame_size, pixel_width);

    if (channel >= 64) {
        throw std::runtime_error(fmt::format("channel={} must be less than 64.", channel));
    }

    if (vlan_enabled) {
        /* No known HSB version supports VLAN at the moment. */
        throw std::runtime_error("VLAN is not supported");
    }

    // Clearing DP_VP_MASK should be unnecessary-- we should only
    // be here following a reset, but be defensive and make
    // sure we're not transmitting anything while we update.
    hololink_->and_uint32(hif_address_ + DP_VP_MASK, ~vp_mask_);

    size_t header_size = 46U; // 14 Ethernet + 12 avtpdu + 12 gisf + 8 coe
    if (vlan_enabled) {
        header_size += 4U; // 4 VLAN header
    }
    configure_common(frame_size, header_size);
    const uint32_t enable_1722B = 1;
    uint32_t line_threshold_log2 = 0;
    while (true) {
        if ((1u << line_threshold_log2) >= pixel_width) {
            break;
        }
        line_threshold_log2++;
        if (line_threshold_log2 > 127) {
            throw std::runtime_error(fmt::format("pixel_width={} is too large.", pixel_width));
        }
    }
    uint32_t cfg_1722B = (enable_1722B << 24) | (line_threshold_log2 << 25) | channel;
    hololink_->write_uint32(vp_address_ + DP_QP, cfg_1722B);

    // Restore the DP_VP_MASK to re-enable the sensor.
    hololink_->or_uint32(hif_address_ + DP_VP_MASK, vp_mask_);
}

void DataChannel::unconfigure()
{
    if (fpga_uuid_ != LEOPARD_EAGLE_UUID) { // for all boards that are not leopard eagle

        // This stops transmission.
        hololink_->and_uint32(hif_address_ + DP_VP_MASK, ~vp_mask_);
        // Let any in-transit data flush out.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // Clear the ROCE configuration.
        hololink_->write_uint32(vp_address_ + DP_BUFFER_MASK, 0);
        hololink_->write_uint32(vp_address_ + DP_BUFFER_LENGTH, 0);
        hololink_->write_uint32(vp_address_ + DP_QP, 0);
        hololink_->write_uint32(vp_address_ + DP_RKEY, 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_0, 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_1, 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_2, 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_3, 0);
        hololink_->write_uint32(vp_address_ + DP_HOST_MAC_LOW, 0);
        hololink_->write_uint32(vp_address_ + DP_HOST_MAC_HIGH, 0);
        hololink_->write_uint32(vp_address_ + DP_HOST_IP, 0);
        hololink_->write_uint32(vp_address_ + DP_HOST_UDP_PORT, 0);
    } else {
        // skip for now as this causes to lose ping on vb1940-aio
    }
}

void DataChannel::configure_socket(int socket_fd)
{
    const std::string& peer_ip = this->peer_ip();
    auto [local_ip, local_device, local_mac] = core::local_ip_and_mac(peer_ip);

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
    auto fpga_uuid = metadata.get<std::string>("fpga_uuid").value();
    EnumerationStrategy& enumeration_strategy = hololink::Enumerator::get_uuid_strategy(fpga_uuid);
    enumeration_strategy.use_data_plane_configuration(metadata, data_plane);
}

/* static */ void DataChannel::use_sensor(Metadata& metadata, int64_t sensor_number)
{
    auto fpga_uuid = metadata.get<std::string>("fpga_uuid").value();
    EnumerationStrategy& enumeration_strategy = hololink::Enumerator::get_uuid_strategy(fpga_uuid);
    enumeration_strategy.use_sensor(metadata, sensor_number);
}

void DataChannel::disable_packetizer()
{
    hololink_->write_uint32(sif_address_ + PACKETIZER_MODE, 0);
}

void DataChannel::enable_packetizer_10()
{
    const uint32_t base = sif_address_;
    hololink_->write_uint32(base + PACKETIZER_MODE, 0x11f1fff7);
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000000); // RAM: 0 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000010); // RAM: 1 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AA6AA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000020); // RAM: 2 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x02020000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000040); // RAM: 4 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x05000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000050); // RAM: 5 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00001865); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000060); // RAM: 6 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20300000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000070); // RAM: 7 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x1C007061); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000090); // RAM: 9 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x40044400); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a0); // RAM: 10 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x50000600); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b0); // RAM: 11 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20802080); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c0); // RAM: 12 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x9362E000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d0); // RAM: 13 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01B18626); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000140); // RAM: 20 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000001); // RAM: 0 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAABFE); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000011); // RAM: 1 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000081); // RAM: 8 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00010000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000091); // RAM: 9 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x440CCC81); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a1); // RAM: 10 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01555600); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b1); // RAM: 11 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x18006085); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c1); // RAM: 12 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x9362E000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d1); // RAM: 13 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01B10724); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000f1); // RAM: 15 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00222200); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000101); // RAM: 16 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xA802A800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000111); // RAM: 17 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xE0000002); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000121); // RAM: 18 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x6DB0F000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000131); // RAM: 19 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x0000C001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000141); // RAM: 20 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000002); // RAM: 0 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAABFE); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000012); // RAM: 1 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AA6AA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000022); // RAM: 2 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000032); // RAM: 3 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x11110000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000042); // RAM: 4 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x05000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000052); // RAM: 5 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00001865); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000062); // RAM: 6 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20300000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000072); // RAM: 7 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x1C007061); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000092); // RAM: 9 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00111100); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a2); // RAM: 10 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xA9557800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b2); // RAM: 11 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xD80C668D); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c2); // RAM: 12 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xDB63E001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d2); // RAM: 13 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01C18022); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000f2); // RAM: 15 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00222200); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000102); // RAM: 16 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xA802A800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000112); // RAM: 17 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xE0000002); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000122); // RAM: 18 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x6DB0F000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000132); // RAM: 19 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x0000C001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000142); // RAM: 20 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000003); // RAM: 0 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x155AA6A9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000013); // RAM: 1 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x15555555); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000023); // RAM: 2 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00010000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000033); // RAM: 3 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00444401); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000043); // RAM: 4 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x02AAA000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000053); // RAM: 5 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x3C00C108); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000063); // RAM: 6 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x9B7FC05C); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000073); // RAM: 7 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x03624AC0); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a3); // RAM: 10 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x05555000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b3); // RAM: 11 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xC0000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c3); // RAM: 12 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x013FE01F); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d3); // RAM: 13 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00018206); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000143); // RAM: 20 ELEMENT:3
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000004); // RAM: 0 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x155AA6A9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000014); // RAM: 1 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x15555555); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000024); // RAM: 2 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00800000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000034); // RAM: 3 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00444480); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000044); // RAM: 4 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xF2A15000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000054); // RAM: 5 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xB006840F); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000064); // RAM: 6 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x0B7FC07F); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000074); // RAM: 7 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x038344CC); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a4); // RAM: 10 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x05555000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b4); // RAM: 11 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xC0000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c4); // RAM: 12 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x013FE01F); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d4); // RAM: 13 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00018206); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000144); // RAM: 20 ELEMENT:4
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000005); // RAM: 0 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x155AAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000015); // RAM: 1 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x15555555); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000035); // RAM: 3 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00444400); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000045); // RAM: 4 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xA2A00000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000055); // RAM: 5 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xF000820A); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000065); // RAM: 6 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x64BA1FFF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000075); // RAM: 7 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x6360C8C3); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000085); // RAM: 8 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x40400000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000095); // RAM: 9 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x44226600); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a5); // RAM: 10 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x50501000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b5); // RAM: 11 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xE0C04145); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c5); // RAM: 12 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x06DF807F); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d5); // RAM: 13 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00003C78); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000145); // RAM: 20 ELEMENT:5
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000006); // RAM: 0 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AA6A9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000016); // RAM: 1 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x15555555); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000026); // RAM: 2 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x40400000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000036); // RAM: 3 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x88191100); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000046); // RAM: 4 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000056); // RAM: 5 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x80000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000066); // RAM: 6 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x66F9DFFF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000076); // RAM: 7 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x63E3CCCF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000096); // RAM: 9 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00222200); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a6); // RAM: 10 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x55500000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b6); // RAM: 11 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xC0000005); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c6); // RAM: 12 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xB7FEE1FF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d6); // RAM: 13 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00318607); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000146); // RAM: 20 ELEMENT:6
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000007); // RAM: 0 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000017); // RAM: 1 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x15555555); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000037); // RAM: 3 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00222200); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000047); // RAM: 4 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x50005000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000057); // RAM: 5 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xC0180C11); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000067); // RAM: 6 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00A01FFF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000077); // RAM: 7 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x7FE0F033); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000087); // RAM: 8 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20200000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000097); // RAM: 9 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x44440000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a7); // RAM: 10 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x50001000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b7); // RAM: 11 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xE0C020C0); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c7); // RAM: 12 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xB05807FF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d7); // RAM: 13 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x003078F9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000147); // RAM: 20 ELEMENT:7
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000008); // RAM: 0 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AA6A9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000018); // RAM: 1 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x155556A9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000028); // RAM: 2 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20200000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000038); // RAM: 3 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00222200); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000048); // RAM: 4 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x52054000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000058); // RAM: 5 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2003C31F); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000068); // RAM: 6 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xD03FDFE0); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000078); // RAM: 7 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x60E0CC8A); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b8); // RAM: 11 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xC0000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c8); // RAM: 12 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x9362FFFF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d8); // RAM: 13 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01B18626); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000148); // RAM: 20 ELEMENT:8
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000009); // RAM: 0 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000019); // RAM: 1 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x155556A9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000039); // RAM: 3 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00222200); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000049); // RAM: 4 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x05555000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000059); // RAM: 5 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00180804); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000069); // RAM: 6 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xB66FDFFC); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000079); // RAM: 7 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x7CE0F073); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000089); // RAM: 8 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x10100000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000099); // RAM: 9 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x22113300); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a9); // RAM: 10 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xAD02A800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b9); // RAM: 11 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20006187); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c9); // RAM: 12 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xEB3FE7FE); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d9); // RAM: 13 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01B1F8DB); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000149); // RAM: 20 ELEMENT:9
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000000a); // RAM: 0 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000001a); // RAM: 1 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x155556AA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000002a); // RAM: 2 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x10100000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000003a); // RAM: 3 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x44440000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000004a); // RAM: 4 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x07555000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000005a); // RAM: 5 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2000C30A); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000006a); // RAM: 6 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x403FDFC0); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000007a); // RAM: 7 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x6000C082); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000009a); // RAM: 9 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00555500); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000aa); // RAM: 10 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xF802F800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000ba); // RAM: 11 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x26003043); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000ca); // RAM: 12 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xDB63FFFE); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000da); // RAM: 13 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01C18022); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000014a); // RAM: 20 ELEMENT:10
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000000b); // RAM: 0 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000001b); // RAM: 1 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x155AA6A9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000003b); // RAM: 3 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00111100); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000004b); // RAM: 4 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x05555000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000006b); // RAM: 6 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x49BFDF00); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000007b); // RAM: 7 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x7CE0F072); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000008b); // RAM: 8 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x08080000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000009b); // RAM: 9 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x22220000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000ab); // RAM: 10 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x07AAA800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000bb); // RAM: 11 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20006185); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000cb); // RAM: 12 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x35FFE600); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000db); // RAM: 13 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01B1DA9F); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000014b); // RAM: 20 ELEMENT:11
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000000c); // RAM: 0 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000001c); // RAM: 1 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x155AA6AA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000002c); // RAM: 2 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x08080000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000003c); // RAM: 3 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00111100); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000004c); // RAM: 4 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x55500000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000005c); // RAM: 5 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000005); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000006c); // RAM: 6 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x4DBF1F00); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000007c); // RAM: 7 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x7C00F872); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000ac); // RAM: 10 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x52ABF800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000bc); // RAM: 11 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x21801005); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000cc); // RAM: 12 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x05BFFFC0); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000dc); // RAM: 13 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01C1A266); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000014c); // RAM: 20 ELEMENT:12
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000000d); // RAM: 0 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000001d); // RAM: 1 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AA6A9); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000003d); // RAM: 3 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00111100); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000004d); // RAM: 4 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x55500000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000005d); // RAM: 5 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00200045); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000006d); // RAM: 6 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x24300000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000007d); // RAM: 7 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x1CC07061); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000008d); // RAM: 8 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x04040000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000009d); // RAM: 9 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x11089980); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000ad); // RAM: 10 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xA8280800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000bd); // RAM: 11 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x0001882A); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000cd); // RAM: 12 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xB25D1000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000dd); // RAM: 13 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x31B06461); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000014d); // RAM: 20 ELEMENT:13
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000000e); // RAM: 0 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000001e); // RAM: 1 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2A9AA6AA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000002e); // RAM: 2 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x04040000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000003e); // RAM: 3 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x22220000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000006e); // RAM: 6 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20300000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000007e); // RAM: 7 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x1C007061); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000009e); // RAM: 9 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x002AAA80); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000ae); // RAM: 10 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xAAA80000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000be); // RAM: 11 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20000002); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000ce); // RAM: 12 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xB37CF000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000de); // RAM: 13 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x31F1E667); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000014e); // RAM: 20 ELEMENT:14
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000000f); // RAM: 0 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000001f); // RAM: 1 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000008f); // RAM: 8 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x02020000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000009f); // RAM: 9 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x11554400); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000af); // RAM: 10 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x78005800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000bf); // RAM: 11 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x0601B459); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000cf); // RAM: 12 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x80501000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000df); // RAM: 13 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x3FF07819); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x0000014f); // RAM: 20 ELEMENT:15
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
}

void DataChannel::enable_packetizer_12()
{
    const uint32_t base = sif_address_;
    hololink_->write_uint32(base + PACKETIZER_MODE, 0x11210007);
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000000); // RAM: 0 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xAAAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000010); // RAM: 1 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x5AA5AAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000030); // RAM: 3 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00111100); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000040); // RAM: 4 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x55500000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000050); // RAM: 5 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000005); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000060); // RAM: 6 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x6DBC3000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000070); // RAM: 7 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00008103); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000090); // RAM: 9 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x44226600); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a0); // RAM: 10 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xA8002800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b0); // RAM: 11 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x0000020A); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c0); // RAM: 12 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x02400000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d0); // RAM: 13 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000810); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000140); // RAM: 20 ELEMENT:0
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000001); // RAM: 0 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xAAAAAAFF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000011); // RAM: 1 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xA5AAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000021); // RAM: 2 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x02020000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000041); // RAM: 4 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x05000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000051); // RAM: 5 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00001865); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000061); // RAM: 6 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x60300000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000071); // RAM: 7 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x0000F1E3); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000081); // RAM: 8 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x40400000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000091); // RAM: 9 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00111100); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a1); // RAM: 10 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x78005000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b1); // RAM: 11 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00001455); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000f1); // RAM: 15 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00222200); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000101); // RAM: 16 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xA802A800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000111); // RAM: 17 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xF0000002); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000121); // RAM: 18 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x6DB0F000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000141); // RAM: 20 ELEMENT:1
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000002); // RAM: 0 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xAAAAAAAF); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000012); // RAM: 1 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0xAAAAAAAA); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000092); // RAM: 9 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x004CCC80); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000a2); // RAM: 10 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x50000000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000b2); // RAM: 11 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00002080); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000c2); // RAM: 12 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x92406000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000d2); // RAM: 13 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000408); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000e2); // RAM: 14 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x20200000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x000000f2); // RAM: 15 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x22002200); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000102); // RAM: 16 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x28002800); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000112); // RAM: 17 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x30001860); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000122); // RAM: 18 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x01B03000); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000132); // RAM: 19 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x0000078F); // DATA
    hololink_->write_uint32(base + PACKETIZER_RAM, 0x00000142); // RAM: 20 ELEMENT:2
    hololink_->write_uint32(base + PACKETIZER_DATA, 0x00000001); // DATA
}

class FrameEndSequencer
    : public Hololink::Sequencer {
public:
    FrameEndSequencer(std::shared_ptr<Hololink> hololink, Hololink::Event event)
        : Hololink::Sequencer()
        , hololink_(hololink)
        , event_(event)
    {
    }

    void enable() override
    {
        done();
        assign_location(event_);
        write(*hololink_);
        hololink_->configure_apb_event(event_, location());
    }

public:
    std::shared_ptr<Hololink> hololink_;
    Hololink::Event event_;
};

std::shared_ptr<Hololink::Sequencer> DataChannel::frame_end_sequencer()
{
    if (!frame_end_event_valid_) {
        throw std::runtime_error(fmt::format("frame_end_sequencer isn't available for sensor={}.", sensor_));
    }
    auto r = std::make_shared<FrameEndSequencer>(hololink_, frame_end_event_);
    return r;
}

} // namespace hololink
