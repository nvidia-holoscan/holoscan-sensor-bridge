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
    packetizer_program_ = std::make_shared<NullPacketizerProgram>();
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

    // Enable the current packetizer program.
    packetizer_program_->enable(*hololink_, sif_address_);
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

    // Disable the packetizer program.
    packetizer_program_->disable(*hololink_, sif_address_);
}

void DataChannel::configure_socket(int socket_fd, uint16_t udp_port)
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
        address.sin_port = htons(udp_port);
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

void DataChannel::set_packetizer_program(std::shared_ptr<PacketizerProgram> program)
{
    if (!program) {
        throw std::runtime_error("Invalid packetizer program");
    }
    packetizer_program_ = program;
}

} // namespace hololink
